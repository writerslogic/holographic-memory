# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Official-protocol reader inputs carry no evidence labels and never cover held-out questions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import longmemeval_official_qa as qa


def entry():
    return {"question_id": "q", "question": "Which hostel?", "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_dates": ["2023/05/02 (Tue) 09:00", "2023/04/01 (Sat) 08:00"],
            "haystack_session_ids": ["answer_s1", "s2"],
            "haystack_sessions": [
                [{"role": "user", "content": "u0", "has_answer": True},
                 {"role": "assistant", "content": "a1"},
                 {"role": "user", "content": "u2"}],
                [{"role": "user", "content": "v0"}, {"role": "assistant", "content": "b1", "has_answer": True}]]}


class OfficialPromptTest(unittest.TestCase):
    def test_rounds_are_label_free_date_sorted_and_input_is_untouched(self):
        raw = entry()
        before = copy.deepcopy(raw)
        prompt = qa.official_prompt(raw, ["answer_s1_1", "noans_s1_3", "s2_1"], topk=3)
        self.assertEqual(raw, before)
        self.assertNotIn("has_answer", prompt)
        history = prompt.split("History Chats:\n\n", 1)[1].split("\n\nCurrent Date:", 1)[0]
        self.assertLess(history.index('"v0"'), history.index('"u0"'))
        self.assertIn(json.dumps([{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a1"}]), history)
        self.assertIn(json.dumps([{"role": "user", "content": "u2"}]), history)
        self.assertIn(json.dumps([{"role": "user", "content": "v0"}, {"role": "assistant", "content": "b1"}]), history)

    def test_topk_keeps_the_highest_ranked_rounds(self):
        prompt = qa.official_prompt(entry(), ["s2_1", "answer_s1_1", "noans_s1_3"], topk=1)
        self.assertIn('"v0"', prompt)
        self.assertNotIn('"u0"', prompt)

    def test_ties_ignore_the_label_bearing_id_text(self):
        ids = ["answer_s1_1", "noans_s1_3", "sharegpt_x_1", "s2_1"]
        ranks = qa._label_free_ranks({t: 1.0 for t in ids})
        relabeled = qa._label_free_ranks({t: 1.0 for t in reversed(ids)})
        self.assertEqual(ranks, relabeled)
        hashed = sorted(ids, key=lambda t: qa.hashlib.sha256(t.encode()).digest())
        self.assertEqual([t for t, _ in sorted(ranks.items(), key=lambda item: item[1])], hashed)

    def test_held_out_questions_are_refused(self):
        heldout = json.loads(qa.SPLIT.read_text())["heldout"][0]
        with self.assertRaises(ValueError):
            qa.rankings([{"question_id": heldout}], 0.0)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _SyncClient:
    def __init__(self):
        self.sent = []
        self.chat = _Obj(completions=_Obj(create=self.create))

    def create(self, **kwargs):
        self.sent.append(kwargs)
        usage = _Obj(prompt_tokens=100, completion_tokens=5, prompt_tokens_details=_Obj(cached_tokens=0))
        return _Obj(usage=usage, model=kwargs["model"],
                    choices=[_Obj(message=_Obj(content=" sync answer "))])


def _output_line(custom_id, body):
    if body is None:
        return {"custom_id": custom_id, "response": {"status_code": 500, "body": {"error": {"message": "boom"}}}}
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": body}}


class _BatchClient:
    """Batch API stand-in: answers every request in the input file except those whose prompt is in `fail`."""

    def __init__(self, fail=()):
        self.fail, self.uploads, self.created, self.store = set(fail), [], 0, {}
        self.files = _Obj(create=self.upload, content=lambda fid: _Obj(text=self.store[fid]))
        self.batches = _Obj(create=self.create_batch, retrieve=self.retrieve)

    def upload(self, file, purpose):
        assert purpose == "batch"
        self.uploads.append(file[1])
        self.store["in"] = file[1].decode()
        return _Obj(id="in")

    def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        assert (endpoint, completion_window) == ("/v1/chat/completions", "24h")
        self.created += 1
        out = []
        for raw in self.store[input_file_id].splitlines():
            request = json.loads(raw)
            prompt = request["body"]["messages"][0]["content"]
            body = None if prompt in self.fail else {
                "model": request["body"]["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
                "choices": [{"message": {"content": f" batch {prompt} "}}]}
            out.append(json.dumps(_output_line(request["custom_id"], body)))
        self.store["out"] = "\n".join(out) + "\n"
        return _Obj(id=f"batch_{self.created}")

    def retrieve(self, batch_id):
        return _Obj(id=batch_id, status="completed", request_counts=None, output_file_id="out",
                    error_file_id=None, errors=None)


class BatchModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.saved = qa.CACHE, qa.LEDGER, qa.BATCH_STATE
        qa.CACHE, qa.LEDGER, qa.BATCH_STATE = tmp / "calls", tmp / "spend.json", tmp / "batches"

    def tearDown(self):
        qa.CACHE, qa.LEDGER, qa.BATCH_STATE = self.saved
        self.tmp.cleanup()

    def test_batch_body_is_the_synchronous_request_and_lands_in_its_cache_entry(self):
        prompt = qa.official_prompt(entry(), ["answer_s1_1", "s2_1"], topk=2)
        for model, max_tokens in (("gpt-4o-2024-08-06", qa.GEN_LENGTH), (qa.JUDGE_MODEL, 10)):
            lines, ids = qa.batch_lines(model, {"q": prompt}, max_tokens, repeat=3)
            client = _SyncClient()
            qa.call(client, "t", model, prompt, max_tokens, repeat=3)
            self.assertEqual(json.loads(lines[0])["body"], client.sent[0])
            self.assertEqual(json.loads(lines[0])["url"], "/v1/chat/completions")
            self.assertTrue((qa.CACHE / f"{ids['r0000']}.json").exists())
            self.assertEqual(qa.batch_lines(model, {"q": prompt}, max_tokens, repeat=3), ([], {}))

    def test_failed_lines_stop_the_run_and_a_rerun_resubmits_only_them(self):
        client = _BatchClient(fail={"p2"})
        prompts = {"a": "p1", "b": "p2"}
        model = "gpt-4o-2024-08-06"
        with self.assertRaises(SystemExit):
            qa.batch_stage(client, "t", "reader", model, prompts, 800, 0, 60, sync_retry=False)
        ledger = json.loads(qa.LEDGER.read_text())
        self.assertEqual(ledger["batches"][0]["succeeded"], 1)
        self.assertEqual(ledger["batches"][0]["failed"], 1)
        expected = 100 * qa.BATCH_PRICES[model][0] + 5 * qa.BATCH_PRICES[model][2]
        self.assertAlmostEqual(ledger["total_usd"], expected)
        self.assertAlmostEqual(ledger["by_run_usd"]["t"], expected)
        client.fail.clear()
        out = qa.batch_stage(client, "t", "reader", model, prompts, 800, 0, 60, sync_retry=False)
        self.assertEqual(client.created, 2)
        self.assertEqual(client.uploads[1].decode().count("\n"), 1)
        self.assertEqual({k: v["text"] for k, v in out.items()}, {"a": "batch p1", "b": "batch p2"})
        self.assertTrue(all(v["batch"] for v in out.values()))
        self.assertAlmostEqual(json.loads(qa.LEDGER.read_text())["total_usd"], 2 * expected)

    def test_sync_retry_answers_failed_lines_at_the_full_rate(self):
        batch = _BatchClient(fail={"p2"})
        sync = _SyncClient()
        api = _Obj(files=batch.files, batches=batch.batches, chat=sync.chat)
        out = qa.batch_stage(api, "t", "judge", qa.JUDGE_MODEL, {"a": "p1", "b": "p2"}, 10, 0, 60, sync_retry=True)
        self.assertEqual(out["b"]["text"], "sync answer")
        self.assertEqual(len(sync.sent), 1)
        self.assertNotIn("batch", out["b"])


if __name__ == "__main__":
    unittest.main()
