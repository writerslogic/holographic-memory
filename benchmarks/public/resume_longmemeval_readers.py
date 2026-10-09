# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["openai==3.27.0", "python-dotenv==1.2.4", "tiktoken==0.14.0", "tokenizers==0.23.3"]
# ///
"""Retry unchanged failed dev readers after external API credit restoration."""
import argparse
import copy
import json
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from dotenv import load_dotenv

import longmemeval_reader_api as reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.attempt <= 6:
        parser.error("attempt must be between 1 and 6")
    load_dotenv(Path.home() / ".env", override=True)
    protocol = reader.load_protocol()
    report = json.loads(reader.RESULT.read_text())
    assert report["protocol_sha256"] == reader.sha(reader.PROTOCOL)
    tasks = []
    for row in report["rows"]:
        for name in protocol["arms"]:
            previous = row["systems"]["matched_gpt54"]["arms"][name]["reader"]
            if previous["status"] == "pending" or (previous["status"] == "error" and
                    previous.get("call", {}).get("status") in ("error", "budget_exhausted")):
                tasks.append((row, name))
    random.Random(20261009).shuffle(tasks)
    api = reader.client()
    report["artifacts"] = [a for a in report["artifacts"] if a["path"] != __file__]
    report["artifacts"].append({"path": __file__, "sha256": reader.sha(__file__)})

    def run(task):
        row, name = task
        arm = row["systems"]["matched_gpt54"]["arms"][name]
        assert arm["reader_input_sha256"] == reader.digest(arm["reader_input"])
        req = reader.request(protocol, arm["reader_input"], "reader")
        old = arm["reader"].get("call")
        if old is not None:
            assert old["request"] == req
        call_id = "reader:" + row["qid"] + ":" + name + ":restored-credit-attempt:" + str(args.attempt)
        return row, name, reader.api_call(api, protocol, req, call_id)

    def record(result):
        row, name, call = result
        arm = row["systems"]["matched_gpt54"]["arms"][name]
        previous = arm["reader"]
        prior = copy.deepcopy(previous.get("prior_attempts", []))
        if "call" in previous and previous["call"]["cache_key"] != call["cache_key"]:
            prior.append(previous["call"])
        for attempt in range(1, args.attempt):
            call_id = "reader:" + row["qid"] + ":" + name + ":restored-credit-attempt:" + str(attempt)
            key = reader.digest(reader.canonical({"request": call["request"], "call_id": call_id,
                                                  "protocol_sha256": report["protocol_sha256"]}))
            path = reader.CACHE / (key + ".json")
            if path.exists() and key != call["cache_key"] and key not in {c["cache_key"] for c in prior}:
                prior.append(json.loads(path.read_text()))
        times = {e["cache_key"]: e["unix_time"] for e in json.loads(reader.LEDGER.read_text())["entries"]}
        prior.sort(key=lambda c: times[c["cache_key"]])
        parsed = reader.parse_answer(call["raw_text"]) if call["status"] == "ok" else None
        arm["reader"] = {"status": "ok" if parsed else "error", "raw_text": call["raw_text"], "parsed": parsed,
                         "call": call, "prior_attempts": prior}
        arm["scores"] = reader.scores(arm)
        reader.save_report(report)
        print(json.dumps({"qid": row["qid"], "arm": name, "status": arm["reader"]["status"],
                          "http_status": call.get("http_status"), "spend_usd": report["spend_usd"]}), flush=True)
        return call["status"] == "ok"

    if not tasks:
        reader.save_report(report)
        return
    # A rejected preflight prevents another queue of rejected requests.
    if not record(run(tasks.pop(0))):
        raise SystemExit("unchanged preflight rejected; remaining readers were not launched")
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {}
        for _ in range(min(4, len(tasks))):
            future = executor.submit(run, tasks.pop(0))
            pending[future] = True
        stopped = False
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                del pending[future]
                result = future.result()
                success = record(result)
                call = result[2]
                if not success and (call.get("http_status") in (401, 403, 429) or call["status"] == "budget_exhausted"):
                    stopped = True
            while tasks and not stopped and len(pending) < 4:
                pending[executor.submit(run, tasks.pop(0))] = True
        if stopped:
            raise SystemExit("API or budget rejection; remaining readers were not launched")


if __name__ == "__main__":
    main()
