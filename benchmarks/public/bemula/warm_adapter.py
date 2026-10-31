"""Bemula `MemoryAdapter` backed by warm-server's memory API.

Storage and retrieval are WARM: every observed turn is one fact (text, turn time, speaker,
an embedding), recall is hybrid lexical + embedding search over the store, and a forget
request ends in `forget` with `purge`, which leaves the store as if the facts were never
written (docs/memory-api/README.md, "Forgetting").

Which stored facts a natural-language request covers is adapter policy, decided by the
agent model: a request is detected in a user turn, every stored fact is reviewed against
it, matching facts are purged, and any that also carry unrelated content are re-stored
with the covered information removed. The request turn and the acknowledgement after it
are not stored; a value the request asks to keep (a replacement) is.

The adapter reads only `Turn.speaker`, `Turn.text` and `Turn.timestamp`, never the
benchmark's gold annotations.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import httpx

from bemula.baselines.base import MemoryAdapter
from bemula.schemas import Turn

# A cheap screen in front of the request classifier: a user turn that asks for removal
# names one of these verbs. The classifier makes the decision.
FORGET_CUE = re.compile(
    r"\b(forget|forgot|delete|erase|remove|wipe|scrub|purge|clear|discard|drop|"
    r"don'?t (?:remember|keep|store|retain|save)|do not (?:remember|keep|store|retain|save)|"
    r"stop (?:remembering|storing|keeping|tracking)|no longer (?:remember|keep|store|retain)|"
    r"unlearn|get rid of)\b",
    re.IGNORECASE,
)

MAX_WORDS = 3000
SCAN_BATCH = 20
SCAN_CHARS = 1500
WRITE_BATCH = 256

CLASSIFY_PROMPT = """You manage a personal assistant's long-term memory. Decide whether the user's message asks the assistant to forget, delete or stop retaining information it has stored about the user.

Message: {text}

Return ONLY JSON: {{"forget": true or false, "what": "<the information to forget, stated precisely, including any time period or category>", "keep": ["<each new fact the user wants remembered instead, as a standalone sentence>"]}}
"what" is empty and "keep" is [] when "forget" is false."""

SCAN_PROMPT = """You manage a personal assistant's long-term memory. The user asked: "{request}"
Information to forget: {what}

Below are stored memory items (id, time, speaker, text). Flag every item that states, repeats, paraphrases or lets someone infer any of the information to forget, whether said by the user or echoed by the assistant. For a time-period request, use the item time and any dates in the text. For a category request, include every item about that category.

For each flagged item give "rest": the item text with all information to forget removed, or "" if nothing else worth keeping remains.

Items:
{items}

Return ONLY JSON: {{"flagged": [{{"id": "<id>", "rest": "<text>"}}]}} (empty list if none)."""

ANSWER_PROMPT = """You are a personal assistant answering from your memory of past conversations with the user. Use ONLY the memory items below. If they do not contain the answer, say you don't have that information. Keep the answer short.

Memory:
{context}

Question: {question}"""


def _millis(timestamp: str) -> int | None:
    try:
        return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, AttributeError):
        return None


def _split_words(text: str) -> list[str]:
    words = text.split(" ")
    if len(words) <= MAX_WORDS:
        return [text]
    return [" ".join(words[i : i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]


class WarmAdapter(MemoryAdapter):
    name = "warm"

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        server_url: str,
        top_k: int = 5,
        log_path: str | None = None,
        scan_workers: int = 4,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k
        self.scan_workers = scan_workers
        self._http = httpx.Client(base_url=server_url, timeout=120.0)
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None
        self._log_lock = threading.Lock()
        self.reset()

    # -- WARM calls ---------------------------------------------------------------------

    def _call(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(
            f"/api/v1/memory/{op}", json=body, headers={"X-Tenant-ID": self._tenant}
        )
        resp.raise_for_status()
        return resp.json()

    def _remember(self, facts: list[dict[str, Any]]) -> None:
        if not facts:
            return
        vectors = self.embedder.embed([f["text"] for f in facts])
        for fact, vec in zip(facts, vectors):
            fact["embedding"] = [float(x) for x in vec]
        self._call("remember", {"facts": facts})
        self._ids.extend(f["id"] for f in facts)

    def _emit(self, record: dict[str, Any]) -> None:
        if self._log is None:
            return
        record["tenant"] = self._tenant
        record["instance_id"] = self._instance_id
        record["variant"] = self._variant
        with self._log_lock:
            self._log.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._log.flush()

    # -- MemoryAdapter -----------------------------------------------------------------

    def begin_instance(self, instance: Any) -> None:
        self._instance_id = instance.instance_id
        self._resets = 0

    def reset(self) -> None:
        # The runner resets once before the control copy and once before the forget copy.
        resets = getattr(self, "_resets", 0)
        self._variant = "control" if resets == 0 else "forget"
        self._resets = resets + 1
        self._instance_id = getattr(self, "_instance_id", None)
        self._tenant = f"bemula-{uuid.uuid4().hex}"
        self._pending: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._purged: set[str] = set()
        self._seq = 0
        self._skip_ack = False

    def observe(self, turn: Turn) -> None:
        text = turn.text or ""
        if turn.speaker == "user" and FORGET_CUE.search(text.replace("’", "'")):
            request = self._classify(text)
            if request.get("forget"):
                self._flush()
                self._apply_forget(text, request, turn.timestamp)
                self._skip_ack = True
                return
        if turn.speaker == "assistant" and self._skip_ack:
            self._skip_ack = False
            return
        self._skip_ack = False
        self._queue(text, turn.speaker, turn.timestamp)
        if len(self._pending) >= WRITE_BATCH:
            self._flush()

    def end_variant(self) -> None:
        self._flush()

    def query(self, question: str) -> str:
        self._flush()
        if not self._ids:
            return "I don't have any relevant information stored."
        vec = self.embedder.embed([question])[0]
        results = self._call(
            "recall",
            {"query": question, "k": self.top_k, "embedding": [float(x) for x in vec]},
        )["results"]
        leaked = [r["id"] for r in results if r["id"] in self._purged]
        context = "\n".join(
            f"- [{(r.get('metadata') or {}).get('time', '')}] "
            f"{(r.get('metadata') or {}).get('speaker', '')}: {r.get('text') or ''}"
            for r in results
        )
        answer = self.llm.chat(
            [{"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="warm_answer",
        ).strip()
        self._emit(
            {
                "kind": "query",
                "question": question,
                "ids": [r["id"] for r in results],
                "context": context,
                "answer": answer,
                "purged_returned": leaked,
            }
        )
        return answer

    # -- internals ---------------------------------------------------------------------

    def _queue(self, text: str, speaker: str, timestamp: str) -> None:
        for part in _split_words(text):
            if not part.strip():
                continue
            self._seq += 1
            fact: dict[str, Any] = {
                "id": f"t{self._seq:06d}",
                "text": part,
                "metadata": {"speaker": speaker, "time": timestamp},
            }
            when = _millis(timestamp)
            if when is not None:
                fact["eventTime"] = when
            self._pending.append(fact)

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        for i in range(0, len(pending), WRITE_BATCH):
            self._remember(pending[i : i + WRITE_BATCH])

    def _classify(self, text: str) -> dict[str, Any]:
        try:
            out = self.llm.chat_json(
                [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text)}],
                temperature=0.0,
                max_tokens=512,
                task="warm_classify",
            )
        except ValueError:
            self._emit({"kind": "classify_failure", "text": text})
            return {"forget": False}
        return out if isinstance(out, dict) else {"forget": False}

    def _scan(self, request: str, what: str, batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Review a batch; a failed call is retried on each half, down to single facts.

        Returns the flagged facts and the number of facts whose review still failed.
        """
        try:
            return self._scan_once(request, what, batch), 0
        except ValueError:
            if len(batch) == 1:
                return [], 1
        mid = len(batch) // 2
        left, lf = self._scan(request, what, batch[:mid])
        right, rf = self._scan(request, what, batch[mid:])
        return left + right, lf + rf

    def _scan_once(self, request: str, what: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = "\n".join(
            f"[{f['id']}] ({(f.get('metadata') or {}).get('time', '')}, "
            f"{(f.get('metadata') or {}).get('speaker', '')}) "
            f"{(f.get('text') or '')[:SCAN_CHARS]}"
            for f in batch
        )
        out = self.llm.chat_json(
            [{"role": "user", "content": SCAN_PROMPT.format(request=request, what=what, items=items)}],
            temperature=0.0,
            max_tokens=4096,
            task="warm_scan",
        )
        if not isinstance(out, dict) or not isinstance(out.get("flagged"), list):
            raise ValueError("review reply has no flagged list")
        known = {f["id"] for f in batch}
        flagged = out["flagged"]
        return [x for x in flagged if isinstance(x, dict) and x.get("id") in known]

    def _apply_forget(self, request: str, parsed: dict[str, Any], timestamp: str) -> None:
        what = str(parsed.get("what") or request)
        facts = []
        for fid in self._ids:
            fact = self._call("get", {"id": fid}).get("fact")
            if fact is not None:
                facts.append(fact)
        batches = [facts[i : i + SCAN_BATCH] for i in range(0, len(facts), SCAN_BATCH)]
        with ThreadPoolExecutor(max_workers=self.scan_workers) as pool:
            parts = list(pool.map(lambda b: self._scan(request, what, b), batches))
        flagged = [x for part, _ in parts for x in part]
        scan_failures = sum(failed for _, failed in parts)
        by_id = {f["id"]: f for f in facts}
        ids = sorted({x["id"] for x in flagged})
        if ids:
            self._call("forget", {"scope": {"kind": "ids", "ids": ids}, "purge": True})
            self._purged.update(ids)
            gone = set(ids)
            self._ids = [i for i in self._ids if i not in gone]
        rewrites = []
        for x in flagged:
            rest = str(x.get("rest") or "").strip()
            original = by_id[x["id"]]
            meta = original.get("metadata") or {}
            if rest and rest != (original.get("text") or "").strip():
                self._seq += 1
                fact: dict[str, Any] = {
                    "id": f"t{self._seq:06d}",
                    "text": rest,
                    "metadata": {"speaker": meta.get("speaker", ""), "time": meta.get("time", "")},
                }
                if original.get("eventTime") is not None:
                    fact["eventTime"] = original["eventTime"]
                rewrites.append(fact)
        for keep in parsed.get("keep") or []:
            if isinstance(keep, str) and keep.strip():
                self._seq += 1
                fact = {
                    "id": f"t{self._seq:06d}",
                    "text": keep.strip(),
                    "metadata": {"speaker": "user", "time": timestamp},
                }
                when = _millis(timestamp)
                if when is not None:
                    fact["eventTime"] = when
                rewrites.append(fact)
        self._remember(rewrites)
        self._emit(
            {
                "kind": "forget",
                "request": request,
                "what": what,
                "scanned": len(facts),
                "purged": ids,
                "rewritten": len(rewrites),
                "scan_failures": scan_failures,
            }
        )
