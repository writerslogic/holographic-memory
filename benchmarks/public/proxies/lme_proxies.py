"""LongMemEval S-dev proxies on the cached Qwen3-Embedding-0.6B turn embeddings of the dev
export (`~/.cache/hms-bench/holo_dev`, the 100 frozen dev questions
held-out ids untouched).
Dense-exact ranking is the arm
the official eval_utils scores it. Usage:
  uv run --with numpy python lme_proxies.py <diversify|event_dates>"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/public"))
import longmemeval_pipeline as P  # noqa: E402

CACHE = Path.home() / ".cache/hms-bench"
EXPORT = CACHE / "holo_dev"
EVAL = CACHE / "downloads/longmemeval_eval_utils_9e0b455f4ef0.py"
OUT = ROOT / "benchmarks/results/proxies_2026-10_lme.json"


def write(name, payload):
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"note": "LongMemEval S dev proxies (frozen dev ids only), dense-exact Qwen3-Embedding-0.6B arm"}
    doc[name] = payload
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", name)


def load():
    spec = importlib.util.spec_from_file_location("lme_eval_utils", EVAL)
    ev = importlib.util.module_from_spec(spec)
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec.loader.exec_module(ev)
    tx = json.loads((EXPORT / "turns_dev.json").read_text())
    dim = json.loads((EXPORT / "dim.json").read_text())["dim"]
    temb = np.fromfile(EXPORT / "turn_emb.f16", dtype=np.float16).reshape(-1, dim).astype(np.float32)
    qemb = np.fromfile(EXPORT / "q_emb.f16", dtype=np.float16).reshape(-1, dim).astype(np.float32)
    return ev, tx, temb, qemb


def questions_of(tx):
    qs, rows = [], []
    row = 0
    for q in tx["questions"]:
        sess = [P.Session(s["sid"], P.parse_date(s["date"]), s["date"], [(t, "", "") for t in s["tids"]]) for s in q["sessions"]]
        qs.append(P.Question(q["qid"], q["qtype"], q["question"], q["date"], P.parse_date(q["date"]), sess))
        spans = []
        for s in q["sessions"]:
            spans.append((row, row + len(s["tids"]), s["sid"], s["tids"]))
            row += len(s["tids"])
        rows.append(spans)
    return qs, rows


def per_type(m):
    return {t: {"n": v["n_questions"], "R5": round(v["recall_all@5"], 3), "N10": round(v["ndcg_any@10"], 3)} for t, v in m.items()}


def diversify():
    ev, tx, temb, qemb = load()
    qs, rows = questions_of(tx)
    variants = {}

    def rank_all(policy):
        turn_rank, sess_rank = {}, {}
        for qi, (q, spans) in enumerate(zip(qs, rows)):
            qv = qemb[qi]
            items = []  # (score, sid, tid)
            for a, b, sid, tids in spans:
                sc = temb[a:b] @ qv
                items += [(float(s), sid, t) for s, t in zip(sc, tids)]
            items.sort(key=lambda x: -x[0])
            turn_rank[q.qid] = policy(items)
            best = {}
            for s, sid, _ in items:
                best.setdefault(sid, s)
            sess_rank[q.qid] = sorted(best, key=lambda k: -best[k])
        return turn_rank, sess_rank

    def flat(items):
        return [t for _, _, t in items]

    def quota(c):
        def f(items):
            cnt, head, tail = {}, [], []
            for s, sid, t in items:
                if cnt.get(sid, 0) < c:
                    cnt[sid] = cnt.get(sid, 0) + 1
                    head.append(t)
                else:
                    tail.append(t)
            return head + tail
        return f

    def round_robin(m):
        # best turn of each of the top-m sessions first (by session max), then the flat order
        def f(items):
            first, out = {}, []
            for s, sid, t in items:
                if sid not in first:
                    first[sid] = t
            lead = list(first.values())[:m]
            out = lead + [t for _, _, t in items if t not in set(lead)]
            return out
        return f

    policies = {"flat_max (baseline)": flat, "quota_1_per_session_first": quota(1), "quota_2_per_session_first": quota(2), "quota_3_per_session_first": quota(3), "round_robin_top3_sessions": round_robin(3), "round_robin_top5_sessions": round_robin(5)}
    for name, pol in policies.items():
        tr, sr = rank_all(pol)
        m_turn = P.evaluate(qs, tr, "turn", ev)
        m_sess = P.evaluate(qs, sr, "session", ev)
        variants[name] = {"turn": per_type(m_turn), "session_overall": per_type(m_sess)["overall"]}
        print(name, variants[name]["turn"]["overall"], variants[name]["turn"].get("multi-session"))
    return {"question": "Does per-session diversification of the turn ranking raise multi-session recall_all@5 without hurting the other types? (axiom M1: a session is a flat set of turns; max aggregation)", "results": variants,
            "decision_rule": "a diversification policy goes to the full pipeline (cached-score sweep, spec C) only if multi-session turn R5 rises >= 0.04 (one question = 0.042) with overall turn R5 not down and no other type down >= 0.02; n is 24 and 84, so CIs are wide and this is a go/no-go for a measurement, not a claim"}


DATE_RE = re.compile(r"\b(20\d\d[/-]\d\d?[/-]\d\d?|\d\d?/\d\d?/20\d\d|(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.? \d{1,2}(st|nd|rd|th)?(,? 20\d\d)?)\b", re.I)
REL_RE = re.compile(r"\b(yesterday|today|tomorrow|last (week|month|year|night|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|next (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(\d+|a|an|two|three|four|five|six|seven|eight|nine|ten|few|couple of) (days?|weeks?|months?|years?|hours?) (ago|from now|later|earlier|before|after)|this (morning|afternoon|evening|week|month|year|weekend)|in (20\d\d|\d+ (days|weeks|months|years))|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I)


def event_dates():
    tx = json.loads((EXPORT / "turns_dev.json").read_text())
    text = {}
    with (CACHE / "longmemeval_s/turn.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            text[x["id"]] = x["text"]
    per_type = {}
    examples = []
    for q in tx["questions"]:
        ev_turns = [(s, t) for s in q["sessions"] if s["sid"].startswith("answer_") for t in s["tids"]]
        hits_abs = hits_rel = n = missing = 0
        for s, t in ev_turns:
            tt = text.get(t)
            if tt is None:
                missing += 1
                continue
            n += 1
            a = DATE_RE.search(tt)
            r = REL_RE.search(tt)
            hits_abs += bool(a)
            hits_rel += bool(r)
            if (a or r) and q["qtype"] == "temporal-reasoning" and len(examples) < 12:
                examples.append({"qid": q["qid"], "session_date": s["date"], "match": (a or r).group(0), "snippet": tt[max(0, (a or r).start() - 60):(a or r).end() + 60]})
        d = per_type.setdefault(q["qtype"], {"questions": 0, "evidence_turns": 0, "with_absolute_date": 0, "with_relative_time": 0, "questions_with_any": 0, "missing_text": 0})
        d["questions"] += 1
        d["evidence_turns"] += n
        d["with_absolute_date"] += hits_abs
        d["with_relative_time"] += hits_rel
        d["questions_with_any"] += int(hits_abs + hits_rel > 0)
        d["missing_text"] += missing
    for d in per_type.values():
        d["share_turns_with_time_expression"] = round((d["with_absolute_date"] + d["with_relative_time"]) / max(d["evidence_turns"], 1), 3)
        d["share_questions_with_any"] = round(d["questions_with_any"] / d["questions"], 3)
    return {"question": "How often do evidence turns state a time (absolute date or relative expression) other than the session date? (axiom M2: a fact's date is its session date)", "results": per_type, "examples_temporal": examples,
            "decision_rule": "valid-time extraction (spec C lever ii, bitemporal mapping 3.4) is kept in the reader spec only if >= 30% of temporal-reasoning evidence turns carry a time expression; otherwise it is dropped from the spec"}


if __name__ == "__main__":
    what = sys.argv[1]
    t = time.time()
    r = diversify() if what == "diversify" else event_dates()
    r["minutes"] = round((time.time() - t) / 60, 1)
    write(what, r)
