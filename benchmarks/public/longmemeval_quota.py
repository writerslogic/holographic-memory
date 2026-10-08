# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Per-session quota on the tuned LongMemEval turn ranking, measured on cached dev scores only.

  uv run --script benchmarks/public/longmemeval_quota.py <run dir> <longmemeval_s_cleaned.json>

Leaf M1 of docs/research/IDEAS-2026-10.md: the dense-only proxy passed its rule by one question,
so the policy is re-measured on the full tuned pipeline (longmemeval_config.json) using the
cached scores of a Modal dev run (`modal volume get hms-lme runs/<tag>`; no compute). The turn
ranking is post-processed: at most c turns per session before any session's (c+1)-th; the rest
of the order is unchanged. Writes benchmarks/results/longmemeval_dev_quota.json. Decision rule
(stated before the run): adopt only if multi-session turn recall_all@5 rises >= 0.04 with overall
turn recall_all@5 not down and no type down >= 0.02. Held-out questions are never loaded.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import longmemeval_pipeline as P  # noqa: E402
from evaluate import lme_eval_utils  # noqa: E402

QUOTAS = (1, 2, 3, 4)
RULE = {"multi_session_R5_min_gain": 0.04, "overall_R5_min_gain": 0.0, "max_type_loss": 0.02}


def quota(order: list[str], session_of: dict[str, str], c: int) -> list[str]:
    cnt, head, tail = {}, [], []
    for t in order:
        s = session_of[t]
        if cnt.get(s, 0) < c:
            cnt[s] = cnt.get(s, 0) + 1
            head.append(t)
        else:
            tail.append(t)
    return head + tail


def main(run_dir: str, data: str) -> None:
    rd = Path(run_dir)
    ev, ev_sha = lme_eval_utils()
    dev = P.split_ids(HERE / "longmemeval_split.json", "dev")
    qs = P.load(data, dev)
    scores = P.load_scores(rd / "scores.json")
    targets = json.loads((rd / "targets.json").read_text())
    qfile = json.loads((rd / "queries.json").read_text()) if (rd / "queries.json").exists() else {}
    queries, llm = qfile.get("by_key"), qfile.get("llm", "")
    rerank = json.loads((rd / "rerank.json").read_text()) if (rd / "rerank.json").exists() else None
    cfg = json.loads((HERE / "longmemeval_config.json").read_text())
    rankings, base = P.run(qs, scores, targets, cfg["turn"], cfg["session"], ev, queries, llm, rerank)
    session_of = {q.qid: {tid: s.sid for s in q.sessions for tid, _, _ in s.turns} for q in qs}

    def summary(m):
        return {t: {"n": v["n_questions"], "R5": v["recall_all@5"], "R10": v["recall_all@10"],
                    "N10": v["ndcg_any@10"]} for t, v in m.items()}

    out = {"question": "Does a per-session quota on the tuned turn ranking raise multi-session recall_all@5 "
                       "without hurting the other types? (IDEAS-2026-10 leaf M1)",
           "run_dir": str(rd), "config": cfg["turn"], "eval_sha256": ev_sha, "n_dev": len(qs),
           "n_scored": sum(q.scored for q in qs), "decision_rule": RULE,
           "baseline": summary(base["turn"]), "quota": {}, "decision": None}
    b = base["turn"]
    print("baseline", b["overall"]["recall_all@5"], b["multi-session"]["recall_all@5"])
    for c in QUOTAS:
        r = {q.qid: quota(rankings["turn"][q.qid], session_of[q.qid], c) for q in qs}
        m = P.evaluate(qs, r, "turn", ev)
        d_ms = m["multi-session"]["recall_all@5"] - b["multi-session"]["recall_all@5"]
        d_all = m["overall"]["recall_all@5"] - b["overall"]["recall_all@5"]
        losses = {t: b[t]["recall_all@5"] - m[t]["recall_all@5"] for t in b if t != "overall"}
        passed = (d_ms >= RULE["multi_session_R5_min_gain"] and d_all >= RULE["overall_R5_min_gain"]
                  and max(losses.values()) < RULE["max_type_loss"])
        out["quota"][str(c)] = {"metrics": summary(m), "delta_multi_session_R5": d_ms, "delta_overall_R5": d_all,
                                "type_losses_R5": losses, "passes_rule": passed}
        print(f"quota {c}: overall R5 {m['overall']['recall_all@5']:.4f} ({d_all:+.4f}) "
              f"multi-session {m['multi-session']['recall_all@5']:.4f} ({d_ms:+.4f}) "
              f"max loss {max(losses.values()):+.4f} pass={passed}")
    winners = [c for c, v in out["quota"].items() if v["passes_rule"]]
    out["decision"] = (f"adopt quota {winners[0]}" if winners else
                       "no quota passes the rule on the tuned pipeline; not adopted")
    dest = HERE.parent / "results" / "longmemeval_dev_quota.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(out["decision"], "->", dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
