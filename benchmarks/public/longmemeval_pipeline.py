# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""LongMemEval retrieval pipeline: keys, model-input manifests, fusion, and official scoring.

CPU-only logic shared by the local dev sweeps and the Modal job (longmemeval_modal.py):

  keys        multi-granularity keys per question: user turns, rounds (user turn + assistant reply),
              sessions (concatenated user turns, the official session value), LLM-extracted user
              facts (separate keys) and fact-expanded turns/sessions (K = V + facts)
  scores      public-bench lme-scores gives the HMS document API's BM25 and exact-cosine score of
              every key for every query variant (original question, LLM rewrite, sub-queries)
  fuse        weighted RRF of lexical and dense ranks (the document API's own hybrid formula) or a
              convex blend of normalized scores; key scores aggregated to turns and sessions; ranked
              lists merged across key kinds and query variants; optional session prior for turns,
              soft boost for sessions inside an LLM-parsed time range, cross-encoder re-rank blend
  evaluate    LongMemEval's own eval_utils.py (pinned commit, unmodified) on official corpus ids

Targets follow the official flat index (src/retrieval/run_retrieval.py): turn ids `<sid>_<n>` for
user turns, session ids for sessions, `answer` rewritten to `noans` where no user turn has the answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

EMBED_TASK = (
    "Given a question a user asks about their earlier conversations with an AI assistant, "
    "retrieve the user messages that contain the information needed to answer it"
)
RERANK_TASK = (
    "Given a question a user asks about their earlier conversations with an AI assistant, "
    "judge whether the conversation excerpt contains information needed to answer it"
)
ROUND_CHARS = 2000  # assistant replies are long; the round key keeps the user turn plus the reply's start

FACT_PROMPT = """Below are the user's messages from one conversation with an AI assistant, numbered. The conversation took place on {date}.

List every piece of personal information the user reveals: facts about themselves, people they know, possessions, purchases, places, events, plans, preferences, opinions, habits, numbers and dates. Write each as one short self-contained sentence in the third person ("The user ...") that keeps the specifics (names, items, quantities). Resolve relative dates ("yesterday", "last month") against the conversation date and state the absolute date. Skip generic requests that reveal nothing about the user.

Answer with JSON only: {{"facts": [{{"turn": <message number>, "fact": "<sentence>"}}]}}. Use {{"facts": []}} if there are none.

{messages}"""

QUERY_PROMPT = """A user is asking an AI assistant a question about their earlier conversations with it. Today is {today}.

Question: {question}

Answer with JSON only, with these fields:
"rewrite": the question restated as a search query that names the key entities, with likely synonyms;
"subqueries": a list of up to 4 short search queries, one per separate fact the answer needs (for example each event to compare, each item to count, the old and the new value of something that changed); an empty list if one search suffices;
"time_range": ["YYYY/MM/DD", "YYYY/MM/DD"] if the question restricts when the relevant conversations happened (for example "last week", "in March", "two months ago", "the first time"... only when a calendar range is implied), resolved against today; otherwise null."""


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


def parse_date(s: str) -> date | None:
    m = re.match(r"\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s or "")
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


# --------------------------------------------------------------------------------------- data


@dataclass
class Session:
    sid: str  # official session id (noans rewrite applied)
    date: date | None
    date_text: str
    turns: list[tuple[str, str, str]]  # (official turn id, user text, following assistant text)

    @property
    def user_text(self) -> str:
        return " ".join(t for _, t, _ in self.turns)

    def fact_input(self) -> str:
        # Pasted documents make some user turns very long; the extractor sees each turn's start.
        return "\n".join(f"[{i + 1}] {t[:1500]}" for i, (_, t, _) in enumerate(self.turns))[:24000]


@dataclass
class Question:
    qid: str
    qtype: str
    text: str
    date_text: str
    date: date | None
    sessions: list[Session] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        return "_abs" not in self.qid and any("answer" in tid for s in self.sessions for tid, _, _ in s.turns)


def load(path: str | Path, ids: set[str] | None = None) -> list[Question]:
    out = []
    for e in json.loads(Path(path).read_text()):
        if ids is not None and e["question_id"] not in ids:
            continue
        q = Question(e["question_id"], e["question_type"], e["question"], e["question_date"], parse_date(e["question_date"]))
        for sid, d, sess in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]):
            turns = []
            for i, t in enumerate(sess):
                if t["role"] != "user":
                    continue
                tid = f"{sid}_{i + 1}"
                if "answer" in sid and not t.get("has_answer"):
                    tid = tid.replace("answer", "noans")
                reply = sess[i + 1]["content"] if i + 1 < len(sess) and sess[i + 1]["role"] == "assistant" else ""
                turns.append((tid, t["content"], reply))
            if not turns:
                continue
            osid = sid
            if "answer" in sid and not any("answer" in tid for tid, _, _ in turns):
                osid = sid.replace("answer", "noans")
            q.sessions.append(Session(osid, parse_date(d), d, turns))
        out.append(q)
    return out


def split_ids(split_file: Path, part: str) -> set[str]:
    return set(json.loads(split_file.read_text())[part])


# --------------------------------------------------------------------------------------- keys

KINDS = ("turn", "round", "session", "fact", "turn_exp", "session_exp")


def fact_key(s: Session, llm: str) -> str:
    return sha("facts", llm, FACT_PROMPT, s.date_text, s.fact_input())


def query_key(q: Question, llm: str) -> str:
    return sha("query", llm, QUERY_PROMPT, q.date_text, q.text)


def session_facts(s: Session, facts: dict, llm: str) -> list[tuple[int, str]]:
    """(turn index, fact) pairs; facts with an out-of-range turn are attributed to the session only (-1)."""
    out = []
    for f in (facts.get(fact_key(s, llm)) or {}).get("facts", []):
        if not isinstance(f, dict) or not isinstance(f.get("fact"), str) or not f["fact"].strip():
            continue
        t = f.get("turn")
        i = int(t) - 1 if isinstance(t, int) or (isinstance(t, str) and t.isdigit()) else -1
        out.append((i if 0 <= i < len(s.turns) else -1, f["fact"].strip()))
    return out


def build_keys(q: Question, kinds, facts: dict | None = None, llm: str = "") -> list[tuple[str, str, str | None, str]]:
    """(kind, text, turn target or None, session target) for every key of the question."""
    keys = []
    for s in q.sessions:
        sf = session_facts(s, facts or {}, llm) if facts is not None else []
        for i, (tid, user, reply) in enumerate(s.turns):
            if "turn" in kinds:
                keys.append(("turn", user, tid, s.sid))
            if "round" in kinds:
                keys.append(("round", f"user: {user}\nassistant: {reply}"[:ROUND_CHARS], tid, s.sid))
            if "turn_exp" in kinds:
                mine = [f for j, f in sf if j == i]
                keys.append(("turn_exp", " ".join(mine + [user]), tid, s.sid))
        if "session" in kinds:
            keys.append(("session", s.user_text, None, s.sid))
        if "session_exp" in kinds:
            keys.append(("session_exp", " ".join([f for _, f in sf] + [s.user_text]), None, s.sid))
        if "fact" in kinds:
            for j, f in sf:
                keys.append(("fact", f, s.turns[j][0] if j >= 0 else None, s.sid))
    return keys


def query_variants(q: Question, queries: dict | None, llm: str) -> list[tuple[str, str]]:
    out = [("orig", q.text)]
    r = (queries or {}).get(query_key(q, llm)) or {}
    if isinstance(r.get("rewrite"), str) and r["rewrite"].strip():
        out.append(("rewrite", r["rewrite"].strip()))
    for n, s in enumerate(r.get("subqueries") or []):
        if isinstance(s, str) and s.strip() and n < 4:
            out.append((f"sub{n}", s.strip()))
    return out


def time_range(q: Question, queries: dict | None, llm: str) -> tuple[date, date] | None:
    r = (queries or {}).get(query_key(q, llm)) or {}
    tr = r.get("time_range")
    if not isinstance(tr, list) or len(tr) != 2:
        return None
    a, b = parse_date(str(tr[0])), parse_date(str(tr[1]))
    if a is None or b is None:
        return None
    return (min(a, b), max(a, b))


# --------------------------------------------------------------------------- score manifests


def write_score_inputs(questions, kinds, facts, llm, queries, embed, out: Path) -> dict:
    """Write items.jsonl / queries.jsonl / .f32 for public-bench lme-scores.

    `embed(texts, is_query)` returns L2-normalized float32 rows. Key ids are `<kind>|<n>`; the
    returned map gives each key id its (turn target, session target) per question.
    """
    out.mkdir(parents=True, exist_ok=True)
    targets, item_rows, query_rows = {}, [], []
    item_texts, query_texts = {}, {}
    for q in questions:
        keys = build_keys(q, kinds, facts, llm)
        targets[q.qid] = {}
        for n, (kind, text, tid, sid) in enumerate(keys):
            kid = f"{kind}|{n}"
            targets[q.qid][kid] = (tid, sid)
            item_rows.append({"q": q.qid, "id": kid, "text": text, "e": item_texts.setdefault(text, len(item_texts))})
        for v, text in query_variants(q, queries, llm):
            query_rows.append({"q": q.qid, "v": v, "text": text, "e": query_texts.setdefault(text, len(query_texts))})
    for name, rows in (("items", item_rows), ("queries", query_rows)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    ie, qe = embed(list(item_texts), False), embed(list(query_texts), True)
    ie.astype("<f4").tofile(out / "items.f32")
    qe.astype("<f4").tofile(out / "queries.f32")
    (out / "targets.json").write_text(json.dumps(targets))
    return {"emb_dim": int(ie.shape[1]), "keys": len(item_rows), "unique_keys": len(item_texts), "queries": len(query_rows)}


# ------------------------------------------------------------------------------------ fusion

DEFAULT = {
    "kinds": {"turn": 1.0, "session": 1.0},  # weight of each key kind's ranked list in the merge
    "variants": {"orig": 1.0},  # 'orig', 'rewrite', 'sub' (all sub-queries share the weight)
    "w_lex": 1.0,
    "w_sem": 1.0,
    "fuse": "rrf",  # 'rrf' (document API hybrid, k=60) or 'cvx' (min-max normalized blend)
    "agg": "max",  # key scores -> target: 'max', 'sum2', 'sum3', 'rrf'
    "k0": 60.0,  # RRF constant of the merge across lists
    "sess_prior": 0.0,  # turn ranking: weight of the turn's session rank
    "fact_as": None,  # merge fact keys into this kind's list (max over keys) instead of their own list
    "time_w": 0.0,  # sessions dated inside the parsed time range
    "rerank_w": 0.0,
    "rerank_n": 20,
}


def _fuse_keys(rows, cfg) -> dict[str, float]:
    ids = [r[0] for r in rows]
    lex = np.array([r[1] for r in rows], dtype=float)
    sem = np.array([-1.0 if r[2] is None else r[2] for r in rows], dtype=float)
    if cfg["fuse"] == "cvx":
        def mm(a):
            span = a.max() - a.min() if len(a) else 0.0
            return (a - a.min()) / span if span > 0 else np.zeros_like(a)
        s = cfg["w_lex"] * mm(lex) + cfg["w_sem"] * mm(sem)
    else:
        s = np.zeros(len(ids))
        for vals, w, mask in ((lex, cfg["w_lex"], lex > 0), (sem, cfg["w_sem"], np.ones(len(ids), bool))):
            if w <= 0:
                continue
            order = sorted(np.flatnonzero(mask), key=lambda i: (-vals[i], i))
            for r, i in enumerate(order):
                s[i] += w / (61.0 + r)
    return dict(zip(ids, s.tolist()))


def _aggregate(scores: dict[str, float], key_target: dict[str, str | None], agg: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    if agg == "rrf":
        for r, kid in enumerate(sorted(scores, key=lambda k: -scores[k])):
            t = key_target.get(kid)
            if t is not None:
                per.setdefault(t, []).append(1.0 / (60.0 + r))
        return {t: float(sum(v)) for t, v in per.items()}
    for kid, s in scores.items():
        t = key_target.get(kid)
        if t is not None:
            per.setdefault(t, []).append(s)
    n = {"max": 1, "sum2": 2, "sum3": 3}[agg]
    return {t: float(sum(sorted(v, reverse=True)[:n])) for t, v in per.items()}


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    return {t: r for r, t in enumerate(sorted(scores, key=lambda t: (-scores[t], t)))}


def rank_question(q: Question, raw: dict, targets: dict, cfg: dict, level: str,
                  trange=None, rerank: dict | None = None, session_ranking: list[str] | None = None) -> list[str]:
    """Full ranking of the question's official targets at `level` ('turn' or 'session')."""
    all_targets = [tid for s in q.sessions for tid, _, _ in s.turns] if level == "turn" else [s.sid for s in q.sessions]
    turn_session = {tid: s.sid for s in q.sessions for tid, _, _ in s.turns}
    k0 = cfg["k0"]
    final = {t: 0.0 for t in all_targets}
    for variant, rows in raw.items():
        vw = cfg["variants"].get("sub" if variant.startswith("sub") else variant, 0.0)
        if vw <= 0:
            continue
        by_kind: dict[str, list] = {}
        for r in rows:
            kind = r[0].split("|", 1)[0]
            if kind == "fact" and cfg.get("fact_as"):
                kind = cfg["fact_as"]  # the paper's "separate" join: facts share the value keys' index
            by_kind.setdefault(kind, []).append(r)
        for kind, krows in by_kind.items():
            kw = cfg["kinds"].get(kind, 0.0)
            if kw <= 0:
                continue
            idx = 0 if level == "turn" else 1
            kt = {r[0]: targets[r[0]][idx] for r in krows}
            agg = _aggregate(_fuse_keys(krows, cfg), kt, cfg["agg"])
            ranks = _ranks(agg)
            # Partial lists (facts exist for some sessions only) rank absent targets last, so
            # merely having a key of this kind is not rewarded.
            worst = len(all_targets)
            for t in final:
                final[t] += kw * vw / (k0 + ranks.get(t, worst) + 1)
    if level == "turn" and cfg["sess_prior"] > 0 and session_ranking:
        srank = {s: r for r, s in enumerate(session_ranking)}
        for t in final:
            final[t] += cfg["sess_prior"] / (k0 + srank.get(turn_session[t], len(srank)) + 1)
    if cfg["time_w"] > 0 and trange is not None:
        sdate = {s.sid: s.date for s in q.sessions}
        for t in final:
            d = sdate.get(turn_session.get(t, t))
            if d is not None and trange[0] <= d <= trange[1]:
                final[t] += cfg["time_w"] / (k0 + 1)
    if cfg["rerank_w"] > 0 and rerank:
        head = sorted(final, key=lambda t: (-final[t], all_targets.index(t)))[: cfg["rerank_n"]]
        scored = {t: rerank[t] for t in head if t in rerank}
        for t, r in _ranks(scored).items():
            final[t] += cfg["rerank_w"] / (k0 + r + 1)
    pos = {t: i for i, t in enumerate(all_targets)}
    return sorted(all_targets, key=lambda t: (-final[t], pos[t]))


# -------------------------------------------------------------------------------- evaluation


def evaluate(questions: list[Question], rankings: dict[str, list[str]], level: str, ev) -> dict:
    """Official recall_any / recall_all / ndcg_any at 5 and 10, overall and per question type."""
    rows = {}
    for q in questions:
        if not q.scored:
            continue
        ids = [tid for s in q.sessions for tid, _, _ in s.turns] if level == "turn" else [s.sid for s in q.sessions]
        pos = {c: i for i, c in enumerate(ids)}
        rank = [pos[c] for c in rankings[q.qid]]
        correct = list({c for c in ids if "answer" in c})
        m = {}
        for k in (5, 10):
            r_any, r_all, nd = ev.evaluate_retrieval(rank, correct, ids, k=k)
            m.update({f"recall_any@{k}": r_any, f"recall_all@{k}": r_all, f"ndcg_any@{k}": nd})
        rows[q.qid] = (q.qtype, m)
    out = {}
    groups = [("overall", list(rows))] + [(t, [k for k, v in rows.items() if v[0] == t]) for t in sorted({v[0] for v in rows.values()})]
    for g, members in groups:
        out[g] = {"n_questions": len(members)}
        for name in rows[members[0]][1]:
            out[g][name] = float(np.mean([rows[k][1][name] for k in members]))
    return out


def objective(metrics: dict) -> float:
    o = metrics["overall"]
    return float(np.mean([o["recall_all@5"], o["recall_all@10"], o["ndcg_any@5"], o["ndcg_any@10"]]))


def run(questions, scores: dict, targets: dict, cfg_turn: dict, cfg_sess: dict, ev,
        queries=None, llm="", rerank=None) -> tuple[dict, dict]:
    """Rankings and official metrics for both levels. `rerank[level][qid][target]` holds cross-encoder scores."""
    rankings = {"session": {}, "turn": {}}
    for q in questions:
        tr = time_range(q, queries, llm)
        rs = rank_question(q, scores[q.qid], targets[q.qid], cfg_sess, "session", tr, (rerank or {}).get("session", {}).get(q.qid))
        rankings["session"][q.qid] = rs
        rankings["turn"][q.qid] = rank_question(
            q, scores[q.qid], targets[q.qid], cfg_turn, "turn", tr, (rerank or {}).get("turn", {}).get(q.qid), rs
        )
    return rankings, {lvl: evaluate(questions, rankings[lvl], lvl, ev) for lvl in ("session", "turn")}


def load_scores(path: Path) -> dict:
    return json.loads(Path(path).read_text())["scores"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")
