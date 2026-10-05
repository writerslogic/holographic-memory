# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "modal>=1.0"]
# ///
"""Holographic fact memory prototype on the LongMemEval_S dev split (docs/RESEARCH-OPPORTUNITIES.md 5.7).

Facts become hypervectors by binding attribute, value and time; an entity's facts are superposed into
one trace; a question is parsed into (entity, attribute, time), the trace is unbound and cleaned up
against the store's value codebook. Component vectors are sign codes of the embedder's embeddings, so
near-synonymous strings share bits. Numpy only; nothing here touches src/.

Modal stages (volume `hms-lme`, content-keyed caches, one cumulative ledger per --tag):
  uvx modal run benchmarks/public/holographic_memory_proto.py --stage export --out <dir>
      cached dev facts -> one LLM structuring pass (entity, attribute, value, when) -> query parse
      -> component embeddings -> tuned pipeline's top-1 turn -> <dir>/export_dev.json + emb files
  uvx modal run benchmarks/public/holographic_memory_proto.py --stage judge --candidates <json> --out <json>
      LLM judge of (question, candidate record) pairs against the gold answer, cached by content
Local stages (numpy):
  uv run --script benchmarks/public/holographic_memory_proto.py capacity <dir> <out.json>
  uv run --script benchmarks/public/holographic_memory_proto.py answer <dir> <candidates.json> <state.json>
  uv run --script benchmarks/public/holographic_memory_proto.py report <dir> <state.json> <judged.json> <out.json> [capacity.json]
  uv run --script benchmarks/public/holographic_memory_proto.py rank <dir> <eval_utils.py> <out.json>   (turn-atom variant)
  uvx modal run benchmarks/public/holographic_memory_proto.py --stage export-turns --out <dir>
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import longmemeval_modal as M  # noqa: E402
import longmemeval_pipeline as P  # noqa: E402

V = M.V
TAG = "holo-struct-dev"
STRUCT_CHUNK = 25

STRUCT_PROMPT = """Below are facts about a user, extracted from one conversation with an AI assistant and numbered.

Rewrite each fact as one or more structured records with these fields:
"n": the fact's number;
"entity": who or what the record is about: "user" for the user themself, otherwise the person, pet, object, place or organization as the user names it (for example "sister", "dog Max", "car");
"attribute": what is stated about the entity, as a short noun phrase (for example "favorite color", "job", "city", "dog's name", "purchase", "plan", "opinion of X");
"value": the stated value, as short as possible but keeping names, numbers, units and specifics;
"when": the time the fact refers to, exactly as stated in the fact (for example "last week", "in March", "next year"), or null if none.
A fact that states several things becomes several records. Keep every specific.

Answer with JSON only: {{"records": [{{"n": 1, "entity": "...", "attribute": "...", "value": "...", "when": null}}]}}.

Facts:
{facts}"""

QUERY_PROMPT = """A user asks an AI assistant a question about their earlier conversations with it. Today is {today}.

Question: {question}

The assistant's memory stores records of the form (entity, attribute, value, date), where entity is "user" for the user themself or the person, pet, object, place or organization as the user names it, and attribute is a short noun phrase such as "favorite color", "job", "dog's name" or "purchase". Turn the question into one lookup.

Answer with JSON only:
{{"entity": "<entity>", "attribute": "<attribute>", "wants": "<latest|earliest|in_range|any>", "time_range": ["YYYY/MM/DD", "YYYY/MM/DD"] or null, "needs_aggregation": true or false}}
"wants" is "latest" when the question asks for the current or most recent value (for example "now", "currently", "what did I change it to"), "earliest" for the first value, "in_range" when the question names a period (resolved against today into "time_range"), otherwise "any". "needs_aggregation" is true when the answer needs several records combined (a count, a total, a comparison of events, a list), false otherwise."""

JUDGE_PROMPT = """A user asked an AI assistant a question about their earlier conversations with it. A memory system returned one record. Decide whether the record answers the question correctly.

Question: {question}
Gold answer: {answer}
Returned record: {candidate}

The record is correct if it states the gold answer or information from which the gold answer follows directly; a record that gives a different value, an unrelated fact, or no usable information is incorrect. Answer with JSON only: {{"correct": true or false}}."""


# -------------------------------------------------------------------------------------- Modal

app = modal.App("hms-holo")
_sources = ("longmemeval_pipeline", "longmemeval_modal", "holographic_memory_proto")
_gpu_kw = dict(M._gpu_kw, image=M.gpu_image.add_local_python_source(*_sources))
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy<2.3").env(M.HF).add_local_python_source(*_sources)


@app.function(gpu="L4", **_gpu_kw)
def holo_llm_l4(*a):
    return M._llm(*a)


@app.function(gpu="T4", **_gpu_kw)
def holo_embed_t4(*a):
    return M._embed(*a)


# longmemeval_modal's stage helpers dispatch through its FN table; this app registers its own functions.
M.FN = {("llm", "L4"): holo_llm_l4, ("embed", "T4"): holo_embed_t4}


def _llm_id() -> str:
    m = M.MODELS["small"]["llm"]
    return f"{m[0]}@{m[1]}"


def struct_inputs(s: P.Session, facts: dict, llm: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """(cache key, [(turn index, fact)]) per chunk of the session's cached facts."""
    sf = P.session_facts(s, facts, llm)
    out = []
    for i in range(0, len(sf), STRUCT_CHUNK):
        chunk = sf[i:i + STRUCT_CHUNK]
        text = "\n".join(f"[{n + 1}] {f}" for n, (_, f) in enumerate(chunk))
        out.append((P.sha("holo-struct", llm, STRUCT_PROMPT, text), chunk))
    return out


def _struct_prompt(chunk) -> str:
    return STRUCT_PROMPT.format(facts="\n".join(f"[{n + 1}] {f}" for n, (_, f) in enumerate(chunk)))


def _query_key(q: P.Question, llm: str) -> str:
    return P.sha("holo-query", llm, QUERY_PROMPT, q.date_text, q.text)


def judge_key(llm: str, question: str, answer: str, cand: str) -> str:
    return P.sha("holo-judge", llm, JUDGE_PROMPT, question, answer, cand)


def _records(parsed: dict | None, chunk) -> list[dict]:
    out = []
    for r in (parsed or {}).get("records", []) or []:
        if not isinstance(r, dict):
            continue
        n = r.get("n")
        n = int(n) if isinstance(n, int) or (isinstance(n, str) and n.isdigit()) else 0
        if not (1 <= n <= len(chunk)):
            continue
        e, a, v = (str(r.get(k) or "").strip() for k in ("entity", "attribute", "value"))
        if not (e and a and v):
            continue
        w = r.get("when")
        out.append({"n": n, "entity": e, "attribute": a, "value": v, "when": str(w).strip() if w else None})
    return out


def _top1_run(finished: str) -> Path:
    for d in sorted((V / "runs").iterdir()):
        r = d / "result.json"
        if r.exists() and json.loads(r.read_text()).get("finished") == finished:
            return d
    raise RuntimeError(f"no run under {V / 'runs'} finished at {finished}")


@app.function(image=cpu_image, volumes={"/vol": M.VOL}, timeout=6 * 3600, cpu=4, memory=16384)
def holo_driver(stage: str, payload: dict, cap: float, tag: str) -> dict:
    import numpy as np

    budget = M.Budget(cap, V / "ledger" / f"{tag}.json")
    llm = _llm_id()
    ids = set(payload["dev_ids"])
    data = M._dataset("s")
    qs = P.load(data, ids)
    raw = {e["question_id"]: str(e["answer"]) for e in json.loads(data.read_text()) if e["question_id"] in ids}

    if stage == "judge":
        byq = {q.qid: q for q in qs}
        need = {}
        for c in payload["candidates"]:
            q = byq[c["q"]]
            need[judge_key(llm, q.text, raw[q.qid], c["cand"])] = JUDGE_PROMPT.format(
                question=q.text, answer=raw[q.qid], candidate=c["cand"][:6000])
        cache = M._llm_stage("holo-judge", "small", need, budget)
        verdicts = {}
        for k in need:
            v = cache.get(k)
            verdicts[k] = v.get("correct") if isinstance(v, dict) and isinstance(v.get("correct"), bool) else None
        budget.persist()
        return {"verdicts": verdicts, "cost": {"estimated_usd": round(budget.total(), 4), "calls": budget.log}}

    if stage == "export-turns":
        # turn-atom variant: user-turn embeddings (cached pipeline keys) and the LLM rewrite's query embedding
        run = _top1_run(payload["top1_finished"])
        queries = json.loads((run / "queries.json").read_text())["by_key"]
        key_cache, query_cache = M.EmbedCache("small", False), M.EmbedCache("small", True)
        out_q, texts, rewrites = [], [], []
        for q in qs:
            r = queries.get(P.query_key(q, llm)) or {}
            rw = r["rewrite"].strip() if isinstance(r.get("rewrite"), str) and r["rewrite"].strip() else q.text
            rewrites.append(rw)
            sess = []
            for s in q.sessions:
                sess.append({"sid": s.sid, "date": s.date_text, "tids": [t for t, _, _ in s.turns]})
                texts.extend(u for _, u, _ in s.turns)
            out_q.append({"qid": q.qid, "qtype": q.qtype, "question": q.text, "date": q.date_text, "rewrite": rw, "sessions": sess})
        query_cache.fill(sorted(set(rewrites)), "small", budget)
        key_cache.fill(sorted(set(texts)), "small", budget)
        budget.persist()
        return {"export": {"questions": out_q, "cost": {"estimated_usd": round(budget.total(), 4)}},
                "turn_emb": key_cache.get(texts).astype(np.float16).tobytes(),
                "q_emb": query_cache.get([q.text for q in qs]).astype(np.float16).tobytes(),
                "rewrite_emb": query_cache.get(rewrites).astype(np.float16).tobytes()}

    facts = M._load_json_cache(M._cache_dir("facts", *M.MODELS["small"]["llm"][:2]))
    missing = {P.fact_key(s, llm) for q in qs for s in q.sessions} - set(facts)
    if missing:
        raise RuntimeError(f"{len(missing)} dev sessions have no cached fact extraction; not regenerating")

    need = {}
    chunks = {}
    for q in qs:
        for s in q.sessions:
            for key, chunk in struct_inputs(s, facts, llm):
                chunks[key] = chunk
                need[key] = _struct_prompt(chunk)
    struct = M._llm_stage("holo-struct", "small", need, budget)
    qneed = {_query_key(q, llm): QUERY_PROMPT.format(today=q.date_text, question=q.text) for q in qs}
    parsed_q = M._llm_stage("holo-query", "small", qneed, budget)

    struct_out = {k: {"records": _records(struct.get(k), c), "facts": c} for k, c in chunks.items()}
    strings = set()
    for v in struct_out.values():
        for r in v["records"]:
            strings.update((r["entity"], r["attribute"], r["value"]))
        strings.update(f for _, f in v["facts"])
    parsed = {}
    for q in qs:
        p = parsed_q.get(_query_key(q, llm)) or {}
        parsed[q.qid] = {k: p.get(k) for k in ("entity", "attribute", "wants", "time_range", "needs_aggregation")}
        strings.update(str(p.get(k) or "") for k in ("entity", "attribute"))
    strings.discard("")
    strings = sorted(strings)
    key_cache, query_cache = M.EmbedCache("small", False), M.EmbedCache("small", True)
    key_cache.fill(strings, "small", budget)
    query_cache.fill(sorted({q.text for q in qs}), "small", budget)
    emb = key_cache.get(strings).astype(np.float16)
    qemb = query_cache.get([q.text for q in qs]).astype(np.float16)

    run = _top1_run(payload["top1_finished"])
    scores = P.load_scores(run / "scores.json")
    targets = json.loads((run / "targets.json").read_text())
    rerank = json.loads((run / "rerank.json").read_text()) if (run / "rerank.json").exists() else None
    queries = json.loads((run / "queries.json").read_text())["by_key"] if (run / "queries.json").exists() else None
    cfg = payload["top1_config"]
    rankings, _ = P.run(qs, scores, targets, cfg["turn"], cfg["session"], M._eval_utils()[0], queries, llm, rerank)

    out_q = []
    for q in qs:
        turn = {tid: (s, user) for s in q.sessions for tid, user, _ in s.turns}
        tid = rankings["turn"][q.qid][0]
        s, user = turn[tid]
        out_q.append({
            "qid": q.qid, "qtype": q.qtype, "question": q.text, "date": q.date_text, "answer": raw[q.qid],
            "parsed": parsed[q.qid],
            "top1": {"tid": tid, "sid": s.sid, "date": s.date_text, "text": f"Conversation date: {s.date_text}\nuser: {user[:3000]}"},
            "top5": [{"tid": t, "sid": turn[t][0].sid, "date": turn[t][0].date_text,
                      "text": f"Conversation date: {turn[t][0].date_text}\nuser: {turn[t][1][:1100]}"}
                     for t in rankings["turn"][q.qid][:5]],
            "session_ranking": rankings["session"][q.qid][:10],
            "sessions": [{"sid": s.sid, "date": s.date_text, "turn_ids": [t for t, _, _ in s.turns],
                          "skeys": [k for k, _ in struct_inputs(s, facts, llm)]} for s in q.sessions],
        })
    budget.persist()
    export = {"llm": llm, "embed": M.MODELS["small"]["embed"][0], "top1_run": run.name, "questions": out_q,
              "struct": struct_out, "strings": strings,
              "cost": {"estimated_usd": round(budget.total(), 4), "calls": budget.log}}
    return {"export": export, "emb": emb.tobytes(), "qemb": qemb.tobytes(), "dim": int(emb.shape[1])}


@app.local_entrypoint()
def main(stage: str = "export", cap: float = 5.0, tag: str = TAG, out: str = "", candidates: str = ""):
    split = json.loads((HERE / "longmemeval_split.json").read_text())
    payload = {"dev_ids": split["dev"]}
    if stage == "export":
        final = json.loads((HERE.parents[1] / "benchmarks" / "results" / "longmemeval_dev_final.json").read_text())
        payload.update(top1_finished=final["finished"], top1_config=final["config"])
        r = holo_driver.remote(stage, payload, cap, tag)
        d = Path(out or ".")
        d.mkdir(parents=True, exist_ok=True)
        (d / "export_dev.json").write_text(json.dumps(r["export"]))
        (d / "emb.f16").write_bytes(r["emb"])
        (d / "qemb.f16").write_bytes(r["qemb"])
        (d / "dim.json").write_text(json.dumps({"dim": r["dim"]}))
        print(f"exported {len(r['export']['questions'])} questions, {len(r['export']['strings'])} strings, "
              f"cost ${r['export']['cost']['estimated_usd']:.3f}")
    elif stage == "export-turns":
        final = json.loads((HERE.parents[1] / "benchmarks" / "results" / "longmemeval_dev_final.json").read_text())
        payload.update(top1_finished=final["finished"])
        r = holo_driver.remote(stage, payload, cap, tag)
        d = Path(out or ".")
        d.mkdir(parents=True, exist_ok=True)
        (d / "turns_dev.json").write_text(json.dumps(r["export"]))
        for k in ("turn_emb", "q_emb", "rewrite_emb"):
            (d / f"{k}.f16").write_bytes(r[k])
        print(f"exported turns for {len(r['export']['questions'])} questions, cost ${r['export']['cost']['estimated_usd']:.3f}")
    elif stage == "judge":
        payload["candidates"] = json.loads(Path(candidates).read_text())
        r = holo_driver.remote(stage, payload, cap, tag)
        Path(out).write_text(json.dumps(r))
        print(f"judged {len(r['verdicts'])}, cost ${r['cost']['estimated_usd']:.3f}")
    else:
        sys.exit("bad --stage")


# -------------------------------------------------------------------------------------- numpy

FRACS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # fraction of trace bits flipped
DELETE = [0.0, 0.1, 0.25, 0.5]  # fraction of storage deleted
SHARDS = 8
FINE_M = 5  # sessions resolved at the fine level after the coarse (user-trace) stage


def _bucket(d: date | None, origin: date) -> int:
    d = d or origin
    return (d.year - origin.year) * 12 + d.month - origin.month


class Codes:
    """Sign codes of embeddings under one fixed Gaussian projection, packed bits.

    Embeddings are centered on `mean` first: the raw Qwen3 embeddings share a common direction
    (mean pairwise cosine 0.43 over the dev strings), so uncentered sign codes agree on 64% of their
    bits and every superposition collapses onto that shared component (measured in `capacity`)."""

    def __init__(self, dim: int, D: int, mean=None, seed: int = 7):
        import numpy as np
        self.D = D
        self.R = np.random.default_rng(seed).standard_normal((dim, D), dtype=np.float32)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.rng = np.random.default_rng(seed + 1)

    def of(self, emb):
        import numpy as np
        x = emb.astype(np.float32)
        if self.mean is not None:
            x = x - self.mean
        return np.packbits((x @ self.R) > 0, axis=-1)

    def random(self, n: int):
        import numpy as np
        return np.packbits(self.rng.integers(0, 2, (n, self.D), dtype=np.uint8), axis=-1)


POP = None


def hamming(a, b):
    """Hamming distance between packed code a (1-D) and rows of b."""
    import numpy as np
    global POP
    if POP is None:
        POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    return POP[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int64)


def majority(rows, tie):
    """Majority vote over packed rows; even ties broken by the packed vector `tie`."""
    import numpy as np
    bits = np.unpackbits(rows, axis=-1).astype(np.int32).sum(axis=0)
    n = rows.shape[0]
    out = bits * 2 > n
    if n % 2 == 0:
        out = np.where(bits * 2 == n, np.unpackbits(tie).astype(bool), out)
    return np.packbits(out)


def flip(code, frac: float, rng):
    """Flip a `frac` fraction of a packed code's bits."""
    import numpy as np
    nbits = code.size * 8
    k = int(round(frac * nbits))
    if k == 0:
        return code.copy()
    mask = np.zeros(nbits, dtype=np.uint8)
    mask[rng.choice(nbits, k, replace=False)] = 1
    return np.bitwise_xor(code, np.packbits(mask))


def time_codes(n_buckets: int, D: int, rng):
    """Month codes: adjacent months share all but D/(2 n) bits, far months are uncorrelated."""
    import numpy as np
    base = rng.integers(0, 2, D, dtype=np.uint8)
    order = rng.permutation(D)
    step = max(1, D // (2 * n_buckets))
    out = []
    for b in range(n_buckets):
        c = base.copy()
        c[order[: b * step]] ^= 1
        out.append(np.packbits(c))
    return np.stack(out)


def xor(*codes):
    import numpy as np
    out = codes[0]
    for c in codes[1:]:
        out = np.bitwise_xor(out, c)
    return out


# capacity ---------------------------------------------------------------------------------


def capacity(export_dir: Path, out: Path) -> None:
    """Facts per trace versus cleanup accuracy: BSC (XOR / majority) and HRR (circular convolution),
    sign codes of real attribute / value embeddings against random codes, flat and two-level traces."""
    import numpy as np
    ex = json.loads((export_dir / "export_dev.json").read_text())
    dim = json.loads((export_dir / "dim.json").read_text())["dim"]
    emb = np.fromfile(export_dir / "emb.f16", dtype=np.float16).reshape(-1, dim)
    pos = {s: i for i, s in enumerate(ex["strings"])}
    attrs = sorted({r["attribute"] for v in ex["struct"].values() for r in v["records"]})
    vals = sorted({r["value"] for v in ex["struct"].values() for r in v["records"]})
    rng = np.random.default_rng(0)
    vals = [vals[i] for i in rng.choice(len(vals), min(4000, len(vals)), replace=False)]
    ea, ev = emb[[pos[a] for a in attrs]], emb[[pos[v] for v in vals]]
    Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    trials = 160
    res = {"codebook_values": len(vals), "attributes": len(attrs), "queries_per_K": trials, "K": Ks, "curves": {},
           "note": "accuracy = cleanup picks the stored value among the codebook; attributes sampled without "
                   "replacement, values with replacement; two-level = facts in sessions of 8, session traces superposed"}

    def run_bsc(ca, cv, D, two_level):
        tie = np.packbits(rng.integers(0, 2, D, dtype=np.uint8))
        acc = []
        for K in Ks:
            hit, tot = 0, 0
            for _ in range(max(1, trials // K)):
                ai = rng.choice(len(attrs), K, replace=False)
                vi = rng.choice(len(vals), K, replace=True)
                fv = np.bitwise_xor(ca[ai], cv[vi])
                if two_level and K > 8:
                    sess = np.stack([majority(fv[i:i + 8], tie) for i in range(0, K, 8)])
                    trace = majority(sess, tie)
                else:
                    trace = majority(fv, tie)
                probe = np.bitwise_xor(trace[None], ca[ai])
                hit += sum(int(np.argmin(hamming(probe[j], cv)) == vi[j]) for j in range(K))
                tot += K
            acc.append(round(hit / tot, 3))
        return acc

    mean = emb.astype(np.float32).mean(axis=0)
    for D in (8192, 16384, 32768, 65536):
        C = Codes(dim, D, mean)
        kinds = [("sign", (C.of(ea), C.of(ev))), ("random", (C.random(len(attrs)), C.random(len(vals))))]
        if D == 16384:
            raw = Codes(dim, D)
            kinds.append(("sign_uncentered", (raw.of(ea), raw.of(ev))))
        for kind, (ca, cv) in kinds:
            res["curves"][f"bsc_{kind}_D{D}"] = run_bsc(ca, cv, D, False)
            print(f"bsc {kind} D={D}: {res['curves'][f'bsc_{kind}_D{D}']}", flush=True)
            if kind == "sign" and D in (16384, 65536):
                res["curves"][f"bsc_{kind}_two_level_D{D}"] = run_bsc(ca, cv, D, True)
                print(f"bsc two-level {kind} D={D}: {res['curves'][f'bsc_{kind}_two_level_D{D}']}", flush=True)
        if D > 16384:
            continue
        # HRR: bipolar codes of the same sign projection, circular convolution binding, sum superposition
        fa = np.fft.rfft(np.where(np.unpackbits(C.of(ea), axis=-1) > 0, 1.0, -1.0), axis=-1)
        bv = np.where(np.unpackbits(C.of(ev), axis=-1) > 0, 1.0, -1.0)
        fv = np.fft.rfft(bv, axis=-1)
        acc = []
        for K in Ks:
            hit, tot = 0, 0
            for _ in range(max(1, trials // K)):
                ai = rng.choice(len(attrs), K, replace=False)
                vi = rng.choice(len(vals), K, replace=True)
                trace = (fa[ai] * fv[vi]).sum(axis=0)
                probe = np.fft.irfft(trace[None] * np.conj(fa[ai]), n=D, axis=-1)
                hit += int((np.argmax(probe @ bv.T, axis=1) == vi).sum())
                tot += K
            acc.append(round(hit / tot, 3))
        res["curves"][f"hrr_sign_D{D}"] = acc
        print(f"hrr sign D={D}: {acc}", flush=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


# memory -----------------------------------------------------------------------------------


class Memory:
    """One question's holographic memory.

    records   (entity, attribute, value, bucket, session index, sid, tid, fact)
    flat      per entity: majority of A (x) V (x) T over its records (and a no-time variant)
    sessions  per entity and session: majority of A (x) V (x) T (x) S over the session's records
    user      per entity: majority of its session traces (the hierarchical trace)
    """

    def __init__(self, q: dict, ex: dict, C: Codes, emb, pos: dict, origin: date, tcodes, seed: int):
        import numpy as np
        self.C, self.D = C, C.D
        self.sess = []  # (sid, bucket, date)
        self.recs = []
        for j, s in enumerate(q["sessions"]):
            sd = P.parse_date(s["date"])
            self.sess.append((s["sid"], _bucket(sd, origin), sd))
            for k in s["skeys"]:
                st = ex["struct"][k]
                for r in st["records"]:
                    ti, fact = st["facts"][r["n"] - 1]
                    tid = s["turn_ids"][ti] if ti >= 0 else None
                    self.recs.append((r["entity"], r["attribute"], r["value"], self.sess[-1][1], j, s["sid"], tid, fact))
        self.entities = sorted({r[0] for r in self.recs})
        self.values = sorted({r[2] for r in self.recs})
        self.vpos = {v: i for i, v in enumerate(self.values)}
        attrs = sorted({r[1] for r in self.recs})
        self.acode = dict(zip(attrs, C.of(emb[[pos[a] for a in attrs]]))) if attrs else {}
        self.ecode = C.of(emb[[pos[e] for e in self.entities]]) if self.entities else None
        self.vcode = C.of(emb[[pos[v] for v in self.values]]) if self.values else None
        self.tcodes = tcodes
        rng = np.random.default_rng(seed)
        self.scode = np.packbits(rng.integers(0, 2, (len(self.sess), C.D), dtype=np.uint8), axis=-1)
        self.tie = np.packbits(rng.integers(0, 2, C.D, dtype=np.uint8))
        self.by_entity = {e: [r for r in self.recs if r[0] == e] for e in self.entities}
        self.fvec = {}  # record index -> A (x) V (x) T
        self.fvec_nt = {}
        self.fvec_s = {}  # A (x) V (x) T (x) S
        for i, r in enumerate(self.recs):
            av = np.bitwise_xor(self.acode[r[1]], self.vcode[self.vpos[r[2]]])
            self.fvec_nt[i] = av
            self.fvec[i] = np.bitwise_xor(av, tcodes[r[3]])
            self.fvec_s[i] = np.bitwise_xor(self.fvec[i], self.scode[r[4]])
        self.idx = {e: [i for i, r in enumerate(self.recs) if r[0] == e] for e in self.entities}
        self.flat = {e: majority(np.stack([self.fvec[i] for i in ii]), self.tie) for e, ii in self.idx.items()}
        self.flat_nt = {e: majority(np.stack([self.fvec_nt[i] for i in ii]), self.tie) for e, ii in self.idx.items()}
        self.sess_traces = {}  # e -> {j: trace}
        self.user = {}
        for e, ii in self.idx.items():
            byj = {}
            for i in ii:
                byj.setdefault(self.recs[i][4], []).append(self.fvec_s[i])
            self.sess_traces[e] = {j: majority(np.stack(v), self.tie) for j, v in byj.items()}
            self.user[e] = majority(np.stack(list(self.sess_traces[e].values())), self.tie)
        self.load = {e: len(ii) for e, ii in self.idx.items()}
        self.ops = 0

    # -- helpers

    def resolve(self, ecode) -> str | None:
        import numpy as np
        if self.ecode is None:
            return None
        return self.entities[int(np.argmin(hamming(ecode, self.ecode)))]

    def agree(self, probe):
        """Per-value agreement in [-1, 1] of a probe with the value codebook (one cleanup)."""
        self.ops += len(self.values) + 1
        return 1 - 2 * hamming(probe, self.vcode) / self.D

    def session_filter(self, js, wants, trange, origin, qdate):
        out = [j for j in js if self.sess[j][2] is None or qdate is None or self.sess[j][2] <= qdate] or list(js)
        if wants == "in_range" and trange:
            inside = [j for j in out if self.sess[j][2] and trange[0] <= self.sess[j][2] <= trange[1]]
            out = inside or out
        return out

    @staticmethod
    def pick(cands, wants, margin):
        """cands: (score, value index, order key); newest within `margin` of the best, or earliest."""
        best = max(c[0] for c in cands)
        ok = [c for c in cands if c[0] >= best - margin]
        return min(ok, key=lambda c: c[2]) if wants == "earliest" else max(ok, key=lambda c: (c[2], c[0]))

    def provenance(self, e, v, j=None, b=None):
        rs = [r for r in self.by_entity.get(e, []) if r[2] == v and (j is None or r[4] == j) and (b is None or r[3] == b)]
        rs = rs or [r for r in self.recs if r[2] == v]
        return rs[0] if rs else None

    # -- queries

    def query_flat(self, e, acode, wants, trange, origin, qbucket, margin, traces=None, with_time=True):
        """Unbind the entity's flat trace with the attribute (and each month code), clean up."""
        import numpy as np
        trace = (traces or (self.flat if with_time else self.flat_nt))[e]
        probe = np.bitwise_xor(trace, acode)
        if not with_time:
            a = self.agree(probe)
            j = int(np.argmax(a))
            return self.values[j], float(a[j]), None
        bs = sorted({r[3] for r in self.by_entity[e]})
        if wants == "in_range" and trange:
            lo, hi = _bucket(trange[0], origin), _bucket(trange[1], origin)
            bs = [b for b in bs if lo <= b <= hi] or bs
        bs = [b for b in bs if b <= qbucket] or bs
        cands = []
        for b in bs:
            a = self.agree(np.bitwise_xor(probe, self.tcodes[b]))
            j = int(np.argmax(a))
            cands.append((float(a[j]), j, b))
        s, j, b = self.pick(cands, wants, margin)
        return self.values[j], s, b

    def query_sharded(self, e, acode, wants, trange, origin, qbucket, margin, shards):
        """Flat query where the trace is a set of shard traces; agreements are averaged over shards."""
        import numpy as np
        bs = sorted({r[3] for r in self.by_entity[e]})
        if wants == "in_range" and trange:
            lo, hi = _bucket(trange[0], origin), _bucket(trange[1], origin)
            bs = [b for b in bs if lo <= b <= hi] or bs
        bs = [b for b in bs if b <= qbucket] or bs
        cands = []
        for b in bs:
            a = np.mean([self.agree(xor(t, acode, self.tcodes[b])) for t in shards], axis=0)
            j = int(np.argmax(a))
            cands.append((float(a[j]), j, b))
        s, j, b = self.pick(cands, wants, margin)
        return self.values[j], s, b

    def rank_sessions(self, e, acode, wants, trange, origin, qdate, traces):
        """Coarse stage: unbind a trace (the user trace, or shard traces) with attribute, month and
        session code of every candidate session; sessions ranked by their best cleanup agreement."""
        import numpy as np
        js = self.session_filter(sorted(self.sess_traces[e]), wants, trange, origin, qdate)
        out = []
        for j in js:
            key = xor(acode, self.tcodes[self.sess[j][1]], self.scode[j])
            a = np.mean([self.agree(np.bitwise_xor(t, key)) for t in traces], axis=0)
            v = int(np.argmax(a))
            out.append((float(a[v]), v, j))
        return sorted(out, key=lambda c: -c[0])

    def query_hier(self, e, acode, wants, trange, origin, qdate, margin, m=FINE_M):
        """Coarse-to-fine: user trace -> top-m sessions -> each session trace unbound and cleaned up."""
        import numpy as np
        coarse = self.rank_sessions(e, acode, wants, trange, origin, qdate, [self.user[e]])
        fine = []
        for _, _, j in coarse[:m]:
            key = xor(acode, self.tcodes[self.sess[j][1]], self.scode[j])
            a = self.agree(np.bitwise_xor(self.sess_traces[e][j], key))
            v = int(np.argmax(a))
            fine.append((float(a[v]), v, j))
        if not fine:
            return None
        s, v, (_, j) = self.pick([(s, v, (self.sess[j][2] or origin, j)) for s, v, j in fine], wants, margin)
        return self.values[v], s, j, coarse, fine

    def shards(self, K, rng):
        """K shard traces per entity, each the majority of a random half of the entity's records
        (every record in at least one shard); likewise for the session traces (hierarchical shards)."""
        import numpy as np
        flat, hier = {}, {}
        for e, ii in self.idx.items():
            memb = rng.random((len(ii), K)) < 0.5
            for row in memb:
                if not row.any():
                    row[rng.integers(K)] = True
            flat[e] = [majority(np.stack([self.fvec[i] for i, m in zip(ii, memb[:, k]) if m]), self.tie)
                       if memb[:, k].any() else None for k in range(K)]
            js = sorted(self.sess_traces[e])
            memb = rng.random((len(js), K)) < 0.5
            for row in memb:
                if not row.any():
                    row[rng.integers(K)] = True
            hier[e] = [majority(np.stack([self.sess_traces[e][j] for j, m in zip(js, memb[:, k]) if m]), self.tie)
                       if memb[:, k].any() else None for k in range(K)]
        return flat, hier


def _sparse_code(emb, R, k=64):
    import numpy as np
    proj = emb.astype(np.float32) @ R
    out = np.zeros(proj.shape, dtype=np.uint8)
    top = np.argpartition(-proj, k, axis=-1)[:, :k]
    np.put_along_axis(out, top, 1, axis=-1)
    return out


def recall(ranked: list[str], gold: set[str]) -> dict:
    out = {}
    for k in (5, 10):
        head = set(ranked[:k])
        out[f"any@{k}"] = float(bool(head & gold))
        out[f"all@{k}"] = float(gold <= head)
    return out


def evaluate(export_dir: Path, cands_out: Path, state_out: Path, D: int = 16384, margin: float = 0.02) -> None:
    import numpy as np
    ex = json.loads((export_dir / "export_dev.json").read_text())
    dim = json.loads((export_dir / "dim.json").read_text())["dim"]
    emb = np.fromfile(export_dir / "emb.f16", dtype=np.float16).reshape(-1, dim)
    qemb = np.fromfile(export_dir / "qemb.f16", dtype=np.float16).reshape(-1, dim)
    pos = {s: i for i, s in enumerate(ex["strings"])}
    C = Codes(dim, D, emb.astype(np.float32).mean(axis=0))
    rng = np.random.default_rng(11)
    dates = [P.parse_date(s["date"]) for q in ex["questions"] for s in q["sessions"]] + [P.parse_date(q["date"]) for q in ex["questions"]]
    dates = [d for d in dates if d]
    origin = date(min(dates).year, min(dates).month, 1)
    n_buckets = _bucket(max(dates), origin) + 2
    tcodes = time_codes(n_buckets, D, rng)
    Rs = np.random.default_rng(5).standard_normal((dim, 16384), dtype=np.float32)
    cands: dict[tuple[str, str], None] = {}
    state = {"D": D, "margin": margin, "fracs": FRACS, "delete": DELETE, "shards": SHARDS, "fine_m": FINE_M,
             "origin": origin.isoformat(), "buckets": n_buckets, "questions": {}}

    def add(qid, cand):
        cands[(qid, cand)] = None
        return cand

    def rec_cand(e, att, v, prov):
        return f"{e} / {att}: {v}" + (f"\n(source: {prov[7]})" if prov else "")

    for qi, q in enumerate(ex["questions"]):
        qid, p = q["qid"], q["parsed"]
        ent, att = str(p.get("entity") or "user"), str(p.get("attribute") or q["question"])
        ent = ent if ent in pos else "user"
        att = att if att in pos else q["question"]
        ecode = C.of(emb[pos[ent]][None])[0] if ent in pos else C.random(1)[0]
        acode = C.of(emb[pos[att]][None])[0] if att in pos else C.random(1)[0]
        wants = p.get("wants") if p.get("wants") in ("latest", "earliest", "in_range", "any") else "any"
        tr = p.get("time_range")
        trange = None
        if isinstance(tr, list) and len(tr) == 2:
            a, b = P.parse_date(str(tr[0])), P.parse_date(str(tr[1]))
            if a and b:
                trange = (min(a, b), max(a, b))
        qdate = P.parse_date(q["date"])
        qbucket = _bucket(qdate, origin)
        gold = {s["sid"] for s in q["sessions"] if "answer" in s["sid"]}
        all_sids = [s["sid"] for s in q["sessions"]]
        mem = Memory(q, ex, C, emb, pos, origin, tcodes, seed=qi)
        e = mem.resolve(ecode)
        rec = {"qtype": q["qtype"], "answer": q["answer"], "parsed": p, "entity_used": ent, "attribute_used": att,
               "entity_resolved": e, "n_records": len(mem.recs), "n_entities": len(mem.entities), "n_values": len(mem.values),
               "n_sessions": len(mem.sess), "entity_load": mem.load.get(e, 0), "max_load": max(mem.load.values(), default=0),
               "n_gold_sessions": len(gold),
               "top1": add(qid, q["top1"]["text"]), "top1_tid": q["top1"]["tid"],
               "top5": add(qid, "\n\n".join(t["text"] for t in q["top5"])),
               "pipeline_session_recall": recall(q["session_ranking"], gold) if gold else None,
               "holo": {}, "hier": {}, "shard": {}, "control": {}, "ops": {}}
        if e is None:
            state["questions"][qid] = rec
            continue
        # flat traces: with time (plus bit-flip corruption) and without time
        for with_time in (True, False):
            name = "time" if with_time else "notime"
            for frac in (FRACS if with_time else [0.0]):
                traces = None
                if frac > 0:
                    crng = np.random.default_rng(int(frac * 100) + qi * 1000)
                    traces = {k: flip(t, frac, crng) for k, t in mem.flat.items()}
                mem.ops = 0
                v, s, b = mem.query_flat(e, acode, wants, trange, origin, qbucket, margin, traces, with_time)
                prov = mem.provenance(e, v, b=b)
                rec["holo"][f"{name}@{frac}"] = {"value": v, "score": s, "bucket": b, "sid": prov[5] if prov else None,
                                                 "tid": prov[6] if prov else None, "cand": add(qid, rec_cand(e, att, v, prov))}
                if frac == 0:
                    rec["ops"][f"flat_{name}"] = mem.ops
        # hierarchical: user trace -> sessions -> facts; compositional list across sessions
        mem.ops = 0
        h = mem.query_hier(e, acode, wants, trange, origin, qdate, margin)
        rec["ops"]["hier"] = mem.ops
        if h:
            v, s, j, coarse, fine = h
            prov = mem.provenance(e, v, j=j)
            ranked = [mem.sess[j][0] for _, _, j in coarse]
            ranked += [sid for sid in all_sids if sid not in set(ranked)]
            comp = [(mem.sess[j][2], mem.values[vv], mem.provenance(e, mem.values[vv], j=j)) for ss, vv, j in fine]
            comp_text = "records:\n" + "\n".join(f"- {d}: {att} = {vv}" + (f" (source: {pr[7]})" if pr else "") for d, vv, pr in comp)
            rec["hier"] = {"value": v, "score": s, "sid": mem.sess[j][0], "tid": prov[6] if prov else None,
                           "cand": add(qid, rec_cand(e, att, v, prov)), "comp_cand": add(qid, comp_text),
                           "session_recall": recall(ranked, gold) if gold else None,
                           "coarse_top": [(round(s, 4), mem.sess[j][0]) for s, _, j in coarse[:10]],
                           "n_candidate_sessions": len(coarse)}
        # holographic storage: K shards with redundancy, then shard deletion
        srng = np.random.default_rng(qi * 7 + 3)
        flat_sh, hier_sh = mem.shards(SHARDS, srng)
        for frac in DELETE:
            keep = sorted(srng.choice(SHARDS, SHARDS - int(round(frac * SHARDS)), replace=False))
            fs = [t for k, t in enumerate(flat_sh[e]) if k in keep and t is not None]
            hs = [t for k, t in enumerate(hier_sh[e]) if k in keep and t is not None]
            entry = {"kept": len(keep)}
            if fs:
                v, s, b = mem.query_sharded(e, acode, wants, trange, origin, qbucket, margin, fs)
                prov = mem.provenance(e, v, b=b)
                entry.update(value=v, score=s, cand=add(qid, rec_cand(e, att, v, prov)))
            if hs and gold:
                coarse = mem.rank_sessions(e, acode, wants, trange, origin, qdate, hs)
                ranked = [mem.sess[j][0] for _, _, j in coarse]
                ranked += [sid for sid in all_sids if sid not in set(ranked)]
                entry["session_recall"] = recall(ranked, gold)
            rec["shard"][str(frac)] = entry
        # controls: per-fact index of the fact sentences (sparse top-64 codes; dense sign codes),
        # question (query embedding) as the probe; bit-flip corruption and item deletion
        facts = [(r[7], r[5], r[6], r[4]) for r in mem.recs]
        seen, ufacts = set(), []
        for f in facts:
            if f[0] not in seen:
                seen.add(f[0])
                ufacts.append(f)
        if ufacts:
            fe = emb[[pos[f[0]] for f in ufacts]]
            qe = qemb[qi][None]
            qsparse, fsparse = _sparse_code(qe, Rs)[0], _sparse_code(fe, Rs)
            qdense, fdense = C.of(qe)[0], C.of(fe)
            rec["ops"]["dense_index"] = len(ufacts)
            for frac in FRACS:
                crng = np.random.default_rng(int(frac * 100) + qi * 1000 + 7)
                if frac > 0:
                    fs_ = np.bitwise_xor(fsparse, (crng.random(fsparse.shape) < frac).astype(np.uint8))
                    fd_ = np.stack([flip(c, frac, crng) for c in fdense])
                else:
                    fs_, fd_ = fsparse, fdense
                inter = (fs_ & qsparse).sum(axis=1)
                js, jd = int(np.argmax(inter)), int(np.argmin(hamming(qdense, fd_)))
                rec["control"][f"sparse@{frac}"] = {"fact": ufacts[js][0], "overlap": int(inter[js]), "sid": ufacts[js][1],
                                                    "tid": ufacts[js][2], "cand": add(qid, ufacts[js][0])}
                rec["control"][f"dense@{frac}"] = {"fact": ufacts[jd][0], "sid": ufacts[jd][1], "tid": ufacts[jd][2],
                                                   "cand": add(qid, ufacts[jd][0])}
            d0 = hamming(qdense, fdense)
            for frac in DELETE:
                drng = np.random.default_rng(qi * 13 + int(frac * 100))
                alive = np.flatnonzero(drng.random(len(ufacts)) >= frac)
                if len(alive) == 0:
                    rec["control"][f"deleted@{frac}"] = {"cand": None}
                    continue
                order = alive[np.argsort(d0[alive], kind="stable")]
                jd = int(order[0])
                ranked = []
                for i in order:
                    if ufacts[i][1] not in ranked:
                        ranked.append(ufacts[i][1])
                ranked += [sid for sid in all_sids if sid not in set(ranked)]
                rec["control"][f"deleted@{frac}"] = {"fact": ufacts[jd][0], "sid": ufacts[jd][1], "tid": ufacts[jd][2],
                                                     "cand": add(qid, ufacts[jd][0]), "alive": int(len(alive)),
                                                     "session_recall": recall(ranked, gold) if gold else None}
        state["questions"][qid] = rec
        print(f"{qi + 1}/{len(ex['questions'])} {qid} {q['qtype']} recs={len(mem.recs)} load={mem.load.get(e, 0)} "
              f"ops flat={rec['ops'].get('flat_time')} hier={rec['ops'].get('hier')} idx={rec['ops'].get('dense_index')}", flush=True)
    cands_out.write_text(json.dumps([{"q": q, "cand": c} for q, c in cands]))
    state_out.write_text(json.dumps(state))
    print(f"{len(cands)} candidates to judge")


# report -----------------------------------------------------------------------------------


def report(export_dir: Path, state_path: Path, judged_path: Path, out: Path, capacity_path: Path | None) -> None:
    import numpy as np
    ex = json.loads((export_dir / "export_dev.json").read_text())
    state = json.loads(state_path.read_text())
    judged = json.loads(judged_path.read_text())
    qa = {q["qid"]: (q["question"], str(q["answer"])) for q in ex["questions"]}
    verdict = judged["verdicts"]

    def ok(qid, cand):
        return bool(cand) and bool(verdict.get(judge_key(ex["llm"], *qa[qid], cand)))

    rows = state["questions"]
    scored = {k: v for k, v in rows.items() if "_abs" not in k}
    types = sorted({v["qtype"] for v in scored.values()})

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 3) if xs else None

    def acc(sel, fn):
        return mean([fn(k, v) for k, v in sel.items()])

    def table(fn, sel=None):
        sel = sel or scored
        t = {"overall": acc(sel, fn)}
        t.update({ty: acc({k: v for k, v in sel.items() if v["qtype"] == ty}, fn) for ty in types})
        return t

    def holo(k, v, name="time@0.0"):
        h = v["holo"].get(name)
        return ok(k, h["cand"]) if h else False

    def hier(k, v, key="cand"):
        return ok(k, v["hier"].get(key)) if v.get("hier") else False

    def top1(k, v):
        return ok(k, v["top1"])

    def top5(k, v):
        return ok(k, v["top5"])

    def ctrl(k, v, name):
        return ok(k, (v["control"].get(name) or {}).get("cand"))

    def srec(sel, get, key):
        vals = []
        for v in sel.values():
            r = get(v)
            vals.append(r.get(key) if r else None)
        return mean(vals)

    def recall_table(get):
        sel = {k: v for k, v in scored.items() if v.get("pipeline_session_recall")}
        return {key: srec(sel, get, key) for key in ("any@5", "all@5", "any@10", "all@10")}

    n_by_type = {ty: sum(1 for v in scored.values() if v["qtype"] == ty) for ty in types}
    n_by_type["overall"] = len(scored)
    ops = lambda name: mean([v["ops"].get(name) for v in scored.values()])  # noqa: E731
    res = {
        "protocol": "LongMemEval_S dev split (benchmarks/public/longmemeval_split.json), 100 questions; the 6 abstention "
                    "questions are reported separately; answers judged by the same LLM for every system",
        "representation": {"kind": "binary spatter codes: XOR binding, majority superposition", "D": state["D"],
                           "component_codes": f"sign codes of {ex['embed']} embeddings under one fixed Gaussian projection",
                           "time": f"month-bucket codes with graded overlap, {state['buckets']} buckets from {state['origin']}",
                           "session_codes": "random", "margin": state["margin"], "fine_m": state["fine_m"], "shards": state["shards"]},
        "n_questions": n_by_type,
        "accuracy": {
            "holographic_flat": table(holo),
            "holographic_flat_no_time": table(lambda k, v: holo(k, v, "notime@0.0")),
            "holographic_hierarchical": table(hier),
            "holographic_compositional_top5_sessions": table(lambda k, v: hier(k, v, "comp_cand")),
            "retrieval_top1_turn": table(top1),
            "retrieval_top5_turns": table(top5),
            "either_flat_or_top1": table(lambda k, v: holo(k, v) or top1(k, v)),
            "either_any_holographic_or_top1": table(lambda k, v: holo(k, v) or hier(k, v) or hier(k, v, "comp_cand") or top1(k, v)),
            "control_sparse_fact_index": table(lambda k, v: ctrl(k, v, "sparse@0.0")),
            "control_dense_fact_index": table(lambda k, v: ctrl(k, v, "dense@0.0")),
        },
        "session_recall": {
            "note": "scored questions with gold sessions; user trace alone = coarse stage, no index",
            "user_trace_alone": recall_table(lambda v: (v.get("hier") or {}).get("session_recall")),
            "dense_fact_index": recall_table(lambda v: (v["control"].get("deleted@0.0") or {}).get("session_recall")),
            "tuned_pipeline": recall_table(lambda v: v.get("pipeline_session_recall")),
        },
        "query_cost_vector_ops": {
            "note": "D-bit XOR / Hamming operations per question (mean); cleanup against |values| counts |values| ops",
            "flat_time": ops("flat_time"), "flat_no_time": ops("flat_notime"), "hierarchical_coarse_to_fine": ops("hier"),
            "dense_fact_index_scan": ops("dense_index"),
            "values_per_store_mean": mean([v["n_values"] for v in scored.values()]),
            "sessions_per_store_mean": mean([v["n_sessions"] for v in scored.values()]),
        },
        "degradation_bit_flips": {}, "storage_deletion": {}, "temporal": {}, "compositional": {},
        "store_size": {
            "records_per_question_mean": mean([v["n_records"] for v in rows.values()]),
            "entities_per_question_mean": mean([v["n_entities"] for v in rows.values()]),
            "queried_entity_load_mean": mean([v["entity_load"] for v in rows.values()]),
            "queried_entity_load_max": max(v["entity_load"] for v in rows.values()),
            "queried_entity_sessions_mean": mean([(v.get("hier") or {}).get("n_candidate_sessions") for v in rows.values()]),
        },
        "judge": {"model": ex["llm"], "unparsed_verdicts": sum(1 for v in verdict.values() if v is None)},
        "cost_usd": {"export_stage": ex["cost"]["estimated_usd"], "cumulative_tag_ledger": judged["cost"]["estimated_usd"]},
    }
    for frac in state["fracs"]:
        f = str(frac)
        res["degradation_bit_flips"][f] = {
            "holographic_flat": acc(scored, lambda k, v: holo(k, v, f"time@{f}")),
            "holographic_same_answer_as_clean": acc(scored, lambda k, v: (v["holo"].get(f"time@{f}") or {}).get("value") == (v["holo"].get("time@0.0") or {}).get("value")),
            "control_sparse": acc(scored, lambda k, v: ctrl(k, v, f"sparse@{f}")),
            "control_dense": acc(scored, lambda k, v: ctrl(k, v, f"dense@{f}")),
        }
    for frac in state["delete"]:
        f = str(frac)
        res["storage_deletion"][f] = {
            "holographic_shards_answer": acc(scored, lambda k, v: ok(k, (v["shard"].get(f) or {}).get("cand"))),
            "holographic_shards_session_recall": recall_table(lambda v: (v["shard"].get(f) or {}).get("session_recall")),
            "index_answer": acc(scored, lambda k, v: ctrl(k, v, f"deleted@{f}")),
            "index_session_recall": recall_table(lambda v: (v["control"].get(f"deleted@{f}") or {}).get("session_recall")),
        }
    for ty in ("knowledge-update", "temporal-reasoning"):
        sel = {k: v for k, v in scored.items() if v["qtype"] == ty}
        res["temporal"][ty] = {"n": len(sel), "flat_with_time": acc(sel, holo),
                               "flat_without_time": acc(sel, lambda k, v: holo(k, v, "notime@0.0")),
                               "hierarchical": acc(sel, hier), "retrieval_top1": acc(sel, top1)}
    for ty in ("multi-session", "temporal-reasoning", "knowledge-update"):
        sel = {k: v for k, v in scored.items() if v["qtype"] == ty}
        res["compositional"][ty] = {"n": len(sel), "query_by_binding_top5_sessions": acc(sel, lambda k, v: hier(k, v, "comp_cand")),
                                    "query_by_binding_single_value": acc(sel, hier),
                                    "retrieval_top1": acc(sel, top1), "retrieval_top5": acc(sel, top5)}
    abs_rows = {k: v for k, v in rows.items() if "_abs" in k}
    res["abstention"] = {"n": len(abs_rows),
                         "holographic_scores": {k: round((v["holo"].get("time@0.0") or {}).get("score", 0.0), 4) for k, v in abs_rows.items()},
                         "note": "no abstention threshold is applied; compare with score_distribution"}
    res["score_distribution"] = {
        "correct": sorted(round(v["holo"]["time@0.0"]["score"], 4) for k, v in scored.items() if v["holo"].get("time@0.0") and holo(k, v)),
        "incorrect": sorted(round(v["holo"]["time@0.0"]["score"], 4) for k, v in scored.items() if v["holo"].get("time@0.0") and not holo(k, v)),
    }
    res["failures"] = {
        "aggregation_questions": sum(1 for v in scored.values() if v["parsed"].get("needs_aggregation") is True),
        "entity_not_in_store": sum(1 for v in scored.values() if not v["holo"]),
        "flat_wrong_retrieval_right": [k for k, v in scored.items() if top1(k, v) and not holo(k, v)],
        "flat_right_retrieval_wrong": [k for k, v in scored.items() if holo(k, v) and not top1(k, v)],
    }
    if capacity_path and capacity_path.exists():
        res["capacity"] = json.loads(capacity_path.read_text())
    res["per_question"] = {k: {"qtype": v["qtype"], "flat": holo(k, v), "flat_no_time": holo(k, v, "notime@0.0"),
                               "hier": hier(k, v), "comp": hier(k, v, "comp_cand"), "top1": top1(k, v), "top5": top5(k, v),
                               "flat_value": (v["holo"].get("time@0.0") or {}).get("value"),
                               "flat_source": {kk: (v["holo"].get("time@0.0") or {}).get(kk) for kk in ("sid", "tid")},
                               "entity": v.get("entity_resolved"), "attribute": v["attribute_used"], "answer": v["answer"]}
                           for k, v in rows.items()}
    res["finished"] = P.now()
    out.write_text(json.dumps(res, indent=1) + "\n")
    for k in ("accuracy", "session_recall", "degradation_bit_flips", "storage_deletion", "temporal", "compositional", "query_cost_vector_ops"):
        print(k, json.dumps(res[k]))


# turn atoms --------------------------------------------------------------------------------


class TurnMemory:
    """Extraction-free variant: atoms are sign codes of raw user turns and of the cached free-text
    fact sentences. Each atom is bound with a random code of its turn, its session code and the
    session's month code (its key). Session trace = majority of its bound atoms; window traces =
    majority of W consecutive session traces; user trace = majority of all session traces.

    A content query is not a factor of any trace, so it is not unbound; a trace is unbound with an
    atom's key and the result is compared with the query code (one Hamming per atom, no codebook):
    coarse scores come from the user (or window / shard) trace, fine scores from the session trace."""

    def __init__(self, q, codes, atom_sess, atom_tid, dates, tcodes, origin, rng):
        import numpy as np
        D = codes.shape[1] * 8
        self.codes, self.sess_of, self.tid_of = codes, np.asarray(atom_sess), atom_tid
        self.n_sess = len(q["sessions"])
        self.bucket = [_bucket(d, origin) for d in dates]
        self.scode = np.packbits(rng.integers(0, 2, (self.n_sess, D), dtype=np.uint8), axis=-1)
        self.tie = np.packbits(rng.integers(0, 2, D, dtype=np.uint8))
        turns = {}
        for i, t in enumerate(atom_tid):
            turns.setdefault(t if t is not None else ("fact", i), len(turns))
        turncode = np.packbits(rng.integers(0, 2, (len(turns), D), dtype=np.uint8), axis=-1)
        self.akey = np.stack([xor(self.scode[j], tcodes[self.bucket[j]], turncode[turns[t if t is not None else ("fact", i)]])
                              for i, (j, t) in enumerate(zip(atom_sess, atom_tid))])
        self.bound = np.bitwise_xor(codes, self.akey)
        self.atoms = [np.flatnonzero(self.sess_of == j) for j in range(self.n_sess)]
        self.sess = np.stack([majority(self.bound[a], self.tie) if len(a) else self.tie for a in self.atoms])
        self.user = majority(self.sess, self.tie)
        self.ops = 0

    def windows(self, W):
        return [majority(self.sess[i:i + W], self.tie) if W > 1 else self.sess[i] for i in range(0, self.n_sess, W)]

    def _scores(self, qcode, traces, a):
        """Agreement of (trace unbound with each atom key) with the query, averaged over traces."""
        import numpy as np
        D = self.codes.shape[1] * 8
        self.ops += len(traces) * (len(a) + 1)
        return np.mean([1 - 2 * hamming(np.bitwise_xor(t, qcode), self.akey[a]) / D for t in traces], axis=0)

    def coarse(self, qcode, trace_of, shards=None):
        """Session scores from a coarse trace: best atom of the session."""
        import numpy as np
        out = np.full(self.n_sess, -2.0)
        for j in range(self.n_sess):
            a = self.atoms[j]
            if len(a):
                out[j] = self._scores(qcode, shards(j) if shards else [trace_of(j)], a).max()
        return out

    def fine(self, qcode, sess_traces=None, shards=None):
        """Per-atom scores from each session's own trace."""
        import numpy as np
        out = np.full(len(self.codes), -2.0)
        for j in range(self.n_sess):
            a = self.atoms[j]
            if len(a):
                traces = shards(j) if shards else [(sess_traces if sess_traces is not None else self.sess)[j]]
                out[a] = self._scores(qcode, traces, a)
        return out


def _turn_rankings(mem, q, sess_scores, atom_scores, nested: bool):
    """Official target lists: sessions by score; turns by best atom score (nested inside the session
    order when `nested`), atoms without a turn (session-only facts) contribute nothing."""
    sids = [s["sid"] for s in q["sessions"]]
    sorder = sorted(range(mem.n_sess), key=lambda j: (-sess_scores[j], j))
    tscore = {}
    for i, s in enumerate(atom_scores):
        t = mem.tid_of[i]
        if t is not None:
            tscore[t] = max(tscore.get(t, -2.0), float(s))
    tids = [t for s in q["sessions"] for t in s["tids"]]
    tsess = {t: j for j, s in enumerate(q["sessions"]) for t in s["tids"]}
    srank = {j: r for r, j in enumerate(sorder)}
    pos = {t: i for i, t in enumerate(tids)}
    key = (lambda t: (srank[tsess[t]], -tscore.get(t, -2.0), pos[t])) if nested else (lambda t: (-tscore.get(t, -2.0), pos[t]))
    return [sids[j] for j in sorder], sorted(tids, key=key)


def rank_turns(export_dir: Path, eval_utils: Path, out: Path, D: int = 16384) -> None:
    import importlib.util

    import numpy as np
    spec = importlib.util.spec_from_file_location("lme_eval_utils", eval_utils)
    ev = importlib.util.module_from_spec(spec)
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec.loader.exec_module(ev)
    ex = json.loads((export_dir / "export_dev.json").read_text())
    tx = json.loads((export_dir / "turns_dev.json").read_text())
    dim = json.loads((export_dir / "dim.json").read_text())["dim"]
    emb = np.fromfile(export_dir / "emb.f16", dtype=np.float16).reshape(-1, dim)
    temb = np.fromfile(export_dir / "turn_emb.f16", dtype=np.float16).reshape(-1, dim)
    qemb = {"orig": np.fromfile(export_dir / "q_emb.f16", dtype=np.float16).reshape(-1, dim),
            "rewrite": np.fromfile(export_dir / "rewrite_emb.f16", dtype=np.float16).reshape(-1, dim)}
    pos = {s: i for i, s in enumerate(ex["strings"])}
    struct_q = {q["qid"]: q for q in ex["questions"]}
    C = Codes(dim, D, np.concatenate([temb.astype(np.float32), emb.astype(np.float32)]).mean(axis=0))
    rng = np.random.default_rng(3)
    dates = [P.parse_date(s["date"]) for q in tx["questions"] for s in q["sessions"]]
    origin = date(min(d for d in dates if d).year, 1, 1)
    tcodes = time_codes(_bucket(max(d for d in dates if d), origin) + 2, D, rng)
    questions, rankings, sizes, ops = [], {}, [], {}
    # hier_* : coarse from the user trace, fine from session traces; window / per_session: coarse from
    # traces of W sessions; flips corrupt window-4 and session traces; del* keeps a subset of 8 shards
    # of each session trace (coarse = best atom); index_del* drops atoms from the flat scan outright.
    variants = ["hier_nested", "hier_global", "window4", "window16", "per_session", "flat_hamming", "hier_rewrite"]
    for f in FRACS[1:]:
        variants.append(f"hier_flip{f}")
    for f in DELETE[1:]:
        variants += [f"hier_del{f}", f"index_del{f}"]
    rankings = {v: {"session": {}, "turn": {}} for v in variants}
    row = 0
    for qi, q in enumerate(tx["questions"]):
        sess_objs = [P.Session(s["sid"], P.parse_date(s["date"]), s["date"], [(t, "", "") for t in s["tids"]]) for s in q["sessions"]]
        questions.append(P.Question(q["qid"], q["qtype"], q["question"], q["date"], P.parse_date(q["date"]), sess_objs))
        codes, a_sess, a_tid = [], [], []
        sq = struct_q[q["qid"]]
        for j, s in enumerate(q["sessions"]):
            n = len(s["tids"])
            codes.append(C.of(temb[row:row + n]))
            row += n
            a_sess += [j] * n
            a_tid += list(s["tids"])
            facts = [(ti, f) for k in sq["sessions"][j]["skeys"] for ti, f in ex["struct"][k]["facts"]]
            if facts:
                codes.append(C.of(emb[[pos[f] for _, f in facts]]))
                a_sess += [j] * len(facts)
                a_tid += [s["tids"][ti] if ti >= 0 else None for ti, _ in facts]
        codes = np.concatenate(codes)
        mem = TurnMemory(q, codes, a_sess, a_tid, [P.parse_date(s["date"]) for s in q["sessions"]], tcodes, origin, np.random.default_rng(qi))
        sizes.append({"atoms": len(codes), "sessions": mem.n_sess, "atoms_per_session": len(codes) / mem.n_sess})
        qc = {v: C.of(qemb[v][qi][None])[0] for v in ("orig", "rewrite")}

        def put(name, ss, at, nested=True, qq=q):
            s_rank, t_rank = _turn_rankings(mem, qq, ss, at, nested)
            rankings[name]["session"][qq["qid"]] = s_rank
            rankings[name]["turn"][qq["qid"]] = t_rank

        mem.ops = 0
        ss = mem.coarse(qc["orig"], lambda j: mem.user)
        at = mem.fine(qc["orig"])
        ops.setdefault("hier", []).append(mem.ops)
        put("hier_nested", ss, at, True)
        put("hier_global", ss, at, False)
        put("hier_rewrite", mem.coarse(qc["rewrite"], lambda j: mem.user), mem.fine(qc["rewrite"]), True)
        for W in (4, 16):
            w = mem.windows(W)
            put(f"window{W}", mem.coarse(qc["orig"], lambda j, w=w, W=W: w[j // W]), at, True)
        put("per_session", mem.coarse(qc["orig"], lambda j: mem.sess[j]), at, True)
        mem.ops = 0
        ops.setdefault("flat", []).append(len(codes))
        flat_atom = 1 - 2 * hamming(qc["orig"], codes) / D
        flat_sess = np.array([flat_atom[a].max() if len(a) else -1 for a in mem.atoms])
        put("flat_hamming", flat_sess, flat_atom, False)
        for f in FRACS[1:]:
            crng = np.random.default_rng(qi * 100 + int(f * 100))
            w4 = [flip(t, f, crng) for t in mem.windows(4)]
            st = np.stack([flip(t, f, crng) for t in mem.sess])
            put(f"hier_flip{f}", mem.coarse(qc["orig"], lambda j, w=w4: w[j // 4]), mem.fine(qc["orig"], st), False)
        srng = np.random.default_rng(qi * 7 + 1)
        memb_a = srng.random((len(codes), SHARDS)) < 0.5
        for r in memb_a:
            if not r.any():
                r[srng.integers(SHARDS)] = True
        sess_sh = [[majority(mem.bound[a[memb_a[a, k]]], mem.tie) if memb_a[a, k].any() else None for k in range(SHARDS)] for a in mem.atoms]
        for f in DELETE[1:]:
            keep = sorted(srng.choice(SHARDS, SHARDS - int(round(f * SHARDS)), replace=False))
            at = mem.fine(qc["orig"], shards=lambda j, keep=keep: [t for k in keep if (t := sess_sh[j][k]) is not None] or [mem.tie])
            put(f"hier_del{f}", np.array([at[a].max() if len(a) else -2 for a in mem.atoms]), at, False)
            alive = srng.random(len(codes)) >= f
            fa = np.where(alive, flat_atom, -2.0)
            put(f"index_del{f}", np.array([fa[a].max() if len(a) else -2 for a in mem.atoms]), fa, False)
        print(f"{qi + 1}/{len(tx['questions'])} {q['qid'][:12]} atoms={len(codes)} sessions={mem.n_sess}", flush=True)
    metrics = {v: {lvl: P.evaluate(questions, rankings[v][lvl], lvl, ev) for lvl in ("session", "turn")} for v in variants}
    res = {"D": D, "variants": metrics,
           "sizes": {"atoms_per_user_trace_mean": round(float(np.mean([s["atoms"] for s in sizes])), 1),
                     "atoms_per_user_trace_max": max(s["atoms"] for s in sizes),
                     "atoms_per_session_trace_mean": round(float(np.mean([s["atoms_per_session"] for s in sizes])), 1),
                     "sessions_per_user_trace_mean": round(float(np.mean([s["sessions"] for s in sizes])), 1)},
           "query_cost_vector_ops": {k: round(float(np.mean(v)), 1) for k, v in ops.items()},
           "eval_utils_sha256": __import__("hashlib").sha256(eval_utils.read_bytes()).hexdigest()}
    out.write_text(json.dumps(res, indent=1) + "\n")
    for v in variants:
        m = metrics[v]
        print(v, " ".join(f"{lvl[0]}:R5 {m[lvl]['overall']['recall_all@5']:.3f} N10 {m[lvl]['overall']['ndcg_any@10']:.3f}" for lvl in ("session", "turn")))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    t0 = time.time()
    if a[0] == "capacity":
        capacity(Path(a[1]), Path(a[2]))
    elif a[0] == "answer":
        evaluate(Path(a[1]), Path(a[2]), Path(a[3]), D=int(a[4]) if len(a) > 4 else 16384,
                 margin=float(a[5]) if len(a) > 5 else 0.02)
    elif a[0] == "report":
        report(Path(a[1]), Path(a[2]), Path(a[3]), Path(a[4]), Path(a[5]) if len(a) > 5 else None)
    elif a[0] == "rank":
        rank_turns(Path(a[1]), Path(a[2]), Path(a[3]), D=int(a[4]) if len(a) > 4 else 16384)
    else:
        sys.exit(__doc__)
    print(f"{time.time() - t0:.0f}s")
