"""LongMemEval retrieval on Modal: the pipeline of longmemeval_pipeline.py with GPU model stages.

  uvx modal run benchmarks/public/longmemeval_modal.py --dataset s --part dev --models small --cap 5
  uvx modal run benchmarks/public/longmemeval_modal.py --dataset m --part all --models large --cap <USD>

Stages (each cached in the Modal Volume `hms-lme`, keyed by sha256 of model@revision + prompt +
text, written shard by shard with a volume commit, so an interrupted run resumes where it stopped):
  facts    LLM user-fact extraction per session (vLLM)            only if a fact key kind is used
  queries  LLM rewrite / sub-queries / time range per question     only if used by the config
  embed    Qwen3-Embedding keys and query variants (sentence-transformers, model-card prompt)
  scores   HMS document API BM25 + exact cosine per question (public-bench lme-scores, built here)
  rerank   Qwen3-Reranker yes/no probability for the top candidates (model-card prompt)
  rank     fusion config from --config, official eval_utils.py metrics per split part

Hard cost cap: every GPU call reports its wall time (model load included); before each wave the
driver adds the measured spend and the projected cost of the wave and stops if it would pass --cap.
GPU rates are Modal's published per-hour prices (`modal billing rates`, 2026-10-05) plus 15% for CPU
and memory; the authoritative figure is `modal billing report`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # longmemeval_pipeline ships with the images
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # containers import this file from /root
VOL = modal.Volume.from_name("hms-lme", create_if_missing=True)
V = Path("/vol")
HF = {"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"}

LME_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
DATA_SHA256 = {
    "s": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "m": "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f",
}
EVAL_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

MODELS = {
    "small": {
        "embed": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "T4"),
        "rerank": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", "T4"),
        "llm": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "L4"),
    },
    "large": {
        "embed": ("Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "H100"),
        "rerank": ("Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912", "H100"),
        "llm": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "H100"),
    },
}
USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95}
OVERHEAD = 1.15
CPU_USD_PER_HOUR = 8 * 0.0473 + 16 * 0.008

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.11.0", "sentence-transformers==5.1.1", "numpy<2.3")
    .env(HF)
    .add_local_python_source("longmemeval_pipeline")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.89.0")
    .uv_pip_install("numpy<2.3")
    .add_local_dir(
        REPO,
        "/src",
        copy=True,
        ignore=["target", "**/target", ".git", ".venv", "**/.venv", ".claude", "**/node_modules", "figures", "benchmarks/results"],
    )
    .run_commands("cd /src && /root/.cargo/bin/cargo build --release --locked --bin public-bench")
    .env(HF)
    .add_local_python_source("longmemeval_pipeline")
)
app = modal.App("hms-longmemeval")


def _gpu(kind: str, models: str) -> str:
    return MODELS[models][kind][2]


# ------------------------------------------------------------------------------- GPU stages


def _embed(model: str, rev: str, gpu: str, texts: list[str], is_query: bool) -> tuple[bytes, float]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    import longmemeval_pipeline as P

    t = time.time()
    m = SentenceTransformer(
        model,
        revision=rev,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16 if gpu == "T4" else torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    m.max_seq_length = 2048
    # Model card: queries carry "Instruct: {task}\nQuery:{query}", documents no prompt; last-token pooling, L2.
    prompt = f"Instruct: {P.EMBED_TASK}\nQuery:" if is_query else None
    e = m.encode(texts, prompt=prompt, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(e, dtype=np.float16).tobytes(), time.time() - t


def _llm(model: str, rev: str, gpu: str, prompts: list[str]) -> tuple[list[str], float]:
    from vllm import LLM, SamplingParams

    t = time.time()
    llm = LLM(model=model, revision=rev, dtype="half" if gpu == "T4" else "auto", max_model_len=12288,
              gpu_memory_utilization=0.9, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    chats = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
        for p in prompts
    ]
    outs = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=1536))
    return [o.outputs[0].text for o in outs], time.time() - t


RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _rerank(model: str, rev: str, gpu: str, pairs: list[tuple[str, str]]) -> tuple[list[float], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import longmemeval_pipeline as P

    t = time.time()
    tok = AutoTokenizer.from_pretrained(model, revision=rev, padding_side="left")
    net = AutoModelForCausalLM.from_pretrained(
        model, revision=rev, torch_dtype=torch.float16 if gpu == "T4" else torch.bfloat16
    ).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok.encode(RERANK_PREFIX, add_special_tokens=False), tok.encode(RERANK_SUFFIX, add_special_tokens=False)
    max_len = 4096
    texts = [f"<Instruct>: {P.RERANK_TASK}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
    ids = tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
              max_length=max_len - len(pre) - len(suf))["input_ids"]
    ids = [pre + x + suf for x in ids]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    scores = [0.0] * len(ids)
    budget = 32768 if gpu == "H100" else 12288  # tokens per forward batch
    i = 0
    with torch.no_grad():
        while i < len(order):
            j = i + 1
            while j < len(order) and len(ids[order[j]]) * (j + 1 - i) <= budget:
                j += 1
            batch = tok.pad({"input_ids": [ids[k] for k in order[i:j]]}, padding=True, return_tensors="pt")
            logits = net(**{k: v.cuda() for k, v in batch.items()}).logits[:, -1, :]
            pair = torch.stack([logits[:, no], logits[:, yes]], dim=1).float()
            p = torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()
            for k, s in zip(order[i:j], p):
                scores[k] = s
            i = j
    return scores, time.time() - t


_gpu_kw = dict(image=gpu_image, volumes={"/vol": VOL}, timeout=6 * 3600, max_containers=4)


@app.function(gpu="T4", **_gpu_kw)
def embed_t4(*a):
    return _embed(*a)


@app.function(gpu="H100", **_gpu_kw)
def embed_h100(*a):
    return _embed(*a)


@app.function(gpu="L4", **_gpu_kw)
def llm_l4(*a):
    return _llm(*a)


@app.function(gpu="H100", **_gpu_kw)
def llm_h100(*a):
    return _llm(*a)


@app.function(gpu="T4", **_gpu_kw)
def rerank_t4(*a):
    return _rerank(*a)


@app.function(gpu="H100", **_gpu_kw)
def rerank_h100(*a):
    return _rerank(*a)


FN = {("embed", "T4"): embed_t4, ("embed", "H100"): embed_h100, ("llm", "L4"): llm_l4,
      ("llm", "H100"): llm_h100, ("rerank", "T4"): rerank_t4, ("rerank", "H100"): rerank_h100}


# ----------------------------------------------------------------------------------- driver


class Budget:
    def __init__(self, cap: float):
        self.cap, self.spent, self.t0, self.log = cap, 0.0, time.time(), []

    def cpu(self) -> float:
        return (time.time() - self.t0) / 3600 * CPU_USD_PER_HOUR

    def total(self) -> float:
        return self.spent + self.cpu()

    def check(self, projected: float, what: str) -> None:
        if self.total() + projected > self.cap:
            raise RuntimeError(f"cost cap ${self.cap:.2f}: spent ${self.total():.3f}, {what} projected ${projected:.3f}")

    def add(self, stage: str, gpu: str, secs: float, n: int) -> None:
        usd = secs / 3600 * USD_PER_HOUR[gpu] * OVERHEAD
        self.spent += usd
        self.log.append({"stage": stage, "gpu": gpu, "secs": round(secs, 1), "items": n, "usd": round(usd, 4)})
        print(f"[{stage}] {n} items on {gpu}: {secs:.0f}s ${usd:.3f} (total ${self.total():.3f})", flush=True)


def _cache_dir(stage: str, model: str, rev: str) -> Path:
    return V / "cache" / stage / f"{model.replace('/', '__')}@{rev[:12]}"


def _load_json_cache(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        out.update(json.loads(f.read_text()))
    return out


def _waves(stage, kind, models, items, shard, budget, call, store):
    """Run `call(chunk)` over shards of the missing items in waves of 4 containers, storing each shard."""
    gpu = _gpu(kind, models)
    fn = FN[(kind, gpu)]
    shards = [items[i:i + shard] for i in range(0, len(items), shard)]
    per_item = None
    first = True
    while shards:
        wave = shards[:1] if first else shards[:4]
        shards = shards[len(wave):]
        n = sum(len(w) for w in wave)
        projected = (per_item or 0.0) * n / 3600 * USD_PER_HOUR[gpu] * OVERHEAD
        budget.check(projected if per_item else 120 / 3600 * USD_PER_HOUR[gpu] * OVERHEAD * len(wave), f"{stage} wave")
        results = list(fn.starmap([call(w) for w in wave]))
        secs = 0.0
        for w, (res, s) in zip(wave, results):
            store(w, res)
            secs += s
        VOL.commit()
        budget.add(stage, gpu, secs, n)
        per_item = secs / n
        first = False


def _parse_json(text: str) -> dict | None:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def _llm_stage(stage, models, keyed_prompts: dict[str, str], budget) -> dict:
    model, rev, _ = MODELS[models]["llm"]
    d = _cache_dir(stage, model, rev)
    d.mkdir(parents=True, exist_ok=True)
    cache = _load_json_cache(d)
    missing = sorted(k for k in keyed_prompts if k not in cache)
    print(f"[{stage}] {len(keyed_prompts)} needed, {len(missing)} missing", flush=True)

    def store(keys, outs):
        part = {k: _parse_json(o) for k, o in zip(keys, outs)}
        cache.update(part)
        name = hashlib.sha256("".join(keys).encode()).hexdigest()[:24]
        (d / f"{name}.json").write_text(json.dumps(part))

    shard = 4000 if models == "large" else 1500
    _waves(stage, "llm", models, missing, shard, budget,
           lambda keys: (model, rev, _gpu("llm", models), [keyed_prompts[k] for k in keys]), store)
    return cache


class EmbedCache:
    def __init__(self, models: str, is_query: bool):
        self.model, self.rev, _ = MODELS[models]["embed"]
        self.is_query = is_query
        self.d = _cache_dir("embed-query" if is_query else "embed-key", self.model, self.rev)
        self.d.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, tuple[str, int]] = {}
        for f in sorted(self.d.glob("*.json")):
            for i, h in enumerate(json.loads(f.read_text())):
                self.index[h] = (f.stem, i)
        self.dim = None

    def key(self, text: str) -> str:
        import longmemeval_pipeline as P
        return P.sha("embed", self.model, self.rev, str(self.is_query), P.EMBED_TASK if self.is_query else "", text)

    def fill(self, texts: list[str], models: str, budget: Budget) -> None:
        todo = {}
        for t in texts:
            h = self.key(t)
            if h not in self.index:
                todo[h] = t
        print(f"[embed {'query' if self.is_query else 'key'}] {len(texts)} needed, {len(todo)} missing", flush=True)
        hashes = sorted(todo, key=lambda h: len(todo[h]))

        def store(hs, blob):
            name = hashlib.sha256("".join(hs).encode()).hexdigest()[:24]
            (self.d / f"{name}.f16").write_bytes(blob)
            (self.d / f"{name}.json").write_text(json.dumps(hs))
            for i, h in enumerate(hs):
                self.index[h] = (name, i)

        shard = 20000 if models == "large" else 8000
        _waves("embed", "embed", models, hashes, shard, budget,
               lambda hs: (self.model, self.rev, _gpu("embed", models), [todo[h] for h in hs], self.is_query), store)

    def get(self, texts: list[str]):
        import numpy as np
        mm, rows = {}, []
        for t in texts:
            name, i = self.index[self.key(t)]
            if name not in mm:
                raw = np.fromfile(self.d / f"{name}.f16", dtype=np.float16)
                n = len(json.loads((self.d / f"{name}.json").read_text()))
                mm[name] = raw.reshape(n, -1)
            rows.append(mm[name][i])
        e = np.asarray(rows, dtype=np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def _dataset(name: str) -> Path:
    import urllib.request
    p = V / "data" / f"longmemeval_{name}_cleaned.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{LME_REVISION}"
               f"/longmemeval_{name}_cleaned.json")
        urllib.request.urlretrieve(url, p.with_suffix(".part"))
        p.with_suffix(".part").rename(p)
        VOL.commit()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != DATA_SHA256[name]:
        raise RuntimeError(f"{p} sha256 mismatch")
    return p


def _eval_utils():
    import importlib.util
    import urllib.request

    import numpy as np
    p = V / "data" / f"eval_utils_{EVAL_COMMIT[:12]}.py"
    if not p.exists():
        url = f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{EVAL_COMMIT}/src/retrieval/eval_utils.py"
        p.write_bytes(urllib.request.urlopen(url).read())
    if not hasattr(np, "asfarray"):  # removed in NumPy 2; eval_utils' only use is np.asfarray(x)
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec = importlib.util.spec_from_file_location("lme_eval_utils", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(p.read_bytes()).hexdigest()


def _kinds_used(cfgs) -> list[str]:
    return sorted({k for c in cfgs for k, w in c["kinds"].items() if w > 0})


def _rerank_query(q) -> str:
    return f"(asked on {q.date_text}) {q.text}"


def _rerank_doc(q, target: str, level: str) -> str:
    for s in q.sessions:
        if level == "session" and s.sid == target:
            return f"Conversation date: {s.date_text}\n{s.user_text[:12000]}"
        for tid, user, reply in s.turns:
            if tid == target:
                return f"Conversation date: {s.date_text}\nuser: {user[:3000]}\nassistant: {reply[:1500]}"
    raise KeyError(target)


@app.function(image=cpu_image, volumes={"/vol": VOL}, timeout=24 * 3600, cpu=8, memory=16384)
def driver(dataset: str, part: str, models: str, cap: float, config: dict, tag: str,
           all_kinds: bool = False, rerank_pool: tuple[int, int] = (30, 50)) -> dict:

    import longmemeval_pipeline as P

    budget = Budget(cap)
    split = json.loads(config.pop("_split"))
    ids = None if part == "all" else set(split[part])
    qs = P.load(_dataset(dataset), ids)
    cfg_t, cfg_s = config["turn"], config["session"]
    kinds = list(P.KINDS) if all_kinds else _kinds_used([cfg_t, cfg_s])
    llm = MODELS[models]["llm"][0] + "@" + MODELS[models]["llm"][1]
    print(f"{len(qs)} questions, kinds {kinds}", flush=True)

    facts = None
    if any(k in kinds for k in ("fact", "turn_exp", "session_exp")):
        need = {P.fact_key(s, llm): P.FACT_PROMPT.format(date=s.date_text, messages=s.fact_input())
                for q in qs for s in q.sessions}
        facts = _llm_stage("facts", models, need, budget)
    queries = None
    uses_llm_queries = all_kinds or any(
        c["time_w"] > 0 or any(w > 0 for v, w in c["variants"].items() if v != "orig") for c in (cfg_t, cfg_s))
    if uses_llm_queries:
        need = {P.query_key(q, llm): P.QUERY_PROMPT.format(today=q.date_text, question=q.text) for q in qs}
        queries = _llm_stage("queries", models, need, budget)

    key_cache, query_cache = EmbedCache(models, False), EmbedCache(models, True)
    run_dir = V / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    batches = [qs[i:i + 50] for i in range(0, len(qs), 50)]
    all_keys, all_q = set(), set()
    for q in qs:
        all_keys.update(t for _, t, _, _ in P.build_keys(q, kinds, facts, llm))
        all_q.update(t for _, t in P.query_variants(q, queries, llm))
    key_cache.fill(sorted(all_keys), models, budget)
    query_cache.fill(sorted(all_q), models, budget)
    del all_keys, all_q

    scores, targets = {}, {}
    exe = "/src/target/release/public-bench"
    for b in batches:
        tmp = Path("/tmp/lme") / hashlib.sha256("".join(q.qid for q in b).encode()).hexdigest()[:16]
        info = P.write_score_inputs(b, kinds, facts, llm, queries,
                                    lambda texts, isq: (query_cache if isq else key_cache).get(texts), tmp)
        digest = hashlib.sha256()
        for f in ("items.jsonl", "queries.jsonl", "items.f32", "queries.f32"):
            digest.update((tmp / f).read_bytes())
        out = V / "cache" / "scores" / f"{digest.hexdigest()[:32]}.json"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([exe, "lme-scores", "--items", tmp / "items.jsonl", "--item-emb", tmp / "items.f32",
                            "--queries", tmp / "queries.jsonl", "--query-emb", tmp / "queries.f32",
                            "--emb-dim", str(info["emb_dim"]), "--out", out], check=True)
            VOL.commit()
        scores.update(P.load_scores(out))
        targets.update(json.loads((tmp / "targets.json").read_text()))
        print(f"[scores] batch of {len(b)}: {info}", flush=True)
    (run_dir / "scores.json").write_text(json.dumps({"scores": scores}))
    (run_dir / "targets.json").write_text(json.dumps(targets))
    if queries is not None:
        by_key = {P.query_key(q, llm): queries.get(P.query_key(q, llm)) for q in qs}
        (run_dir / "queries.json").write_text(json.dumps({"llm": llm, "by_key": by_key}))

    ev, ev_sha = _eval_utils()
    rerank = None
    if cfg_t["rerank_w"] > 0 or cfg_s["rerank_w"] > 0 or all_kinds:
        rankings, _ = P.run(qs, scores, targets, cfg_t, cfg_s, ev, queries, llm)
        model, rev, _ = MODELS[models]["rerank"]
        d = _cache_dir("rerank", model, rev)
        d.mkdir(parents=True, exist_ok=True)
        cache = _load_json_cache(d)
        want, pairs = {}, {}
        byq = {q.qid: q for q in qs}
        for level, depth in (("session", rerank_pool[0]), ("turn", rerank_pool[1])):
            want[level] = {}
            for qid, ranked in rankings[level].items():
                q = byq[qid]
                for t in ranked[:depth]:
                    qt, dt = _rerank_query(q), _rerank_doc(q, t, level)
                    h = P.sha("rerank", model, rev, P.RERANK_TASK, qt, dt)
                    want[level].setdefault(qid, {})[t] = h
                    if h not in cache:
                        pairs[h] = (qt, dt)
        print(f"[rerank] {sum(len(v) for lv in want.values() for v in lv.values())} pairs, {len(pairs)} missing", flush=True)
        hs = sorted(pairs, key=lambda h: len(pairs[h][1]))

        def store(keys, out):
            part = dict(zip(keys, out))
            cache.update(part)
            (d / f"{hashlib.sha256(''.join(keys).encode()).hexdigest()[:24]}.json").write_text(json.dumps(part))

        _waves("rerank", "rerank", models, hs, 20000 if models == "large" else 6000, budget,
               lambda keys: (model, rev, _gpu("rerank", models), [pairs[h] for h in keys]), store)
        rerank = {lv: {qid: {t: cache[h] for t, h in m.items()} for qid, m in want[lv].items()} for lv in want}
        (run_dir / "rerank.json").write_text(json.dumps(rerank))

    parts = {"all": qs} if part != "all" else {
        "dev": [q for q in qs if q.qid in set(split["dev"])],
        "heldout": [q for q in qs if q.qid in set(split["heldout"])],
        "all": qs,
    }
    metrics = {}
    for name, sub in parts.items():
        _, m = P.run(sub, scores, targets, cfg_t, cfg_s, ev, queries, llm, rerank)
        metrics[name if part == "all" else part] = m
    result = {
        "dataset": {"name": f"longmemeval_{dataset}", "release_revision": LME_REVISION, "sha256": DATA_SHA256[dataset],
                    "part": part, "n_questions": len(qs), "n_scored": sum(q.scored for q in qs)},
        "models": {k: {"id": v[0], "revision": v[1], "gpu": v[2]} for k, v in MODELS[models].items()},
        "config": {"turn": cfg_t, "session": cfg_s, "kinds_indexed": kinds},
        "evaluation": {"code": "xiaowu0162/LongMemEval src/retrieval/eval_utils.py", "commit": EVAL_COMMIT,
                       "sha256": ev_sha},
        "cost": {"estimated_usd": round(budget.total(), 4), "gpu_calls": budget.log,
                 "driver_cpu_usd": round(budget.cpu(), 4), "cap_usd": cap},
        "metrics": metrics,
        "finished": P.now(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    VOL.commit()
    return result


@app.local_entrypoint()
def main(dataset: str = "s", part: str = "dev", models: str = "small", cap: float = 5.0,
         config: str = str(HERE / "longmemeval_config.json"), tag: str = "", all_kinds: bool = False,
         out: str = ""):
    if dataset not in ("s", "m") or part not in ("dev", "heldout", "all") or models not in MODELS:
        sys.exit("bad --dataset / --part / --models")
    if part == "heldout" and dataset == "s":
        sys.exit("held-out questions are scored only in the final M run")
    cfg = json.loads(Path(config).read_text())
    cfg["_split"] = (HERE / "longmemeval_split.json").read_text()
    tag = tag or f"{dataset}-{part}-{models}-{int(time.time())}"
    result = driver.remote(dataset, part, models, cap, cfg, tag, all_kinds)
    o = result["metrics"]
    for name, m in o.items():
        for lvl in ("session", "turn"):
            ov = m[lvl]["overall"]
            print(f"{name} {lvl}: R@5 {ov['recall_all@5']:.3f} R@10 {ov['recall_all@10']:.3f} "
                  f"nDCG@5 {ov['ndcg_any@5']:.3f} nDCG@10 {ov['ndcg_any@10']:.3f}")
    print(f"estimated cost ${result['cost']['estimated_usd']:.3f}; run tag {tag}")
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
