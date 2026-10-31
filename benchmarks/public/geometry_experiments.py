# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Frozen dev-only conversation geometry experiment; protocol: research record section 14."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import time
import zlib
from pathlib import Path

import numpy as np

import longmemeval_pipeline as P

RANKS = (1, 2, 4, 8)
FORMATS = ("f16", "i8")
SEED = 20261007
VERSION = 1
SCAN_BLOCK = 64
DATA_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
EVAL_SHA = "c98b8d1096877a15aa755c9de44fe33c195298466a2eb6f3c0f9f6bde8c72349"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-12):
        raise ValueError("nonfinite or zero embedding")
    return x / norms


def encode(x, fmt):
    if fmt == "f16":
        return {"code": np.asarray(x, dtype=np.float16)}
    levels = 127 if fmt == "i8" else 7
    scale = (np.abs(x).max(axis=1) / levels).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-30))
    code = np.clip(np.rint(x / scale[:, None]), -levels, levels).astype(np.int8)
    if fmt == "i4":
        flat = (code.astype(np.int16) + 8).astype(np.uint8).ravel()
        if len(flat) % 2:
            raise ValueError("packed four-bit shape must be even")
        code = flat[::2] | (flat[1::2] << 4)
    return {"code": code, "scale": scale}


def decode(model, dim):
    if "scale" not in model:
        return model["code"].astype(np.float64)
    if model["code"].ndim == 1:
        code = model["code"]
        flat = np.empty(len(code) * 2, dtype=np.int8)
        flat[::2] = (code & 15).astype(np.int8) - 8
        flat[1::2] = (code >> 4).astype(np.int8) - 8
        code = flat.reshape(-1, dim)
    else:
        code = model["code"]
    return code.astype(np.float64) * model["scale"][:, None]


def array_bytes(model):
    return sum(v.nbytes for v in model.values())


def scan(model, q, dim):
    n = len(model["scale"]) if "scale" in model else len(model["code"])
    result = np.empty(n, dtype=np.float64)
    for start in range(0, n, SCAN_BLOCK):
        end = min(n, start + SCAN_BLOCK)
        code = model["code"]
        block = {"code": code[start:end] if code.ndim == 2 else code[start * dim // 2:end * dim // 2]}
        if "scale" in model:
            block["scale"] = model["scale"][start:end]
        result[start:end] = unit(decode(block, dim)) @ q
    return result


def basis(x):
    values, vectors = np.linalg.eigh(x @ x.T)
    order = np.argsort(-values, kind="stable")
    order = order[values[order] > max(1e-10, float(values.max()) * 1e-10)][:8]
    b = (vectors[:, order].T @ x) / np.sqrt(values[order])[:, None]
    for row in b:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return b, values[order], float(np.maximum(values, 0).sum())


def representatives(x, count):
    first = int(np.argmax(x @ unit(x.mean(axis=0))))
    chosen = [first]
    similarity = x @ x[first]
    while len(chosen) < min(count, len(x)):
        similarity[chosen] = np.inf
        nxt = int(np.argmin(similarity))
        chosen.append(nxt)
        similarity = np.maximum(similarity, x @ x[nxt])
    return x[chosen]


def variants():
    out = ["full_f16", "full_i8", "full_i4", "mean_f16", "mean_i8"]
    for fmt in FORMATS:
        for rank in RANKS:
            out.extend(f"{kind}_{fmt}_k{rank}" for kind in
                       ("energy", "weighted", "support", "medoids_energy", "medoids_support"))
    return out


def fit_session(x):
    b, values, total = basis(x)
    models = {f"mean_{fmt}": encode(unit(x.mean(axis=0))[None], fmt) for fmt in FORMATS}
    dim = x.shape[1]
    for fmt in FORMATS:
        for rank in RANKS:
            r = min(rank, len(b))
            core = encode(b[:r], fmt)
            decoded = decode(core, dim)
            gram = np.linalg.inv(decoded @ decoded.T)
            weight = (values[:r] / total).astype(np.float32)
            energy = dict(core, gram=gram, weight=weight)
            coords = ((x @ decoded.T) @ gram).astype(np.float16)
            support = dict(energy, coords=coords)
            for kind in ("energy", "weighted"):
                models[f"{kind}_{fmt}_k{rank}"] = energy
            models[f"support_{fmt}_k{rank}"] = support
            per_vector = dim * (2 if fmt == "f16" else 1) + (0 if fmt == "f16" else 4)
            for kind, template in (("energy", energy), ("support", support)):
                count = max(1, array_bytes(template) // per_vector)
                model = encode(representatives(x, count), fmt)
                if array_bytes(model) > array_bytes(template):
                    raise ValueError("representative control exceeds geometry budget")
                models[f"medoids_{kind}_{fmt}_k{rank}"] = model
    return models


def source_blocks(question):
    blocks, hashes, raw_sizes = [], [], []
    for s in question.sessions:
        record = {"sid": s.sid, "date": s.date_text,
                  "turns": [[tid, text] for tid, text, _ in s.turns]}
        raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
        packed = zlib.compress(raw, level=6)
        if zlib.decompress(packed) != raw:
            raise ValueError("source block roundtrip failed")
        blocks.append(np.frombuffer(packed, dtype=np.uint8))
        hashes.append(hashlib.sha256(raw).hexdigest())
        raw_sizes.append(len(raw))
    return blocks, hashes, raw_sizes


def fit(args):
    args.work.mkdir(parents=True, exist_ok=True)
    split_path = Path(__file__).with_name("longmemeval_split.json")
    split = json.loads(split_path.read_text())
    tx = json.loads((args.cache / "turns_dev.json").read_text())
    if {q["qid"] for q in tx["questions"]} != set(split["dev"]):
        raise ValueError("export does not match frozen dev split")
    if digest(args.source) != DATA_SHA:
        raise ValueError("cleaned source release hash mismatch")
    questions = P.load(args.source, set(split["dev"]))
    by_id = {q.qid: q for q in questions}
    dim = json.loads((args.cache / "dim.json").read_text())["dim"]
    if not 1 <= dim <= 8192:
        raise ValueError("invalid embedding dimension")
    turns = np.fromfile(args.cache / "turn_emb.f16", dtype=np.float16).reshape(-1, dim)
    queries = np.fromfile(args.cache / "q_emb.f16", dtype=np.float16).reshape(-1, dim)
    if len(queries) != len(tx["questions"]):
        raise ValueError("query rows do not match export")
    manifest = {"version": VERSION, "dim": dim, "variants": variants(), "questions": [],
                "source_sha256": DATA_SHA, "code_sha256": digest(__file__),
                "split_sha256": digest(split_path),
                "inputs": {name: digest(args.cache / name) for name in
                           ("turns_dev.json", "dim.json", "turn_emb.f16", "q_emb.f16")},
                "script_lock_sha256": digest(Path(str(__file__) + ".lock")),
                "scan_block": SCAN_BLOCK}
    start, row = time.perf_counter(), 0
    for qi, q in enumerate(tx["questions"]):
        question = by_id[q["qid"]]
        sids = [s.sid for s in question.sessions]
        tids = [t[0] for s in question.sessions for t in s.turns]
        if sids != [s["sid"] for s in q["sessions"]] or tids != [t for s in q["sessions"] for t in s["tids"]]:
            raise ValueError("cached embedding identity/order differs from source")
        counts = [len(s.turns) for s in question.sessions]
        n = sum(counts)
        raw = turns[row:row + n]
        row += n
        x = unit(raw)
        blocks, hashes, raw_sizes = source_blocks(question)
        entry = {"qid": q["qid"], "qtype": q["qtype"], "scored": question.scored,
                 "sids": sids, "tids": tids, "counts": counts,
                 "source_hashes": hashes, "source_raw_sizes": raw_sizes,
                 "source_block_sizes": [len(b) for b in blocks], "models": {}}
        arrays = {"raw": raw, "query": queries[qi]}
        arrays.update({f"source.{j}": b for j, b in enumerate(blocks)})
        for stream, xx in (("original", x), ("shuffle", x[np.random.default_rng(20261008 + qi).permutation(n)])):
            arrays[f"{stream}.raw"] = raw if stream == "original" else raw[np.random.default_rng(20261008 + qi).permutation(n)]
            models = {f"full_{fmt}": [encode(xx, fmt)] for fmt in ("f16", "i8", "i4")}
            # The float16 oracle is the supplied cache, not a second quantization of it.
            models["full_f16"] = [{"code": arrays[f"{stream}.raw"]}]
            at = 0
            for j, count in enumerate(counts):
                for name, model in fit_session(xx[at:at + count]).items():
                    models.setdefault(name, []).append(model)
                at += count
            sizes = {}
            for name, parts in models.items():
                sizes[name] = []
                for j, model in enumerate(parts):
                    sizes[name].append({key: int(value.nbytes) for key, value in model.items()})
                    arrays.update({f"{stream}.{name}.{j}.{key}": value for key, value in model.items()})
            entry["models"][stream] = sizes
        np.savez(args.work / f"q{qi:03}.npz", **arrays)
        manifest["questions"].append(entry)
        if (qi + 1) % 10 == 0:
            print(f"fit {qi + 1}/100", flush=True)
    if row != len(turns):
        raise ValueError("unconsumed embedding rows")
    manifest["fit_seconds"] = time.perf_counter() - start
    manifest["n_turns"] = row
    (args.work / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"fit complete: {row} turn instances, {manifest['fit_seconds']:.1f}s", flush=True)


def load_models(z, entry, stream, name):
    return [{key: z[f"{stream}.{name}.{j}.{key}"] for key in sizes}
            for j, sizes in enumerate(entry["models"][stream][name])]


def score(models, name, q, counts, dim):
    if name.startswith("full_"):
        atom = scan(models[0], q, dim)
        ends = np.cumsum(counts)
        return np.maximum.reduceat(atom, np.r_[0, ends[:-1]]), atom
    session, atoms = [], []
    for model in models:
        decoded = decode(model, dim)
        z = decoded @ q
        if name.startswith("energy_"):
            value = float(z @ model["gram"] @ z)
        elif name.startswith("weighted_"):
            weighted = z * np.sqrt(model["weight"])
            value = float(weighted @ model["gram"] @ weighted)
        elif name.startswith("support_"):
            atom = model["coords"].astype(np.float64) @ z
            atoms.append(atom)
            value = float(atom.max())
        else:
            value = float((unit(decoded) @ q).max())
        session.append(value)
    return np.array(session), np.concatenate(atoms) if atoms else None


def order(values):
    return np.argsort(-values, kind="stable")


def metrics(session_rank, turn_rank, entry, ev):
    out = {}
    for level, ranking, ids in (("session", session_rank, entry["sids"]),
                                 ("turn", turn_rank, entry["tids"])):
        correct = [v for v in ids if "answer" in v]
        values = {}
        for k in (5, 10):
            any_recall, all_recall, ndcg = ev.evaluate_retrieval(ranking, correct, ids, k=k)
            values.update({f"recall_any@{k}": any_recall, f"recall_all@{k}": all_recall,
                           f"ndcg_any@{k}": ndcg})
        out[level] = values
    return out


def bootstrap(values, differences=None):
    values = np.asarray(values, dtype=np.float64)
    if differences is not None:
        values = values - np.asarray(differences, dtype=np.float64)
    indices = np.random.default_rng(SEED).integers(len(values), size=(1000, len(values)))
    return {"mean": float(values.mean()),
            "ci95": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()}


def accounting(entry, stream, name, dim):
    sizes = entry["models"][stream][name]
    n, ns = len(entry["tids"]), len(entry["sids"])
    model_bytes = sum(sum(part.values()) for part in sizes)
    source = sum(entry["source_block_sizes"])
    directories = 8 * (ns + 1) + 4 * n + 64
    is_full = name.startswith("full_")
    # Each decode uses float64; budget two simultaneous decode/multiply arrays,
    # atom/session scores, orders, selected flags, and one source-block readout.
    rows = min(n, SCAN_BLOCK) if is_full else max(part["code"] // ((2 if "f16" in name else 1) * dim) for part in sizes)
    coords = max((part.get("coords", 0) * 4 for part in sizes), default=0)
    workspace = 32 * rows * dim + coords + 8 * dim + 48 * n + 48 * ns + 2 * max(entry["source_raw_sizes"])
    fine_rows = min(SCAN_BLOCK, sum(sorted(entry["counts"], reverse=True)[:10]))
    fine_workspace = workspace + 32 * fine_rows * dim
    full_vectors = 2 * n * dim
    standalone = is_full or name.startswith("support_")
    return {"model_bytes": model_bytes, "source_bytes": source, "directory_bytes": directories,
            "standalone_resident_bytes": model_bytes + source + directories if standalone else None,
            "standalone_workspace_bytes": workspace if standalone else None,
            "coarse_index_bytes": model_bytes + directories,
            "dense_index_bytes": full_vectors + directories,
            "two_stage_resident_bytes": model_bytes + source + directories + (0 if is_full else full_vectors),
            "two_stage_workspace_bytes": workspace if is_full else fine_workspace,
            "dense_resident_bytes": full_vectors + source + directories}


def recover(z, entry, selected):
    offsets = np.r_[0, np.cumsum(entry["counts"])]
    cache = {}
    for turn in selected:
        j = int(np.searchsorted(offsets, turn, side="right") - 1)
        if j not in cache:
            raw = zlib.decompress(z[f"source.{j}"].tobytes())
            if hashlib.sha256(raw).hexdigest() != entry["source_hashes"][j]:
                raise ValueError("source hash mismatch")
            cache[j] = json.loads(raw)
        tid, text = cache[j]["turns"][int(turn) - int(offsets[j])]
        if tid != entry["tids"][int(turn)] or not isinstance(text, str):
            raise ValueError("source membership mismatch")
    return len(selected)


def query(z, models, name, q, entry, dim, stream, shortlists=(5, 10)):
    ss, atoms = score(models, name, q, entry["counts"], dim)
    sr = order(ss)
    offsets = np.r_[0, np.cumsum(entry["counts"])]
    results = {"session_rank": sr[:10].tolist()}
    if atoms is not None:
        tr = order(atoms)
        results["standalone_turn_rank"] = tr[:10].tolist()
    else:
        results["standalone_turn_rank"] = None
    for m in shortlists:
        if name.startswith("full_"):
            tr = order(atoms)
        else:
            candidates = np.concatenate([np.arange(offsets[j], offsets[j + 1]) for j in sr[:m]])
            exact = np.empty(len(candidates), dtype=np.float64)
            for start in range(0, len(candidates), SCAN_BLOCK):
                block = candidates[start:start + SCAN_BLOCK]
                exact[start:start + SCAN_BLOCK] = unit(z[f"{stream}.raw"][block]) @ q
            tr = candidates[np.lexsort((candidates, -exact))]
        results[f"turn_rank_m{m}"] = tr[:10].tolist()
    recover(z, entry, results["turn_rank_m10"][:5])
    return results


def summarize(rows):
    out = {}
    for stream in ("original", "shuffle"):
        streams = [r for r in rows if r["stream"] == stream and r["scored"]]
        for name in variants():
            current = [r for r in streams if r["variant"] == name]
            groups = ["overall"] + sorted({r["qtype"] for r in current})
            table = {}
            for group in groups:
                rr = [r for r in current if group == "overall" or r["qtype"] == group]
                table[group] = {"n": len(rr), "metrics": {}}
                for setting in ("m5", "m10", "standalone"):
                    if rr[0]["metrics"].get(setting) is None:
                        continue
                    table[group]["metrics"][setting] = {
                        level: {metric: bootstrap([r["metrics"][setting][level][metric] for r in rr])
                                for metric in ("recall_all@5", "ndcg_any@10")}
                        for level in ("session", "turn")}
                for field in ("shortlist_coverage_m5", "shortlist_coverage_m10", "gold_coverage_m10"):
                    table[group][field] = bootstrap([r[field] for r in rr])
            out.setdefault(stream, {})[name] = {
                "groups": table,
                "bytes_mean": {key: float(np.mean([r["bytes"][key] for r in current]))
                               for key, value in current[0]["bytes"].items() if value is not None},
                "p95_query_us": float(np.quantile([np.median(r["timing_us"]) for r in current], 0.95)),
                "query_us_medians": [float(np.median([r["timing_us"][j] for r in current])) for j in range(3)]}
    decisions = []
    for name, result in out["original"].items():
        if not name.startswith(("energy_", "weighted_", "support_")):
            continue
        baseline = out["original"]["full_f16"]
        rr = result["groups"]["overall"]
        bb = baseline["groups"]["overall"]
        control_name = ("medoids_support_" if name.startswith("support_") else "medoids_energy_") + name.split("_", 1)[1]
        cc = out["original"][control_name]["groups"]["overall"]
        quality = all(rr["metrics"]["m10"]["session"][metric]["mean"] >=
                      bb["metrics"]["m10"]["session"][metric]["mean"] - 0.005
                      for metric in ("recall_all@5", "ndcg_any@10"))
        control = all(rr["metrics"]["m10"]["session"][metric]["mean"] >=
                      cc["metrics"]["m10"]["session"][metric]["mean"]
                      for metric in ("recall_all@5", "ndcg_any@10")) and any(
                          rr["metrics"]["m10"]["session"][metric]["mean"] >
                          cc["metrics"]["m10"]["session"][metric]["mean"]
                          for metric in ("recall_all@5", "ndcg_any@10"))
        ratio = result["bytes_mean"]["dense_index_bytes"] / result["bytes_mean"]["coarse_index_bytes"]
        coverage = rr["shortlist_coverage_m10"]["mean"]
        decisions.append({"variant": name, "quality_pass": quality, "control_pass": control,
                          "coarse_reduction": ratio, "coverage": coverage,
                          "proxy_pass": quality and control and ratio >= 4 and coverage >= 0.99,
                          "original_capacity_target_met": False})
    return out, decisions


def evaluate(args):
    manifest = json.loads((args.work / "manifest.json").read_text())
    if manifest["code_sha256"] != digest(__file__):
        raise ValueError("refit after code changes")
    if digest(args.eval_utils) != EVAL_SHA:
        raise ValueError("official scorer hash mismatch")
    spec = importlib.util.spec_from_file_location("lme_eval", args.eval_utils)
    ev = importlib.util.module_from_spec(spec)
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec.loader.exec_module(ev)
    rows, loads = [], []
    dim = manifest["dim"]
    for qi, entry in enumerate(manifest["questions"]):
        with np.load(args.work / f"q{qi:03}.npz", allow_pickle=False) as packed:
            z = {key: packed[key] for key in packed.files}
            q = unit(z["query"])
            for stream in ("original", "shuffle"):
                oracle_atom = unit(z[f"{stream}.raw"]) @ q
                oracle_top = set(order(oracle_atom)[:10].tolist())
                gold = {j for j, tid in enumerate(entry["tids"]) if "answer" in tid}
                offsets = np.r_[0, np.cumsum(entry["counts"])]
                local = []
                for name in variants():
                    models = load_models(z, entry, stream, name)
                    result = query(z, models, name, q, entry, dim, stream)
                    ss, _ = score(models, name, q, entry["counts"], dim)
                    neg, _ = score(models, name, -q, entry["counts"], dim)
                    record = {"qid": entry["qid"], "qtype": entry["qtype"], "scored": entry["scored"],
                              "stream": stream, "variant": name, **result,
                              "bytes": accounting(entry, stream, name, dim), "metrics": {},
                              "sign_score_max_difference": float(np.max(np.abs(ss - neg))),
                              "timing_us": []}
                    for m in (5, 10):
                        tr = record[f"turn_rank_m{m}"]
                        record["metrics"][f"m{m}"] = metrics(record["session_rank"], tr, entry, ev)
                        pool = {int(i) for j in record["session_rank"][:m]
                                for i in range(int(offsets[j]), int(offsets[j + 1]))}
                        record[f"shortlist_coverage_m{m}"] = len(oracle_top & pool) / len(oracle_top)
                        if m == 10:
                            record["gold_coverage_m10"] = len(gold & pool) / len(gold) if gold else 1.0
                    record["metrics"]["standalone"] = (metrics(record["session_rank"], record["standalone_turn_rank"], entry, ev)
                                                           if record["standalone_turn_rank"] is not None else None)
                    local.append((record, models))
                for repeat in range(3):
                    loads.append(os.getloadavg())
                    rotated = local[repeat * 13 % len(local):] + local[:repeat * 13 % len(local)]
                    for record, models in rotated:
                        start = time.perf_counter_ns()
                        result = query(z, models, record["variant"], q, entry, dim, stream, (10,))
                        elapsed = (time.perf_counter_ns() - start) / 1000
                        if result["turn_rank_m10"] != record["turn_rank_m10"]:
                            raise ValueError("nonreproducible query")
                        record["timing_us"].append(elapsed)
                rows.extend(record for record, _ in local)
        if (qi + 1) % 10 == 0:
            print(f"evaluate {qi + 1}/100", flush=True)
    summary, decisions = summarize(rows)
    report = {"protocol": "IDEAS-2026-10 section 14; dev only; no novelty/vendor/capacity claim",
              "manifest": manifest, "eval_utils_sha256": digest(args.eval_utils),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "load_min": np.min(loads, axis=0).tolist(),
                              "load_max": np.max(loads, axis=0).tolist(),
                              "threads": {k: os.environ.get(k) for k in
                                          ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")}},
              "rows": rows, "summary": summary, "decisions": decisions}
    with gzip.open(args.output, "wt") as f:
        json.dump(report, f, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"rows": len(rows), "proxy_passes": [d["variant"] for d in decisions if d["proxy_pass"]],
                      "load_max": report["environment"]["load_max"]}), flush=True)


def self_test():
    x = unit(np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]]))
    for fmt in ("f16", "i8", "i4"):
        model = encode(x, fmt)
        restored = unit(decode(model, 4))
        if not np.allclose(restored, x, atol=1e-3):
            raise ValueError("scalar codec regression")
        for count in (SCAN_BLOCK - 1, SCAN_BLOCK, SCAN_BLOCK + 1):
            repeated = np.resize(x, (count, 4))
            coded = encode(repeated, fmt)
            if not np.allclose(scan(coded, x[0], 4), unit(decode(coded, 4)) @ x[0]):
                raise ValueError("bounded scan boundary regression")
    for count in (1, 7, 8, 9):
        models = fit_session(np.repeat(x[:1], count, axis=0))
        for name in ("energy_f16_k8", "weighted_i8_k8"):
            positive, _ = score([models[name]], name, x[0], [count], 4)
            negative, _ = score([models[name]], name, -x[0], [count], 4)
            if not np.array_equal(positive, negative):
                raise ValueError("projection sign-invariance regression")
        positive, _ = score([models["support_i8_k8"]], "support_i8_k8", x[0], [count], 4)
        negative, _ = score([models["support_i8_k8"]], "support_i8_k8", -x[0], [count], 4)
        if positive[0] < 0.99 or negative[0] > -0.99:
            raise ValueError("signed support orientation regression")
    for invalid in (np.zeros((1, 4)), np.full((1, 4), np.nan), np.full((1, 4), np.inf)):
        try:
            unit(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("invalid embedding accepted")
    print("scalar codecs, duplicate/rank boundaries, sign semantics and invalid inputs passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "evaluate", "self-test"))
    root = Path.home() / ".cache/warm-bench"
    p.add_argument("--cache", type=Path, default=root / "holo_dev")
    p.add_argument("--source", type=Path, default=root / "downloads/longmemeval_s_cleaned.json")
    p.add_argument("--eval-utils", type=Path, default=root / "downloads/longmemeval_eval_utils_9e0b455f4ef0.py")
    p.add_argument("--work", type=Path, default=Path("target/conversation-geometry-validation/models"))
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/geometry_dev.json.gz"))
    args = p.parse_args()
    if args.stage == "self-test":
        self_test()
    else:
        (fit if args.stage == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()
