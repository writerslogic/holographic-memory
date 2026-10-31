# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Independent numerical, source, metric, byte and decision checks for section 14."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
import zlib
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def close(actual, expected, tolerance=1e-10):
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError(f"numerical mismatch: {actual} versus {expected}")


def normalized(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.sqrt(np.einsum("...d,...d->...", a, a))[..., None]


def decompress_model(model, dim):
    code = model["code"]
    if code.dtype == np.uint8:
        signed = np.stack((code % 16, code // 16), axis=1).reshape(-1).astype(np.int16) - 8
        code = signed.reshape(len(model["scale"]), dim)
    a = code.astype(np.float64)
    if "scale" in model:
        a *= model["scale"].reshape(-1, 1)
    return a


def rankings(parts, name, query, counts, dim):
    atoms = []
    sessions = []
    if name.startswith("full_"):
        a = normalized(decompress_model(parts[0], dim)).dot(query)
        cursor = 0
        for count in counts:
            sessions.append(float(max(a[cursor:cursor + count])))
            cursor += count
        atoms = a
    else:
        for part in parts:
            decoded = decompress_model(part, dim)
            coordinate = np.einsum("rd,d->r", decoded, query)
            if name.startswith(("energy_", "weighted_")):
                if name.startswith("weighted_"):
                    coordinate *= np.sqrt(part["weight"])
                sessions.append(float(np.einsum("i,ij,j->", coordinate, part["gram"], coordinate)))
            elif name.startswith("support_"):
                a = part["coords"].astype(np.float64).dot(coordinate)
                atoms.extend(a)
                sessions.append(float(max(a)))
            else:
                sessions.append(float(max(normalized(decoded).dot(query))))
    session_rank = sorted(range(len(sessions)), key=lambda i: (-sessions[i], i))
    atom_rank = sorted(range(len(atoms)), key=lambda i: (-atoms[i], i)) if len(atoms) else None
    return session_rank, atom_rank, np.array(sessions)


def measure(rank, ids, k):
    gold = {i for i, value in enumerate(ids) if "answer" in value}
    found = set(rank[:k])
    discounts = [1.0 if i == 0 else 1 / math.log2(i + 1) for i in range(k)]
    dcg = sum(discounts[i] for i, value in enumerate(rank[:k]) if value in gold)
    ideal = sum(discounts[:min(len(gold), k)])
    return {f"recall_any@{k}": float(bool(found & gold)),
            f"recall_all@{k}": float(gold <= found), f"ndcg_any@{k}": dcg / ideal if ideal else 0.0}


def interval(values):
    a = np.array(values)
    indices = np.random.default_rng(20261007).integers(len(a), size=(1000, len(a)))
    means = sorted(a[indices].sum(axis=1) / len(a))
    def quantile(p):
        f = p * (len(means) - 1)
        lower = int(f)
        return float(means[lower] + (f - lower) * (means[min(lower + 1, len(means) - 1)] - means[lower]))
    return {"mean": statistics.mean(values), "ci95": [quantile(.025), quantile(.975)]}


def byte_counts(entry, parts, name, dim):
    n = sum(entry["counts"])
    ns = len(entry["counts"])
    payload = sum(a.nbytes for part in parts for a in part.values())
    witness = sum(entry["source_block_sizes"])
    directory = 64 + 4 * n + 8 * (ns + 1)
    full = name.startswith("full_")
    rows = min(n, 64) if full else max(part["code"].shape[0] for part in parts)
    coord = max((part["coords"].size * 8 if "coords" in part else 0 for part in parts))
    scratch = 32 * rows * dim + coord + 8 * dim + 48 * (n + ns) + 2 * max(entry["source_raw_sizes"])
    standalone = full or name.startswith("support_")
    return {"model_bytes": payload, "source_bytes": witness, "directory_bytes": directory,
            "standalone_resident_bytes": payload + witness + directory if standalone else None,
            "standalone_workspace_bytes": scratch if standalone else None,
            "coarse_index_bytes": payload + directory, "dense_index_bytes": 2 * n * dim + directory,
            "two_stage_resident_bytes": payload + witness + directory + (0 if full else 2 * n * dim),
            "two_stage_workspace_bytes": scratch + (0 if full else 32 * dim * min(64, sum(sorted(entry["counts"], reverse=True)[:10]))),
            "dense_resident_bytes": 2 * n * dim + witness + directory}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--work", type=Path, default=Path("target/conversation-geometry-validation/models"))
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/warm-bench/holo_dev")
    ap.add_argument("--source", type=Path, default=Path.home() / ".cache/warm-bench/downloads/longmemeval_s_cleaned.json")
    ap.add_argument("--summary", type=Path, default=Path("benchmarks/results/geometry_dev_summary.json"))
    args = ap.parse_args()
    with gzip.open(args.report, "rt") as f:
        report = json.load(f)
    manifest = report["manifest"]
    assert manifest["version"] == 1
    assert manifest["scan_block"] == 64
    assert manifest["code_sha256"] == sha(Path(__file__).with_name("geometry_experiments.py"))
    assert manifest["script_lock_sha256"] == sha(Path(__file__).with_name("geometry_experiments.py.lock"))
    assert report["eval_utils_sha256"] == "c98b8d1096877a15aa755c9de44fe33c195298466a2eb6f3c0f9f6bde8c72349"
    split_file = Path(__file__).with_name("longmemeval_split.json")
    split = json.loads(split_file.read_text())
    assert manifest["split_sha256"] == sha(split_file)
    assert {e["qid"] for e in manifest["questions"]} == set(split["dev"])
    assert not set(split["heldout"]) & {r["qid"] for r in report["rows"]}
    assert sha(args.source) == manifest["source_sha256"]
    for name, checksum in manifest["inputs"].items():
        assert sha(args.cache / name) == checksum
    source = {q["question_id"]: q for q in json.loads(args.source.read_text()) if q["question_id"] in split["dev"]}
    dim = manifest["dim"]
    grouped = {}
    for r in report["rows"]:
        grouped.setdefault(r["qid"], {})[(r["stream"], r["variant"])] = r
    recovered = 0
    checked_models = 0
    for qi, entry in enumerate(manifest["questions"]):
        qsrc = source[entry["qid"]]
        with np.load(args.work / f"q{qi:03}.npz", allow_pickle=False) as packed:
            z = {k: packed[k] for k in packed.files}
        assert all(np.isfinite(a).all() for a in z.values())
        expected_tids, expected_sids = [], []
        for j, (sid, date, conversation) in enumerate(zip(qsrc["haystack_session_ids"], qsrc["haystack_dates"], qsrc["haystack_sessions"])):
            turns = []
            for i, turn in enumerate(conversation):
                if turn["role"] == "user":
                    tid = f"{sid}_{i + 1}"
                    if "answer" in sid and not turn.get("has_answer"):
                        tid = tid.replace("answer", "noans")
                    turns.append([tid, turn["content"]])
            if not turns:
                continue
            if "answer" in sid and not any("answer" in t[0] for t in turns):
                sid = sid.replace("answer", "noans")
            expected_sids.append(sid)
            expected_tids.extend(t[0] for t in turns)
            packed_source = z[f"source.{len(expected_sids) - 1}"]
            raw = zlib.decompress(packed_source.tobytes())
            actual = json.loads(raw)
            assert actual == {"sid": sid, "date": date, "turns": turns}
            assert hashlib.sha256(raw).hexdigest() == entry["source_hashes"][len(expected_sids) - 1]
            assert len(raw) == entry["source_raw_sizes"][len(expected_sids) - 1]
            assert packed_source.nbytes == entry["source_block_sizes"][len(expected_sids) - 1]
            recovered += len(turns)
        assert expected_tids == entry["tids"] and expected_sids == entry["sids"]
        assert entry["scored"] == ("_abs" not in entry["qid"] and any("answer" in t for t in expected_tids))
        assert z["raw"].shape == (len(expected_tids), dim)
        offsets = np.r_[0, np.cumsum(entry["counts"])]
        q = normalized(z["query"])
        for stream in ("original", "shuffle"):
            raw = z[f"{stream}.raw"]
            permutation = np.arange(len(raw)) if stream == "original" else np.random.default_rng(20261008 + qi).permutation(len(raw))
            assert np.array_equal(raw, z["raw"][permutation])
            exact = normalized(raw).dot(q)
            oracle = set(sorted(range(len(raw)), key=lambda i: (-exact[i], i))[:10])
            for name in manifest["variants"]:
                r = grouped[entry["qid"]][(stream, name)]
                parts = [{key: z[f"{stream}.{name}.{j}.{key}"] for key in sizes}
                         for j, sizes in enumerate(entry["models"][stream][name])]
                assert all({key: a.nbytes for key, a in part.items()} == sizes
                           for part, sizes in zip(parts, entry["models"][stream][name]))
                assert byte_counts(entry, parts, name, dim) == r["bytes"]
                sr, ar, ss = rankings(parts, name, q, entry["counts"], dim)
                assert sr[:10] == r["session_rank"]
                assert (ar[:10] if ar is not None else None) == r["standalone_turn_rank"]
                _, _, negative = rankings(parts, name, -q, entry["counts"], dim)
                close(max(abs(ss - negative)), r["sign_score_max_difference"])
                if name.startswith(("energy_", "weighted_")):
                    assert r["sign_score_max_difference"] == 0
                for m in (5, 10):
                    pool = {i for j in sr[:m] for i in range(int(offsets[j]), int(offsets[j + 1]))}
                    tr = ar if name.startswith("full_") else sorted(pool, key=lambda i: (-exact[i], i))
                    assert tr[:10] == r[f"turn_rank_m{m}"]
                    close(len(oracle & pool) / len(oracle), r[f"shortlist_coverage_m{m}"])
                    gold = {i for i, tid in enumerate(entry["tids"]) if "answer" in tid}
                    if m == 10:
                        close(len(gold & pool) / len(gold) if gold else 1., r["gold_coverage_m10"])
                    for level, rank, ids in (("session", sr, entry["sids"]), ("turn", tr, entry["tids"])):
                        for k in (5, 10):
                            for metric, value in measure(rank, ids, k).items():
                                close(value, r["metrics"][f"m{m}"][level][metric])
                if ar is not None:
                    for level, rank, ids in (("session", sr, entry["sids"]), ("turn", ar, entry["tids"])):
                        for k in (5, 10):
                            for metric, value in measure(rank, ids, k).items():
                                close(value, r["metrics"]["standalone"][level][metric])
                if qi < 3 and name.startswith("energy_"):
                    for j, part in enumerate(parts):
                        x = normalized(raw[offsets[j]:offsets[j + 1]])
                        _, singular, right = np.linalg.svd(x, full_matrices=False)
                        b = decompress_model(part, dim)
                        rank = len(b)
                        orthogonal, _ = np.linalg.qr(b.T)
                        captured = np.linalg.norm(np.einsum("dr,sd->rs", orthogonal, right[:rank])) ** 2
                        assert captured > rank * .99
                        close(float(singular[:rank].dot(singular[:rank]) / singular.dot(singular)),
                              float(part["weight"].sum()), 1e-6)
                        assert np.allclose(np.einsum("ij,jd,kd->ik", part["gram"], b, b), np.eye(rank), atol=1e-8)
                        checked_models += 1
        if (qi + 1) % 20 == 0:
            print(f"validated {qi + 1}/100", flush=True)
    assert recovered == manifest["n_turns"]
    assert len(report["rows"]) == len(manifest["questions"]) * 2 * len(manifest["variants"])
    for stream, variants in report["summary"].items():
        for name, table in variants.items():
            current = [r for r in report["rows"] if r["stream"] == stream and r["variant"] == name and r["scored"]]
            for group, summary in table["groups"].items():
                rr = [r for r in current if group == "overall" or r["qtype"] == group]
                assert summary["n"] == len(rr)
                for mode, levels in summary["metrics"].items():
                    for level, fields in levels.items():
                        for field, result in fields.items():
                            expected = interval([r["metrics"][mode][level][field] for r in rr])
                            close(expected["mean"], result["mean"])
                            for a, b in zip(expected["ci95"], result["ci95"]):
                                close(a, b)
                for field in ("shortlist_coverage_m5", "shortlist_coverage_m10", "gold_coverage_m10"):
                    expected = interval([r[field] for r in rr])
                    close(expected["mean"], summary[field]["mean"])
                    for a, b in zip(expected["ci95"], summary[field]["ci95"]):
                        close(a, b)
            for field, value in table["bytes_mean"].items():
                close(statistics.mean(r["bytes"][field] for r in current), value)
            close(float(np.quantile([statistics.median(r["timing_us"]) for r in current], .95)), table["p95_query_us"])
    results = []
    for d in report["decisions"]:
        name = d["variant"]
        table = report["summary"]["original"]
        rr = table[name]["groups"]["overall"]
        baseline = table["full_f16"]["groups"]["overall"]
        control = ("medoids_support_" if name.startswith("support_") else "medoids_energy_") + name.split("_", 1)[1]
        cc = table[control]["groups"]["overall"]
        fields = ("recall_all@5", "ndcg_any@10")
        quality = all(rr["metrics"]["m10"]["session"][f]["mean"] >= baseline["metrics"]["m10"]["session"][f]["mean"] - .005 for f in fields)
        control_ok = all(rr["metrics"]["m10"]["session"][f]["mean"] >= cc["metrics"]["m10"]["session"][f]["mean"] for f in fields) and any(
            rr["metrics"]["m10"]["session"][f]["mean"] > cc["metrics"]["m10"]["session"][f]["mean"] for f in fields)
        reduction = table[name]["bytes_mean"]["dense_index_bytes"] / table[name]["bytes_mean"]["coarse_index_bytes"]
        coverage = rr["shortlist_coverage_m10"]["mean"]
        expected_pass = quality and control_ok and reduction >= 4 and coverage >= .99
        assert d["quality_pass"] == quality and d["control_pass"] == control_ok
        close(d["coarse_reduction"], reduction)
        close(d["coverage"], coverage)
        assert d["proxy_pass"] == expected_pass and not d["original_capacity_target_met"]
        rr_rows = sorted((r for r in report["rows"] if r["stream"] == "original" and r["variant"] == name and r["scored"]), key=lambda r: r["qid"])
        bb_rows = sorted((r for r in report["rows"] if r["stream"] == "original" and r["variant"] == "full_f16" and r["scored"]), key=lambda r: r["qid"])
        contrasts = {level: {f: interval([r["metrics"]["m10"][level][f] - b["metrics"]["m10"][level][f] for r, b in zip(rr_rows, bb_rows)])
                              for f in fields} for level in ("session", "turn")}
        results.append({**d, "paired_contrasts_vs_dense": contrasts})
    summary = {"report_sha256": sha(args.report), "validator_sha256": sha(__file__),
               "validated_rows": len(report["rows"]), "source_turn_instances_recovered": recovered,
               "independent_svd_subspace_checks": checked_models, "scored_dev_questions": len(bb_rows),
               "decisions": results, "proxy_passes": [d["variant"] for d in results if d["proxy_pass"]],
               "overall_superiority_established": False}
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "decisions"}), flush=True)


if __name__ == "__main__":
    main()
