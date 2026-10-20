# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Independent replay of set selection, evidence readout and confirmation statistics."""

import argparse
import ast
import hashlib
import json
import math
import random
import statistics
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_number(a, b):
    assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (a, b)


def resample(values):
    rng = random.Random(20261008)
    samples = []
    for _ in range(2000):
        samples.append(statistics.mean(rng.choices(values, k=len(values))))
    samples.sort()
    result = []
    for fraction in (.025, .975):
        index = fraction * 1999
        low = int(index)
        result.append(samples[low] * (1 - (index - low)) + samples[min(low + 1, 1999)] * (index - low))
    return {"mean": statistics.mean(values), "ci95": result}


def replay(base, facets, alpha, family):
    assert all(Counter(f) == Counter(base) for f in facets)
    if len(facets) < 2:
        return base[:10]
    candidates = []
    for tid in base[:64] + [v for f in facets for v in f[:10]]:
        if tid not in candidates:
            candidates.append(tid)
    priors = [1 / math.sqrt(1 + base.index(t)) for t in candidates]
    agreement = [[1 / math.sqrt(1 + f.index(t)) for f in facets] for t in candidates]
    def value(state):
        maxima = []
        for column in range(len(facets)):
            maxima.append(max(agreement[i][column] for i in state))
        cover = min(maxima) if family == "minimum_beam" else statistics.mean(maxima)
        return alpha * sum(priors[i] for i in state) / 5 + (1 - alpha) * cover
    states = [frozenset([0])]
    for _ in range(min(5, len(candidates)) - 1):
        unique = set()
        for s in states:
            for i in range(len(candidates)):
                if i not in s:
                    unique.add(tuple(sorted(s | {i})))
        ranked = sorted(unique, key=lambda s: (-value(s), s))
        states = [frozenset(s) for s in ranked[:1 if family == "mean_greedy" else 32]]
    selected = sorted([candidates[i] for i in states[0]], key=base.index)
    tail = base.copy()
    for tid in selected:
        tail.remove(tid)
    return (selected + tail)[:10]


def measured(top, ids, k):
    gold = {v for v in ids if "answer" in v}
    covered = set(top[:k])
    weights = [1. if i == 0 else 1 / math.log2(i + 1) for i in range(k)]
    ideal = sum(weights[:min(k, sum(t in gold for t in ids))])
    return {f"recall_any@{k}": float(bool(covered & gold)), f"recall_all@{k}": float(gold <= covered),
            f"ndcg_any@{k}": sum(weights[i] for i, t in enumerate(top[:k]) if t in gold) / ideal if ideal else 0.}


def source_block(archive, j):
    blob = archive.read(f"source.{j}.npy")
    assert blob[:8] == b"\x93NUMPY\x01\x00"
    length = struct.unpack("<H", blob[8:10])[0]
    header = ast.literal_eval(blob[10:10 + length].decode().strip())
    assert header["descr"] == "|u1" and not header["fortran_order"]
    packed = blob[10 + length:]
    assert header["shape"] == (len(packed),)
    raw = zlib.decompress(packed)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--work", type=Path, default=Path("target/conversation-geometry-validation/models"))
    p.add_argument("--summary", type=Path, default=Path("benchmarks/results/evidence_completeness_summary.json"))
    args = p.parse_args()
    r = json.loads(args.report.read_text())
    assert r["code_sha256"] == sha(Path(__file__).with_name("evidence_completeness.py"))
    assert r["script_lock_sha256"] == sha(Path(__file__).with_name("evidence_completeness.py.lock"))
    assert r["source_manifest_sha256"] == sha(args.work / "manifest.json")
    frozen_file = Path(__file__).with_name("longmemeval_split.json")
    frozen = json.loads(frozen_file.read_text())
    assert r["split_sha256"] == sha(frozen_file)
    assert {x["qid"] for x in r["rows"]} == set(frozen["dev"])
    assert not set(frozen["heldout"]) & {x["qid"] for x in r["rows"]}
    manifest = json.loads((args.work / "manifest.json").read_text())
    assert manifest["version"] == 1
    source_count = 0
    for qi, row in enumerate(r["rows"]):
        entry = manifest["questions"][qi]
        assert row["qid"] == entry["qid"] and row["ids"] == entry["tids"]
        assert row["scored"] == ("_abs" not in row["qid"] and any("answer" in t for t in row["ids"]))
        assert Counter(row["baseline_ranking"]) == Counter(row["ids"])
        for name, variant in row["variants"].items():
            if name == "baseline":
                expected = row["baseline_ranking"][:10]
            else:
                family, alpha = name.rsplit("_", 1)
                expected = replay(row["baseline_ranking"], row["facet_rankings"], float(alpha), family)
            assert expected == variant["top10"], (row["qid"], name)
            for k in (5, 10):
                for field, value in measured(expected, row["ids"], k).items():
                    check_number(value, variant["metrics"][field])
            slots = [i for i, t in enumerate(row["ids"]) if t in set(expected[:5])]
            assert variant["source_slots"] == slots
            source_count += len(slots)
        with zipfile.ZipFile(args.work / f"q{qi:03}.npz") as archive:
            for j in range(len(entry["sids"])):
                source, checksum = source_block(archive, j)
                assert checksum == entry["source_hashes"][j]
                assert source["sid"] == entry["sids"][j]
                offset = sum(entry["counts"][:j])
                assert [t[0] for t in source["turns"]] == entry["tids"][offset:offset + entry["counts"][j]]
        if (qi + 1) % 20 == 0:
            print(f"quality replay {qi + 1}/100", flush=True)
    assert source_count == r["source_turn_readouts_verified"]
    rng = random.Random(20261008)
    partition = {"calibration": [], "confirmation": []}
    for kind in sorted({x["qtype"] for x in r["rows"] if x["scored"]}):
        ids = sorted(x["qid"] for x in r["rows"] if x["qtype"] == kind and x["scored"])
        rng.shuffle(ids)
        partition["calibration"] += ids[:len(ids) // 2]
        partition["confirmation"] += ids[len(ids) // 2:]
    assert partition == r["partition"]
    summary = r["summary"]
    for part, table in summary["metrics"].items():
        members = partition[part] if part in partition else [x["qid"] for x in r["rows"] if x["scored"]]
        for name, groups in table.items():
            for kind, statistics_row in groups.items():
                rr = [x for x in r["rows"] if x["qid"] in members and (kind == "overall" or x["qtype"] == kind)]
                assert len(rr) == statistics_row["n"]
                for field in ("recall_all@5", "ndcg_any@10"):
                    result = resample([x["variants"][name]["metrics"][field] for x in rr])
                    check_number(result["mean"], statistics_row[field]["mean"])
                    for a, b in zip(result["ci95"], statistics_row[field]["ci95"]):
                        check_number(a, b)
    selections = {}
    for family in ("minimum_beam", "mean_beam", "mean_greedy"):
        options = [f"{family}_{a:.2f}" for a in (.25, .50, .75)]
        selections[family] = max(options, key=lambda name: (
            summary["metrics"]["calibration"][name]["overall"]["recall_all@5"]["mean"],
            summary["metrics"]["calibration"][name]["overall"]["ndcg_any@10"]["mean"], -float(name.rsplit("_", 1)[1])))
    assert selections == summary["selected"]
    control = max(["baseline", selections["mean_beam"], selections["mean_greedy"]], key=lambda name: (
        summary["metrics"]["calibration"][name]["overall"]["recall_all@5"]["mean"],
        summary["metrics"]["calibration"][name]["overall"]["ndcg_any@10"]["mean"], name == "baseline"))
    assert control == summary["selected_control"]
    proposed = selections["minimum_beam"]
    for comparator, groups in summary["confirmation_comparisons"].items():
        assert comparator in ("baseline", control)
        for kind, values in groups.items():
            rr = [x for x in r["rows"] if x["qid"] in partition["confirmation"] and (kind == "overall" or x["qtype"] == kind)]
            expected = resample([x["variants"][proposed]["metrics"]["recall_all@5"] -
                                 x["variants"][comparator]["metrics"]["recall_all@5"] for x in rr])
            check_number(expected["mean"], values["mean"])
            for a, b in zip(expected["ci95"], values["ci95"]):
                check_number(a, b)
    passed = all(c["overall"]["mean"] >= .05 and c["overall"]["ci95"][0] > 0 and c["multi-session"]["mean"] >= .10
                 and all(v["mean"] > -.02 for k, v in c.items() if k not in ("overall", "multi-session"))
                 for c in summary["confirmation_comparisons"].values())
    assert passed == summary["quality_leap_proxy_pass"] and not summary["overall_superiority_established"]
    result = {"report_sha256": sha(args.report), "validator_sha256": sha(__file__),
              "questions": len(r["rows"]), "variants": len(r["rows"][0]["variants"]),
              "source_readouts_verified": source_count,
              "partition_sizes": {k: len(v) for k, v in partition.items()},
              "selected": selections, "quality_leap_proxy_pass": passed,
              "confirmation_comparisons": summary["confirmation_comparisons"]}
    args.summary.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
