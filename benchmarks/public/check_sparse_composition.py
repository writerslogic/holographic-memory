# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "lz4==4.4.5"]
# ///
"""Independent primitive/count/projection oracle; no producer imports."""

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import lz4.block
import numpy as np

MASK = (1 << 64) - 1


def mix(a, b):
    z = (a * 0x9E3779B97F4A7C15 + b) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def mixes(a, b):
    with np.errstate(over="ignore"):
        z = np.asarray(a, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.asarray(b, dtype=np.uint64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def text_hash(text, seed):
    for b in text.encode():
        seed = mix(seed, b)
    return seed


@lru_cache(maxsize=8192)
def atom(dim, seed):
    k = max(1, dim // 256)
    per_group = k // 64
    if per_group:
        g = mixes(seed, np.arange(64, dtype=np.uint64))
        ids = mixes(g[:, None], np.arange(per_group, dtype=np.uint64)).ravel() % dim
    else:
        ids = mixes(seed, np.arange(k, dtype=np.uint64)) % dim
    values = set(map(int, ids))
    for counter in range(k * 10):
        if len(values) >= k:
            break
        values.add(mix((seed + 0xDEAD) & MASK, counter) % dim)
    return np.array(sorted(values), dtype=np.int64)


@lru_cache(maxsize=8192)
def gram(dim, chars):
    n = len(chars)
    parity = set()
    for offset, char in enumerate(chars):
        parity.symmetric_difference_update(map(int, (atom(dim, mix(mix(1, n), ord(char))) + offset) % dim))
    return np.array(sorted(parity), dtype=np.int64)


def primitives(text, dim):
    text = text.strip()
    words = text.split()
    for n in range(2, 6):
        for value, multiplicity in Counter(text[i:i+n] for i in range(len(text)-n+1)).items():
            yield n - 2, gram(dim, value), multiplicity
    for i, word in enumerate(words):
        yield 4, (atom(dim, text_hash(word, 2)) + i) % dim, 1
    for i, (a, b) in enumerate(zip(words, words[1:])):
        ids = np.setxor1d(atom(dim, text_hash(a, 3)), (atom(dim, text_hash(b, 3)) + 1) % dim)
        yield 5, (ids + i) % dim, 1


def reconstruct(text, dim):
    counts = np.zeros((6, dim), dtype=np.uint64)
    ns = [0] * 6
    for level, indices, multiplicity in primitives(text, dim):
        counts[level, indices] += multiplicity
        ns[level] += multiplicity
    return counts, ns


def decode_counts(row):
    dim = row["dim"]
    assert isinstance(dim, int) and 0 < dim <= 65536
    blob = bytes.fromhex(row["counts_lz4_hex"])
    assert len(blob) <= 8 * 6 * dim + 4096
    assert len(blob) >= 4 and struct.unpack("<I", blob[:4])[0] == 8 * 6 * dim
    decoded = lz4.block.decompress(blob)
    assert len(decoded) == row["count_payload_bytes"] == 8 * 6 * dim
    return np.frombuffer(decoded, dtype="<u8").reshape(6, dim)


def project(counts, ns, dim, seed, text):
    common = math.lcm(*(n for n in ns if n))
    normalized = np.zeros(dim, dtype=object)
    minimum = np.full(dim, common * 1_000_001, dtype=object)
    summed = counts.sum(axis=0)
    support = np.flatnonzero(summed)
    for level, n in zip(counts, ns):
        if n:
            term = level.astype(object) * (common // n)
            normalized += term
            minimum = np.where(level > 0, np.minimum(minimum, term), minimum)
    levels = (counts > 0).sum(axis=0)
    frequency = support[np.lexsort((support, -normalized[support]))]
    hashed = mixes(support, seed)
    orders = {
        "UC": support[np.lexsort((support, -levels[support]))],
        "F": support[np.lexsort((support, -summed[support].astype(np.int64)))],
        "FN": frequency,
        "H": support[np.lexsort((support, hashed))],
        "reserve": support[np.lexsort((support, hashed, minimum[support]))],
    }
    domain = np.arange(dim)
    orders["N"] = domain[np.lexsort((domain, mixes(domain, text_hash(text, seed))))]
    k = max(1, dim // 256)
    result = {}
    for budget in (max(1, k // 2), k, 2*k):
        for arm in ("UC", "F", "FN", "H", "R25", "R50", "N"):
            if arm in ("R25", "R50"):
                reserve = orders["reserve"][:budget // (4 if arm == "R25" else 2)]
                order = np.concatenate((reserve, frequency[~np.isin(frequency, reserve)]))
            else:
                order = orders[arm]
            result[(arm, budget)] = np.sort(order[:budget])
    result[("U", 0)] = support
    return result


def delta_bytes(indices):
    previous = 0
    size = 0
    for index in indices:
        delta = int(index) - previous
        previous = int(index)
        size += max(1, (delta.bit_length() + 6) // 7)
    return size


def scores(query, target, dim):
    intersection = len(np.intersect1d(query, target, assume_unique=True))
    union = len(query) + len(target) - intersection
    raw = intersection / len(query) if len(query) else 1.0
    density = len(target) / dim
    return (intersection / union if union else 1.0, raw,
            max(0.0, (raw - density) / (1 - density)) if density < 1 else 0.0)


def check_row(row, expected, dim):
    assert row["schema"] == 1 and row["dim"] == dim
    for key in ("id", "text", "seed", "rare_position"):
        assert row[key] == expected[key], key
    if expected["rare_id"] is not None:
        assert expected["text"].split()[expected["rare_position"]] == f"rare{expected['rare_id']:04}"
    counts = decode_counts(row)
    independent, ns = reconstruct(row["text"], dim)
    assert ns == row["ns"]
    assert np.array_equal(counts, independent), "primitive count mismatch"
    k = max(1, dim // 256)
    assert len(row["majority"]) == 6
    final = Counter()
    for values, n, level in zip(row["majority"], ns, counts):
        assert values == sorted(set(values))
        eligible = np.flatnonzero(level >= (n + 1) // 2) if n else np.array([], dtype=int)
        assert len(values) == min(k, len(eligible))
        assert set(values) <= set(eligible)
        if values:
            remaining = set(map(int, eligible)) - set(values)
            assert not remaining or min(int(level[i]) for i in values) >= max(int(level[i]) for i in remaining)
        final.update(values)
    production = np.array(sorted(sorted(final, key=lambda i: (-final[i], i))[:k]), dtype=np.int64)
    outputs = project(counts, ns, dim, row["seed"], row["text"])
    outputs[("P", 0)] = production
    assert len(row["rows"]) == len(outputs)
    seen = set()
    probes = [np.sort((atom(dim, text_hash(f"rare{i:04}", 2)) + row["rare_position"]) % dim) for i in range(64)]
    metrics = []
    for result in row["rows"]:
        key = (result["arm"], result["budget"])
        assert key not in seen and key in outputs, "duplicate/unexpected arm"
        seen.add(key)
        v = np.array(result["indices"], dtype=np.int64)
        assert np.array_equal(v, outputs[key]), (key, "projection mismatch")
        assert result["delta_varint_bytes"] == delta_bytes(v)
        assert result["json_bytes"] == len(json.dumps({"dim": dim, "indices": list(map(int, v))}, separators=(",", ":")).encode())
        assert result["operators_pass"] is True
        intersections = [len(np.intersect1d(p, v, assume_unique=True)) for p in probes]
        assert intersections == result["rare_intersections"]
        rare_scores = [n / len(p) for n, p in zip(intersections, probes)]
        rare_id = expected["rare_id"]
        auc = None
        rank = None
        if rare_id is not None:
            positive = rare_scores[rare_id]
            negative = [value for i, value in enumerate(rare_scores) if i != rare_id]
            auc = sum((positive > value) + .5 * (positive == value) for value in negative) / 63
            rank = 1 + sum(value > positive or (value == positive and i < rare_id) for i, value in enumerate(rare_scores) if i != rare_id)
        metrics.append({"arm": key[0], "budget": key[1], "active": len(v),
                        "signature": hashlib.sha256(v.astype("<u4").tobytes()).hexdigest(),
                        "rare_auc": auc, "rare_rank": rank, "rare_scores": rare_scores,
                        "delta_bytes": result["delta_varint_bytes"], "json_bytes": result["json_bytes"],
                        "json_lz4_bytes": result["json_lz4_bytes"], "selection_ns": result["selection_ns"]})
    assert seen == set(outputs)
    assert row["cb_count_payload_bytes"] == dim * 4
    total = counts.sum(axis=0).astype(float)
    lam = total.sum() / dim
    cb = [(float(total[p].sum()) - len(p)*lam) / math.sqrt(len(p)*lam) if lam else 0.0 for p in probes]
    assert np.allclose(cb, row["cb_scores"], rtol=0, atol=1e-12), "counting Bloom score"
    rare_id = expected["rare_id"]
    auc = None if rare_id is None else sum((cb[rare_id] > value) + .5 * (cb[rare_id] == value) for i, value in enumerate(cb) if i != rare_id) / 63
    metrics.append({"arm": "CB", "budget": 0, "rare_auc": auc, "rare_scores": cb,
                    "count_payload_bytes": dim * 4})
    return outputs, metrics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def analyze_block(block, dim, stream):
    if not block:
        return
    bases = {meta["family"]: vectors for meta, vectors in block if meta["variant"] == "base"}
    for meta, vectors in block:
        if not meta["applicable"] or meta["variant"] in ("base", "duplicate", "unrelated"):
            continue
        target = bases[meta["family"]]
        wrong_family = (meta["family"] + 1) % 4 + (meta["family"] // 4) * 4
        for key, query in vectors.items():
            correct = scores(query, target[key], dim)
            unrelated = scores(query, bases[wrong_family][key], dim)
            retrieval = [scores(query, bases[i][key], dim)[0] for i in sorted(bases)]
            rank = 1 + sum(value > correct[0] or (value == correct[0] and i < meta["family"]) for i, value in enumerate(retrieval) if i != meta["family"])
            output = {"id": meta["id"], "dim": dim, "seed": meta["seed"], "length": meta["length"],
                      "family": meta["family"], "variant": meta["variant"], "corpus": meta["corpus"],
                      "arm": key[0], "budget": key[1], "correct": correct, "unrelated": unrelated,
                      "ordering": float(correct[0] > unrelated[0]) + .5 * (correct[0] == unrelated[0]),
                      "rank": rank, "ties": sum(value == correct[0] for value in retrieval),
                      "degenerate_query": len(query) == 0}
            stream.write(json.dumps(output) + "\n")


def check_operators(path):
    expected = [(d, s, n, repeated) for d in (1024,4096,16384,65536) for s in (11,23,47,89,179)
                for n in (0,1,2,3,4,7,8,9,31,32,33,127,128,129,511,512,513) for repeated in (False,True)]
    losses = 0
    def frequency(vs, dim):
        counts = np.zeros(dim, dtype=np.int64)
        for vector in vs:
            counts[vector] += 1
        support = np.flatnonzero(counts)
        return np.sort(support[np.lexsort((support,-counts[support]))][:max(1,dim//256)]), counts
    with gzip.open(path, "rt") as stream:
        for dim, seed, n, repeated in expected:
            row = json.loads(stream.readline())
            assert (row["dim"],row["seed"],row["n"],row["repeated"]) == (dim,seed,n,repeated)
            vs = [atom(dim,mix(seed,0 if repeated else i)) for i in range(n)]
            f, counts = frequency(vs,dim)
            assert list(f) == row["frequency"]
            assert list(np.flatnonzero(counts)) == row["union"]
            a, first = frequency(vs[:n//2],dim)
            b, second = frequency(vs[n//2:],dim)
            assert np.array_equal(counts,first+second)
            projected,_ = frequency([a,b],dim)
            assert list(projected) == row["partition_frequency"]
            eligible = np.flatnonzero(counts >= (n+1)//2) if n else np.array([],dtype=int)
            majority = row["majority"]
            assert majority == sorted(set(majority)) and set(majority) <= set(eligible)
            assert len(majority) == min(max(1,dim//256),len(eligible))
            remaining = set(map(int,eligible))-set(majority)
            assert not remaining or not majority or min(counts[majority]) >= max(counts[list(remaining)])
            losses += not np.array_equal(projected,f)
        assert not stream.readline()
    return {"rows":len(expected), "lossy_partition_disagreements":losses, "count_merge_exact":True}


def main(attempt):
    freeze = json.loads((attempt / "freeze.json").read_text())
    assert sha(attempt / "inputs.jsonl") == freeze["input_sha256"]
    manifest = [json.loads(line) for line in (attempt / "inputs.jsonl").read_text().splitlines()]
    assert len({m["id"] for m in manifest}) == len(manifest) == freeze["inputs"]
    execution_path = attempt / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    parent_path = attempt.parent.parent / "compaction_comprehensive/reconstructive-witness-report.json"
    parent = json.loads(parent_path.read_text())
    golden = {(w["dimensions"], f"parent-{w['corpus']}-{i}"): vector for w in parent["witnesses"]
              if w["corpus"] != "random" for i, vector in enumerate(w["vectors"])}
    for w in parent["witnesses"]:
        if w["corpus"] == "random":
            assert all(list(map(int, atom(w["dimensions"], i))) == v for i, v in enumerate(w["vectors"]))
    operator_result = check_operators(attempt / "operators.jsonl.gz")
    count = 0
    golden_count = 0
    started = time.monotonic()
    with gzip.open(attempt / "checked-metrics.jsonl.gz", "xt") as metrics_stream, gzip.open(attempt / "checked-pairs.jsonl.gz", "xt") as pair_stream:
        for dim in freeze["dimensions"]:
            path = attempt / f"d{dim}.jsonl.gz"
            if execution:
                assert execution["complete"] and sha(path) == execution["outputs"][path.name]
            expected = [m for m in manifest if m["corpus"] not in ("template", "varied") or dim in (1024, 16384)]
            block = []
            block_id = None
            with gzip.open(path, "rt") as stream:
                for meta in expected:
                    line = stream.readline()
                    assert line, "missing row"
                    row = json.loads(line)
                    vectors, metrics = check_row(row, meta, dim)
                    key = (dim, meta["id"])
                    if key in golden:
                        assert list(vectors[("P", 0)]) == golden[key], "parent golden mismatch"
                        golden_count += 1
                    metrics_stream.write(json.dumps({"input": meta, "dim": dim, "metrics": metrics,
                        "timings": {k: row[k] for k in ("primitive_ns", "production_ns", "accumulation_ns", "projection_all_ns")}}) + "\n")
                    current = (meta["seed"], meta["length"]) if meta["corpus"] in ("uniform", "repeated") else None
                    if current != block_id:
                        analyze_block(block, dim, pair_stream)
                        block = []
                        block_id = current
                    if current is not None:
                        block.append((meta, vectors))
                    count += 1
                    if count % 100 == 0:
                        print(json.dumps({"checked": count, "dim": dim, "elapsed_s": round(time.monotonic()-started)}), flush=True)
                assert not stream.readline(), "extra rows"
                analyze_block(block, dim, pair_stream)
    assert count == freeze["expected_requests"]
    result = {"complete": True, "rows_checked": count, "parent_vectors_checked": golden_count,
              "random_parent_vectors_checked": 512, "primitive_counts_independently_reconstructed": True,
              "all_projections_independently_checked": True, "timings_validated": False,
              "migration_verified": False, "adoption_allowed": False,
              "operator_check": operator_result,
              "elapsed_seconds": time.monotonic()-started,
              "outputs": {p.name: sha(p) for p in attempt.glob("checked-*.gz")}}
    (attempt / "independent-check.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    main(parser.parse_args().attempt)
