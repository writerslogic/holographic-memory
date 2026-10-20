# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "lz4==4.4.5"]
# ///
"""Reconstruct encodings, CRC-framed documents, rankings, costs and train metrics."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import zlib

import lz4.block
import numpy as np

from check_sparse_composition import reconstruct
from document_search_compare import peak_rss_bytes


ROOT = Path(__file__).resolve().parents[2]
UNION = "multiscale-capped-union-v1"
MAJORITY = "multiscale-words-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoding(text, dim, encoder, witness=None):
    if encoder == MAJORITY and witness is not None:
        assert witness["dim"] == dim and witness["text"] == text
        assert len(witness["majority"]) == 6
    counts, ns = reconstruct(text, dim)
    k = max(1, dim // 256)
    votes = Counter()
    for index, (level, n) in enumerate(zip(counts, ns)):
        if not n:
            continue
        if encoder == UNION:
            eligible = np.flatnonzero(level)
        else:
            eligible = np.flatnonzero(level >= (n + 1) // 2)
            eligible = sorted(eligible, key=lambda i: (-int(level[i]), int(i)))
            ambiguous = len(eligible) > k and level[eligible[k-1]] == level[eligible[k]]
            if ambiguous:
                assert witness is not None, "ambiguous majority tie needs a separate witness"
                chosen = witness["majority"][index]
                assert chosen == sorted(set(chosen)) and len(chosen) == k
                assert set(chosen) <= set(eligible)
                remaining = set(map(int, eligible)) - set(chosen)
                assert min(int(level[i]) for i in chosen) >= max(int(level[i]) for i in remaining)
                eligible = chosen
            else:
                eligible = eligible[:k]
        votes.update(map(int, eligible))
    output = sorted(sorted(votes, key=lambda i: (-votes[i], i))[:k])
    if encoder == MAJORITY and witness is not None:
        assert output == witness["indices"], "prechange majority identity"
    return output


def frames(path):
    found, offset = [], 0
    with path.open("rb") as source:
        while True:
            header = source.read(16)
            assert len(header) == 16, "truncated header"
            crc, raw, compressed, _ = struct.unpack("<IIII", header)
            if raw == compressed == 0:
                break
            assert 0 < raw <= 50 * 1024 * 1024 and 0 < compressed <= raw
            payload = source.read(compressed)
            assert len(payload) == compressed, "truncated payload"
            decoded = lz4.block.decompress(payload, uncompressed_size=raw) if compressed < raw else payload
            assert len(decoded) == raw and zlib.crc32(decoded) == crc, "frame checksum"
            assert decoded.startswith(b"HMS-TXN-2\0"), "unexpected transaction"
            mutations = json.loads(decoded[10:])
            assert len(mutations) == 1 and set(mutations[0]) == {"Document"}
            found.append(mutations[0]["Document"])
            offset += 16 + compressed
    return found, offset


def similarities(query, vectors):
    expected = {}
    q = set(query)
    for key, vector in vectors.items():
        overlap = len(q & set(vector))
        union = len(query) + len(vector) - overlap
        expected[key] = overlap / union if union else 1.0
    return expected


def result_scores(results, query, vectors):
    expected = similarities(query, vectors)
    assert len(results) == min(10, len(expected)), "missing/extra retrieval result"
    assert len({row["id"] for row in results}) == len(results), "duplicate result"
    scores = []
    for row in results:
        assert row["id"] in expected, "foreign result"
        assert math.isfinite(row["similarity"])
        assert abs(row["similarity"] - expected[row["id"]]) < 1e-12, "incorrect Jaccard score"
        scores.append(row["similarity"])
    assert scores == sorted(scores, reverse=True), "rank order"
    assert scores == sorted(expected.values(), reverse=True)[:10], "inexact top ten"
    return scores


def metrics(results, relevant, ids):
    hits = [ids[row["id"]] in relevant for row in results]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return {"ndcg10": dcg / ideal, "recall10": sum(hits) / len(relevant)}


def check(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    freeze = json.loads((folder / "source-freeze.json").read_text())
    analysis = json.loads((folder / "analysis-freeze.json").read_text())
    assert sha(Path(__file__)) == analysis["checker_sha256"], "analysis source drift"
    assert sha(ROOT / "benchmarks/public/test_encoder_version_diagnostic.py") == analysis["tests_sha256"]
    assert sha(folder / "protocol.json") == freeze["protocol_sha256"]
    assert sha(folder / "input.json") == protocol["input_sha256"]
    assert sha(folder / "baseline-witnesses.json") == freeze["baseline_witnesses_sha256"]
    witness_rows = json.loads((folder / "baseline-witnesses.json").read_text())
    witnesses = {(row["dim"], row["text"]): row for row in witness_rows}
    assert sha(Path(freeze["binary"])) == freeze["binary_sha256"]
    for name, digest in freeze["sources"].items():
        archived = {
            "benchmarks/public/check_encoder_version_diagnostic.py": "checker-v1.py",
            "benchmarks/public/test_encoder_version_diagnostic.py": "oracle-tests-v1.py",
        }
        path = folder / archived[name] if name in archived else ROOT / name
        assert sha(path) == digest, f"source drift: {name}"
    inputs = json.loads((folder / "input.json").read_text())
    documents = {row["id"]: row for row in inputs["documents"]}
    queries = {row["id"]: row for row in inputs["queries"]}
    assert len(documents) == protocol["documents"] == 64
    assert len(queries) == protocol["queries"] == 16
    chunks = {f"hms:chunk:{len(key.encode())}:{key}:0": key for key in documents}
    runs = json.loads((folder / "runs.json").read_text())
    identities, checked, cache, lexical = set(), [], {}, {}
    assert len(runs["rows"]) == len(protocol["dimensions"]) * 2 * protocol["rounds"]
    for receipt in runs["rows"]:
        dim, encoder, repetition = (receipt[key] for key in ("dimensions", "encoder", "round"))
        identity = dim, encoder, repetition
        assert identity not in identities and dim in protocol["dimensions"]
        assert encoder in protocol["encoders"] and 0 <= repetition < protocol["rounds"]
        identities.add(identity)
        report_path = folder / receipt["report"]
        tag = f"{dim}-{repetition}-{encoder}"
        assert receipt["report"] == f"{tag}.json" and receipt["log"] == f"{tag}.log"
        assert sha(report_path) == receipt["report_sha256"]
        log = folder / receipt["log"]
        assert sha(log) == receipt["log_sha256"]
        assert peak_rss_bytes(log.read_text(), "Darwin" if "macOS" in freeze["platform"] else "Linux") == receipt["peak_rss_bytes"]
        gate = max(receipt["load_before"][0], receipt["load_after"][0]) < protocol["load_limit"]
        assert receipt["load_gate_met"] == gate
        raw = json.loads(report_path.read_text())
        assert raw["schema"] == 1 and raw["encoder"] == encoder and raw["dimensions"] == dim
        assert raw["input_bytes"] == (folder / "input.json").stat().st_size
        for key in ("init_ns", "ingest_ns", "compact_ns", "restart_ns"):
            assert type(raw[key]) is int and raw[key] > 0
        vectors = {row["id"]: row["indices"] for row in raw["vectors"]}
        assert len(vectors) == len(raw["vectors"]) == len(chunks) and set(vectors) == set(chunks)
        for row in raw["vectors"]:
            assert row["document_id"] == chunks[row["id"]]
        for key, values in vectors.items():
            source = documents[chunks[key]]["text"]
            cache_key = dim, encoder, source
            if cache_key not in cache:
                cache[cache_key] = encoding(source, dim, encoder, witnesses.get((dim, source)))
            assert values == cache[cache_key], "encoder vector differs from independent primitives"
        store = report_path.with_suffix(".store")
        schema = json.loads((store / "store.json").read_text())
        assert schema["text_encoder"] == encoder and schema["dimensions"] == dim and not schema["encrypted"]
        arena = store / "vectors_data.bin"
        generation = (arena / "CURRENT").read_text()
        assert generation.startswith("gen-") and all(c.isalnum() or c == "-" for c in generation)
        for phase, path in (("after", arena / generation / "seg_0.bin"),):
            decoded, logical = frames(path)
            assert logical == raw[phase]["logicalBytes"] and len(decoded) == raw[phase]["frameCount"] == len(documents)
            assert raw[phase]["checksumsVerified"] and raw[phase]["segmentCount"] == 1
            assert raw[phase]["allocatedBytes"] == path.stat().st_size
            assert {doc["id"] for doc in decoded} == set(documents)
            for doc in decoded:
                original = documents[doc["id"]]
                assert len(doc["chunks"]) == 1 and doc["source_uri"] == original["sourceUri"]
                chunk = doc["chunks"][0]
                assert chunk["text"] == original["text"] and doc["metadata"] == original["metadata"]
                assert chunk["start_byte"] == 0 and chunk["end_byte"] == len(original["text"].encode())
                assert chunk["vector"]["dim"] == dim and chunk["vector"]["indices"] == vectors[chunk["id"]]
        actual_files = {str(path.relative_to(store)): path.stat().st_size for path in store.rglob("*") if path.is_file()}
        assert actual_files == raw["after_files"], "store byte accounting"
        assert len(raw["queries"]) == len(raw["restart_results"]) == len(queries)
        assert {row["id"] for row in raw["queries"]} == set(queries)
        restarted = {row["id"]: row for row in raw["restart_results"]}
        assert set(restarted) == set(queries)
        per_query, query_times = [], []
        for row in raw["queries"]:
            original = queries[row["id"]]
            assert row["text"] == original["text"] and row["relevant_ids"] == original["relevantIds"]
            cache_key = dim, encoder, row["text"]
            if cache_key not in cache:
                cache[cache_key] = encoding(row["text"], dim, encoder, witnesses.get((dim, row["text"])))
            assert row["indices"] == cache[cache_key], "query encoder mismatch"
            scores = result_scores(row["results"], row["indices"], vectors)
            assert result_scores(restarted[row["id"]]["results"], row["indices"], vectors) == scores
            assert restarted[row["id"]]["lexical"] == row["lexical"], "lexical restart change"
            lexical_key = row["id"]
            if lexical_key in lexical:
                assert lexical[lexical_key] == row["lexical"], "text encoder changed lexical document API"
            else:
                lexical[lexical_key] = row["lexical"]
            all_scores = similarities(row["indices"], vectors)
            top_ties = sum(value == max(all_scores.values()) for value in all_scores.values())
            per_query.append({"id": row["id"], "top_score_ties": top_ties,
                              "distinct_scores": len(set(all_scores.values())),
                              **metrics(row["results"], set(row["relevant_ids"]), chunks)})
            assert type(row["query_ns"]) is int and row["query_ns"] > 0
            query_times.append(row["query_ns"])
        checked.append({"dimensions": dim, "encoder": encoder, "round": repetition, "load_gate_met": gate,
                        "ndcg10": statistics.mean(row["ndcg10"] for row in per_query),
                        "recall10": statistics.mean(row["recall10"] for row in per_query), "per_query": per_query,
                        "mean_top_score_ties": statistics.mean(row["top_score_ties"] for row in per_query),
                        "empty_vectors": sum(not v for v in vectors.values()), "unique_vectors": len({tuple(v) for v in vectors.values()}),
                        "logical_bytes": raw["after"]["logicalBytes"], "file_length_bytes": sum(actual_files.values()),
                        "input_bytes": raw["input_bytes"], "peak_rss_bytes": receipt["peak_rss_bytes"],
                        "ingest_ns": raw["ingest_ns"], "query_median_ns": statistics.median(query_times),
                        "compact_ns": raw["compact_ns"], "restart_ns": raw["restart_ns"]})
    expected_ids = {(dim, encoder, repetition) for dim in protocol["dimensions"]
                    for encoder in protocol["encoders"] for repetition in range(protocol["rounds"])}
    assert identities == expected_ids
    cells = []
    for dim in protocol["dimensions"]:
        for encoder in protocol["encoders"]:
            rows = [row for row in checked if row["dimensions"] == dim and row["encoder"] == encoder]
            for key in ("ndcg10", "recall10", "empty_vectors", "unique_vectors", "logical_bytes", "file_length_bytes"):
                assert len({row[key] for row in rows}) == 1, f"nonrepeatable {key}"
            cells.append({"dimensions": dim, "encoder": encoder, "load_gates": sum(row["load_gate_met"] for row in rows),
                          **{key: statistics.median(row[key] for row in rows) for key in (
                              "ndcg10", "recall10", "mean_top_score_ties", "empty_vectors", "unique_vectors", "logical_bytes", "file_length_bytes",
                              "input_bytes", "peak_rss_bytes", "ingest_ns", "query_median_ns", "compact_ns", "restart_ns")}})
    return {"schema": 1, "verified_runs": len(checked), "cells": cells, "checked": checked,
            "source_files_verified": len(freeze["sources"]), "protocol_sha256": freeze["protocol_sha256"],
            "timing_claim_eligible": all(row["load_gate_met"] for row in checked),
            "limitations": protocol["limits"] + "; precompaction frame lengths are producer-only because publication removed those files",
            "analysis_sha256": sha(folder / "analysis-freeze.json"), "adoption": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = check(args.folder)
    with args.out.open("x") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps({"verified_runs": summary["verified_runs"], "timing_claim_eligible": summary["timing_claim_eligible"]}))


if __name__ == "__main__":
    main()
