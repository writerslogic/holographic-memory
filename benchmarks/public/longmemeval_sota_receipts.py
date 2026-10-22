# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Portable native receipts; ordinal references reconstruct sources from pinned dev only."""

import argparse
import gzip
import json
from pathlib import Path

import longmemeval_retrieval as R
import longmemeval_sota as S

NAMES = ("baseline", "selected", "candidate_ceiling")
DEFAULT = R.ROOT / "benchmarks/results/longmemeval_sota_native_receipts_v1.json"


def expand_row(row, sources):
    def anchor(index, score):
        source = sources[index]
        return {key: source[key] for key in ("recordId", "sourceId", "sessionId", "version", "turnIndex")} | {"score": score}

    def drop(index, reason):
        source = sources[index]
        return {key: source[key] for key in ("recordId", "sourceId", "sessionId", "version")} | {"reason": reason}

    indexes = [*row["sources"], *(i for i, _ in row["anchors"]), *(i for i, _ in row["dropped"])]
    R.require(all(type(i) is int and 0 <= i < len(sources) for i in indexes), "invalid receipt source ordinal")
    evidence = {"sources": [sources[i] for i in row["sources"]],
                "anchors": [anchor(i, score) for i, score in row["anchors"]],
                "dropped": [drop(i, reason) for i, reason in row["dropped"]],
                "serializedBytes": row["serialized_bytes"]}
    R.require(R.digest(evidence) == row["evidence_sha256"], "reconstructed native evidence differs")
    return {"evidence": evidence}


def load(entries, path=DEFAULT):
    data = json.loads(path.read_text())
    R.require(data["schema_version"] == 1 and data["dev_sha256"] == R.digest(entries)
              and set(data["rows"]) == set(entries), "native receipt dev identity differs")
    canonical = json.loads((R.ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json").read_text())
    identity = canonical["native"]["identity"]
    R.require(data["native_identity_sha256"] == R.digest(identity), "native binary/profile identity differs")
    receipts = {}
    for qid, entry in entries.items():
        groups, _, _, _ = R.rounds_for(entry)
        sources, _, _, _ = S.inputs(groups)
        row = data["rows"][qid]
        R.require(row["key"] == R.digest([identity, entry]) and set(row["replies"]) == set(NAMES), "native cache identity differs")
        receipts[qid] = {"key": row["key"], "replies": {name: expand_row(row["replies"][name], sources) for name in NAMES}}
    return receipts


def export(entries, archive):
    with gzip.open(archive, "rb") as stream:
        data = stream.read(384 * 1024 * 1024 + 1)
    R.require(len(data) <= 384 * 1024 * 1024, "native receipt archive exceeds limit")
    receipts = json.loads(data)
    R.require(set(receipts) == set(entries), "native receipt membership differs")
    canonical = json.loads((R.ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json").read_text())
    identity = canonical["native"]["identity"]
    rows = {}
    for qid, entry in entries.items():
        groups, originals, _, _ = R.rounds_for(entry)
        sources, _, _, _ = S.inputs(groups)
        positions = {source["sourceId"]: i for i, source in enumerate(sources)}
        row = receipts[qid]
        R.require(row["key"] == R.digest([identity, entry]), "native receipt identity differs")
        replies = {}
        for name in NAMES:
            evidence = row["replies"][name]["evidence"]
            R.native_groups(evidence, originals, groups)
            compact = {"sources": [positions[source["sourceId"]] for source in evidence["sources"]],
                       "anchors": [[positions[anchor["sourceId"]], anchor["score"]] for anchor in evidence["anchors"]],
                       "dropped": [[positions[drop["sourceId"]], drop["reason"]] for drop in evidence["dropped"]],
                       "serialized_bytes": evidence["serializedBytes"], "evidence_sha256": R.digest(evidence)}
            R.require(expand_row(compact, sources)["evidence"] == evidence, "portable receipt round trip differs")
            replies[name] = compact
        rows[qid] = {"key": row["key"], "replies": replies}
    return {"schema_version": 1, "dev_only": True, "dev_sha256": R.digest(entries),
            "native_identity_sha256": R.digest(identity), "original_archive_sha256": R.file_digest(archive),
            "ordinal_definition": "zero-based original source order: haystack session position, then original turn position",
            "source_reconstruction": "Pinned dev source only; original role, text, time and complete premise IDs are hash-checked against native receipts.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    args.output.write_text(json.dumps(export(entries, args.archive), separators=(",", ":")) + "\n")
    load(entries, args.output)
    print(json.dumps({"receipts": len(entries), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
