# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Recover frozen dev user observations from existing geometry source blocks."""

import ast
import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

from longmemeval_answer_evidence import SOURCE_SHA256, _observation_key

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 4096
MAX_SESSIONS = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, f"JSON input exceeds size limit: {path.name}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _uint8_payload(member):
    _require(len(member) <= MAX_MEMBER_BYTES, "source member exceeds size limit")
    _require(member[:6] == b"\x93NUMPY", "source member is not a NumPy array")
    version = member[6:8]
    _require(version in (b"\x01\x00", b"\x02\x00"), "unsupported source array version")
    offset, size_format = (10, "<H") if version == b"\x01\x00" else (12, "<I")
    _require(len(member) >= offset, "truncated source array header")
    header_size = struct.unpack(size_format, member[8:offset])[0]
    _require(0 < header_size <= MAX_HEADER_BYTES, "source array header exceeds size limit")
    end = offset + header_size
    _require(end <= len(member), "truncated source array header")
    try:
        header = ast.literal_eval(member[offset:end].decode("latin1").strip())
    except (ValueError, SyntaxError, TypeError, RecursionError) as error:
        raise ValueError("invalid source array header") from error
    _require(isinstance(header, dict)
             and set(header) == {"descr", "fortran_order", "shape"}, "invalid source array header")
    _require(header["descr"] == "|u1", "source array must have uint8 dtype")
    _require(header["fortran_order"] is False, "source array must use C order")
    shape = header["shape"]
    _require(isinstance(shape, tuple) and len(shape) == 1
             and type(shape[0]) is int and 0 < shape[0] <= MAX_MEMBER_BYTES,
             "source array must be a bounded one-dimensional vector")
    _require(len(member) - end == shape[0], "source array payload length mismatch")
    return member[end:]


def _decode_source(member, expected_hash, expected_raw_size, expected_block_size):
    _require(type(expected_raw_size) is int and 0 < expected_raw_size <= MAX_SOURCE_BYTES,
             "invalid decoded source size")
    _require(type(expected_block_size) is int and 0 < expected_block_size <= MAX_MEMBER_BYTES,
             "invalid compressed source size")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64
             and all(char in "0123456789abcdef" for char in expected_hash), "invalid source hash")
    packed = _uint8_payload(member)
    _require(len(packed) == expected_block_size, "compressed source size mismatch")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(packed, expected_raw_size + 1)
    except zlib.error as error:
        raise ValueError("invalid compressed source") from error
    _require(len(raw) == expected_raw_size, "decoded source size mismatch")
    _require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
             "truncated or trailing compressed source")
    _require(hashlib.sha256(raw).hexdigest() == expected_hash, "source hash mismatch")
    try:
        source = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid source JSON") from error
    _require(isinstance(source, dict) and set(source) == {"sid", "date", "turns"},
             "invalid source record")
    _require(isinstance(source["sid"], str) and source["sid"], "invalid source session ID")
    _require(isinstance(source["date"], str), "invalid source observation time")
    _observation_key(source["date"])
    _require(isinstance(source["turns"], list), "invalid source turns")
    for turn in source["turns"]:
        _require(isinstance(turn, list) and len(turn) == 2
                 and isinstance(turn[0], str) and turn[0] and isinstance(turn[1], str),
                 "invalid source turn")
    return source


def load_dev_sources(work: Path, split_path: Path) -> dict[str, list[dict]]:
    work, split_path = Path(work), Path(split_path)
    split, split_hash = _read_json(split_path, 1024 * 1024)
    _require(isinstance(split, dict), "invalid frozen split")
    for name, count in (("dev", 100), ("heldout", 400)):
        ids = split.get(name)
        _require(isinstance(ids, list) and len(ids) == count
                 and all(isinstance(qid, str) and qid for qid in ids)
                 and len(set(ids)) == count, f"invalid frozen {name} membership")
    dev = set(split["dev"])
    _require(not dev.intersection(split["heldout"]), "overlapping frozen split")
    manifest, _ = _read_json(work / "manifest.json", MAX_MANIFEST_BYTES)
    _require(isinstance(manifest, dict) and type(manifest.get("version")) is int
             and manifest["version"] == 1, "unsupported dev source manifest")
    _require(manifest.get("source_sha256") == SOURCE_SHA256, "unpinned source release")
    _require(manifest.get("split_sha256") == split_hash, "frozen split fingerprint mismatch")
    entries = manifest.get("questions")
    _require(isinstance(entries, list) and len(entries) == 100, "expected exactly 100 dev sources")
    _require(all(isinstance(entry, dict) and isinstance(entry.get("qid"), str)
                 for entry in entries), "invalid dev source question")
    qids = [entry["qid"] for entry in entries]
    _require(len(set(qids)) == 100 and set(qids) == dev, "dev source membership mismatch")
    raw_total, turn_total = 0, 0
    for entry in entries:
        sids, counts, tids = (entry.get(key) for key in ("sids", "counts", "tids"))
        _require(isinstance(sids, list) and 0 < len(sids) <= MAX_SESSIONS
                 and all(isinstance(sid, str) and sid for sid in sids), "invalid source session IDs")
        _require(isinstance(counts, list) and len(counts) == len(sids)
                 and all(type(count) is int and count > 0 for count in counts),
                 "invalid source turn counts")
        _require(isinstance(tids, list) and len(tids) == sum(counts)
                 and all(isinstance(tid, str) and tid for tid in tids), "invalid source turn IDs")
        for key in ("source_hashes", "source_raw_sizes", "source_block_sizes"):
            _require(isinstance(entry.get(key), list) and len(entry[key]) == len(sids),
                     f"invalid manifest {key}")
        _require(all(type(size) is int and 0 < size <= MAX_SOURCE_BYTES
                     for size in entry["source_raw_sizes"]), "invalid decoded source sizes")
        _require(all(type(size) is int and 0 < size <= MAX_MEMBER_BYTES
                     for size in entry["source_block_sizes"]), "invalid compressed source sizes")
        raw_total += sum(entry["source_raw_sizes"])
        turn_total += sum(counts)
    _require(raw_total <= MAX_TOTAL_SOURCE_BYTES, "dev source total exceeds size limit")
    _require(type(manifest.get("n_turns")) is int and manifest["n_turns"] == turn_total,
             "manifest total turn count mismatch")
    recovered = {}
    for index, entry in enumerate(entries):
        sources, tids = [], []
        with zipfile.ZipFile(work / f"q{index:03d}.npz") as archive:
            source_members = [info for info in archive.infolist() if info.filename.startswith("source.")]
            expected_names = {f"source.{j}.npy" for j in range(len(entry["sids"]))}
            _require(len(source_members) == len(expected_names)
                     and {info.filename for info in source_members} == expected_names,
                     f"source member membership mismatch in question {index}")
            for session_index, sid in enumerate(entry["sids"]):
                info = archive.getinfo(f"source.{session_index}.npy")
                _require(0 < info.file_size <= MAX_MEMBER_BYTES
                         and 0 < info.compress_size <= MAX_MEMBER_BYTES,
                         f"source member exceeds size limit in question {index}")
                with archive.open(info) as stream:
                    member = stream.read(MAX_MEMBER_BYTES + 1)
                try:
                    source = _decode_source(member, entry["source_hashes"][session_index],
                                            entry["source_raw_sizes"][session_index],
                                            entry["source_block_sizes"][session_index])
                    _require(source["sid"] == sid, "source session order mismatch")
                    _require(len(source["turns"]) == entry["counts"][session_index],
                             "source turn count mismatch")
                except ValueError as error:
                    raise ValueError(f"question {index}, session {session_index}: {error}") from error
                sources.append(source)
                tids.extend(turn[0] for turn in source["turns"])
        _require(tids == entry["tids"], f"source turn order mismatch in question {index}")
        recovered[entry["qid"]] = sources
    return recovered
