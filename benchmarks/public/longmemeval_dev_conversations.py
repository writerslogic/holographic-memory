# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Decode only frozen dev records from the pinned release's binary record frames."""

import hashlib
import json
import re
from pathlib import Path

from longmemeval_answer_evidence import SOURCE_SHA256, _observation_key

SPLIT_SHA256 = "ef9c00e7bc8d029a4f4cf2c177098c35e3ee347893680e1fe26ee98ed9d16ba7"
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_RELEASE_BYTES = 384 * 1024 * 1024
MAX_SPLIT_BYTES = 1024 * 1024
ID_HEADER = re.compile(rb'        "question_id": "([-A-Za-z0-9_]{1,128})",\n')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate dev JSON field")
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError("invalid dev JSON constant")


def _validate_dev_record(record, qid):
    _require(isinstance(record, dict) and record.get("question_id") == qid, "dev record identity mismatch")
    _require(isinstance(record.get("question"), str), "invalid dev question")
    _require(isinstance(record.get("question_type"), str) and record["question_type"],
             "invalid dev question type")
    _require(isinstance(record.get("question_date"), str), "invalid dev question date")
    _observation_key(record["question_date"])
    sessions, dates, sids = (record.get(key) for key in
                            ("haystack_sessions", "haystack_dates", "haystack_session_ids"))
    _require(isinstance(sessions, list) and isinstance(dates, list) and isinstance(sids, list)
             and len(sessions) == len(dates) == len(sids), "dev session fields are not aligned")
    for session, date, sid in zip(sessions, dates, sids):
        _require(isinstance(date, str) and isinstance(sid, str) and sid, "invalid dev session metadata")
        _observation_key(date)
        _require(isinstance(session, list), "invalid dev conversation")
        for turn in session:
            _require(isinstance(turn, dict) and turn.get("role") in ("user", "assistant")
                     and isinstance(turn.get("content"), str), "invalid dev conversation turn")


def _parse_dev_conversations(stream, dev, heldout, expected_hash):
    _require(isinstance(dev, (set, frozenset)) and isinstance(heldout, (set, frozenset))
             and not dev.intersection(heldout), "invalid disjoint source membership")
    known = dev | heldout
    _require(known and all(isinstance(qid, str) and re.fullmatch(r"[-A-Za-z0-9_]{1,128}", qid)
                           for qid in known), "invalid source question IDs")
    _require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash),
             "invalid expected release fingerprint")
    hasher, file_size = hashlib.sha256(), 0

    def read_line():
        nonlocal file_size
        line = stream.readline(MAX_LINE_BYTES + 1)
        _require(isinstance(line, bytes), "source stream must be binary")
        _require(len(line) <= MAX_LINE_BYTES, "source line exceeds size limit")
        file_size += len(line)
        _require(file_size <= MAX_RELEASE_BYTES, "source release exceeds size limit")
        hasher.update(line)
        return line

    _require(read_line() == b"[\n", "unexpected source root frame")
    seen, decoded = set(), {}
    opening = read_line()
    while True:
        _require(opening == b"    {\n", "unexpected source record opening")
        header = read_line()
        match = ID_HEADER.fullmatch(header)
        _require(match is not None, "unexpected question ID header format")
        qid = match[1].decode("ascii")
        _require(qid in known, "unknown source question ID")
        _require(qid not in seen, "duplicate source question ID")
        seen.add(qid)
        record_size = len(opening) + len(header)
        _require(record_size <= MAX_RECORD_BYTES, "source record exceeds size limit")
        retained = bytearray(opening + header) if qid in dev else None
        while True:
            line = read_line()
            _require(line, "truncated source record")
            record_size += len(line)
            _require(record_size <= MAX_RECORD_BYTES, "source record exceeds size limit")
            if line in (b"    },\n", b"    }\n"):
                if retained is not None:
                    retained.extend(b"    }\n")
                break
            _require(line not in (b"    {\n", b"[\n", b"]\n", b"]"),
                     "unexpected source record frame")
            if retained is not None:
                retained.extend(line)
        if retained is not None:
            try:
                record = json.loads(retained, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ValueError("invalid dev record JSON") from error
            _validate_dev_record(record, qid)
            decoded[qid] = record
        if line == b"    }\n":
            _require(read_line() in (b"]", b"]\n"), "unexpected source root closing")
            _require(read_line() == b"", "trailing source release bytes")
            break
        opening = read_line()
    _require(seen == known and set(decoded) == dev, "source question membership incomplete")
    _require(hasher.hexdigest() == expected_hash, "pinned source release fingerprint mismatch")
    return decoded


def load_dev_conversations(source_path: Path, split_path: Path) -> dict[str, dict]:
    with Path(split_path).open("rb") as stream:
        split_bytes = stream.read(MAX_SPLIT_BYTES + 1)
    _require(len(split_bytes) <= MAX_SPLIT_BYTES, "source split exceeds size limit")
    _require(hashlib.sha256(split_bytes).hexdigest() == SPLIT_SHA256, "frozen split fingerprint mismatch")
    split = json.loads(split_bytes)
    _require(isinstance(split, dict), "invalid frozen split")
    for name, count in (("dev", 100), ("heldout", 400)):
        ids = split.get(name)
        _require(isinstance(ids, list) and len(ids) == count
                 and all(isinstance(qid, str) and qid for qid in ids)
                 and len(set(ids)) == count, f"invalid frozen {name} membership")
    with Path(source_path).open("rb") as stream:
        return _parse_dev_conversations(stream, set(split["dev"]), set(split["heldout"]), SOURCE_SHA256)
