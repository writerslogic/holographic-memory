# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference encoder for the Rust conversation-context-v1 source contract."""

import hashlib
import json
import unicodedata

FIELDS = ("recordId", "sourceId", "sessionId", "version", "turnIndex", "role", "observationTime", "text", "premiseRecordIds")
MAX_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 4096


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def record_id(session, version, index, identity):
    return "conversation:source:" + hashlib.sha256(wire([1, "source", [session, version, index, identity]])).hexdigest()


def canonical(sources):
    return [{field: source[field] for field in FIELDS} for source in sorted(
        sources, key=lambda source: (source["sessionId"], source["version"], source["turnIndex"]))]


def pack(sources, max_bytes=MAX_BYTES):
    if type(max_bytes) is not int or not 2 <= max_bytes <= MAX_BYTES or len(sources) > MAX_SOURCES:
        raise ValueError("invalid context bounds")
    sources = canonical(sources)
    if sum(len(source["text"].encode()) for source in sources) > MAX_BYTES:
        raise ValueError("context allocation exceeds limit")
    sessions = sorted({(source["sessionId"], source["version"]) for source in sources})
    session_index = {identity: index for index, identity in enumerate(sessions)}
    source_index = {source["recordId"]: index for index, source in enumerate(sources)}
    if len(source_index) != len(sources):
        raise ValueError("duplicate context source")
    turns = []
    for source in sources:
        if source["recordId"] != record_id(source["sessionId"], source["version"], source["turnIndex"], source["sourceId"]):
            raise ValueError("source identity differs")
        premises = [source_index.get(identity, identity) for identity in source["premiseRecordIds"]]
        turns.append([session_index[(source["sessionId"], source["version"])], source["turnIndex"],
                      source["sourceId"], source["role"], source["observationTime"], source["text"], premises])
    result = {"schemaVersion": 1, "sufficiency": "unverified", "sourcesSha256": hashlib.sha256(wire(sources)).hexdigest(),
              "sessions": [list(session) for session in sessions], "turns": turns}
    if len(wire(result)) > max_bytes:
        raise ValueError("context exceeds byte budget")
    if unpack(result) != sources:
        raise ValueError("context round trip differs")
    return result


def unpack(context):
    if set(context) != {"schemaVersion", "sufficiency", "sourcesSha256", "sessions", "turns"} or type(context["schemaVersion"]) is not int or context["schemaVersion"] != 1 or context["sufficiency"] != "unverified":
        raise ValueError("invalid context schema")
    sessions, turns = context["sessions"], context["turns"]
    if len(turns) > MAX_SOURCES or len(sessions) > len(turns) or sessions != sorted(sessions) or len({tuple(s) for s in sessions}) != len(sessions):
        raise ValueError("invalid context cardinality")
    sources, identities, used, total = [], set(), set(), 0
    previous = None
    for turn in turns:
        if len(turn) != 7:
            raise ValueError("invalid turn arity")
        si, index, identity, role, observed, text, premises = turn
        if type(si) is not int or not 0 <= si < len(sessions) or type(index) is not int or not 0 <= index < MAX_SOURCES:
            raise ValueError("invalid context ordinal")
        session, version = sessions[si]
        if any(not isinstance(label, str) or not label or len(label.encode()) > 256 or any(unicodedata.category(ch) == "Cc" for ch in label) for label in (session, version, identity)):
            raise ValueError("invalid context identity")
        if role not in ("user", "assistant") or type(observed) is not int or not -(2**63) <= observed < 2**63 or not isinstance(text, str):
            raise ValueError("invalid context source")
        total += len(text.encode())
        if len(text.encode()) > 60 * 1024 or total > MAX_BYTES or (si, identity) in identities or (previous is not None and previous >= (si, index)):
            raise ValueError("invalid context source bounds or order")
        local = [p for p in premises if type(p) is int]
        external = [p for p in premises if type(p) is str]
        if len(premises) > 4 or len(local) + len(external) != len(premises) or any(not 0 <= p < len(sources) for p in local) or local != sorted(set(local)) or len(set(external)) != len(external):
            raise ValueError("invalid context premises")
        if any(len(p) != 84 or not p.startswith("conversation:source:") or any(ch not in "0123456789abcdefABCDEF" for ch in p[20:]) for p in external):
            raise ValueError("invalid unresolved premise")
        if any(sources[p]["sessionId"] != session or sources[p]["version"] != version or sources[p]["turnIndex"] >= index for p in local):
            raise ValueError("context premise scope differs")
        sources.append({"recordId": record_id(session, version, index, identity), "sourceId": identity,
                        "sessionId": session, "version": version, "turnIndex": index, "role": role,
                        "observationTime": observed, "text": text,
                        "premiseRecordIds": [sources[p]["recordId"] if type(p) is int else p for p in premises]})
        identities.add((si, identity))
        used.add(si)
        previous = (si, index)
    available = {source["recordId"] for source in sources}
    if any(type(p) is str and p in available for turn in turns for p in turn[6]):
        raise ValueError("resolvable premise must use local ordinal")
    if len(used) != len(sessions) or hashlib.sha256(wire(sources)).hexdigest() != context["sourcesSha256"]:
        raise ValueError("context source digest differs")
    return sources
