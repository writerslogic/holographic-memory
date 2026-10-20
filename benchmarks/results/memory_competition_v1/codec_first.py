"""Lossless, bounded source packets for memory evidence.

Packets preserve complete source text, roles, observation times and occurrence
order. They assert no answer or temporal inference. IDs are lowercase SHA-256
handles, encoded as base64url without padding in the packet.
"""

import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Sequence

Source = tuple[str, str, str, str, str]
MAX_PACKET_BYTES = 1_048_576
MAX_SOURCE_BYTES = 4_194_304
MAX_GROUPS = 4096
MAX_OCCURRENCES = 65_536
MAX_CANDIDATES = 64
_HANDLE = re.compile(r"[0-9a-f]{64}\Z")
_COMPACT_HANDLE = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _handle(value: str) -> str:
    _require(isinstance(value, str) and _HANDLE.fullmatch(value) is not None,
             "source handles must be lowercase SHA-256 hex")
    return base64.urlsafe_b64encode(bytes.fromhex(value)).decode().rstrip("=")


def _expand_handle(value: str) -> str:
    _require(isinstance(value, str) and _COMPACT_HANDLE.fullmatch(value) is not None,
             "invalid compact source handle")
    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except binascii.Error as error:
        raise ValueError("invalid compact source handle") from error
    result = decoded.hex()
    _require(len(decoded) == 32 and _handle(result) == value,
             "noncanonical compact source handle")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_groups(groups: Sequence[Sequence[Source]]) -> None:
    _require(isinstance(groups, (list, tuple)) and len(groups) <= MAX_GROUPS,
             "invalid source group count")
    occurrences = 0
    source_bytes = 0
    sources: dict[str, tuple[str, str]] = {}
    for group in groups:
        _require(isinstance(group, (list, tuple)) and 1 <= len(group) <= 16,
                 "source groups require 1..=16 occurrences")
        occurrences += len(group)
        _require(occurrences <= MAX_OCCURRENCES, "too many source occurrences")
        for source in group:
            _require(isinstance(source, (list, tuple)) and len(source) == 5,
                     "invalid source observation")
            turn, session, observed, role, text = source
            _handle(turn)
            _handle(session)
            _require(isinstance(observed, str) and 1 <= len(observed) <= 64,
                     "invalid observation time")
            _require(role in ("user", "assistant"), "invalid source role")
            _require(isinstance(text, str), "source text must be a string")
            try:
                source_bytes += len(text.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("source text is not valid UTF-8") from error
            _require(source_bytes <= MAX_SOURCE_BYTES, "expanded source text exceeds limit")
            previous = sources.setdefault(turn, (role, text))
            _require(previous == (role, text), "source handle has conflicting role or text")


def encode_source_packet(groups: Sequence[Sequence[Source]]) -> str:
    """Serialize complete observations, factoring shared text and session metadata."""
    _validate_groups(groups)
    turns: list[list[str]] = []
    sessions: list[list[str]] = []
    occurrences: list[list[list[int]]] = []
    turn_indices: dict[str, int] = {}
    session_indices: dict[tuple[str, str], int] = {}
    for group in groups:
        retained = []
        for turn, session, observed, role, text in group:
            if turn not in turn_indices:
                turn_indices[turn] = len(turns)
                turns.append([_handle(turn), role, text])
            key = (session, observed)
            if key not in session_indices:
                session_indices[key] = len(sessions)
                sessions.append([_handle(session), observed])
            retained.append([turn_indices[turn], session_indices[key]])
        occurrences.append(retained)
    packet = _canonical({"v": 3, "t": turns, "s": sessions, "g": occurrences, "i": []})
    _require(len(packet.encode("utf-8")) <= MAX_PACKET_BYTES, "source packet exceeds limit")
    return packet


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate packet key")
        result[key] = value
    return result


def decode_source_packet(packet: str) -> list[list[Source]]:
    """Validate and reconstruct each original observation without interpretation."""
    _require(isinstance(packet, str), "source packet must be text")
    try:
        size = len(packet.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("source packet is not valid UTF-8") from error
    _require(size <= MAX_PACKET_BYTES, "source packet exceeds limit")
    try:
        value = json.loads(packet, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid source packet JSON") from error
    _require(isinstance(value, dict) and set(value) == {"v", "t", "s", "g", "i"},
             "invalid source packet fields")
    _require(type(value["v"]) is int and value["v"] == 3 and value["i"] == [],
             "invalid packet version or inferred answers")
    turns, sessions, groups = value["t"], value["s"], value["g"]
    _require(all(isinstance(table, list) and len(table) <= MAX_OCCURRENCES
                 for table in (turns, sessions)), "invalid source table size")
    _require(isinstance(groups, list) and len(groups) <= MAX_GROUPS,
             "invalid source group count")
    decoded_turns = []
    for turn in turns:
        _require(isinstance(turn, list) and len(turn) == 3, "invalid source table row")
        decoded_turns.append((_expand_handle(turn[0]), turn[1], turn[2]))
    decoded_sessions = []
    for session in sessions:
        _require(isinstance(session, list) and len(session) == 2, "invalid session table row")
        decoded_sessions.append((_expand_handle(session[0]), session[1]))
    result = []
    occurrences = 0
    source_bytes = 0
    for group in groups:
        _require(isinstance(group, list) and 1 <= len(group) <= 16,
                 "source groups require 1..=16 occurrences")
        occurrences += len(group)
        _require(occurrences <= MAX_OCCURRENCES, "too many source occurrences")
        retained = []
        for occurrence in group:
            _require(isinstance(occurrence, list) and len(occurrence) == 2,
                     "invalid source occurrence")
            ti, si = occurrence
            _require(type(ti) is int and 0 <= ti < len(decoded_turns)
                     and type(si) is int and 0 <= si < len(decoded_sessions),
                     "source table index out of range")
            turn, role, text = decoded_turns[ti]
            session, observed = decoded_sessions[si]
            _require(isinstance(text, str), "source text must be a string")
            try:
                source_bytes += len(text.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("source text is not valid UTF-8") from error
            _require(source_bytes <= MAX_SOURCE_BYTES, "expanded source text exceeds limit")
            retained.append((turn, session, observed, role, text))
        result.append(retained)
    _validate_groups(result)
    _require(encode_source_packet(result) == packet, "noncanonical or unused source table rows")
    return result


@dataclass(frozen=True)
class EvidenceSelection:
    selected: tuple[int, ...]
    packet: str
    packet_bytes: int


def complete_evidence(
    candidates: Sequence[Sequence[Source]],
    selected: Sequence[int],
    budget_bytes: int,
) -> EvidenceSelection:
    """Retain a source selection and append ranked complete groups within its byte cap.

    Candidate order is the caller's retrieval order. Existing selected groups
    remain mandatory; a budget that cannot contain them fails explicitly.
    """
    _require(type(budget_bytes) is int and 1 <= budget_bytes <= MAX_PACKET_BYTES,
             "invalid evidence budget")
    _require(isinstance(candidates, (list, tuple)) and 1 <= len(candidates) <= MAX_CANDIDATES,
             "expected 1..=64 ranked candidates")
    _validate_groups(candidates)
    _require(isinstance(selected, (list, tuple))
             and all(type(i) is int and 0 <= i < len(candidates) for i in selected)
             and len(selected) == len(set(selected)),
             "invalid mandatory selection")
    retained = list(selected)
    packet = encode_source_packet([candidates[i] for i in retained])
    _require(len(packet.encode("utf-8")) <= budget_bytes, "mandatory evidence exceeds budget")
    chosen = set(retained)
    for index, group in enumerate(candidates):
        if index in chosen:
            continue
        try:
            trial = encode_source_packet([candidates[i] for i in retained] + [group])
        except ValueError as error:
            if str(error) == "source packet exceeds limit":
                continue
            raise
        if len(trial.encode("utf-8")) <= budget_bytes:
            retained.append(index)
            chosen.add(index)
            packet = trial
    return EvidenceSelection(tuple(retained), packet, len(packet.encode("utf-8")))
