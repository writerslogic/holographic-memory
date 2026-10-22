# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build a raw-source verification packet from citations and question-only retrieval."""

import copy

from longmemeval_answer_evidence import _observation_key

MAX_GROUPS = 4096
MAX_SOURCES = 131072
MAX_TEXT_BYTES = 16 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def _validate_source(source):
    _require(isinstance(source, dict) and _identity(source.get("source_id"))
             and _identity(source.get("session_id")), "invalid source identity")
    _require(all(type(source.get(key)) is int and 0 <= source[key] < 2 ** 31
                 for key in ("session_index", "turn_index")), "invalid source occurrence")
    _require(source.get("role") in ("user", "assistant") and isinstance(source.get("text"), str)
             and len(source["text"].encode("utf-8")) <= MAX_TEXT_BYTES, "invalid source role or text")
    _require(isinstance(source.get("observation_time"), str), "invalid source observation time")
    _observation_key(source["observation_time"])


def _content(source):
    return source["role"], source["text"], source["observation_time"]


def select_verifier_sources(original_groups, prior_seeds, counterevidence_group_ids,
                            serialize_callback, count_callback, cap=24000):
    _require(type(cap) is int and cap > 0, "verification cap must be a positive integer")
    _require(isinstance(original_groups, list) and 0 < len(original_groups) <= MAX_GROUPS,
             "invalid original source groups")
    _require(isinstance(prior_seeds, list) and len(prior_seeds) <= 16, "invalid prior seed count")
    groups, sources, owner, aliases, positions, sessions = {}, {}, {}, {}, {}, {}
    text_bytes = 0
    for group in original_groups:
        _require(isinstance(group, dict) and _identity(group.get("group_id")), "invalid original group identity")
        gid = group["group_id"]
        _require(gid not in groups, "duplicate original group identity")
        _require(isinstance(group.get("sources"), list) and group["sources"], "empty original source group")
        groups[gid] = group
        key = group.get("frozen_selected_id_sha256")
        _require(key is None or _identity(key), "invalid original alias identity")
        for source in group["sources"]:
            _validate_source(source)
            sid = source["source_id"]
            if sid in sources:
                _require(sources[sid] == source, "conflicting original source identity")
            else:
                position = source["session_index"], source["turn_index"]
                _require(position not in positions, "conflicting original source occurrence")
                positions[position] = sid
                sources[sid], owner[sid] = source, gid
                text_bytes += len(source["text"].encode("utf-8"))
                _require(len(sources) <= MAX_SOURCES and text_bytes <= MAX_TEXT_BYTES,
                         "original source allocation exceeds limit")
                sessions.setdefault(source["session_index"], []).append(sid)
            if key is not None:
                aliases.setdefault(key, []).append(sid)
    for members in sessions.values():
        members.sort(key=lambda sid: sources[sid]["turn_index"])
        first = sources[members[0]]
        _require(all((sources[sid]["session_id"], sources[sid]["observation_time"])
                     == (first["session_id"], first["observation_time"]) for sid in members),
                 "conflicting original session metadata")
    _require(isinstance(counterevidence_group_ids, list)
             and all(_identity(gid) and gid in groups for gid in counterevidence_group_ids)
             and len(set(counterevidence_group_ids)) == len(counterevidence_group_ids),
             "invalid counterevidence group order")
    diagnostics = {"mapped_citations": [], "unmapped_citations": [], "ambiguous_citations": [],
                   "dropped_cited_source_ids": [], "dropped_premise_source_ids": [],
                   "dropped_counterevidence_source_ids": []}
    cited = []
    for prior_index, seed in enumerate(prior_seeds):
        _require(isinstance(seed, dict) and set(seed) == {"source_groups", "cited_source_ids"},
                 "prior seed must contain only source groups and cited source IDs")
        _require(isinstance(seed["source_groups"], list) and len(seed["source_groups"]) <= MAX_GROUPS
                 and isinstance(seed["cited_source_ids"], list)
                 and len(seed["cited_source_ids"]) <= MAX_SOURCES, "invalid stripped prior packet")
        previous = {}
        for group in seed["source_groups"]:
            _require(isinstance(group, dict) and isinstance(group.get("sources"), list), "invalid prior group")
            for source in group["sources"]:
                if isinstance(source, dict) and _identity(source.get("source_id")):
                    previous.setdefault(source["source_id"], []).append((source, group.get("frozen_selected_id_sha256")))
        for citation in dict.fromkeys(sid for sid in seed["cited_source_ids"] if _identity(sid)):
            matches, reason = [], "citation_not_in_prior_packet"
            records = previous.get(citation, [])
            if records and any(source != records[0][0] for source, _ in records):
                reason, records = "conflicting_prior_source_identity", []
            for old, key in records:
                try:
                    _validate_source(old)
                except ValueError:
                    reason = "invalid_prior_source_metadata"
                    continue
                if citation in sources:
                    target = sources[citation]
                    fields = ("session_index", "turn_index", "session_id")
                    if _content(old) == _content(target) and all(old[field] == target[field] for field in fields):
                        matches.append(citation)
                    else:
                        reason = "prior_source_conflicts_with_original"
                else:
                    reason = "no_original_alias_content_match"
                    if _identity(key):
                        matches.extend(sid for sid in aliases.get(key, [])
                                       if _content(sources[sid]) == _content(old))
            matches = list(dict.fromkeys(matches))
            item = {"prior_index": prior_index, "cited_source_id": citation}
            if matches:
                item["original_source_ids"] = matches
                diagnostics["mapped_citations"].append(item)
                if len(matches) > 1:
                    diagnostics["ambiguous_citations"].append(copy.deepcopy(item))
                cited.extend(matches)
            else:
                diagnostics["unmapped_citations"].append({**item, "reason": reason})
        for citation in seed["cited_source_ids"]:
            if not _identity(citation):
                diagnostics["unmapped_citations"].append({"prior_index": prior_index,
                                                         "cited_source_id": None,
                                                         "reason": "invalid_citation_identity"})
    cited = list(dict.fromkeys(cited))
    premise, cited_sessions = [], list(dict.fromkeys(sources[sid]["session_index"] for sid in cited))
    for sid in cited:
        source = sources[sid]
        users = [candidate for candidate in sessions[source["session_index"]]
                 if sources[candidate]["role"] == "user" and sources[candidate]["turn_index"] < source["turn_index"]]
        if users:
            premise.append(users[-1])
    for index in cited_sessions:
        premise.extend([sid for sid in sessions[index] if sources[sid]["role"] == "user"][:2])
    counter = [source["source_id"] for gid in counterevidence_group_ids for source in groups[gid]["sources"]]
    selected, attempted = [], set()

    def packet(ids):
        selected_ids, by_group = set(ids), {}
        for sid in ids:
            by_group.setdefault(owner[sid], [])
        for gid in by_group:
            unique = dict.fromkeys(source["source_id"] for source in groups[gid]["sources"])
            by_group[gid] = [copy.deepcopy(sources[sid]) for sid in unique if sid in selected_ids and owner[sid] == gid]
        return [{**copy.deepcopy(groups[gid]), "selection_rank": rank, "sources": members}
                for rank, (gid, members) in enumerate(by_group.items())]

    def count(ids):
        serialized = serialize_callback(packet(ids))
        _require(isinstance(serialized, str), "verification serialization must be text")
        value = count_callback(serialized)
        _require(type(value) is int and value >= 0, "invalid verification token count")
        return value

    tokens = count([])
    base_exceeds_cap = tokens > cap
    for priority, candidates in (("cited", cited), ("premise", premise), ("counterevidence", counter)):
        for sid in candidates:
            if sid in attempted:
                continue
            attempted.add(sid)
            proposed = count(selected + [sid]) if not base_exceeds_cap else cap + 1
            if proposed <= cap:
                selected.append(sid)
                tokens = proposed
            else:
                diagnostics[f"dropped_{priority}_source_ids"].append(sid)
    kept = set(selected)
    diagnostics["dropped_prior_citations"] = [
        {**item, "dropped_original_source_ids": [sid for sid in item["original_source_ids"] if sid not in kept]}
        for item in diagnostics["mapped_citations"] if any(sid not in kept for sid in item["original_source_ids"])]
    return {"source_groups": packet(selected), "input_tokens": tokens, "cap": cap,
            "selection_failure": base_exceeds_cap or not selected,
            "failure_reason": "question_exceeds_cap" if base_exceeds_cap else ("empty_evidence_packet" if not selected else None),
            "diagnostics": diagnostics,
            "source_policy": "complete cited original turns; preceding/opening user premises; question-ranked counterevidence"}
