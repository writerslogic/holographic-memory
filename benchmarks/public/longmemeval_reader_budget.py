# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fit verbatim reader sources while prioritizing complete user turns."""

import copy


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def cap_sources_user_first(arm, serialize_callback, count_callback, cap):
    _require(type(cap) is int and cap > 0, "reader token cap must be a positive integer")
    _require(isinstance(arm, dict), "reader evidence arm must be an object")
    groups = arm.get("source_groups")
    _require(isinstance(groups, list) and 1 <= len(groups) <= 64, "expected 1..=64 source groups")
    group_ids, source_ids = set(), set()
    for group in groups:
        _require(isinstance(group, dict), "source group must be an object")
        group_id = group.get("group_id")
        _require(isinstance(group_id, str) and group_id, "invalid group identity")
        _require(group_id not in group_ids, "duplicate selected group")
        group_ids.add(group_id)
        rank = group.get("selection_rank")
        _require(type(rank) is int and rank >= 0, "invalid selection rank")
        sources = group.get("sources")
        _require(isinstance(sources, list) and sources, "source group must contain sources")
        has_user = False
        for source in sources:
            _require(isinstance(source, dict), "source must be an object")
            source_id = source.get("source_id")
            _require(isinstance(source_id, str) and source_id, "invalid source identity")
            _require(source_id not in source_ids, "duplicate source occurrence")
            source_ids.add(source_id)
            _require(source.get("role") in ("user", "assistant"), "unsupported source role")
            _require(isinstance(source.get("text"), str), "source text must be text")
            has_user |= source["role"] == "user"
        _require(has_user, "source group must contain a user turn")
    _require(groups[0]["selection_rank"] == 0, "reader evidence requires frozen anchor")

    def count(retained):
        serialized = serialize_callback(retained)
        _require(isinstance(serialized, str), "complete reader serialization must be text")
        tokens = count_callback(serialized)
        _require(type(tokens) is int and tokens >= 0, "tokenizer returned invalid count")
        return tokens

    result = copy.deepcopy(arm)
    original = result["source_groups"]
    uncapped = count(original)
    retained = [dict(group, sources=[source for source in group["sources"] if source["role"] == "user"])
                for group in original]
    current = count(retained)
    dropped = []
    for group in sorted(retained[1:], key=lambda group: group["selection_rank"], reverse=True):
        if current <= cap:
            break
        retained.remove(group)
        dropped.append(group["group_id"])
        current = count(retained)
    failed = current > cap
    if not failed:
        by_id = {group["group_id"]: group for group in original}
        for group in sorted(retained, key=lambda group: group["selection_rank"]):
            sources = by_id[group["group_id"]]["sources"]
            included = {source["source_id"] for source in group["sources"]}
            for source in sources:
                if source["role"] != "assistant":
                    continue
                previous = group["sources"]
                group["sources"] = [item for item in sources
                                    if item["source_id"] in included or item["source_id"] == source["source_id"]]
                proposed = count(retained)
                if proposed <= cap:
                    included.add(source["source_id"])
                    current = proposed
                else:
                    group["sources"] = previous

    kept = {source["source_id"] for group in retained for source in group["sources"]}
    result["source_groups"] = retained
    result["reader_cap"] = {
        "cap": cap,
        "input_tokens": current,
        "uncapped_input_tokens": uncapped,
        "dropped_group_ids": dropped,
        "selection_failure": failed,
        "failure_reason": "anchor_user_input_exceeds_cap" if failed else None,
        "source_policy": "user_first",
        "omitted_assistant_source_ids": [source["source_id"] for group in original for source in group["sources"]
                                         if source["role"] == "assistant" and source["source_id"] not in kept],
        "preserved_user_source_ids": [source["source_id"] for group in retained for source in group["sources"]
                                      if source["role"] == "user"],
    }
    return result
