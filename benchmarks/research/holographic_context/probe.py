# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen synthetic episode probe; see PROTOCOL.md for claims and limits."""

import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/public"))
import longmemeval_sota_context as C

SEEDS = [("development", 15630), ("held_out", 21425), ("held_out", 21065), ("held_out", 24948)]
COUNTS = [64, 256]
DIMENSIONS = [512, 2048]
QUERIES = 16
MAX_BYTES = 4096
MAX_SOURCES = 9
MAX_ROUNDS = 5


def digest(value):
    return hashlib.sha256(C.wire(value)).hexdigest()


def corpus(seed, count):
    rng = random.Random(seed)
    places = 8 if count == 64 else 16
    days = count // (places * 2)
    identities = [(place, day, version) for place in range(places)
                  for day in range(days) for version in range(2)]
    rng.shuffle(identities)
    episodes = []
    for ordinal, (place, day, version) in enumerate(identities):
        session = f"session-{seed}-{place}-{day}"
        version = f"v{version}"
        keys = [f"place:{place}", f"day:{day}", f"version:{version}",
                f"unique:{seed}-{ordinal}"]
        texts = ["Please arrange a journey.",
                 f"Rare fact: gate 東京-{seed}-{ordinal}-Zoë.",
                 "Keep the destination previously suggested.",
                 "That destination remains available.",
                 f"The place cue is {keys[0]}.",
                 "The place is recorded.",
                 f"The day cue is {keys[1]}.",
                 "The selected day and place are confirmed.",
                 f"The final cues are {keys[2]} and {keys[3]}."]
        sources = []
        for index, text in enumerate(texts):
            previous_users = list(range(0, index, 2))[:2]
            user = index if index % 2 == 0 else index - 1
            if user != index:
                previous_users.append(user)
            if user > 0:
                previous_users.append(user - 1)
            premises = sorted(set(previous_users))
            sources.append({
                "recordId": C.record_id(session, version, index, f"turn-{index}"),
                "sourceId": f"turn-{index}", "sessionId": session,
                "version": version, "turnIndex": index,
                "role": "user" if index % 2 == 0 else "assistant",
                "observationTime": 1_704_067_200_000 + ordinal * 100 + index,
                "text": text,
                "premiseRecordIds": [sources[i]["recordId"] for i in premises],
            })
        episodes.append({"keys": keys, "sources": sources})
    return episodes


def context(episode, transitive):
    sources = episode["sources"]
    by_id = {source["recordId"]: source for source in sources}
    chosen = {sources[-1]["recordId"]: sources[-1]}
    pending = list(sources[-1]["premiseRecordIds"])
    while pending:
        identity = pending.pop()
        if identity in chosen:
            continue
        source = by_id[identity]
        chosen[identity] = source
        if transitive:
            pending.extend(source["premiseRecordIds"])
    assert len(chosen) <= MAX_SOURCES
    assert len({source["turnIndex"] // 2 for source in chosen.values()}) <= MAX_ROUNDS
    packed = C.pack(list(chosen.values()), MAX_BYTES)
    return packed, C.unpack(packed)


def phase_code(seed, identity, dimension):
    rng = random.Random(hashlib.sha256(C.wire([seed, identity])).digest())
    return [cmath.rect(1.0, rng.random() * math.tau) for _ in range(dimension)]


def make_phase(seed, episodes, dimension):
    keys = sorted({key for episode in episodes for key in episode["keys"]})
    codes = {key: phase_code(seed, ["cue", key], dimension) for key in keys}
    episode_codes = [phase_code(seed, ["episode", index], dimension)
                     for index in range(len(episodes))]
    memory = [0j] * dimension
    for episode, code in zip(episodes, episode_codes):
        for key in episode["keys"]:
            memory = [value + cue * coordinate
                      for value, cue, coordinate in zip(memory, codes[key], code)]
    return memory, codes, episode_codes


def locate_phase(query, memory, codes, episode_codes):
    estimate = [0j] * len(memory)
    for key in query:
        estimate = [value + stored * cue.conjugate()
                    for value, stored, cue in zip(estimate, memory, codes[key])]
    scores = [sum((value * coordinate.conjugate()).real
                  for value, coordinate in zip(estimate, code))
              for code in episode_codes]
    return max(range(len(scores)), key=scores.__getitem__)


def trial(split, seed, count, dimension):
    episodes = corpus(seed, count)
    postings = {}
    for index, episode in enumerate(episodes):
        for key in episode["keys"]:
            postings.setdefault(key, set()).add(index)
    exact_bytes = 4 * (sum(len(ids) for ids in postings.values()) + len(postings) + 1)
    phase_bytes = 16 * dimension
    assert exact_bytes <= phase_bytes
    started = time.perf_counter_ns()
    memory, codes, episode_codes = make_phase(seed, episodes, dimension)
    phase_build_ns = time.perf_counter_ns() - started
    order = list(range(count))
    random.Random(seed + 1).shuffle(order)
    rows = []
    for condition in ("unique_partial_cue", "split_cues_temporal_distractors"):
        for target in order[:QUERIES]:
            episode = episodes[target]
            query = episode["keys"][-1:] if condition == "unique_partial_cue" else episode["keys"][:3]
            started = time.perf_counter_ns()
            exact = set.intersection(*(postings[key] for key in query))
            assert exact == {target}
            exact_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            phase_raw = locate_phase(query, memory, codes, episode_codes)
            phase_ns = time.perf_counter_ns() - started
            phase_verified = phase_raw if set(query) <= set(episodes[phase_raw]["keys"]) else None
            for method, located, raw, transitive, elapsed in [
                ("oracle_existing_context", target, target, False, None),
                ("oracle_transitive_context", target, target, True, None),
                ("exact_episode_intersection", next(iter(exact)), next(iter(exact)), True, exact_ns),
                ("phase_episode_verified", phase_verified, phase_raw, True, phase_ns),
            ]:
                packed, selected = (context(episodes[located], transitive) if located is not None
                                    else (C.pack([], MAX_BYTES), []))
                rare_id = episode["sources"][1]["recordId"]
                rows.append({
                    "split": split, "seed": seed, "episodes": count, "dimension": dimension,
                    "condition": condition, "target": target, "method": method,
                    "episode_correct": raw == target, "abstained": located is None,
                    "rare_source_recalled": any(source["recordId"] == rare_id for source in selected),
                    "wrong_episode_returned": located is not None and located != target,
                    "sources": len(selected), "context_bytes": len(C.wire(packed)),
                    "sources_sha256": packed["sourcesSha256"], "locator_ns": elapsed,
                })
    return {
        "split": split, "seed": seed, "episodes": count, "dimension": dimension,
        "corpus_sha256": digest(episodes), "exact_posting_payload_bytes": exact_bytes,
        "phase_memory_payload_bytes": phase_bytes,
        "phase_codebook_working_payload_bytes": 16 * dimension * (len(codes) + len(episode_codes)),
        "phase_build_ns": phase_build_ns,
    }, rows


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["split"], row["episodes"], row["dimension"], row["condition"], row["method"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        latency = [row["locator_ns"] for row in selected if row["locator_ns"] is not None]
        result.append(dict(zip(("split", "episodes", "dimension", "condition", "method"), key)) | {
            "queries": len(selected),
            **{field: sum(row[field] for row in selected) for field in
               ("episode_correct", "rare_source_recalled", "abstained", "wrong_episode_returned")},
            "mean_sources": statistics.mean(row["sources"] for row in selected),
            "mean_context_bytes": statistics.mean(row["context_bytes"] for row in selected),
            "locator_median_ns": statistics.median(latency) if latency else None,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings, rows = [], []
    for split, seed in SEEDS:
        for count in COUNTS:
            for dimension in DIMENSIONS:
                setting, generated = trial(split, seed, count, dimension)
                settings.append(setting)
                rows.extend(generated)
    result = {
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "context_encoder_sha256": hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "source_budget": MAX_SOURCES, "round_budget": MAX_ROUNDS, "byte_budget": MAX_BYTES,
        "settings": settings, "summary": aggregate(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for method in ("oracle_existing_context", "oracle_transitive_context", "exact_episode_intersection", "phase_episode_verified"):
        selected = [row for row in rows if row["split"] == "held_out" and row["method"] == method]
        print(json.dumps({"method": method, "held_out_config_query_observations": len(selected),
                          "rare_source_recalled": sum(row["rare_source_recalled"] for row in selected),
                          "episode_correct": sum(row["episode_correct"] for row in selected)}))


if __name__ == "__main__":
    main()
