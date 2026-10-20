# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent fixture, training, geometry, codec, accounting and decision checks."""

import argparse
import copy
import gzip
import json
import math
import statistics
import struct
import zlib
from pathlib import Path

KINDS = ("flat", "curved", "metric_only", "transport_only", "dependency", "low_rank", "packed", "lz4")
STREAMS = ("hierarchy", "flat", "heldout", "drift", "deletion", "supersession")
LOADS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
MASK = (1 << 64) - 1


def mix(a, b):
    z = (a * 0x9E3779B97F4A7C15 + b) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def uniform(seed, key):
    return (mix(seed, key) >> 11) / (1 << 53)


def nearest_integer(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def entity(seed, i, stream, learning):
    group = (mix(seed ^ 0x31, i) % 7 if learning else 7) if stream == "heldout" else mix(seed ^ 0x31, i) % 8
    angle = math.tau * (group + 0.15 + 0.7 * uniform(seed ^ 0x32, i)) / 8
    radius = 0.93 * math.sqrt(uniform(seed ^ 0x33, i)) if stream == "flat" else (0.25, 0.5, 0.75, 0.93)[mix(seed ^ 0x33, i) % 4]
    return {"point": [nearest_integer(32768 * radius * math.cos(angle)), nearest_integer(32768 * radius * math.sin(angle))], "category": group}


def distance(a, b, curvature):
    chord = math.hypot(a[0] - b[0], a[1] - b[1])
    if curvature == 0:
        return 2 * chord
    denominator = math.sqrt((1 - curvature * sum(x * x for x in a)) * (1 - curvature * sum(x * x for x in b)))
    return 2 / math.sqrt(curvature) * math.asinh(math.sqrt(curvature) * chord / denominator)


def shift(a, b, curvature):
    cross = b[0] * a[1] - b[1] * a[0]
    dot = sum(x * y for x, y in zip(a, b))
    return nearest_integer(2 * math.atan2(curvature * cross, 1 - curvature * dot) / math.tau * 16) % 16


def coordinates(e):
    return [x / 32768 for x in e["point"]]


def structured(e, attribute, stream):
    x, y = coordinates(e)
    if stream == "flat":
        return (math.floor((x + 1) * 4) + attribute * 3) % 16
    centers = [[0.7 * math.cos(math.tau * (g + 0.5) / 8), 0.7 * math.sin(math.tau * (g + 0.5) / 8)] for g in range(8)]
    chart = min(range(8), key=lambda c: (distance([x, y], centers[c], 1), c))
    return (chart * 2 + attribute * 3 + shift(centers[chart], [x, y], 1)) % 16


def labels(seed, e, i, stream, rate):
    return [mix(seed ^ 0x42, i * 4 + a) % 16 if mix(seed ^ 0x41, i * 4 + a) % 100 < rate else structured(e, a, stream) for a in range(4)]


def learning(seed, stream, rate):
    es = [entity(seed, i, stream, True) for i in range(128)]
    return es, [labels(seed, e, i, stream, rate) for i, e in enumerate(es)]


def learning_crc(es, vs):
    return zlib.crc32(b"".join(struct.pack("<hhB", *e["point"], e["category"]) + bytes(v) for e, v in zip(es, vs)))


def fixture(seed, stream, rate):
    es = [entity(seed, i, stream, False) for i in range(LOADS[-1] // 4)]
    initial = []
    for i, e in enumerate(es):
        for a, value in enumerate(labels(seed, e, i, stream, rate)):
            initial.append((value, mix(seed ^ 0x51, (i * 4 + a) // 32) & 0xFFFFFFFF, 1 + i // 8, 1))
    current = list(initial)
    for slot, observation in enumerate(current):
        if stream == "deletion" and slot % 5 == 0:
            current[slot] = None
        elif stream == "drift" or (stream == "supersession" and slot % 5 == 0):
            value, source, tick, _ = observation
            current[slot] = ((value + (3 if stream == "drift" else 7)) % 16, source, tick + 10000, 2)
    return es, initial, current


def observation_bytes(o):
    return b"\0" if o is None else b"\1" + struct.pack("<BIIH", *o)


def fixture_crc(es, observations):
    return zlib.crc32(b"".join(struct.pack("<hhB", *e["point"], e["category"]) for e in es) + b"".join(map(observation_bytes, observations)))


def predictions(model, es, observations=None, vs=None):
    result = []
    for i, e in enumerate(es):
        if model["kind"] == "dependency":
            parent = vs[i][0] if vs is not None else (observations[i * 4][0] if observations[i * 4] is not None else 0)
            index = (e["category"] * 16 + parent) * 4
            result.extend(model["table"][index:index + 4])
        elif model["kind"] == "low_rank":
            result.extend((model["table"][e["category"]] + o) % 16 for o in model["offsets"])
        elif model["kind"] in ("packed", "lz4"):
            result.extend([0] * 4)
        else:
            centers = [[x / 32768 for x in c["point"]] for c in model["charts"]]
            point = coordinates(e)
            chart = min(range(8), key=lambda c: (distance(point, centers[c], model["metric"]), c))
            phase = shift(centers[chart], point, model["transport"])
            result.extend((x + phase) % 16 for x in model["charts"][chart]["phases"])
    return result


def mode(counts):
    return max(range(16), key=lambda i: (counts[i], -i))


def fit_labels(model, es, vs):
    counts = [[[0] * 16 for _ in range(4)] for _ in range(8)]
    centers = [[x / 32768 for x in c["point"]] for c in model["charts"]]
    for e, v in zip(es, vs):
        p = coordinates(e)
        c = min(range(8), key=lambda i: (distance(p, centers[i], model["metric"]), i))
        phase = shift(centers[c], p, model["transport"])
        for a, x in enumerate(v):
            counts[c][a][(x - phase) % 16] += 1
    for chart, count in zip(model["charts"], counts):
        chart["phases"] = [mode(c) for c in count]


def errors(model, es, vs):
    return sum(a != b for a, b in zip(predictions(model, es, vs=vs), (v for row in vs for v in row)))


def parameter_bytes(model):
    return len(model["charts"]) * 8 + len(model["table"]) + len(model["offsets"])


def train(kind, es, vs, cal, cvs):
    empty = dict(kind=kind, version=1, metric=0.0, transport=0.0, charts=[], table=[], offsets=[])
    if kind in ("packed", "lz4"):
        return empty
    if kind == "dependency":
        counts = [[0] * 16 for _ in range(8 * 16 * 4)]
        for e, v in zip(es, vs):
            for a, x in enumerate(v):
                counts[(e["category"] * 16 + v[0]) * 4 + a][x] += 1
        empty["table"] = [mode(c) for c in counts]
        return empty
    if kind == "low_rank":
        empty["table"], empty["offsets"] = [0] * 8, [0] * 4
        for _ in range(4):
            for group in range(8):
                counts = [0] * 16
                for e, v in zip(es, vs):
                    if e["category"] == group:
                        for a, x in enumerate(v):
                            counts[(x - empty["offsets"][a]) % 16] += 1
                empty["table"][group] = mode(counts)
            for a in range(4):
                counts = [0] * 16
                for e, v in zip(es, vs):
                    counts[(v[a] - empty["table"][e["category"]]) % 16] += 1
                empty["offsets"][a] = mode(counts)
        return empty
    grids = {"flat": [(0, 0)], "curved": [(c, t) for c in (0.25, 0.5, 1) for t in (0, c)], "metric_only": [(c, 0) for c in (0.25, 0.5, 1)], "transport_only": [(0, c) for c in (0.25, 0.5, 1)]}
    best, score = None, math.inf
    for metric, transport in grids[kind]:
        model = copy.deepcopy(empty)
        model["metric"], model["transport"] = metric, transport
        model["charts"] = [dict(point=es[i * len(es) // 8]["point"], phases=[0] * 4) for i in range(8)]
        fit_labels(model, es, vs)
        loss = errors(model, es, vs)
        for sweep in range(2):
            for c in range(8):
                previous = copy.deepcopy(model)
                model["charts"][c]["point"] = es[(c * 37 + sweep * 71 + 13) % len(es)]["point"]
                fit_labels(model, es, vs)
                candidate = errors(model, es, vs)
                if candidate < loss:
                    loss = candidate
                else:
                    model = previous
        candidate = 33 + parameter_bytes(model) + (errors(model, cal, cvs) + 1) // 2
        if candidate < score:
            best, score = model, candidate
    return best


def decompress(data):
    output = bytearray()
    i = 0
    while i < len(data):
        token = data[i]
        i += 1
        literals = token >> 4
        if literals == 15:
            while True:
                x = data[i]
                i += 1
                literals += x
                if x != 255:
                    break
        assert i + literals <= len(data)
        output.extend(data[i:i + literals])
        i += literals
        if i == len(data):
            break
        offset = int.from_bytes(data[i:i + 2], "little")
        i += 2
        assert 0 < offset <= len(output)
        length = token & 15
        if length == 15:
            while True:
                x = data[i]
                i += 1
                length += x
                if x != 255:
                    break
        for _ in range(length + 4):
            output.append(output[-offset])
        assert len(output) <= 640
    return bytes(output)


def validate_blocks(payload, offsets, flags, expected):
    assert offsets[0] == 0 and offsets[-1] == len(payload)
    assert len(offsets) == len(flags) + 1 == len(expected) + 1
    for i, raw in enumerate(flags):
        assert raw in (0, 1) and offsets[i] <= offsets[i + 1]
        block = payload[offsets[i]:offsets[i + 1]]
        decoded = block if raw else decompress(block)
        assert decoded == expected[i], "lossless block oracle disagrees"
        assert len(block) <= len(decoded)


def packed(values):
    return bytes(values[i] | ((values[i + 1] if i + 1 < len(values) else 0) << 4) for i in range(0, len(values), 2))


def witness_blocks(observations):
    observed = [o for o in observations if o is not None]
    return [b"".join(struct.pack("<I", o[1]) for o in group) + b"".join(struct.pack("<I", o[2]) for o in group) + b"".join(struct.pack("<H", o[3]) for o in group) for start in range(0, len(observed), 64) for group in [observed[start:start + 64]]]


def values_for(kind, observations, predicted):
    return [o[0] for slot, o in enumerate(observations) if o is not None and (kind in ("packed", "lz4") or (kind == "dependency" and slot % 4 == 0) or predicted[slot] != o[0])]


def accounting(row, model, observations, predicted, common, initial=False):
    prefix = "initial_" if initial else ""
    n, kind = row["slots"], row["kind"]
    witness_payload = bytes.fromhex(common[prefix + "witness_payload_hex"])
    offsets, flags = common[prefix + "witness_offsets"], common[prefix + "witness_raw"]
    validate_blocks(witness_payload, offsets, flags, witness_blocks(observations))
    witness_bytes = len(witness_payload) + 4 * len(offsets) + len(flags)
    values = values_for(kind, observations, predicted)
    raw_values = packed(values)
    if kind == "lz4":
        payload = bytes.fromhex(row[prefix + "lz4_payload_hex"])
        value_offsets, value_flags = row[prefix + "block_offsets"], row[prefix + "raw_blocks"]
        validate_blocks(payload, value_offsets, value_flags, [raw_values[i:i + 128] for i in range(0, len(raw_values), 128)])
        value_bytes = len(payload)
        block_bytes = len(value_offsets) * 4 + len(value_flags)
    else:
        assert row[prefix + "lz4_payload_hex"] is None
        assert not row[prefix + "block_offsets"] and not row[prefix + "raw_blocks"]
        value_bytes, block_bytes = len(raw_values), 0
    expected = dict(entity_bytes=n // 4 * 6, membership_bytes=(n + 63) // 64 * 12, witness_bytes=witness_bytes, parameter_bytes=parameter_bytes(model), correction_index_bytes=0 if kind in ("packed", "lz4") else (n + 63) // 64 * 12, value_bytes=value_bytes, block_index_bytes=block_bytes, metadata_bytes=264)
    expected["resident_bytes"] = sum(expected.values())
    expected["serialized_bytes"] = 32 + 33 + parameter_bytes(model) + n // 4 * 5 + witness_bytes + expected["membership_bytes"] + expected["correction_index_bytes"] + value_bytes + block_bytes
    expected["query_workspace_bytes"] = 640
    expected["total_query_bytes"] = expected["resident_bytes"] + 640
    assert expected == row[prefix + "accounting"], (kind, expected, row[prefix + "accounting"])
    return expected, 0 if kind in ("packed", "lz4") else len(values)


def interval(values, samples):
    draws = sorted(sum(values[i] for i in sample) / len(values) for sample in samples)
    return [draws[49], draws[1949]]


def analyze(report, indexed):
    count = report["args"]["seeds"]
    samples = [[mix(0xB007 + draw, i) % count for i in range(count)] for draw in range(2000)]
    conditions, capacities = [], []
    seed_indices = range(count)
    for stream in STREAMS:
        for rate in (1, 10, 50, 100):
            group = {kind: {} for kind in KINDS}
            for n in LOADS:
                arms = {}
                for kind in KINDS:
                    rs = [indexed[(stream, rate, report["args"]["seed"] + i, n, kind)] for i in seed_indices]
                    group[kind][n] = rs
                    recall = [r["exact"] / r["observed"] for r in rs]
                    arms[kind] = dict(recall=statistics.mean(recall), recall_ci=interval(recall, samples), mean_query_bytes=statistics.mean(r["accounting"]["total_query_bytes"] for r in rs), mean_serialized_bytes=statistics.mean(r["accounting"]["serialized_bytes"] for r in rs), max_charged_peak_bytes=max(r["charged_peak_bytes"] for r in rs), mean_exceptions=statistics.mean(r["exceptions"] for r in rs), mean_p95_ns=statistics.mean(r["p95_ns"] for r in rs), mean_consolidation_ns=statistics.mean(r["consolidation_ns"] for r in rs))
                contrasts = {}
                curved = group["curved"][n]
                for kind in ("flat", "metric_only", "transport_only", "dependency", "low_rank", "packed", "lz4"):
                    baseline = group[kind][n]
                    contrasts[kind] = dict(query_bytes_ratio=interval([a["accounting"]["total_query_bytes"] / b["accounting"]["total_query_bytes"] for a, b in zip(curved, baseline)], samples), p95_ratio=interval([a["p95_ns"] / b["p95_ns"] for a, b in zip(curved, baseline)], samples), exception_reduction=interval([(b["exceptions"] - a["exceptions"]) / a["observed"] for a, b in zip(curved, baseline)], samples))
                conditions.append(dict(stream=stream, rate=rate, slots=n, arms=arms, contrasts=contrasts, mechanism_pass=all(contrasts[k]["exception_reduction"][0] > 0 for k in ("flat", "metric_only", "transport_only"))))
            capacity, qualifying = {}, {}
            for kind in KINDS:
                last = 0
                for n in LOADS:
                    rs = group[kind][n]
                    if max(r["charged_peak_bytes"] for r in rs) > report["args"]["budget"] or interval([r["exact"] / r["observed"] for r in rs], samples)[0] < 0.95:
                        break
                    last = n
                qualifying[kind] = last
                capacity[kind] = dict(slots=last, admitted=group[kind][last][0]["observed"] if last else 0, censored=last == LOADS[-1])
            comparisons = {}
            for kind in KINDS:
                if kind == "curved":
                    continue
                c, b = capacity["curved"], capacity[kind]
                if c["slots"] and b["slots"]:
                    ratio = interval([a["p95_ns"] / b["p95_ns"] for a, b in zip(group["curved"][c["slots"]], group[kind][b["slots"]])], samples)
                    comparisons[kind] = dict(p95_ratio=ratio, pass_target=c["admitted"] >= 2 * b["admitted"] and ratio[1] <= 1 and not b["censored"])
                else:
                    comparisons[kind] = dict(pass_target=False)
            capacities.append(dict(stream=stream, rate=rate, capacities=capacity, comparisons=comparisons, primary_pass=all(v["pass_target"] for v in comparisons.values())))
    return dict(protocol=report["protocol"], seeds=count, conditions=conditions, capacities=capacities, primary_passes=sum(x["primary_pass"] for x in capacities), mechanism_passes=sum(x["mechanism_pass"] for x in conditions))


def check(report):
    assert report["protocol"] == "manifold-schema-v1"
    args = report["args"]
    assert 2 <= args["seeds"] <= 30 and args["budget"] == 65536
    assert dict(report["layout"]) == dict(entity=6, witness=12, observation_option=20, schema=72, chart=8, workspace=640)
    root = Path(__file__).resolve().parents[2]
    for name, crc in report["sources"]:
        assert zlib.crc32((root / name).read_bytes()) == crc, f"stale source: {name}"
    banks = report["model_cases"]
    assert len(banks) == 6 * 4 * args["seeds"]
    assert len(report["rows"]) == len(banks) * len(LOADS) * len(KINDS)
    cases = {(b["stream"], b["rate"], b["seed"]) for b in banks}
    assert len(cases) == len(banks)
    assert cases == {(s, r, seed) for s in STREAMS for r in (1, 10, 50, 100) for seed in range(args["seed"], args["seed"] + args["seeds"])}
    rows_by_case = {i: [] for i in range(len(banks))}
    common = {}
    for row in report["rows"]:
        rows_by_case[row["model_case"]].append(row)
        if row["kind"] == "flat":
            common[(row["model_case"], row["slots"])] = row
        else:
            assert row["witness_payload_hex"] is None and row["initial_witness_payload_hex"] is None
    indexed = {}
    for index, bank in enumerate(banks):
        seed, stream, rate = bank["seed"], bank["stream"], bank["rate"]
        es, vs = learning(mix(seed, 0x713A), stream, rate)
        cal, cvs = learning(mix(seed, 0xCA11), stream, rate)
        assert learning_crc(es, vs) == bank["training_crc"]
        assert learning_crc(cal, cvs) == bank["calibration_crc"]
        assert mix(seed, 0x713A) != mix(seed, 0xCA11) != seed
        if seed == args["seed"]:
            for k, kind in enumerate(KINDS):
                expected = train(kind, es, vs, cal, cvs)
                assert expected == bank["models"][k], (stream, rate, kind, "training replay")
                nxt = train(kind, es, [[(v + 3) % 16 for v in row] for row in vs], cal, [[(v + 3) % 16 for v in row] for row in cvs]) if stream == "drift" else copy.deepcopy(expected)
                nxt["version"] = 2
                assert nxt == bank["next_models"][k], (stream, rate, kind, "migration replay")
        entities, initial, current = fixture(seed, stream, rate)
        old_predictions = {kind: predictions(bank["models"][k], entities, initial) for k, kind in enumerate(KINDS)}
        new_predictions = {kind: predictions(bank["next_models"][k], entities, current) for k, kind in enumerate(KINDS)}
        assert len(rows_by_case[index]) == len(LOADS) * 8
        for row in rows_by_case[index]:
            n, kind = row["slots"], row["kind"]
            key = (stream, rate, seed, n, kind)
            assert n in LOADS and key not in indexed
            indexed[key] = row
            k = KINDS.index(kind)
            assert bank["models"][k]["version"] == 1 and bank["next_models"][k]["version"] == 2
            observed = sum(o is not None for o in current[:n])
            assert row["fixture_crc"] == fixture_crc(entities[:n // 4], current[:n])
            assert row["initial_crc"] == fixture_crc(entities[:n // 4], initial[:n])
            assert row["recovered_crc"] == zlib.crc32(b"".join(map(observation_bytes, current[:n])))
            assert row["observed"] == row["exact"] == observed
            assert row["absent"] == row["absent_correct"] == n - observed
            assert row["version_rejected"]
            assert row["prediction_crc"] == (None if kind in ("packed", "lz4") else zlib.crc32(bytes(new_predictions[kind][:n])))
            new, exceptions = accounting(row, bank["next_models"][k], current[:n], new_predictions[kind][:n], common[(index, n)])
            old, _ = accounting(row, bank["models"][k], initial[:n], old_predictions[kind][:n], common[(index, n)], initial=True)
            assert row["exceptions"] == exceptions
            workspace = n * 20 + n * 13 + 640 * 4 + 8192 + 4096
            training = {"dependency": 32768, "low_rank": 64, "packed": 0, "lz4": 0}.get(kind, 2048) + 3 * (72 + parameter_bytes(bank["models"][k])) + 2 * 128 * 10 + 512 + (1024 if stream == "drift" else 0)
            build = old["total_query_bytes"] + n * 13 + 640 * 4 + 8192 + 4096
            peak = old["resident_bytes"] + new["resident_bytes"] + workspace
            assert row["consolidation_workspace_bytes"] == workspace
            assert row["training_peak_bytes"] == training
            assert row["initial_build_peak_bytes"] == build
            assert row["consolidation_peak_bytes"] == peak
            assert row["charged_peak_bytes"] == max(new["total_query_bytes"], training, build, peak)
            assert row["p95_ns"] == statistics.median(row["p95_rounds_ns"]) > 0
            assert row["build_ns"] > 0 and row["consolidation_ns"] > 0
        if index % (args["seeds"] * 4) == 0:
            print(f"validated {stream}", flush=True)
    return analyze(report, indexed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.report.suffix == ".gz" else open
    with opener(args.report, "rt") as source:
        report = json.load(source)
    summary = check(report)
    summary["report_crc"] = zlib.crc32(args.report.read_bytes())
    summary["checker_crc"] = zlib.crc32(Path(__file__).read_bytes())
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"validated {len(report['rows'])} rows; primary passes {summary['primary_passes']}; mechanism passes {summary['mechanism_passes']}")


if __name__ == "__main__":
    main()
