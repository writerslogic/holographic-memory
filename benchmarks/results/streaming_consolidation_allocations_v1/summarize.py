# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


METRICS = ("elapsed_ns", "isolated_peak_rss_kib", "isolated_cgroup_peak", "written_bytes")


def read_arm(path, expected_live):
    raw = path.read_bytes()
    text = raw.decode()
    records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    times = [r for r in records if "elapsed_ns" in r]
    hashes = [r["files_sha256"] for r in records if "files_sha256" in r]
    if len(times) != 1 or len(hashes) != 1 or not hashes[0]:
        raise ValueError(f"{path}: expected one timing and nonempty file manifest")
    if re.findall(r"Exit status: (\d+)", text) != ["0"]:
        raise ValueError(f"{path}: missing successful completion")
    row = times[0]
    if row["mode"] != "Merged" or row["live"] != expected_live or row["threads"] != 8:
        raise ValueError(f"{path}: unexpected consolidation state")
    if any(type(row[key]) is not int or row[key] <= 0 for key in METRICS):
        raise ValueError(f"{path}: nonpositive or invalid measurement")
    for name, digest in hashes[0].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: invalid file manifest")
    return dict(metrics=row, files_sha256=hashes[0], log_sha256=hashlib.sha256(raw).hexdigest())


def assess(arms):
    pairs = []
    for number in (1, 2, 3):
        parent, candidate = (arms[f"{number}-{arm}"] for arm in ("parent", "candidate"))
        pairs.append(dict(
            round=number,
            order=["candidate", "parent"] if number == 2 else ["parent", "candidate"],
            ratios={key: candidate["metrics"][key] / parent["metrics"][key] for key in METRICS},
            identical_files=parent["files_sha256"] == candidate["files_sha256"],
        ))
    medians = {key: statistics.median(p["ratios"][key] for p in pairs) for key in METRICS}
    faster = sum(p["ratios"]["elapsed_ns"] < 1 for p in pairs)
    checks = dict(median_time_below_0_98=medians["elapsed_ns"] < 0.98,
                  at_least_two_faster=faster >= 2,
                  median_rss_at_most_1_05=medians["isolated_peak_rss_kib"] <= 1.05,
                  identical_files=all(p["identical_files"] for p in pairs))
    return dict(arms=arms, pairs=pairs, median_ratios=medians, faster_pairs=faster,
                acceptance=checks, decision="eligible_for_scaled_validation" if all(checks.values()) else "rejected",
                limitation="Development consolidation only; allocation counts do not establish a performance win.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--live", type=int, default=211428)
    args = parser.parse_args()
    arms = {f"{n}-{arm}": read_arm(args.evidence / f"{n}-{arm}.log", args.live)
            for n in (1, 2, 3) for arm in ("parent", "candidate")}
    print(json.dumps(assess(arms), indent=2))


if __name__ == "__main__":
    main()
