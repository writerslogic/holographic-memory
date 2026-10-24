# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(directory):
    aggregates = defaultdict(list)
    losses = []
    for name in ("run-synthetic-v1", "run-hybrid-v1"):
        cases = [json.loads(line) for line in (directory / name / "cases.jsonl").read_text().splitlines()]
        for case in cases:
            for row in case["methods"]:
                configuration = {key: row[key] for key in
                                 ("method", "width", "bandwidth", "phase_width_budget") if key in row}
                key = (name, case["scenario"], json.dumps(configuration, sort_keys=True))
                aggregates[key].append(row)
                if case["scenario"] not in ("mixed", "rare", "interior_rare"):
                    continue
                if row["method"] == "hybrid":
                    controls = [control for control in case["methods"]
                                if control["method"] in ("geometry48", "reservoir48")]
                elif row["method"] in ("phase_sum", "phase_mean"):
                    controls = [control for control in case["methods"] if control["method"] == "centroid"
                                or (control["method"] == "reservoir" and control["phase_width_budget"] == row["width"])]
                else:
                    continue
                for control in controls:
                    for fanout, value in row["coverage"].items():
                        difference = value - control["coverage"][fanout]
                        if difference < 0:
                            losses.append({"screen": name, "scenario": case["scenario"],
                                           "nominal_rows": case["nominal_rows"], "data_seed": case["seed"],
                                           "projection_seed": row["projection_seed"], **configuration,
                                           "control": control["method"], "fanout": int(fanout),
                                           "coverage": value, "control_coverage": control["coverage"][fanout],
                                           "difference": difference})
    rows = []
    for (screen, scenario, configuration), records in aggregates.items():
        coverage = {}
        for fanout in ("1", "2", "4", "8"):
            values = [row["coverage"][fanout] for row in records]
            coverage[fanout] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
        rows.append({"screen": screen, "scenario": scenario, **json.loads(configuration),
                     "records": len(records), "coverage": coverage})
    return {"aggregates": rows, "all_negative_mixed_rare_comparisons": losses,
            "loss_scope": "Every fanout in mixed/rare/interior_rare, phase versus centroid and equal-budget reservoir; hybrid versus equal-budget geometry48 and reservoir48. Raw files preserve all other comparisons."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
