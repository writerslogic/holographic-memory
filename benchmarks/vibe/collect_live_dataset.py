"""Collect and validate a completed exact-S VIBE dataset without rerunning queries."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location("live_runner", Path(__file__).with_name("modal_vibe.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=runner.DATASETS)
parser.add_argument("--audit", action="store_true")
args = parser.parse_args()
dataset = args.dataset
revision = "12c80decb094d57ee13ba187a01bf1de8788ca2f"
remote = "/vibe/run-s-20261023"
target = Path(__file__).resolve().parents[1] / "results/vibe_live_s_20261023"
records = []
for entry in runner.VOL.iterdir(f"{remote}/ledger/{dataset}"):
    if not entry.path.endswith(".json"):
        continue
    data = b"".join(runner.VOL.read_file(entry.path))
    record = json.loads(data)
    assert record["dataset"] == dataset
    label = "hms" if record["job"] == f"hms@{revision[:12]}" else record["job"]
    assert "/" not in label and ".." not in label
    (target / f"{dataset}-{label}-ledger.json").write_bytes(data)
    records.append(record)
assert records and all(record["state"] in ("done", "failed") for record in records), "Arms are not final"
hms = next(record for record in records if record["job"] == f"hms@{revision[:12]}")
assert hms["hms_rev"] == revision
completed = [record for record in records if record["job"] != "_download" and record["state"] == "done"]
for record in completed:
    validation = record["validation"]
    assert validation["done"] and validation["expected"] == validation["valid"] == validation["markers"]
    assert validation["test_count"] > 0
    if record["job"] == hms["job"]:
        assert validation["expected"] == 180
    elif hms["state"] == "done":
        assert validation["test_count"] == hms["validation"]["test_count"]

for suffix, name in [("json", "standing.json"), ("summary.parquet", "summary.parquet")]:
    data = b"".join(runner.VOL.read_file(f"{remote}/standings/{dataset}@{revision[:12]}.{suffix}"))
    (target / f"{dataset}-{name}").write_bytes(data)
data = b"".join(runner.VOL.read_file(f"{remote}/logs/{dataset}/export.log"))
(target / f"{dataset}-export.txt").write_bytes(data)
standing = json.loads((target / f"{dataset}-standing.json").read_text())
assert standing["hms_commit"] == revision
expected_algorithms = {record["algorithm"] for record in completed}
assert expected_algorithms <= standing["best_qps_at_recall"].keys(), "Export omits completed arms"
cohorts = {record["algorithm"]: runner.measurement_containers(record) for record in completed}
receipt = {"dataset": dataset, "hms_commit": revision, "final_states": {record["job"]: record["state"] for record in records},
           "measurement_containers": cohorts, "generated": standing["generated"],
           "validations": {record["algorithm"]: record["validation"] for record in completed}}
(target / f"{dataset}-collection-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
if args.audit:
    subprocess.run(["modal", "run", str(Path(__file__).with_name("modal_vibe.py")) + "::audit_dataset_artifacts",
                    "--dataset", dataset, "--hms-rev", revision], check=True)
for name, local in [("receipt.json", "artifact-receipt.json"), ("raw-results.sha256.json", "raw-results.sha256.json")]:
    path = f"{remote}/artifacts/{dataset}@{revision[:12]}/{name}"
    (target / f"{dataset}-{local}").write_bytes(b"".join(runner.VOL.read_file(path)))
print(json.dumps(receipt))
