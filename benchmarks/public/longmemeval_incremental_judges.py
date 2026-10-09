# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["openai==3.27.0", "python-dotenv==1.2.4", "tiktoken==0.14.0", "tokenizers==0.23.3"]
# ///
"""Import completed frozen local readers and judge new arms with the frozen producer."""

import argparse
import json
import time
from pathlib import Path

import longmemeval_reader_api as producer
from dotenv import load_dotenv


def completed_records(data):
    lines = data[:data.rfind(b"\n") + 1].decode("utf-8")
    return lines, [json.loads(line) for line in lines.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-output", type=Path, default=Path(
        "target/superiority-validation/weak-reader/weak_reader_outputs.jsonl"))
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env", override=True)
    protocol = producer.load_protocol()
    report = json.loads(producer.RESULT.read_text())
    if not all(arm["judge"]["status"] != "pending" for row in report["rows"]
               for arm in row["systems"]["matched_gpt54"]["arms"].values()):
        raise ValueError("matched judges must finish before transferring report ownership")
    if sum(len(entry["outcomes"]) for entry in report["instability"]["entries"]) != 400:
        raise ValueError("judge repeat study must finish before transferring report ownership")
    for path in (Path(__file__), Path("benchmarks/public/longmemeval_weak_reader.py")):
        artifact = {"path": str(path), "sha256": producer.sha(path)}
        existing = [item for item in report["artifacts"] if Path(item["path"]).resolve() == path.resolve()]
        if existing:
            if existing != [artifact]:
                raise ValueError("incremental judge artifact fingerprint changed")
        else:
            report["artifacts"].append(artifact)
    expected = {(row["qid"], name) for row in report["rows"] for name in protocol["arms"]}
    snapshot = Path("target/superiority-validation/weak_reader_import_snapshot.jsonl")
    producer.save_report(report)
    while True:
        complete_lines, records = completed_records(args.weak_output.read_bytes())
        identities = [(record["qid"], record["arm"]) for record in records]
        if len(set(identities)) != len(identities) or not set(identities) <= expected:
            raise ValueError("local reader output has duplicate or unexpected arms")
        pending = {(row["qid"], name) for row in report["rows"]
                   for name, arm in row["systems"]["weak_qwen4b"]["arms"].items()
                   if arm["reader"]["status"] == "pending"}
        if pending.intersection(identities):
            temporary = snapshot.with_suffix(".pending")
            temporary.write_text(complete_lines)
            temporary.replace(snapshot)
            producer.import_weak(report, snapshot)
        if any(arm["judge"]["status"] == "pending" and arm["reader"]["status"] != "pending"
               for row in report["rows"] for arm in row["systems"]["weak_qwen4b"]["arms"].values()):
            producer.run_judges(report, protocol)
            print(json.dumps({"imported_weak_arms": len(identities),
                              "spend_usd": report["spend_usd"]}), flush=True)
        if set(identities) == expected:
            break
        time.sleep(30)
    print("COMPLETE: 300 local readers imported and judged", flush=True)


if __name__ == "__main__":
    main()
