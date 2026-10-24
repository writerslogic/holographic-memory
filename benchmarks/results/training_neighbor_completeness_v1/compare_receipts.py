# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "mixed-comparison.json").read_text())
records = {}
for phase, receipt in summary["component_receipts"].items():
    payload = (root / receipt["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
    metadata, *rows = map(json.loads, gzip.decompress(payload).splitlines())
    assert metadata["kind"] == "metadata" and metadata["phase"] == phase
    records[phase] = rows

before, after = records["baseline"], records["repaired"]
assert len(before) == len(after) == summary["queries"]
result_fields = {"ids", "scores", "evaluations"}
changed = []
short = []
for index, (a, b) in enumerate(zip(before, after)):
    assert {k: v for k, v in a.items() if k not in result_fields} == {
        k: v for k, v in b.items() if k not in result_fields
    }, index
    if a != b:
        changed.append(index)
    if len(a["ids"]) < min(a["k"], a["n"]):
        short.append(index)
    assert len(b["ids"]) == min(b["k"], b["n"]), index
    assert len(set(b["ids"])) == len(b["ids"]), index
    assert len(b["scores"]) == len(b["ids"]), index

assert changed == short
assert len(short) == summary["formerly_short_repaired"]
assert len(before) - len(short) == summary["formerly_complete_unchanged"]
assert changed[0] == summary["first_divergence_zero_based_query"]
assert before[changed[0]] == summary["first_before"]
assert after[changed[0]] == summary["first_after"]
assert dict(Counter(str(before[i]["blocks"]) for i in changed)) == summary[
    "changed_by_block_bits"
]
print(
    f"PASS: {len(before)} matched component calls; "
    f"{len(before) - len(short)} complete calls preserve IDs, score bits and evaluation counts; "
    f"all {len(short)} short calls repaired."
)

connected = []
for name in ("connected-before.txt", "connected-after.txt"):
    connected.append(
        [line for line in (root / name).read_text().splitlines() if line.startswith("CONNECTED ")]
    )
assert len(connected[0]) == len(connected[1]) == 12
assert connected[0] == connected[1]
print("PASS: 12 connected aggregates match across all 960 searches, including evaluation counts.")
