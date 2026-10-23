# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a LoCoMo/LongMemEval run for re-scoring after a spend cap (402) or exhausted credit (429).

Reads the newest manifest in WORK, and for every question in ``audit.empty_generated_answers``
removes only the ``cutoff_results`` key from its per-question file, so the next
``--evaluate-only`` scores those questions again and leaves every other result untouched.

    python clear_failed.py WORK
"""

import json
import sys
from pathlib import Path


def main() -> None:
    work = Path(sys.argv[1])
    manifest = max(work.glob("manifest-*.json"), key=lambda p: p.stat().st_mtime)
    failed = json.loads(manifest.read_text())["audit"]["empty_generated_answers"]
    cleared = 0
    for qid in failed:
        for path in work.glob(f"harness-out/predicted_*/{qid}.json"):
            data = json.loads(path.read_text())
            if data.pop("cutoff_results", None) is not None:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                cleared += 1
    print(f"{manifest.name}: {len(failed)} failed questions, cleared {cleared} files")


if __name__ == "__main__":
    main()
