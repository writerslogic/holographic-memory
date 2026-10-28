# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Clear empty cutoffs only when every logged model call failed before completion.

Mixed runs need per-question transport evidence that these logs do not provide.
Successful empty completions, including length truncations, must remain scored WRONG.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def clear_failed(work: Path) -> int:
    manifest = max(work.glob("manifest-*.json"), key=lambda p: p.stat().st_mtime)
    receipt = json.loads(manifest.read_text())
    failed = receipt["audit"]["empty_generated_answers"]
    if not failed:
        return 0
    if not isinstance(failed, list) or any(
        not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", qid) for qid in failed
    ):
        raise ValueError("invalid question IDs in failure receipt")
    llm = receipt["llm"]
    counts = [llm.get(key) for key in ("calls", "errors", "blocked")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("model-call failure counts are missing or invalid")
    calls, errors, blocked = counts
    if calls == 0 or errors + blocked != calls or llm.get("finish_reasons") or llm.get("resolved_models"):
        raise ValueError("successful or unclassified model calls exist; empty answers alone cannot justify selective rescoring")
    updates = {}
    for qid in failed:
        for path in work.glob(f"harness-out/predicted_*/{qid}.json"):
            data = json.loads(path.read_text())
            if data.get("question_id") != qid or path.is_symlink():
                raise ValueError("prediction identity or path does not match failure receipt")
            cutoffs = data.get("cutoff_results")
            if cutoffs is None:
                continue
            if not isinstance(cutoffs, dict) or any(
                not isinstance(result, dict) or result.get("generated_answer") != "" for result in cutoffs.values()
            ):
                raise ValueError("failure receipt includes a nonempty or invalid answer")
            data.pop("cutoff_results")
            updates[path] = data
    for path, data in updates.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return len(updates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    try:
        cleared = clear_failed(args.work)
    except (ValueError, KeyError) as error:
        sys.exit(str(error))
    print(f"Cleared {cleared} entirely failed question files")


if __name__ == "__main__":
    main()
