# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Restore completed unscored workers without reopening or rerunning predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def restore(receipt_path):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1 or not isinstance(receipt.get("files"), list) or len(receipt["files"]) > 1000:
        raise ValueError("invalid or oversized checkpoint receipt")
    completed = {Path(name).name.removesuffix("-validation.json") for name in receipt["completed_result_files"]}
    restored = []
    for item in receipt["files"]:
        source, target = Path(item["retained_path"]), Path(item["original_path"])
        if source.parent.resolve(strict=True) != receipt_path.parent or target.parent.resolve(strict=True) != receipt_path.parent.parent or source.name != target.name or source.is_symlink() or target.is_symlink():
            raise ValueError("checkpoint artifact is outside its owned directory")
        if source.name.split("-validation", 1)[0] not in completed:
            continue
        if sha(source) != item["sha256"]:
            raise ValueError("retained checkpoint artifact changed")
        if target.exists():
            if sha(target) != item["sha256"]:
                raise ValueError("refusing to overwrite changed campaign artifact")
        else:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ann-checkpoint-", delete=False) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if sha(target) != item["sha256"]:
                    raise ValueError("concurrent campaign artifact differs") from None
            finally:
                temporary.unlink()
        restored.append(target.name)
    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps({"restored": restore(args.receipt), "predictions_scored": False}, indent=2))
