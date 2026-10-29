# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Restore checksum-pinned historical benchmark inputs from local Git objects."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 128 * 1024 * 1024


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def restore(root, revision, record, *, max_bytes=MAX_BYTES):
    root = root.resolve(strict=True)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("invalid source revision")
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256", "mode"}:
        raise ValueError("invalid historical input record")
    name, size, expected, mode = (record[key] for key in ("path", "bytes", "sha256", "mode"))
    if not isinstance(name, str) or len(name) > 4096 or "\0" in name:
        raise ValueError("invalid historical input path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("benchmarks", "results"):
        raise ValueError("historical input must be under benchmarks/results")
    if type(size) is not int or not 0 <= size <= max_bytes:
        raise ValueError("historical input exceeds its byte limit")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid historical input checksum")
    if type(mode) is not int or mode not in (0o644, 0o755):
        raise ValueError("invalid historical input mode")
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("historical input path escapes the checkout or is a symlink")
    if target.exists():
        if not target.is_file() or target.stat().st_size != size or file_hash(target) != expected:
            raise FileExistsError("existing input differs: " + name)
        return False
    object_name = revision + ":" + relative.as_posix()
    result = subprocess.run(["git", "cat-file", "-s", object_name], cwd=root,
                            capture_output=True, check=False, text=True)
    if result.returncode:
        raise ValueError("historical Git object is missing; fetch revision " + revision)
    if int(result.stdout.strip()) != size:
        raise ValueError("historical Git object has an unexpected size")
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".hms-restore-", delete=False) as output:
            staged = Path(output.name)
            subprocess.run(["git", "cat-file", "blob", object_name], cwd=root, stdout=output,
                           stderr=subprocess.PIPE, check=True)
            output.flush()
            os.fsync(output.fileno())
        if staged.stat().st_size != size or file_hash(staged) != expected:
            raise ValueError("historical input checksum differs")
        staged.chmod(mode)
        os.link(staged, target)
        if os.name == "posix":
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if staged is not None:
            staged.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--all", action="store_true", help="restore every recorded input")
    choices.add_argument("--path", action="append", help="restore an exact recorded path or directory prefix")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "benchmarks/evidence/retired-inputs.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    if not args.all and not args.path:
        for record in records:
            print(str(record["bytes"]) + "\t" + record["path"])
        return
    selected = records if args.all else [record for record in records if any(
        record["path"] == path.rstrip("/") or record["path"].startswith(path.rstrip("/") + "/")
        for path in args.path)]
    if not selected:
        parser.error("no historical inputs match the requested path")
    for record in selected:
        changed = restore(ROOT, manifest["revision"], record)
        print(("restored " if changed else "verified ") + record["path"])


if __name__ == "__main__":
    main()
