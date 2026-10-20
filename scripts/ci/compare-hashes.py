# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare two SHA256SUMS files from independent builds of the same commit.

usage: compare-hashes.py FIRST SECOND [--allow NAME ...]

Exit 1 if any artefact differs, is missing from one side, or the sets differ. --allow names (shell patterns)
artefacts whose difference is documented as expected (docs/RELEASING.md); they are reported
but do not fail the comparison. A documented exception that has started to match is also
reported so the exception list can shrink.
"""
import argparse
import fnmatch
import pathlib
import sys


def read(path: str) -> dict[str, str]:
    sums = {}
    for line in pathlib.Path(path).read_text().splitlines():
        digest, _, name = line.partition("  ")
        sums[name.strip()] = digest
    return sums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--allow", nargs="*", default=[])
    args = parser.parse_args()
    first, second = read(args.first), read(args.second)
    failed = False
    for name in sorted(first.keys() | second.keys()):
        a, b = first.get(name), second.get(name)
        allowed = any(fnmatch.fnmatch(name, pattern) for pattern in args.allow)
        if a is None or b is None:
            print(f"MISSING  {name} (first={a} second={b})")
            failed = failed or not allowed
        elif a == b:
            print(f"same     {name} {a}" + ("  (documented exception now matches)" if allowed else ""))
        else:
            print(f"DIFFERS  {name}\n           {a}\n           {b}" + ("  (documented exception)" if allowed else ""))
            failed = failed or not allowed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
