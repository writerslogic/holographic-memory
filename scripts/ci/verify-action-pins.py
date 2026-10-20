# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fail unless every `uses:` in .github is a full commit SHA whose version comment resolves to it.

Usage: verify-action-pins.py [--offline]

--offline only checks the shape (40-hex SHA plus a version comment); without it each
`owner/repo@sha # ref` is resolved through the GitHub API (`gh api`) and the commit for
`ref` must equal the pinned SHA (a branch name such as `stable` may be ahead of the pin). Local (`./`) and `docker://` references are skipped.
"""
import json
import pathlib
import re
import subprocess
import sys

USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+?)(?:\s+#\s*(?P<comment>.*))?\s*$")
SHA = re.compile(r"^[0-9a-f]{40}$")
VERSION = re.compile(r"^v?\d")


def is_ancestor(repo: str, pinned: str, ref: str) -> bool:
    """A moving branch comment (`stable`) is valid while the pin is reachable from it."""
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/compare/{pinned}...{ref}", "--jq", ".status"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() in ("identical", "ahead")


def files(root: pathlib.Path):
    yield from sorted(root.glob(".github/workflows/*.y*ml"))
    yield from sorted(root.glob(".github/actions/**/action.y*ml"))


def resolve(repo: str, ref: str, cache: dict) -> str | None:
    key = (repo, ref)
    if key not in cache:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/commits/{ref}", "--jq", ".sha"],
            capture_output=True,
            text=True,
        )
        cache[key] = proc.stdout.strip() if proc.returncode == 0 else None
    return cache[key]


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    root = pathlib.Path(__file__).resolve().parents[2]
    problems: list[str] = []
    cache: dict = {}
    seen = 0
    for path in files(root):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = USES.match(line)
            if not match:
                continue
            ref = match["ref"].strip("'\"")
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            seen += 1
            where = f"{path.relative_to(root)}:{number}"
            action, _, pinned = ref.partition("@")
            if not SHA.match(pinned):
                problems.append(f"{where}: {ref} is not pinned to a commit SHA")
                continue
            comment = (match["comment"] or "").split()
            if not comment:
                problems.append(f"{where}: {ref} has no version comment")
                continue
            if offline:
                continue
            repo = "/".join(action.split("/")[:2])
            actual = resolve(repo, comment[0], cache)
            if actual is None:
                problems.append(f"{where}: cannot resolve {repo}@{comment[0]}")
            elif actual != pinned and not (
                not VERSION.match(comment[0]) and is_ancestor(repo, pinned, comment[0])
            ):
                problems.append(f"{where}: {repo}@{comment[0]} is {actual}, pinned {pinned}")
    print(json.dumps({"checked": seen, "problems": len(problems)}))
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
