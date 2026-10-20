# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runs the README code blocks marked `quickstart-check` so the README cannot rot.

A fenced block opened with ```js quickstart-check is run with Node from the repository root.
A block opened with ```rust quickstart-check is placed in the body of a `main` returning
`anyhow::Result<()>` inside the quickstart crate (examples/quickstart/rust) and run with cargo.

Usage: python scripts/check-readme-snippets.py [js|rust|all]
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^```(\w+) quickstart-check\n(.*?)^```$", re.S | re.M)
RUST_WRAPPER = "fn main() -> anyhow::Result<()> {{\n{body}\n    Ok(())\n}}\n"


def blocks(text: str, language: str) -> list[str]:
    return [body for lang, body in FENCE.findall(text) if lang == language]


def run_js(body: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", dir=ROOT, delete=False) as handle:
        handle.write(body)
    try:
        subprocess.run(["node", handle.name], cwd=ROOT, check=True)
    finally:
        Path(handle.name).unlink()


def run_rust(body: str) -> None:
    crate = ROOT / "examples" / "quickstart" / "rust"
    target = crate / "src" / "bin" / "readme_snippet.rs"
    target.parent.mkdir(exist_ok=True)
    target.write_text(RUST_WRAPPER.format(body=body))
    try:
        subprocess.run(
            ["cargo", "run", "--release", "--manifest-path", str(crate / "Cargo.toml"), "--bin", "readme_snippet"],
            cwd=ROOT,
            check=True,
        )
    finally:
        target.unlink()
        if not any(target.parent.iterdir()):
            target.parent.rmdir()


class ExtractionTests(unittest.TestCase):
    def test_only_marked_blocks_of_the_language_are_extracted(self):
        text = "```js\nskip()\n```\n```js quickstart-check\nrun()\n```\n```rust quickstart-check\nlet x = 1;\n```\n"
        self.assertEqual(blocks(text, "js"), ["run()\n"])
        self.assertEqual(blocks(text, "rust"), ["let x = 1;\n"])


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    text = (ROOT / "README.md").read_text()
    found = 0
    for language, runner in (("js", run_js), ("rust", run_rust)):
        if which in (language, "all"):
            for body in blocks(text, language):
                found += 1
                runner(body)
    if found == 0:
        print("no quickstart-check blocks found in README.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
