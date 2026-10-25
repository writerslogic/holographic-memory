# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hook: send relative links that leave the documentation set to GitHub.

Pages link to benchmark result files, source files, examples and session logs that are
repository files but not site pages. Rewriting those links to the file on GitHub, at the ref the
site is built from, keeps strict link validation on without hiding a genuinely broken link: a
target that does not exist in the repository is left alone and fails the build.
"""

import os
import posixpath
import re
from pathlib import Path

REPO = "https://github.com/writerslogic/holographic-memory"
LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
FENCE = re.compile(r"^\s*(```|~~~)")


def _github_url(ref: str, repo_path: str, is_dir: bool) -> str:
    kind = "tree" if is_dir else "blob"
    return f"{REPO}/{kind}/{ref}/{repo_path}"


def on_page_markdown(markdown, page, config, files):
    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = docs_dir.parent
    ref = os.environ.get("HMS_DOCS_REF", "main")
    page_dir = posixpath.dirname(page.file.src_uri)

    def rewrite(match):
        target = match.group(2)
        if SCHEME.match(target) or target.startswith(("#", "/")):
            return match.group(0)
        path, sep, anchor = target.partition("#")
        if not path:
            return match.group(0)
        docs_relative = posixpath.normpath(posixpath.join(page_dir, path))
        known = files.get_file_from_path(docs_relative)
        if known is not None and not known.inclusion.is_excluded():
            return match.group(0)
        if docs_relative.startswith(".."):
            repo_path = posixpath.normpath(posixpath.join("docs", page_dir, path))
        else:
            repo_path = posixpath.normpath(posixpath.join("docs", docs_relative))
        absolute = (repo_root / repo_path).resolve()
        if repo_root not in absolute.parents or not absolute.exists():
            return match.group(0)
        url = _github_url(ref, repo_path, absolute.is_dir()) + (sep + anchor if anchor else "")
        return f"{match.group(1)}{url}{match.group(3)}"

    out, in_fence = [], False
    for line in markdown.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and page.file.src_uri.startswith("api/node/"):
            # TypeDoc emits matrix suffixes as [][]; autorefs treats them as an empty reference.
            line = re.sub(r"(?<=[`)])(?:\[\]){2,}", lambda m: m[0].replace("[", r"\["), line)
        out.append(line if in_fence else LINK.sub(rewrite, line))
    return "\n".join(out)
