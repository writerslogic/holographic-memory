#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Writes the root files of the versioned docs site and repairs canonical URLs.

Usage: docs_finalize.py <site-dir> <public-base-url>

Every versioned build names the matching page under /latest/ as its canonical URL so search
engines index one copy. A page that exists in a version but not in latest (new in dev, or removed
since) would then point at a 404, so its canonical is rewritten to itself. The root gets the
redirect to latest/, the 404 page (GitHub Pages serves only a root 404.html), robots.txt and a
sitemap pointer limited to latest/.
"""

import re
import shutil
import sys
from pathlib import Path

CANONICAL = re.compile(r'(<link rel="canonical" href=")([^"]*)(")')
OG_URL = re.compile(r'(<meta property="og:url" content=")([^"]*)(")')


def page_url(base: str, version: str, relative: Path) -> str:
    parent = relative.parent.as_posix()
    return f"{base}/{version}/" + ("" if parent == "." else parent + "/")


def repair_canonicals(site: Path, base: str) -> int:
    latest = site / "latest"
    fixed = 0
    for version_dir in sorted(p for p in site.iterdir() if p.is_dir() and p.name != "latest"):
        for html_file in version_dir.rglob("*.html"):
            relative = html_file.relative_to(version_dir)
            if relative.name != "index.html" or (latest / relative).is_file():
                continue
            text = html_file.read_text(encoding="utf-8")
            own = page_url(base, version_dir.name, relative)
            new = CANONICAL.sub(lambda m: m.group(1) + own + m.group(3), text)
            new = OG_URL.sub(lambda m: m.group(1) + own + m.group(3), new)
            if new != text:
                html_file.write_text(new, encoding="utf-8")
                fixed += 1
    return fixed


def main() -> None:
    site, base = Path(sys.argv[1]), sys.argv[2].rstrip("/")
    fixed = repair_canonicals(site, base)
    shutil.copyfile(site / "latest" / "404.html", site / "404.html")
    (site / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        "<title>Holographic Memory</title>\n"
        '<link rel="canonical" href="latest/">\n'
        '<meta http-equiv="refresh" content="0; url=latest/">\n'
        '<a href="latest/">Holographic Memory documentation</a>\n',
        encoding="utf-8",
    )
    (site / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {base}/latest/sitemap.xml\n", encoding="utf-8"
    )
    print(f"finalized {site}: {fixed} canonical URLs repaired")


if __name__ == "__main__":
    main()
