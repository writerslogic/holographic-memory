# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned Lucene jars (Maven Central, sha256-verified) and the compiled Java comparator."""

import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .common import CACHE, ROOT

JARS = CACHE / "competitive" / "jars"
SOURCE = ROOT / "benchmarks/competitive/lucene/LuceneBm25.java"
VERSION = "9.12.3"
PINNED = {
    "lucene-core": "b64a3f8098a7572034fb30085cdee01b34ec81fb0e5a31b471536af58dc6c01b",
    "lucene-analysis-common": "fa571bd7caf0f0b4faf46a72ca004a7836f348c31d92bd522dddcc3d128d287e",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure():
    """Download (once) and verify the jars, compile the comparator, return the classpath."""
    JARS.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, digest in PINNED.items():
        path = JARS / f"{name}-{VERSION}.jar"
        if not path.exists():
            url = f"https://repo1.maven.org/maven2/org/apache/lucene/{name}/{VERSION}/{name}-{VERSION}.jar"
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        if _sha(path) != digest:
            raise ValueError(f"{path.name} does not match its pinned sha256")
        paths.append(path)
    classes = JARS / "classes"
    stamp = classes / f".{_sha(SOURCE)[:16]}"
    if not stamp.exists():
        shutil.rmtree(classes, ignore_errors=True)
        classes.mkdir()
        subprocess.run(["javac", "--release", "17", "-cp", ":".join(map(str, paths)), "-d", str(classes), str(SOURCE)], check=True)
        stamp.write_text("")
    return ":".join([*map(str, paths), str(classes)]), paths


def footprint():
    _, paths = ensure()
    return {"bytes": sum(p.stat().st_size for p in paths), "distributions": len(paths), "note": "two jars; also requires a JVM (not counted)"}


def java_version():
    out = subprocess.run(["java", "-version"], capture_output=True, text=True)
    return (out.stderr or out.stdout).splitlines()[0]
