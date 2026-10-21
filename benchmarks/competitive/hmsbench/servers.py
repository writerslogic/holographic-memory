# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Where each server binary comes from (standard library only; also imported by the server worker).

Downloaded servers are pinned to one release artifact per platform with its sha256 and unpacked
under the cache; `install` refuses a download whose digest differs. PostgreSQL (+pgvector) and
Redis 8 (+ its query engine module) come from the platform package manager instead: Homebrew on
macOS, apt on the Modal image (see `SYSTEM_PACKAGES`). Those are recorded by the version the
binary reports, not by an artifact digest; every override is an environment variable.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

CACHE = Path(os.environ.get("HMS_BENCH_DATA", Path.home() / ".cache" / "hms-bench"))
SERVER_ROOT = CACHE / "competitive" / "servers"
PLATFORM = (platform.system(), platform.machine().replace("aarch64", "arm64").replace("amd64", "x86_64"))
MAC_ARM = ("Darwin", "arm64")
LINUX_X86 = ("Linux", "x86_64")

# name -> version, binary path inside the unpacked artifact, {platform: (url, sha256, digest source)}.
ARTIFACTS = {
    "qdrant": {"version": "1.19.2", "binary": "qdrant", "files": {
        MAC_ARM: ("https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-aarch64-apple-darwin.tar.gz",
                  "d24dfd25f4584684b337dd30ee016472cd25f53fc2501f1cd058ab3d7ee0b019", "GitHub release asset digest"),
        LINUX_X86: ("https://github.com/qdrant/qdrant/releases/download/v1.19.2/qdrant-x86_64-unknown-linux-gnu.tar.gz",
                    "34a788a09a4cb278b5c6d2af96d9a6db30c8c7e88ae3550455f7a4d425bb8d4b", "GitHub release asset digest")}},
    "weaviate": {"version": "1.40.1", "binary": "weaviate", "files": {
        MAC_ARM: ("https://github.com/weaviate/weaviate/releases/download/v1.40.1/weaviate-v1.40.1-darwin-all.zip",
                  "b839453b510c17a9a3101691f0e78151c6fd2d8cb13115ffe9c8099684fa40fc", "GitHub release asset digest"),
        LINUX_X86: ("https://github.com/weaviate/weaviate/releases/download/v1.40.1/weaviate-v1.40.1-linux-amd64.tar.gz",
                    "ed3234edb870310bb0d73f247781f9b597e4ca34e65d95c3ba8d5bed4dcbd13e", "GitHub release asset digest")}},
    "meilisearch": {"version": "1.55.0", "binary": "meilisearch", "files": {
        MAC_ARM: ("https://github.com/meilisearch/meilisearch/releases/download/v1.55.0/meilisearch-macos-apple-silicon",
                  "e02b98f5dc6e4206d1bc7fa755c2649e46143b96f34e7a79ea482c0b5512acfa", "GitHub release asset digest"),
        LINUX_X86: ("https://github.com/meilisearch/meilisearch/releases/download/v1.55.0/meilisearch-linux-amd64",
                    "7e33f5889235f14290074a19a6d3574bd9b10fb2ee8731eabb7dafd0714b0513", "GitHub release asset digest")}},
    # Typesense publishes no digests; these were computed at pin time (trust on first use).
    "typesense": {"version": "30.2", "binary": "typesense-server", "files": {
        MAC_ARM: ("https://dl.typesense.org/releases/30.2/typesense-server-30.2-darwin-arm64.tar.gz",
                  "7d8d6d0c33930ad20ea23dd184250547b16615944be891b2078e8a075152fa7e", "computed at pin time (no published digest)"),
        LINUX_X86: ("https://dl.typesense.org/releases/30.2/typesense-server-30.2-linux-amd64.tar.gz",
                    "cd791605e7c6cc7be457794f534cd4b9f9d361781adb10e40020efc22840ebc3", "computed at pin time (no published digest)")}},
    # Linux only: there is no macOS distribution and the k-NN plugin ships Linux native libraries.
    "opensearch": {"version": "3.9.0", "binary": "opensearch-3.9.0/bin/opensearch", "sha512": True, "files": {
        LINUX_X86: ("https://artifacts.opensearch.org/releases/bundle/opensearch/3.9.0/opensearch-3.9.0-linux-x64.tar.gz",
                    "a70e52a93ad66d9497dd734f5cc48eb4a8a8c4a92c47491a7ad0f1a898383d11c2948d4ba4159b2361d961bf221efa0fd02084e6389532df960b1ebe0c0a2099",
                    "published .sha512")}},
    # macOS PostgreSQL: zonky's relocatable build (a jar holding one .txz). Homebrew's keg-only
    # postgresql@18 resolves its share directory to an unlinked absolute path, so it cannot be
    # used without changing the Homebrew install.
    "postgres": {"version": "18.6.0", "binary": "bin/postgres", "nested": ".txz", "files": {
        MAC_ARM: ("https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-darwin-arm64v8/18.6.0/"
                  "embedded-postgres-binaries-darwin-arm64v8-18.6.0.jar",
                  "8f6ed6f38195b50a40712ada8b2a12060b6193c1a6977a6eb5cff053bb5a1b89", "Maven Central .sha256")}},
}

# Package-manager installs. Paths are the defaults; HMS_BENCH_PG_BIN, HMS_BENCH_PGVECTOR_SHARE,
# HMS_BENCH_PGVECTOR_LIB, HMS_BENCH_REDIS_BIN and HMS_BENCH_REDIS_MODULE override them.
SYSTEM_PACKAGES = {
    "postgres": {
        "Darwin": {"install": "PostgreSQL: the pinned zonky artifact; pgvector: brew install pgvector (0.8.7 when pinned)",
                   "bin": None,
                   # PostgreSQL 18's extension_control_path and dynamic_library_path load pgvector
                   # from its Homebrew keg (built for PostgreSQL 18; one major version, one ABI).
                   "pgvector_share": "/opt/homebrew/opt/pgvector/share/postgresql@18",
                   "pgvector_lib": "/opt/homebrew/opt/pgvector/lib/postgresql@18"},
        "Linux": {"install": "apt (PGDG): postgresql-18 postgresql-18-pgvector", "bin": "/usr/lib/postgresql/18/bin",
                  "pgvector_share": None, "pgvector_lib": None},
    },
    "redis": {
        "Darwin": {"install": "brew install redis (8.10.2 when pinned; bundles redisearch.so)", "bin": "/opt/homebrew/opt/redis/bin",
                   "module": "/opt/homebrew/opt/redis/lib/redis/modules/redisearch.so"},
        "Linux": {"install": "apt (packages.redis.io): redis 8.x, which bundles the query engine module", "bin": "/usr/bin",
                  "module": "/usr/lib/redis/modules/redisearch.so"},
    },
}


class NotAvailable(Exception):
    """The server has no artifact or package for this platform; the reason is recorded."""


def artifact_dir(name):
    return SERVER_ROOT / f"{name}-{ARTIFACTS[name]['version']}"


def artifact_binary(name):
    path = artifact_dir(name) / ARTIFACTS[name]["binary"]
    if not path.exists():
        raise NotAvailable(f"{name} {ARTIFACTS[name]['version']} is not installed; run `hms-bench install-servers {name}`")
    return path


def _digest(path, sha512=False):
    h = hashlib.sha512() if sha512 else hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install(name):
    """Download, verify and unpack the pinned artifact for this platform. Idempotent."""
    spec = ARTIFACTS[name]
    if PLATFORM not in spec["files"]:
        raise NotAvailable(f"{name} {spec['version']} publishes no artifact for {PLATFORM[0]} {PLATFORM[1]}")
    url, digest, _ = spec["files"][PLATFORM]
    target = artifact_dir(name)
    marker = target / ".verified"
    if marker.exists() and marker.read_text() == digest:
        return target / spec["binary"]
    download = SERVER_ROOT / "dl" / url.rsplit("/", 1)[1]
    download.parent.mkdir(parents=True, exist_ok=True)
    if not download.exists() or _digest(download, spec.get("sha512")) != digest:
        tmp = download.with_suffix(download.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        got = _digest(tmp, spec.get("sha512"))
        if got != digest:
            tmp.unlink()
            raise RuntimeError(f"{name}: digest mismatch for {url}: expected {digest}, got {got}")
        tmp.replace(download)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    if url.endswith((".tar.gz", ".tgz")):
        with tarfile.open(download) as tar:
            tar.extractall(target, filter="data")
    elif url.endswith((".zip", ".jar")):
        with zipfile.ZipFile(download) as z:
            z.extractall(target)
        for inner in target.glob("*" + spec.get("nested", "\0")):
            with tarfile.open(inner) as tar:
                tar.extractall(target, filter="data")
            inner.unlink()
    else:
        shutil.copy2(download, target / spec["binary"])
    binary = target / spec["binary"]
    if not binary.exists():
        raise RuntimeError(f"{name}: {spec['binary']} not found in {url}")
    # zipfile drops the executable bit; tar keeps it, but re-setting it is harmless.
    binary.chmod(0o755)
    if name == "opensearch":
        for sub in ("jdk/bin", "bin"):
            for p in (target / "opensearch-3.9.0" / sub).glob("*"):
                p.chmod(p.stat().st_mode | 0o111)
    marker.write_text(digest)
    return binary


def system_package(name):
    """Paths of a package-manager install for this OS, with environment overrides applied."""
    spec = SYSTEM_PACKAGES[name].get(PLATFORM[0])
    if spec is None:
        raise NotAvailable(f"{name}: no package source configured for {PLATFORM[0]}")
    spec = dict(spec)
    if name == "postgres":
        if spec["bin"] is None and PLATFORM in ARTIFACTS["postgres"]["files"]:
            spec["bin"] = str(artifact_dir("postgres") / "bin")
        spec["bin"] = os.environ.get("HMS_BENCH_PG_BIN", spec["bin"])
        spec["pgvector_share"] = os.environ.get("HMS_BENCH_PGVECTOR_SHARE", spec["pgvector_share"])
        spec["pgvector_lib"] = os.environ.get("HMS_BENCH_PGVECTOR_LIB", spec["pgvector_lib"])
        if not (Path(spec["bin"]) / "postgres").exists():
            raise NotAvailable(f"postgres not found in {spec['bin']} (install: {spec['install']})")
        if spec["pgvector_lib"] and not any(Path(spec["pgvector_lib"]).glob("vector.*")):
            raise NotAvailable(f"pgvector not found in {spec['pgvector_lib']} (install: {spec['install']})")
    else:
        spec["bin"] = os.environ.get("HMS_BENCH_REDIS_BIN", spec["bin"])
        spec["module"] = os.environ.get("HMS_BENCH_REDIS_MODULE", spec["module"])
        if not (Path(spec["bin"]) / "redis-server").exists() or not Path(spec["module"]).exists():
            raise NotAvailable(f"redis-server or {spec['module']} not found (install: {spec['install']})")
    return spec


def prepare(name):
    """Make server `name` runnable here: install its pinned artifact for this platform and check
    any package-manager parts. Raises NotAvailable with the reason when it cannot run here."""
    pinned = name in ARTIFACTS and PLATFORM in ARTIFACTS[name]["files"]
    if pinned:
        install(name)
    if name in SYSTEM_PACKAGES:
        system_package(name)
    elif not pinned:
        raise NotAvailable(f"{name} {ARTIFACTS[name]['version']} publishes no artifact for {PLATFORM[0]} {PLATFORM[1]}")


def provenance(name):
    """How the binary for `name` was obtained, for the results file."""
    out = {}
    if name in ARTIFACTS and PLATFORM in ARTIFACTS[name]["files"]:
        spec = ARTIFACTS[name]
        url, digest, source = spec["files"][PLATFORM]
        out = {"version": spec["version"], "url": url, "sha512" if spec.get("sha512") else "sha256": digest, "digest_source": source,
               "bytes": _dir_bytes(artifact_dir(name))}
    if name not in SYSTEM_PACKAGES:
        return out
    spec = system_package(name)
    out["install"] = spec["install"]
    if name == "postgres":
        out["postgres"] = subprocess.run([str(Path(spec["bin"]) / "postgres"), "--version"], capture_output=True, text=True).stdout.strip()
    else:
        out["redis"] = subprocess.run([str(Path(spec["bin"]) / "redis-server"), "--version"], capture_output=True, text=True).stdout.strip()
    return out


def _dir_bytes(path):
    path = Path(path)
    return sum(min(p.stat().st_size, p.stat().st_blocks * 512) for p in path.rglob("*") if p.is_file()) if path.exists() else 0
