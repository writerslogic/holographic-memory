# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Restore the benchmark evidence that lives in the evidence-v1 GitHub release.

The evidence directories and large result files are too big for git. benchmarks/evidence/MANIFEST.json
lists them with the release assets that carry them. This script downloads the assets (resuming a
partial download), checks every asset's sha256, extracts the sources to their original repo paths
and re-verifies each restored file against the archive's per-file sha256 index.

    python scripts/fetch_evidence.py --list
    python scripts/fetch_evidence.py                       # everything (hundreds of GB unpacked)
    python scripts/fetch_evidence.py benchmarks/results/native_harness_v1

Re-running is safe: verified assets stay in the cache and restored files that already match are
skipped. Nothing outside the destination root is ever written.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "evidence" / "MANIFEST.json"
SUPPORTED_SCHEMA = 1


class EvidenceError(Exception):
    """A verification or extraction failure that must stop the run."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(entries):
    """Canonical digest of a source's index entries; must match scripts/pack_evidence.py."""
    digest = hashlib.sha256()
    for entry in entries:
        if entry["type"] == "file":
            line = f"F\0{entry['path']}\0{entry['size']}\0{entry['sha256']}\n"
        elif entry["type"] == "symlink":
            line = f"L\0{entry['path']}\0{entry['target']}\n"
        else:
            line = f"D\0{entry['path']}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def safe_relative(name):
    """Reject archive member names that could leave the destination root."""
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "\0" in name or "\\" in name:
        raise EvidenceError(f"unsafe archive member name: {name!r}")
    return path


def within(path, sources):
    return any(path == source or path.startswith(source + "/") for source in sources)


def manifest_sources(manifest, wanted):
    sources = manifest["sources"]
    if not wanted:
        return sources
    known = {source["path"] for source in sources}
    unknown = [path for path in wanted if path not in known]
    if unknown:
        raise EvidenceError(f"not in the manifest: {', '.join(unknown)}")
    return [source for source in sources if source["path"] in wanted]


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != SUPPORTED_SCHEMA:
        raise EvidenceError(f"unsupported manifest schema {manifest.get('schema_version')!r}")
    return manifest


# ----- download -----------------------------------------------------------------------------------

def open_range(url, start):
    """Open url from byte offset start; returns (response, offset the body really starts at)."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    response = urllib.request.urlopen(request, timeout=60)
    status = getattr(response, "status", 200)
    if start and status != 206:  # the server ignored Range and sent the whole body
        return response, 0
    return response, start


def download(url, destination, size, sha256, retries=5):
    """Fetch url to destination, resuming `<destination>.partial`, and verify size and sha256.

    A verified file is kept; a download whose hash does not match is deleted and reported.
    """
    destination = Path(destination)
    marker = destination.with_name(destination.name + ".verified")
    if destination.exists() and destination.stat().st_size == size and marker.exists() \
            and marker.read_text().strip() == sha256:
        return
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if have < size:
            try:
                response, offset = open_range(url, have)
                with response, open(partial, "r+b" if have and offset == have else "wb") as handle:
                    handle.seek(offset)
                    handle.truncate(offset)
                    for block in iter(lambda: response.read(CHUNK), b""):
                        handle.write(block)
            except (urllib.error.URLError, OSError, ConnectionError) as exc:
                if attempt == retries - 1:
                    raise EvidenceError(f"download of {url} failed: {exc}") from exc
                continue
        if partial.stat().st_size == size:
            break
    if not partial.exists() or partial.stat().st_size != size:
        raise EvidenceError(f"{url}: expected {size} bytes, got {partial.stat().st_size if partial.exists() else 0}")
    actual = sha256_file(partial)
    if actual != sha256:
        partial.unlink()
        raise EvidenceError(f"{destination.name}: sha256 {actual} does not match the manifest's {sha256}")
    os.replace(partial, destination)
    marker.write_text(sha256 + "\n")


# ----- decompression ------------------------------------------------------------------------------

class PartStream:
    """Read the concatenation of several part files as one stream."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.handle = None

    def read(self, size=-1):
        out = bytearray()
        while size < 0 or len(out) < size:
            if self.handle is None:
                if not self.paths:
                    break
                self.handle = open(self.paths.pop(0), "rb")
            block = self.handle.read(CHUNK if size < 0 else size - len(out))
            if not block:
                self.handle.close()
                self.handle = None
                continue
            out += block
        return bytes(out)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def zstd_stream(paths):
    """Return (tar-readable binary stream, closer) over the decompressed concatenated parts."""
    try:
        from compression import zstd  # Python 3.14+
        raw = PartStream(paths)
        return zstd.ZstdFile(raw, "rb"), raw.close
    except ImportError:
        pass
    if shutil.which("zstd") is None:
        raise EvidenceError("zstd decompression needs Python 3.14+ or the zstd command line tool")
    process = subprocess.Popen(["zstd", "-d", "-q", "-c"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    import threading

    def feed():
        try:
            for path in paths:
                with open(path, "rb") as handle:
                    shutil.copyfileobj(handle, process.stdin, CHUNK)
        except BrokenPipeError:
            pass
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

    threading.Thread(target=feed, daemon=True).start()

    def close():
        process.stdout.close()
        process.wait()

    return process.stdout, close


# ----- extraction ---------------------------------------------------------------------------------

def file_matches(path, entry):
    try:
        return path.stat().st_size == entry["size"] and sha256_file(path) == entry["sha256"]
    except OSError:
        return False


def refuse_symlinked_parents(root, relative, wanted):
    """Stop a member from being written through a symlink inside the restored source.

    Directories above the source (a user may symlink benchmarks/results elsewhere) are theirs;
    everything from the source root down is created by this run.
    """
    top = next(len(Path(source).parts) for source in wanted if within(relative.as_posix(), [source]))
    current = root.joinpath(*relative.parts[:top - 1]) if top > 1 else root
    for part in relative.parts[top - 1:-1]:
        current = current / part
        if current.is_symlink():
            raise EvidenceError(f"{relative}: parent {part!r} is a symlink")


def extract(paths, root, index, wanted):
    """Stream the parts into root, keeping only members under `wanted` and listed in `index`."""
    stream, close = zstd_stream(paths)
    expected = {entry["path"]: entry for entry in index}
    written = skipped = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                relative = safe_relative(member.name)
                name = relative.as_posix()
                if not within(name, wanted):
                    continue
                entry = expected.get(name)
                if entry is None:
                    raise EvidenceError(f"{name}: member is not in the archive index")
                refuse_symlinked_parents(root, relative, wanted)
                target = root / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.issym():
                    if entry["type"] != "symlink" or member.linkname != entry["target"]:
                        raise EvidenceError(f"{name}: symlink differs from the index")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_symlink() and os.readlink(target) == member.linkname:
                        skipped += 1
                        continue
                    if target.is_symlink() or target.exists():
                        target.unlink()
                    os.symlink(member.linkname, target)
                    written += 1
                elif member.isreg():
                    if entry["type"] != "file" or member.size != entry["size"]:
                        raise EvidenceError(f"{name}: member size differs from the index")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file() and not target.is_symlink() and file_matches(target, entry):
                        skipped += 1
                        continue
                    temporary = target.with_name(target.name + ".fetch-partial")
                    digest = hashlib.sha256()
                    source = tar.extractfile(member)
                    with open(temporary, "wb") as handle:
                        for block in iter(lambda: source.read(CHUNK), b""):
                            digest.update(block)
                            handle.write(block)
                    if digest.hexdigest() != entry["sha256"]:
                        temporary.unlink()
                        raise EvidenceError(f"{name}: extracted bytes do not match the index sha256")
                    os.chmod(temporary, member.mode & 0o777)
                    os.utime(temporary, (member.mtime, member.mtime))
                    os.replace(temporary, target)
                    written += 1
                else:
                    raise EvidenceError(f"{name}: unsupported member type {member.type!r}")
    finally:
        close()
    return written, skipped


def verify_restored(root, index, wanted):
    """Every indexed file under `wanted` must exist with the recorded size and sha256."""
    problems = []
    for entry in index:
        if not within(entry["path"], wanted):
            continue
        path = root / entry["path"]
        if entry["type"] == "dir":
            if not path.is_dir():
                problems.append(f"missing directory {entry['path']}")
        elif entry["type"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != entry["target"]:
                problems.append(f"missing or changed symlink {entry['path']}")
        elif not file_matches(path, entry):
            problems.append(f"missing or changed file {entry['path']}")
    return problems


# ----- driver -------------------------------------------------------------------------------------

def asset_url(manifest, asset, base_url):
    if base_url:
        return base_url.rstrip("/") + "/" + asset["name"]
    return asset["url"]


def fetch_archive(manifest, archive, sources, root, cache, base_url):
    assets = [a for a in manifest["assets"] if a["archive"] == archive]
    index_asset = next((a for a in assets if a["role"] == "index"), None)
    parts = [a for a in assets if a["role"] == "part"]
    if index_asset is None or not parts:
        raise EvidenceError(f"manifest has no complete asset list for {archive}")
    paths = {}
    for asset in [index_asset] + parts:
        destination = Path(cache) / asset["name"]
        download(asset_url(manifest, asset, base_url), destination, asset["size"], asset["sha256"])
        paths[asset["name"]] = destination
    index = [json.loads(line) for line in paths[index_asset["name"]].read_text().splitlines()]
    wanted = [source["path"] for source in sources]
    for source in sources:
        below = [e for e in index if within(e["path"], [source["path"]])]
        if tree_digest(below) != source["sha256"] and source["kind"] == "dir":
            raise EvidenceError(f"{source['path']}: index does not match the manifest digest")
        if source["kind"] == "file" and (len(below) != 1 or below[0].get("sha256") != source["sha256"]):
            raise EvidenceError(f"{source['path']}: index does not match the manifest sha256")
    written, skipped = extract([paths[p["name"]] for p in parts], root, index, wanted)
    problems = verify_restored(root, index, wanted)
    if problems:
        raise EvidenceError(f"{archive}: {len(problems)} restored entries failed verification, first: {problems[0]}")
    return written, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="manifest source paths to restore (default: all)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dest", default=str(REPO_ROOT), help="root the original paths are restored under")
    parser.add_argument("--cache", default=None, help="download cache (default: <dest>/.evidence-cache)")
    parser.add_argument("--base-url", default=None, help="fetch assets from this URL prefix instead of the manifest's")
    parser.add_argument("--list", action="store_true", help="list sources and sizes, then exit")
    parser.add_argument("--keep-cache", action="store_true", help="keep downloaded archives after extraction")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        sources = manifest_sources(manifest, args.paths)
        if args.list:
            for source in sources:
                print(f"{source['size']:>16,}  {source['kind']:<4} {source['path']}")
            return 0
        root = Path(args.dest).resolve()
        cache = Path(args.cache) if args.cache else root / ".evidence-cache"
        by_archive = {}
        for source in sources:
            by_archive.setdefault(source["archive"], []).append(source)
        for archive, group in by_archive.items():
            written, skipped = fetch_archive(manifest, archive, group, root, cache, args.base_url)
            print(f"{archive}: restored {written} entries, {skipped} already present, verified")
            if not args.keep_cache:
                for asset in manifest["assets"]:
                    if asset["archive"] == archive:
                        for suffix in ("", ".verified"):
                            (cache / (asset["name"] + suffix)).unlink(missing_ok=True)
    except EvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
