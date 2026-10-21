# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pack untracked benchmark evidence into split tar.zst archives and write the manifest.

pack      stream sources into <name>.tar.zst.NNN parts plus a per-file sha256 index
manifest  combine the packed archives into benchmarks/evidence/MANIFEST.json

scripts/fetch_evidence.py is the inverse: it verifies the parts and restores the sources.
Parts concatenate in order to one zstd stream, so `cat NAME.tar.zst.* | zstd -d | tar -x`
also works. Members are stored under their ROOT-relative paths; symlinks stay symlinks.
"""
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

DEFAULT_PART_BYTES = 1_800_000_000  # GitHub caps a release asset at 2 GiB
CHUNK = 8 * 1024 * 1024


class HashingReader:
    """File wrapper that hashes and counts the bytes tarfile pulls through it."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def walk(root, rel):
    """Yield rel and everything below it in sorted order without following symlinks."""
    top = os.path.join(root, rel)
    if not os.path.isdir(top) or os.path.islink(top):
        yield rel
        return
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        dirnames.sort()
        reldir = os.path.relpath(dirpath, root)
        yield reldir
        for name in sorted(filenames):
            yield os.path.join(reldir, name)
        for name in list(dirnames):
            if os.path.islink(os.path.join(dirpath, name)):
                dirnames.remove(name)
                yield os.path.join(reldir, name)


def add_member(tar, root, member, index):
    full = os.path.join(root, member)
    mode = os.lstat(full).st_mode
    tar.inodes.clear()  # store every file in full; a hard link would carry no data
    info = tar.gettarinfo(full, arcname=member)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if stat.S_ISDIR(mode):
        tar.addfile(info)
        index.write(json.dumps({"path": member, "type": "dir"}) + "\n")
    elif stat.S_ISLNK(mode):
        tar.addfile(info)
        index.write(json.dumps({"path": member, "type": "symlink", "target": info.linkname}) + "\n")
    elif stat.S_ISREG(mode):
        with open(full, "rb") as handle:
            reader = HashingReader(handle)
            tar.addfile(info, reader)
        if reader.count != info.size:
            raise RuntimeError(f"{member} changed size while it was archived")
        index.write(json.dumps({"path": member, "type": "file", "size": info.size,
                                "sha256": reader.digest.hexdigest()}) + "\n")
    else:
        raise RuntimeError(f"unsupported file type: {member}")


def pack(name, outdir, root, rels, part_bytes=DEFAULT_PART_BYTES):
    """Archive rels (ROOT-relative) as NAME.tar.zst.NNN; returns the part records."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    zstd = subprocess.Popen(["zstd", "-T0", "-3", "-q", "-c"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    with open(outdir / f"{name}.index.jsonl", "w") as index:

        def produce():
            try:
                with tarfile.open(fileobj=zstd.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for rel in rels:
                        for member in walk(root, rel):
                            add_member(tar, root, member, index)
            except Exception as exc:  # surfaced once the pipe has drained
                errors.append(exc)
            finally:
                zstd.stdin.close()

        thread = threading.Thread(target=produce)
        thread.start()
        parts = []
        handle = digest = path = None
        written = 0
        try:
            while True:
                data = zstd.stdout.read(CHUNK)
                if not data:
                    break
                view = memoryview(data)
                while len(view):
                    if handle is None:
                        path = outdir / f"{name}.tar.zst.{len(parts):03d}"
                        handle, digest, written = open(path, "wb"), hashlib.sha256(), 0
                    take = min(len(view), part_bytes - written)
                    handle.write(view[:take])
                    digest.update(view[:take])
                    written += take
                    view = view[take:]
                    if written == part_bytes:
                        handle.close()
                        parts.append({"name": path.name, "size": written, "sha256": digest.hexdigest()})
                        handle = None
        finally:
            if handle is not None:
                handle.close()
                parts.append({"name": path.name, "size": written, "sha256": digest.hexdigest()})
        thread.join()
    zstd.stdout.close()
    if zstd.wait() != 0 or errors:
        raise SystemExit(f"packing {name} failed: zstd rc={zstd.returncode} errors={errors}")
    (outdir / f"{name}.parts.json").write_text(json.dumps(parts, indent=1) + "\n")
    return parts


def tree_digest(entries):
    """sha256 over the canonical lines of a source's index entries (already path-sorted)."""
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(outdir, repo, tag, archives):
    """archives maps an archive name to the source paths it carries."""
    base = f"https://github.com/{repo}/releases/download/{tag}/"
    sources, assets = [], []
    for archive, paths in archives.items():
        index_path = Path(outdir) / f"{archive}.index.jsonl"
        entries = [json.loads(line) for line in index_path.read_text().splitlines()]
        parts = json.loads((Path(outdir) / f"{archive}.parts.json").read_text())
        assets.append({"name": index_path.name, "archive": archive, "role": "index",
                       "size": index_path.stat().st_size, "sha256": sha256_file(index_path),
                       "url": base + index_path.name})
        for part in parts:
            assets.append({**part, "archive": archive, "role": "part", "url": base + part["name"]})
        for path in paths:
            below = [e for e in entries if e["path"] == path or e["path"].startswith(path + "/")]
            if not below:
                raise SystemExit(f"{path} is not in the {archive} index")
            files = [e for e in below if e["type"] == "file"]
            urls = [base + part["name"] for part in parts] + [base + index_path.name]
            if len(below) == 1 and files:  # a single file source carries its own hash
                sources.append({"path": path, "kind": "file", "size": files[0]["size"],
                                "sha256": files[0]["sha256"], "archive": archive, "asset_urls": urls})
            else:
                sources.append({"path": path, "kind": "dir", "size": sum(e["size"] for e in files),
                                "files": len(files), "entries": len(below),
                                "sha256": tree_digest(below), "archive": archive, "asset_urls": urls})
    return {"schema_version": 1, "repo": repo, "tag": tag, "base_url": base,
            "format": "NAME.tar.zst.NNN parts concatenate to one zstd-compressed tar; NAME.index.jsonl lists every member",
            "sources": sources, "assets": assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="archive sources into split tar.zst parts")
    p.add_argument("name")
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    p.add_argument("paths", nargs="+", help="ROOT-relative files or directories")
    m = sub.add_parser("manifest", help="write MANIFEST.json from packed archives")
    m.add_argument("--out", required=True, help="directory holding the packed archives")
    m.add_argument("--repo", default="writerslogic/holographic-memory")
    m.add_argument("--tag", default="evidence-v1")
    m.add_argument("--archive", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    m.add_argument("--manifest", required=True, help="where to write MANIFEST.json")
    args = parser.parse_args(argv)
    if args.command == "pack":
        parts = pack(args.name, args.out, args.root, args.paths, args.part_bytes)
        print(args.name, "parts", len(parts), "bytes", sum(x["size"] for x in parts))
        return 0
    archives = {}
    for spec in args.archive:
        name, _, paths = spec.partition("=")
        archives[name] = paths.split(",")
    manifest = build_manifest(args.out, args.repo, args.tag, archives)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
