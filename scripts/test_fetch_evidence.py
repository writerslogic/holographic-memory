# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests of the evidence pack/fetch verification: hashes, resume, unsafe archive members.

Run: python -m unittest scripts/test_fetch_evidence.py   (needs the zstd command line tool)
"""
import http.server
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_evidence as fetch  # noqa: E402
import pack_evidence as pack  # noqa: E402

PAYLOAD = random.Random(7).randbytes(300_000)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves server.directory with Range support and optional single-byte corruption."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        name = self.path.lstrip("/")
        data = (self.server.directory / name).read_bytes()
        if name in self.server.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        start = 0
        header = self.headers.get("Range")
        self.server.requests.append((name, header))
        if header and not self.server.ignore_range:
            start = int(header.split("=")[1].rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@unittest.skipUnless(shutil.which("zstd"), "zstd command line tool required")
class EvidenceRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self.tmp / "src"
        results = self.src / "benchmarks" / "results"
        (results / "ev" / "sub").mkdir(parents=True)
        (results / "ev" / "empty").mkdir()
        (results / "ev" / "a.txt").write_text("alpha\n")
        (results / "ev" / "sub" / "b.bin").write_bytes(PAYLOAD)
        os.symlink("a.txt", results / "ev" / "rel-link")
        os.symlink(str(results / "ev" / "a.txt"), results / "ev" / "abs-link")
        (results / "big.json").write_text(json.dumps({"rows": list(range(500))}))
        self.out = self.tmp / "out"
        self.pack_all(40_000)

    def pack_all(self, part_bytes):
        for name, rels in (("ev", ["benchmarks/results/ev"]), ("loose", ["benchmarks/results/big.json"])):
            pack.pack(name, self.out, self.src, rels, part_bytes)
        self.manifest = pack.build_manifest(
            self.out, "example/repo", "evidence-test",
            {"ev": ["benchmarks/results/ev"], "loose": ["benchmarks/results/big.json"]})
        self.manifest_path = self.tmp / "MANIFEST.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def serve(self, corrupt=(), ignore_range=False):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        server.directory, server.corrupt, server.requests = self.out, set(corrupt), []
        server.ignore_range = ignore_range
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def run_fetch(self, base, dest, *extra):
        return fetch.main(["--manifest", str(self.manifest_path), "--dest", str(dest), "--base-url", base,
                           "--cache", str(self.tmp / "cache"), "--keep-cache", *extra])

    def test_restores_original_paths_symlinks_and_is_idempotent(self):
        _, base = self.serve()
        dest = self.tmp / "dest"
        self.assertEqual(self.run_fetch(base, dest), 0)
        results = dest / "benchmarks" / "results"
        self.assertEqual((results / "ev" / "sub" / "b.bin").read_bytes(), PAYLOAD)
        self.assertTrue((results / "ev" / "empty").is_dir())
        self.assertEqual(os.readlink(results / "ev" / "rel-link"), "a.txt")
        self.assertEqual(os.readlink(results / "ev" / "abs-link"), str(self.src / "benchmarks" / "results" / "ev" / "a.txt"))
        self.assertEqual((results / "big.json").read_text(), (self.src / "benchmarks/results/big.json").read_text())
        self.assertGreater(len([a for a in self.manifest["assets"] if a["role"] == "part" and a["archive"] == "ev"]), 1)
        # A second run extracts nothing: every restored entry already verifies.
        before = (results / "ev" / "a.txt").stat().st_mtime_ns
        self.assertEqual(self.run_fetch(base, dest), 0)
        self.assertEqual((results / "ev" / "a.txt").stat().st_mtime_ns, before)

    def test_corrupt_asset_is_rejected_and_not_kept(self):
        part = next(a["name"] for a in self.manifest["assets"] if a["role"] == "part" and a["archive"] == "ev")
        _, base = self.serve(corrupt=[part])
        dest = self.tmp / "dest"
        self.assertEqual(self.run_fetch(base, dest), 1)
        cache = self.tmp / "cache"
        self.assertFalse((cache / part).exists())
        self.assertFalse((cache / (part + ".partial")).exists())
        self.assertFalse((dest / "benchmarks" / "results" / "ev").exists())

    def test_download_resumes_a_partial_file_with_a_range_request(self):
        part = next(a for a in self.manifest["assets"] if a["role"] == "part" and a["archive"] == "ev")
        cache = self.tmp / "cache"
        cache.mkdir()
        data = (self.out / part["name"]).read_bytes()
        (cache / (part["name"] + ".partial")).write_bytes(data[:12_345])
        server, base = self.serve()
        self.assertEqual(self.run_fetch(base, self.tmp / "dest"), 0)
        self.assertIn((part["name"], "bytes=12345-"), server.requests)

    def test_server_ignoring_range_restarts_the_download_and_still_verifies(self):
        part = next(a for a in self.manifest["assets"] if a["role"] == "part" and a["archive"] == "ev")
        cache = self.tmp / "cache"
        cache.mkdir()
        (cache / (part["name"] + ".partial")).write_bytes(b"stale bytes that are not a prefix")
        server, base = self.serve(ignore_range=True)
        self.assertEqual(self.run_fetch(base, self.tmp / "dest"), 0)
        self.assertIn((part["name"], "bytes=33-"), server.requests)

    def test_changed_restored_file_is_detected_and_repaired(self):
        _, base = self.serve()
        dest = self.tmp / "dest"
        self.assertEqual(self.run_fetch(base, dest), 0)
        target = dest / "benchmarks" / "results" / "ev" / "a.txt"
        target.write_text("tampered\n")
        index = [json.loads(line) for line in (self.out / "ev.index.jsonl").read_text().splitlines()]
        problems = fetch.verify_restored(dest, index, ["benchmarks/results/ev"])
        self.assertEqual(problems, ["missing or changed file benchmarks/results/ev/a.txt"])
        self.assertEqual(self.run_fetch(base, dest), 0)
        self.assertEqual(target.read_text(), "alpha\n")

    def test_manifest_digest_mismatch_stops_before_extraction(self):
        for source in self.manifest["sources"]:
            if source["kind"] == "dir":
                source["sha256"] = "0" * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        _, base = self.serve()
        dest = self.tmp / "dest"
        self.assertEqual(self.run_fetch(base, dest), 1)
        self.assertFalse((dest / "benchmarks").exists())

    def test_only_selected_sources_are_restored(self):
        _, base = self.serve()
        dest = self.tmp / "dest"
        self.assertEqual(self.run_fetch(base, dest, "benchmarks/results/big.json"), 0)
        self.assertTrue((dest / "benchmarks" / "results" / "big.json").is_file())
        self.assertFalse((dest / "benchmarks" / "results" / "ev").exists())
        self.assertEqual(self.run_fetch(base, dest, "benchmarks/results/nope"), 1)

    def test_part_boundaries_limit_minus_one_limit_plus_one(self):
        # Total compressed size of the ev archive decides where the split lands.
        whole = pack.pack("ev", self.tmp / "whole", self.src, ["benchmarks/results/ev"], 10**9)
        total = whole[0]["size"]
        for limit, expected in ((total - 1, 2), (total, 1), (total + 1, 1)):
            parts = pack.pack("ev", self.tmp / f"p{limit}", self.src, ["benchmarks/results/ev"], limit)
            self.assertEqual(len(parts), expected, limit)
            self.assertEqual(sum(p["size"] for p in parts), total)
            joined = b"".join((self.tmp / f"p{limit}" / p["name"]).read_bytes() for p in parts)
            self.assertEqual(joined, (self.tmp / "whole" / whole[0]["name"]).read_bytes())


@unittest.skipUnless(shutil.which("zstd"), "zstd command line tool required")
class UnsafeMembers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "root"
        self.root.mkdir()

    def archive(self, build):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            build(tar)
        part = self.tmp / "a.tar.zst.000"
        part.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=buffer.getvalue(), capture_output=True, check=True).stdout)
        return [part]

    @staticmethod
    def add_file(tar, name, data=b"x"):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def index_for(self, *names):
        digest = __import__("hashlib").sha256(b"x").hexdigest()
        return [{"path": n, "type": "file", "size": 1, "sha256": digest} for n in names]

    def test_parent_traversal_and_absolute_names_are_refused(self):
        for name in ("benchmarks/../../escape", "/etc/escape", "benchmarks/results/ev/../../../escape"):
            parts = self.archive(lambda tar: self.add_file(tar, name))
            with self.assertRaises(fetch.EvidenceError, msg=name):
                fetch.extract(parts, self.root, self.index_for(name), ["benchmarks/results/ev"])
        self.assertEqual(list(self.tmp.glob("escape")), [])

    def test_member_through_a_symlink_created_by_the_archive_is_refused(self):
        outside = self.tmp / "outside"
        outside.mkdir()

        def build(tar):
            link = tarfile.TarInfo("benchmarks/results/ev/link")
            link.type, link.linkname = tarfile.SYMTYPE, str(outside)
            tar.addfile(link)
            self.add_file(tar, "benchmarks/results/ev/link/pwned")

        index = [{"path": "benchmarks/results/ev/link", "type": "symlink", "target": str(outside)}] + \
            self.index_for("benchmarks/results/ev/link/pwned")
        with self.assertRaises(fetch.EvidenceError):
            fetch.extract(self.archive(build), self.root, index, ["benchmarks/results/ev"])
        self.assertEqual(list(outside.iterdir()), [])

    def test_hard_links_devices_and_unindexed_members_are_refused(self):
        def hardlink(tar):
            info = tarfile.TarInfo("benchmarks/results/ev/h")
            info.type, info.linkname = tarfile.LNKTYPE, "benchmarks/results/ev/a"
            tar.addfile(info)

        def fifo(tar):
            info = tarfile.TarInfo("benchmarks/results/ev/f")
            info.type = tarfile.FIFOTYPE
            tar.addfile(info)

        for build, name in ((hardlink, "h"), (fifo, "f")):
            index = [{"path": f"benchmarks/results/ev/{name}", "type": "file", "size": 0, "sha256": "0" * 64}]
            with self.assertRaises(fetch.EvidenceError):
                fetch.extract(self.archive(build), self.root, index, ["benchmarks/results/ev"])
        with self.assertRaises(fetch.EvidenceError):
            fetch.extract(self.archive(lambda tar: self.add_file(tar, "benchmarks/results/ev/extra")),
                          self.root, [], ["benchmarks/results/ev"])

    def test_extracted_bytes_must_match_the_index_hash(self):
        name = "benchmarks/results/ev/f"
        index = [{"path": name, "type": "file", "size": 1, "sha256": "0" * 64}]
        with self.assertRaises(fetch.EvidenceError):
            fetch.extract(self.archive(lambda tar: self.add_file(tar, name)), self.root, index, ["benchmarks/results/ev"])
        self.assertFalse((self.root / name).exists())
        self.assertFalse((self.root / (name + ".fetch-partial")).exists())


if __name__ == "__main__":
    unittest.main()
