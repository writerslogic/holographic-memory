# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Security and privacy matrix (category 3B): drives src/bin/research/security-probe.rs and the
comparators, writes benchmarks/results/security_matrix_v1.json.

Cells: network egress during a workload; encryption-at-rest overhead (QPS, peak RSS); tamper
detection over random single-byte flips; signed audit-log verification; key zeroization (Linux
memory scan, run separately on a Linux host and merged with --merge); constant-time secret
comparison (Welch t); cargo deny. Every cell that cannot run records one exclusion line.

Usage: security_matrix.py --probe <binary> --out <json> [--flips 1000] [--docs 2000]
       [--queries 2000] [--repeats 3] [--lucene-jar <anserini fatjar>] [--lucene-core <jar>]
       [--merge <json from the linux host>]
"""
import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PASSPHRASE = "probe-passphrase-3B-" + "x" * 12


def run(cmd, stdin=None, timeout=600, env=None, cwd=None):
    """Runs a command; returns (returncode, stdout, stderr, seconds). Negative codes are signals."""
    t = time.monotonic()
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return p.returncode, p.stdout, p.stderr, time.monotonic() - t
    except subprocess.TimeoutExpired:
        return -999, "", "timeout", time.monotonic() - t


def time_wrapped(cmd, stdin=None, env=None):
    """Runs under /usr/bin/time to get the peak RSS in bytes; returns (rc, stdout, rss_bytes)."""
    if sys.platform == "darwin":
        rc, out, err, _ = run(["/usr/bin/time", "-l", *cmd], stdin=stdin, env=env)
        m = re.search(r"(\d+)\s+maximum resident set size", err)
        rss = int(m.group(1)) if m else None
    else:
        rc, out, err, _ = run(["/usr/bin/time", "-v", *cmd], stdin=stdin, env=env)
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
        rss = int(m.group(1)) * 1024 if m else None
    return rc, out, rss


def classify(rc, stdout):
    """Verdict of one tamper trial from the probe's exit code (used by the tests)."""
    if rc == 0:
        return "served_identical"
    if rc == 3:
        return "served_different"
    if rc == 2:
        return "detected"
    return "crashed"


def used_length(data):
    """Bytes up to and including the last non-zero byte: arena segments are preallocated
    sparse files, and a flip in their zero tail would test nothing."""
    end = len(data)
    while end > 0 and data[end - 1] == 0:
        end -= 1
    return end


def flip_byte(path, rng):
    data = bytearray(path.read_bytes())
    used = used_length(data)
    if used == 0:
        return None
    at = rng.randrange(used)
    old = data[at]
    data[at] = old ^ (1 << rng.randrange(8))
    path.write_bytes(bytes(data))
    return at


def clone_tree(src, dst):
    """Copies a store with clones or sparse-aware copies (stores hold sparse arena files)."""
    if dst.exists():
        shutil.rmtree(dst)
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", "-R", str(src), str(dst)], check=True)
    else:
        subprocess.run(["cp", "-R", "--sparse=always", str(src), str(dst)], check=True)


def store_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_size > 0)


def tamper_warm(probe, seed_dir, expected_path, flips, encrypted, rng):
    counts = {"detected": 0, "served_different": 0, "served_identical": 0, "crashed": 0}
    per_file = {}
    with tempfile.TemporaryDirectory(prefix="warm-tamper-") as tmp:
        for _ in range(flips):
            work = Path(tmp) / "store"
            clone_tree(seed_dir, work)
            files = store_files(work)
            target = rng.choice(files)
            flip_byte(target, rng)
            cmd = [probe, "open", "--store", str(work), "--expect", str(expected_path)]
            if encrypted:
                cmd.append("--encrypted")
            rc, out, err, _ = run(cmd, stdin=PASSPHRASE + "\n" if encrypted else None, timeout=120)
            verdict = classify(rc, out)
            counts[verdict] += 1
            per_file.setdefault(target.name, {"detected": 0, "served_different": 0, "served_identical": 0, "crashed": 0})[verdict] += 1
    return {"flips": flips, "counts": counts, "per_file": per_file}


def build_sqlite(path, docs):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, text TEXT)")
    con.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(id UNINDEXED, text)")
    rng = random.Random(11)
    words = ["ledger", "harbor", "signal", "copper", "meadow", "quartz", "lantern", "timber", "orbit", "canyon", "velvet", "beacon"]
    for n in range(docs):
        text = f"document {n} " + " ".join(rng.choice(words) for _ in range(40))
        con.execute("INSERT INTO docs VALUES (?, ?)", (f"d{n:06}", text))
        con.execute("INSERT INTO docs_fts VALUES (?, ?)", (f"d{n:06}", text))
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()


def sqlite_query(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    # Bytes, not str: a flipped byte inside a text value is served by SQLite without an error,
    # and must count as served-different rather than raise on decoding.
    con.text_factory = bytes
    try:
        rows = con.execute("SELECT id FROM docs_fts WHERE docs_fts MATCH 'ledger harbor' ORDER BY rank LIMIT 10").fetchall()
        rows += con.execute("SELECT id, length(text) FROM docs ORDER BY id LIMIT 50").fetchall()
        check = con.execute("PRAGMA integrity_check").fetchone()[0]
        return rows, check.decode("utf-8", "replace") if isinstance(check, bytes) else check
    finally:
        con.close()


def tamper_sqlite(seed_db, flips, rng):
    pristine, _ = sqlite_query(seed_db)
    counts = {"detected": 0, "served_different": 0, "served_identical": 0, "crashed": 0}
    with tempfile.TemporaryDirectory(prefix="sqlite-tamper-") as tmp:
        for _ in range(flips):
            work = Path(tmp) / "t.db"
            shutil.copyfile(seed_db, work)
            flip_byte(work, rng)
            try:
                rows, check = sqlite_query(work)
            except (sqlite3.DatabaseError, UnicodeDecodeError):
                # DatabaseError, or an SQLite error whose message carries the flipped bytes
                # (Python decodes error text as UTF-8 before raising).
                counts["detected"] += 1
                continue
            if check != "ok":
                counts["detected"] += 1
            elif rows != pristine:
                counts["served_different"] += 1
            else:
                counts["served_identical"] += 1
    return {"flips": flips, "counts": counts, "detection_rule": "DatabaseError (or an undecodable SQLite error message) on open/query, or PRAGMA integrity_check != ok"}


def lucene_available(jar):
    return bool(jar) and Path(jar).exists() and shutil.which("java") is not None


def build_lucene(jar, index_dir, docs):
    with tempfile.TemporaryDirectory(prefix="lucene-coll-") as coll:
        rng = random.Random(11)
        words = ["ledger", "harbor", "signal", "copper", "meadow", "quartz", "lantern", "timber", "orbit", "canyon", "velvet", "beacon"]
        with open(Path(coll) / "docs.jsonl", "w") as f:
            for n in range(docs):
                f.write(json.dumps({"id": f"d{n:06}", "contents": f"document {n} " + " ".join(rng.choice(words) for _ in range(40))}) + "\n")
        rc, out, err, secs = run(["java", "-cp", jar, "io.anserini.index.IndexCollection", "-collection", "JsonCollection",
                                  "-input", coll, "-index", str(index_dir), "-generator", "DefaultLuceneDocumentGenerator",
                                  "-threads", "1", "-storeRaw"], timeout=1200)
        return rc, secs, (err or out)[-400:]


def lucene_check(core_jar, index_dir):
    rc, out, err, _ = run(["java", "-cp", core_jar, "org.apache.lucene.index.CheckIndex", str(index_dir)], timeout=300)
    text = out + err
    if "No problems were detected" in text:
        return "served_identical"
    if rc != 0 or "broken" in text or "ERROR" in text or "corrupt" in text.lower():
        return "detected"
    return "served_different"


def tamper_lucene(core_jar, index_dir, flips, rng):
    counts = {"detected": 0, "served_different": 0, "served_identical": 0, "crashed": 0}
    with tempfile.TemporaryDirectory(prefix="lucene-tamper-") as tmp:
        for _ in range(flips):
            work = Path(tmp) / "idx"
            clone_tree(index_dir, work)
            target = rng.choice([p for p in store_files(work) if p.name != "write.lock"])
            flip_byte(target, rng)
            counts[lucene_check(core_jar, work)] += 1
    return {"flips": flips, "counts": counts, "detection_rule": "org.apache.lucene.index.CheckIndex reports a problem; served_identical = 'No problems were detected' (a CheckIndex pass, not a query comparison)"}


def egress(probe, docs, queries):
    """Per-process network bytes of a full probe workload (macOS nettop, plus the process's
    open network sockets from lsof); Linux uses /proc/net/dev deltas."""
    with tempfile.TemporaryDirectory(prefix="warm-egress-") as tmp:
        # Ten times the query count so the workload outlives several nettop samples.
        cmd = [probe, "bench", "--store", str(Path(tmp) / "store"), "--docs", str(docs), "--queries", str(queries * 10)]
        if sys.platform == "darwin" and shutil.which("nettop"):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            samples, sockets = [], []
            while proc.poll() is None:
                rc, out, _, _ = run(["nettop", "-p", str(proc.pid), "-x", "-L", "1", "-J", "bytes_in,bytes_out"], timeout=30)
                for line in out.splitlines():
                    if line.startswith("security-probe"):
                        parts = line.strip().split(",")
                        if len(parts) >= 3:
                            samples.append((int(parts[1] or 0), int(parts[2] or 0)))
                rc, out, _, _ = run(["lsof", "-a", "-p", str(proc.pid), "-i", "-n", "-P"], timeout=30)
                sockets.append(len([l for l in out.splitlines()[1:] if l.strip()]))
            out, err = proc.communicate()
            last = samples[-1] if samples else (None, None)
            return {"method": "nettop -p <pid> sampled until exit (cumulative per-process bytes) and lsof -i socket counts", "bytes_in": last[0], "bytes_out": last[1],
                    "samples": len(samples), "max_open_network_sockets": max(sockets) if sockets else None, "socket_samples": len(sockets),
                    "workload": json.loads(out.strip().splitlines()[-1]) if out.strip() else None, "rc": proc.returncode}
        before = Path("/proc/net/dev").read_text() if Path("/proc/net/dev").exists() else None
        rc, out, err, secs = run(cmd, timeout=1200)
        after = Path("/proc/net/dev").read_text() if Path("/proc/net/dev").exists() else None
        return {"method": "/proc/net/dev host-wide deltas (not per process)", "before": before, "after": after, "rc": rc,
                "workload": json.loads(out.strip().splitlines()[-1]) if out.strip() else None}


def overhead(probe, docs, queries, repeats):
    rows = []
    for r in range(repeats):
        for encrypted in (False, True):
            with tempfile.TemporaryDirectory(prefix="warm-ovh-") as tmp:
                cmd = [probe, "bench", "--store", str(Path(tmp) / "store"), "--docs", str(docs), "--queries", str(queries)]
                if encrypted:
                    cmd.append("--encrypted")
                rc, out, rss = time_wrapped(cmd, stdin=PASSPHRASE + "\n" if encrypted else None)
                row = json.loads(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else {"error": rc}
                row.update({"repeat": r, "encrypted": encrypted, "peak_rss_bytes": rss})
                rows.append(row)
    def med(key, enc):
        vals = [x[key] for x in rows if x.get("encrypted") == enc and key in x and x[key] is not None]
        return statistics.median(vals) if vals else None
    return {"rows": rows, "median": {"plain": {"qps": med("qps", False), "seed_s": med("seed_s", False), "peak_rss_bytes": med("peak_rss_bytes", False), "store_bytes": med("store_bytes", False)},
                                     "encrypted": {"qps": med("qps", True), "seed_s": med("seed_s", True), "peak_rss_bytes": med("peak_rss_bytes", True), "store_bytes": med("store_bytes", True)}}}


def audit_cell(probe, docs, rng):
    with tempfile.TemporaryDirectory(prefix="warm-audit-") as tmp:
        store = Path(tmp) / "store"
        rc, out, err, _ = run([probe, "seed", "--store", str(store), "--docs", str(docs), "--audit"], timeout=600)
        if rc != 0:
            return {"error": err[-300:]}
        rc0, out0, _, _ = run([probe, "audit-verify", "--store", str(store)], timeout=120)
        intact = json.loads(out0.strip().splitlines()[-1]) if out0.strip() else {}
        log = store / "audit.bin"
        size = log.stat().st_size
        # Flip one byte inside a signature region of a random entry (entries are 105 bytes).
        entry = rng.randrange(size // 105)
        at = entry * 105 + 41 + rng.randrange(64)
        data = bytearray(log.read_bytes())
        data[at] ^= 0x01
        log.write_bytes(bytes(data))
        rc1, out1, _, _ = run([probe, "audit-verify", "--store", str(store)], timeout=120)
        flipped = json.loads(out1.strip().splitlines()[-1]) if out1.strip() else {}
        return {"intact": intact, "intact_rc": rc0, "flipped_signature_byte": {"entry": entry, "offset": at, **flipped}, "flipped_rc": rc1,
                "pass": rc0 == 0 and intact.get("failed") == 0 and intact.get("verified", 0) > 0 and flipped.get("failed", 0) >= 1}


def cargo_deny():
    rc, out, err, _ = run(["cargo", "deny", "check"], timeout=900, cwd=ROOT)
    text = (out + err)
    return {"rc": rc, "advisories_ok": "advisories ok" in text, "output_tail": text[-1500:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flips", type=int, default=1000)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lucene-jar", default=str(Path.home() / ".cache/warm-bench/jars/anserini-2.3.0-fatjar.jar"))
    ap.add_argument("--lucene-core", default=str(Path.home() / ".cache/warm-bench/jars/lucene-core-9.12.3.jar"))
    ap.add_argument("--merge", help="JSON written by this script on another host; its memscan and ct cells are merged")
    ap.add_argument("--update", help="existing output JSON whose cells not run this time are kept")
    ap.add_argument("--cells", default="egress,overhead,tamper,audit,memscan,ct,deny")
    a = ap.parse_args()
    cells = set(a.cells.split(","))
    rng = random.Random(3)
    probe = str(Path(a.probe).resolve())
    result = {"schema_version": 1, "host": os.uname().sysname + " " + os.uname().machine, "probe": probe, "passphrase_source": "stdin",
              "exclusions": [], "cells": {}}
    if a.update and Path(a.update).exists():
        prior = json.loads(Path(a.update).read_text())
        result["cells"] = {k: v for k, v in prior.get("cells", {}).items()}
        result["exclusions"] = list(prior.get("exclusions", []))
    def exclude(line):
        if line not in result["exclusions"]:
            result["exclusions"].append(line)
    if "egress" in cells:
        result["cells"]["egress"] = {"warm": egress(probe, a.docs, a.queries)}
        exclude("managed Mem0 and Zep egress: no mem0ai or zep SDK and no API key is installed in this environment; WARM measured alone")
    if "overhead" in cells:
        result["cells"]["encryption_overhead"] = {"warm": overhead(probe, a.docs, a.queries, a.repeats)}
        exclude("SQLite with SQLCipher: sqlcipher is not installed on this host")
        exclude("Lucene on an encrypted filesystem: no query driver with the pinned jars; encryption overhead measured for WARM only")
    if "tamper" in cells:
        with tempfile.TemporaryDirectory(prefix="warm-seed-") as tmp:
            seeds = {}
            for encrypted in (False, True):
                store = Path(tmp) / ("enc" if encrypted else "plain")
                cmd = [probe, "seed", "--store", str(store), "--docs", str(min(a.docs, 400))]
                if encrypted:
                    cmd.append("--encrypted")
                rc, out, err, _ = run(cmd, stdin=PASSPHRASE + "\n" if encrypted else None, timeout=600)
                info = json.loads(out.strip().splitlines()[-1])
                expected = Path(tmp) / f"expected-{int(encrypted)}.json"
                expected.write_text(json.dumps(info["expected"]))
                seeds[encrypted] = (store, expected, info["files"], info["store_bytes"])
            tamper = {}
            for encrypted in (False, True):
                store, expected, files, size = seeds[encrypted]
                tamper["warm_encrypted" if encrypted else "warm_plain"] = {"files": files, "store_bytes": size, **tamper_warm(probe, store, expected, a.flips, encrypted, rng)}
            db = Path(tmp) / "control.db"
            build_sqlite(db, min(a.docs, 400))
            tamper["sqlite"] = tamper_sqlite(db, a.flips, rng)
            if lucene_available(a.lucene_jar) and Path(a.lucene_core).exists():
                idx = Path(tmp) / "lucene"
                rc, secs, tail = build_lucene(a.lucene_jar, idx, min(a.docs, 400))
                if rc == 0:
                    tamper["lucene"] = tamper_lucene(a.lucene_core, idx, a.flips, rng)
                else:
                    exclude(f"Lucene tamper cell: anserini IndexCollection failed ({tail[-120:]})")
            else:
                exclude("Lucene tamper cell: anserini or lucene-core jar or java missing")
            result["cells"]["tamper"] = tamper
    if "audit" in cells:
        result["cells"]["audit"] = {"warm": audit_cell(probe, 200, rng)}
    if "memscan" in cells:
        if sys.platform.startswith("linux"):
            with tempfile.TemporaryDirectory(prefix="warm-memscan-") as tmp:
                store = Path(tmp) / "store"
                run([probe, "seed", "--store", str(store), "--docs", "200", "--encrypted"], stdin=PASSPHRASE + "\n", timeout=600)
                rc, out, err, _ = run([probe, "memscan", "--store", str(store)], stdin=PASSPHRASE + "\n", timeout=600)
                result["cells"]["memscan"] = {"warm": json.loads(out.strip().splitlines()[-1]) if out.strip() else {"error": err[-300:]}}
        else:
            exclude("key zeroization scan: /proc/self/mem is Linux only; run on a Linux host and merged")
    if "ct" in cells:
        rc, out, err, _ = run([probe, "ct", "--iters", "300000"], timeout=600)
        result["cells"]["constant_time"] = {"warm_server_ct_eq": json.loads(out.strip().splitlines()[-1]) if out.strip() else {"error": err[-300:]}}
    if "deny" in cells:
        result["cells"]["cargo_deny"] = cargo_deny()
    if a.merge:
        other = json.loads(Path(a.merge).read_text())
        for key in ("memscan", "constant_time"):
            if key in other.get("cells", {}):
                result["cells"].setdefault(key, {})[f"{key}_from_{other.get('host', 'other host')}"] = other["cells"][key]
        result["exclusions"] = [e for e in result["exclusions"] if "merged" not in e] + [f"merged {k} from {other.get('host')}" for k in ("memscan", "constant_time") if k in other.get("cells", {})]
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (v if k != "cells" else list(v)) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
