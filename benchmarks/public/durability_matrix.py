# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durability and operations matrix: HMS against SQLite (WAL) and Lucene on one host.

Cells per system (each a pass/fail with counts and times):
  kill_writes        SIGKILL at N random points during committed batch writes; a committed batch
                     that is missing or served with wrong text after recovery fails the cell.
  kill_compaction    the same with the kill placed inside a compaction (HMS compact, SQLite
                     wal_checkpoint + VACUUM, Lucene forceMerge).
  power_loss         SIGKILL under a userspace write barrier (LD_PRELOAD shim, Linux): every
                     write after a file's last fsync/fdatasync/msync is discarded before
                     recovery, which models a power cut that loses the OS cache.
  torn_write         one 512-byte sector near a store file's end overwritten with random bytes;
                     classified as detected (open or integrity check reports it), benign (every
                     committed document still served correctly) or silent (served wrong data).
  bit_flip           one random bit flipped in a random store file; classified the same way.
  readers            readers verify documents while a writer rewrites them and a compaction
                     runs; any torn read fails.
  snapshot_restore   a snapshot of a quiescent store opened from another directory serves
                     every document.
  old_binary         an older release opening a store whose format declares a newer version
                     must refuse it.
Recovery time is the median open-plus-recovery time of the kill cells; bytes lost is the
difference between the store's size at the kill and after recovery (negative when recovery
discards an unfinished tail) together with the count of committed documents lost.

Usage: durability_matrix.py --driver <durability-driver> --work <dir> --out <json>
       [--systems hms,sqlite,lucene] [--kill-points 200] [--flips 50] [--torn 50] [--power 50]
       [--docs 4000] [--batch 20] [--java-home DIR] [--lucene-jar JAR] [--old-lucene-jar JAR]
       [--old-hms-admin BIN] [--shim-cc gcc]
"""
import argparse
import json
import os
import platform
import random
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SQLITE_SCRIPT = r'''
import json, os, sqlite3, sys, time
mode, store, docs, batch, compact_every, revision = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
def payload(n, rev):
    h = ((n * 0x9E3779B97F4A7C15) ^ (rev << 48)) & 0xFFFFFFFFFFFFFFFF
    return f"document {n} revision {rev} marker {h:016x} text body filler words"
path = os.path.join(store, "docs.sqlite")
if mode == "write":
    os.makedirs(store, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=FULL")
    conn.execute("CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, rev INTEGER, text TEXT)")
    b = 0
    for first in range(0, docs, batch):
        conn.execute("BEGIN")
        conn.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?)", [(n, revision, payload(n, revision)) for n in range(first, min(first + batch, docs))])
        conn.execute("COMMIT")
        print(f"committed {b}", flush=True)
        if compact_every and (b + 1) % compact_every == 0:
            print(f"compacting {b}", flush=True)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"); conn.execute("VACUUM")
            print(f"compacted {b}", flush=True)
        b += 1
    conn.close(); print("done", flush=True)
elif mode == "verify":
    committed = int(sys.argv[7])
    t = time.monotonic()
    try:
        conn = sqlite3.connect(path)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        rows = dict((r[0], (r[1], r[2])) for r in conn.execute("SELECT id, rev, text FROM docs"))
    except Exception as e:
        print(json.dumps({"open_ok": False, "error": str(e), "recovery_ms": (time.monotonic() - t) * 1e3})); sys.exit(0)
    recovery_ms = (time.monotonic() - t) * 1e3
    limit = min(committed * batch, docs)
    present = sum(1 for n in range(limit) if rows.get(n) == (revision, payload(n, revision)))
    missing = sum(1 for n in range(limit) if n not in rows or (revision > 0 and rows.get(n) == (revision - 1, payload(n, revision - 1))))
    mismatched = limit - present - missing
    extra = sum(1 for n in range(limit, docs) if rows.get(n) == (revision, payload(n, revision)))
    size = sum(os.path.getsize(os.path.join(store, f)) for f in os.listdir(store))
    print(json.dumps({"open_ok": True, "integrity": integrity, "detected": integrity != "ok", "recovery_ms": recovery_ms, "checked": limit, "present": present, "missing": missing, "mismatched": mismatched, "uncommitted_present": extra, "store_bytes": size}))
elif mode == "readers":
    import threading
    readers = int(sys.argv[7]); stop = False; checks = [0]; bad = [0]
    def reader(r):
        c = sqlite3.connect(path); n = r
        while not stop:
            row = c.execute("SELECT rev, text FROM docs WHERE id=?", (n,)).fetchone()
            checks[0] += 1
            if row is None or row != (row[0], payload(n, row[0])) or row[0] not in (0, 1):
                bad[0] += 1
            n = (n + readers) % docs
    ts = [threading.Thread(target=reader, args=(r,)) for r in range(readers)]
    [t.start() for t in ts]
    w = sqlite3.connect(path, isolation_level=None)
    t0 = time.monotonic(); w.execute("PRAGMA wal_checkpoint(TRUNCATE)"); w.execute("VACUUM"); cm = (time.monotonic() - t0) * 1e3
    for first in range(0, docs, batch):
        w.execute("BEGIN"); w.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?)", [(n, 1, payload(n, 1)) for n in range(first, min(first + batch, docs))]); w.execute("COMMIT")
    w.execute("PRAGMA wal_checkpoint(TRUNCATE)"); w.execute("VACUUM")
    stop = True; [t.join() for t in ts]
    print(json.dumps({"reads": checks[0], "torn_reads": bad[0], "writer": {"batches": -(-docs // batch), "compaction_ms": cm}}))
elif mode == "snapshot":
    dst = sys.argv[7]; os.makedirs(dst, exist_ok=True)
    src = sqlite3.connect(path); out = sqlite3.connect(os.path.join(dst, "docs.sqlite")); src.backup(out); out.close(); print("ok")
'''

JAVA_SOURCE = r'''
import java.io.*; import java.nio.file.*; import java.util.*;
import org.apache.lucene.document.*; import org.apache.lucene.index.*; import org.apache.lucene.search.*; import org.apache.lucene.store.*;
public final class DurabilityLucene {
  static String payload(long n, long rev) { long h = (n * 0x9E3779B97F4A7C15L) ^ (rev << 48); return "document " + n + " revision " + rev + " marker " + String.format("%016x", h) + " text body filler words"; }
  static Document doc(long n, long rev) { Document d = new Document(); d.add(new StringField("id", "d" + n, Field.Store.YES)); d.add(new StoredField("rev", rev)); d.add(new StoredField("text", payload(n, rev))); return d; }
  static String stored(IndexSearcher s, long n) throws IOException { TopDocs t = s.search(new TermQuery(new Term("id", "d" + n)), 1); if (t.totalHits.value == 0) return null; Document d = s.storedFields().document(t.scoreDocs[0].doc); return d.get("rev") + "|" + d.get("text"); }
  public static void main(String[] a) throws Exception {
    String mode = a[0]; Path store = Paths.get(a[1]); int docs = Integer.parseInt(a[2]), batch = Integer.parseInt(a[3]), compactEvery = Integer.parseInt(a[4]); long rev = Long.parseLong(a[5]);
    if (mode.equals("write")) {
      Files.createDirectories(store); IndexWriterConfig c = new IndexWriterConfig(); c.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
      try (FSDirectory dir = FSDirectory.open(store); IndexWriter w = new IndexWriter(dir, c)) {
        int b = 0; for (int first = 0; first < docs; first += batch) { for (int n = first; n < Math.min(first + batch, docs); n++) w.updateDocument(new Term("id", "d" + n), doc(n, rev)); w.commit(); System.out.println("committed " + b); System.out.flush();
          if (compactEvery > 0 && (b + 1) % compactEvery == 0) { System.out.println("compacting " + b); System.out.flush(); w.forceMerge(1); w.commit(); System.out.println("compacted " + b); System.out.flush(); } b++; }
      } System.out.println("done");
    } else if (mode.equals("verify")) {
      int committed = Integer.parseInt(a[6]); long t = System.nanoTime(); boolean detected = false; String err = null; Map<String, Object> out = new LinkedHashMap<>();
      try (FSDirectory dir = FSDirectory.open(store)) {
        CheckIndex ci = new CheckIndex(dir); ci.setChecksumsOnly(true); ci.setInfoStream(new PrintStream(OutputStream.nullOutputStream())); CheckIndex.Status st = ci.checkIndex(); detected = !st.clean;
        try (DirectoryReader r = DirectoryReader.open(dir)) {
          double recovery = (System.nanoTime() - t) / 1e6; IndexSearcher s = new IndexSearcher(r); int limit = Math.min(committed * batch, docs); int present = 0, missing = 0, mismatched = 0, extra = 0;
          for (long n = 0; n < limit; n++) { String v = stored(s, n); if (v == null || (rev > 0 && v.equals((rev - 1) + "|" + payload(n, rev - 1)))) missing++; else if (v.equals(rev + "|" + payload(n, rev))) present++; else mismatched++; }
          for (long n = limit; n < docs; n++) { String v = stored(s, n); if (v != null && v.equals(rev + "|" + payload(n, rev))) extra++; }
          long size = 0; try (DirectoryStream<Path> ds = Files.newDirectoryStream(store)) { for (Path p : ds) size += Files.size(p); }
          System.out.println("{\"open_ok\": true, \"detected\": " + detected + ", \"recovery_ms\": " + recovery + ", \"checked\": " + limit + ", \"present\": " + present + ", \"missing\": " + missing + ", \"mismatched\": " + mismatched + ", \"uncommitted_present\": " + extra + ", \"store_bytes\": " + size + "}"); return;
        }
      } catch (Exception e) { err = e.toString().replace("\"", "'").replace("\n", " "); }
      System.out.println("{\"open_ok\": false, \"detected\": true, \"error\": \"" + err + "\", \"recovery_ms\": " + (System.nanoTime() - t) / 1e6 + "}");
    } else if (mode.equals("readers")) {
      int readers = Integer.parseInt(a[6]); final boolean[] stop = {false}; final long[] checks = {0}, bad = {0};
      try (FSDirectory dir = FSDirectory.open(store); IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
        List<Thread> ts = new ArrayList<>();
        for (int r = 0; r < readers; r++) { final int r0 = r; Thread th = new Thread(() -> { try { DirectoryReader rd = DirectoryReader.open(dir); long n = r0; while (!stop[0]) { DirectoryReader nr = DirectoryReader.openIfChanged(rd); if (nr != null) { rd.close(); rd = nr; } String v = stored(new IndexSearcher(rd), n); synchronized (checks) { checks[0]++; if (v == null || !(v.equals("0|" + payload(n, 0)) || v.equals("1|" + payload(n, 1)))) bad[0]++; } n = (n + readers) % docs; } rd.close(); } catch (Exception e) { synchronized (checks) { bad[0] += 1000000; } } }); th.start(); ts.add(th); }
        long t0 = System.nanoTime(); w.forceMerge(1); w.commit(); double cm = (System.nanoTime() - t0) / 1e6;
        for (int first = 0; first < docs; first += batch) { for (int n = first; n < Math.min(first + batch, docs); n++) w.updateDocument(new Term("id", "d" + n), doc(n, 1)); w.commit(); }
        w.forceMerge(1); w.commit(); stop[0] = true; for (Thread th : ts) th.join();
        System.out.println("{\"reads\": " + checks[0] + ", \"torn_reads\": " + bad[0] + ", \"writer\": {\"batches\": " + ((docs + batch - 1) / batch) + ", \"compaction_ms\": " + cm + "}}");
      }
    } else if (mode.equals("open_old")) {
      try (FSDirectory dir = FSDirectory.open(store); DirectoryReader r = DirectoryReader.open(dir)) { System.out.println("opened " + r.numDocs()); } catch (Exception e) { System.out.println("rejected " + e.getClass().getSimpleName()); }
    }
  }
}
'''

SHIM_SOURCE = r'''
// Userspace write barrier: after a file's fsync/fdatasync/msync, its content is copied under
// SHADOW_ROOT keyed by device and inode (renames keep the inode, so no rename interception
// is needed); the harness restores a store from those copies after SIGKILL, so every write
// after the last sync is lost, as after a power cut. Files under STORE_ROOT only.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static const char *root(void) { return getenv("DURABILITY_STORE_ROOT"); }
static const char *shadow(void) { return getenv("DURABILITY_SHADOW_ROOT"); }
static void dbg(const char *what, const char *a) { if (getenv("DURABILITY_SHIM_DEBUG")) fprintf(stderr, "shim %s %s\n", what, a); }
static void copy_file(const char *src) {
  const char *r = root(), *s = shadow(); if (!r || !s || strncmp(src, r, strlen(r)) != 0) return;
  struct stat st; if (stat(src, &st) != 0 || !S_ISREG(st.st_mode)) return;
  // The copy is built under a temporary name and renamed over the key, so a kill that lands
  // mid-copy leaves the previous complete copy in place.
  char dst[PATH_MAX], tmp[PATH_MAX]; snprintf(dst, sizeof dst, "%s/%llu-%llu", s, (unsigned long long)st.st_dev, (unsigned long long)st.st_ino); snprintf(tmp, sizeof tmp, "%s.part", dst);
  int in = open(src, O_RDONLY); if (in < 0) return;
  int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (out < 0) { close(in); return; }
  if (ftruncate(out, st.st_size) != 0) { close(in); close(out); return; }
  // Copy only data extents, so a sparse arena stays sparse and the copy stays cheap.
  off_t off = 0; char buf[1 << 16];
  while (off < st.st_size) {
    off_t data = lseek(in, off, SEEK_DATA); if (data < 0) break;
    off_t hole = lseek(in, data, SEEK_HOLE); if (hole < 0) hole = st.st_size;
    off_t pos = data;
    while (pos < hole) { size_t want = (size_t)(hole - pos) < sizeof buf ? (size_t)(hole - pos) : sizeof buf; ssize_t n = pread(in, buf, want, pos); if (n <= 0) break; ssize_t w = 0; while (w < n) { ssize_t k = pwrite(out, buf + w, n - w, pos + w); if (k <= 0) break; w += k; } pos += n; }
    off = hole;
  }
  close(in); close(out); if (rename(tmp, dst) != 0) unlink(tmp); dbg("synced", src);
}
static void fd_path(int fd, char *out) { char link[64]; snprintf(link, sizeof link, "/proc/self/fd/%d", fd); ssize_t n = readlink(link, out, PATH_MAX - 1); out[n > 0 ? n : 0] = 0; }
static void sync_fd(int fd) { char path[PATH_MAX]; fd_path(fd, path); if (path[0] == '/') copy_file(path); }
int fsync(int fd) { static int (*real)(int); if (!real) real = dlsym(RTLD_NEXT, "fsync"); int rc = real(fd); sync_fd(fd); return rc; }
int fdatasync(int fd) { static int (*real)(int); if (!real) real = dlsym(RTLD_NEXT, "fdatasync"); int rc = real(fd); sync_fd(fd); return rc; }
int msync(void *addr, size_t len, int flags) {
  static int (*real)(void *, size_t, int); if (!real) real = dlsym(RTLD_NEXT, "msync"); int rc = real(addr, len, flags);
  FILE *maps = fopen("/proc/self/maps", "r"); if (!maps) return rc; char line[PATH_MAX + 128]; unsigned long a = (unsigned long)addr;
  while (fgets(line, sizeof line, maps)) { unsigned long lo, hi; char path[PATH_MAX]; path[0] = 0; if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %s", &lo, &hi, path) >= 2 && a >= lo && a < hi && path[0] == '/') { copy_file(path); break; } }
  fclose(maps); return rc;
}
'''


def now_ms():
    return time.monotonic() * 1e3


def dir_bytes(path):
    """Allocated bytes (sparse arenas count what they use, not their reserved size); a file
    that vanishes while a live writer merges or compacts is skipped."""
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_blocks * 512
        except FileNotFoundError:
            continue
    return total


def store_files(path):
    return sorted(p for p in Path(path).rglob("*") if p.is_file() and p.stat().st_blocks > 0)


def written_extent(path):
    """Bytes of a file that carry data: the allocated extent, capped at its size."""
    st = path.stat()
    return max(1, min(st.st_size, st.st_blocks * 512 + 4096))


def copy_store(src, dst):
    if platform.system() == "Linux":
        subprocess.run(["cp", "-a", "--sparse=always", src, dst], check=True)
    elif platform.system() == "Darwin":
        subprocess.run(["cp", "-ac", src, dst], check=True)
    else:
        shutil.copytree(src, dst)


class System:
    name = ""

    def __init__(self, a, work):
        self.a, self.work = a, work

    def env(self):
        return dict(os.environ)

    def writer(self, store, compact_every=0, revision=0):
        raise NotImplementedError

    def verify(self, store, committed, revision=0):
        out = self.run(self.verify_cmd(store, committed, revision))
        line = [l for l in out.splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"open_ok": False, "error": out[-400:], "recovery_ms": None}

    def run(self, cmd, timeout=600):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=self.env())
        return p.stdout + ("" if p.returncode == 0 else f"\n[exit {p.returncode}] " + p.stderr[-800:])


class Hms(System):
    name = "hms"

    def writer(self, store, compact_every=0, revision=0):
        return [self.a.driver, "write", "--store", store, "--docs", str(self.a.docs), "--batch", str(self.a.batch),
                "--compact-every", str(compact_every), "--revision", str(revision)]

    def verify_cmd(self, store, committed, revision=0):
        return [self.a.driver, "verify", "--store", store, "--docs", str(self.a.docs), "--batch", str(self.a.batch),
                "--committed", str(committed), "--revision", str(revision)]

    def readers_cmd(self, store):
        return [self.a.driver, "readers", "--store", store, "--docs", str(self.a.docs), "--batch", str(self.a.batch)]

    def verify(self, store, committed, revision=0):
        v = super().verify(store, committed, revision)
        if self.a.hms_admin and v.get("open_ok"):
            # The engine's integrity scan (every arena frame's CRC32) is the detection report.
            p = subprocess.run([self.a.hms_admin, "verify", store], capture_output=True, text=True, timeout=600)
            v["detected"] = p.returncode != 0
            if p.returncode != 0:
                v["integrity"] = (p.stderr or p.stdout)[-200:]
        return v

    def snapshot(self, store, dst):
        copy_store(store, dst)

    def old_binary(self, store):
        if not self.a.old_hms_admin:
            return {"excluded": "no old hms-admin binary given (--old-hms-admin)"}
        bumped = store + "-bumped"
        copy_store(store, bumped)
        manifests = sorted(Path(bumped).rglob("format.json")) or [Path(bumped) / "store.json"]
        manifest = manifests[0]
        if not manifest.exists():
            return {"excluded": "store has no format manifest"}
        m = json.loads(manifest.read_text())
        key = next((k for k in m if "version" in k), None)
        if key is None:
            return {"excluded": f"format.json has no version field: {sorted(m)}"}
        m[key] = (m[key] + 1) if isinstance(m[key], int) else str(m[key]) + ".next"
        manifest.write_text(json.dumps(m))
        p = subprocess.run([self.a.old_hms_admin, *self.a.old_hms_admin_args.split(), bumped], capture_output=True, text=True, timeout=120)
        rejected = p.returncode != 0
        return {"pass": rejected, "exit": p.returncode, "stderr": p.stderr[-300:], "field": key, "method": "format manifest version bumped by one; the older release must refuse to open it"}


class Sqlite(System):
    name = "sqlite"

    def __init__(self, a, work):
        super().__init__(a, work)
        self.script = work / "durability_sqlite.py"
        self.script.write_text(SQLITE_SCRIPT)

    def base(self, mode, store, compact_every, revision):
        return [sys.executable, "-I", str(self.script), mode, store, str(self.a.docs), str(self.a.batch), str(compact_every), str(revision)]

    def writer(self, store, compact_every=0, revision=0):
        return self.base("write", store, compact_every, revision)

    def verify_cmd(self, store, committed, revision=0):
        return self.base("verify", store, 0, revision) + [str(committed)]

    def readers_cmd(self, store):
        return self.base("readers", store, 0, 0) + ["3"]

    def snapshot(self, store, dst):
        self.run(self.base("snapshot", store, 0, 0) + [dst])

    def old_binary(self, store):
        bumped = store + "-bumped"
        copy_store(store, bumped)
        db = Path(bumped) / "docs.sqlite"
        data = bytearray(db.read_bytes())
        data[18] = 3  # file format write version: unknown to every released SQLite
        data[19] = 3
        db.write_bytes(data)
        try:
            conn = sqlite3.connect(str(db))
            conn.execute("SELECT count(*) FROM docs").fetchone()
            return {"pass": False, "method": "header write/read version bytes set to 3", "library": sqlite3.sqlite_version}
        except sqlite3.DatabaseError as e:
            return {"pass": True, "error": str(e), "method": "header write/read version bytes set to 3; the library refuses an unknown newer format", "library": sqlite3.sqlite_version}


class Lucene(System):
    name = "lucene"

    def __init__(self, a, work):
        super().__init__(a, work)
        self.java = str(Path(a.java_home) / "bin" / "java") if a.java_home else "java"
        javac = str(Path(a.java_home) / "bin" / "javac") if a.java_home else "javac"
        src = work / "DurabilityLucene.java"
        src.write_text(JAVA_SOURCE)
        self.classes = work / "lucene-classes"
        self.classes.mkdir(exist_ok=True)
        subprocess.run([javac, "-cp", a.lucene_jar, "-d", str(self.classes), str(src)], check=True, capture_output=True, text=True)
        self.cp = f"{self.classes}{os.pathsep}{a.lucene_jar}"

    def base(self, mode, store, compact_every, revision):
        return [self.java, "-Xmx512m", "-cp", self.cp, "DurabilityLucene", mode, store, str(self.a.docs), str(self.a.batch), str(compact_every), str(revision)]

    def writer(self, store, compact_every=0, revision=0):
        return self.base("write", store, compact_every, revision)

    def verify_cmd(self, store, committed, revision=0):
        return self.base("verify", store, 0, revision) + [str(committed)]

    def readers_cmd(self, store):
        return self.base("readers", store, 0, 0) + ["3"]

    def snapshot(self, store, dst):
        copy_store(store, dst)

    def old_binary(self, store):
        if not self.a.old_lucene_jar:
            return {"excluded": "no older lucene-core jar given (--old-lucene-jar)"}
        p = subprocess.run([self.java, "-cp", f"{self.classes}{os.pathsep}{self.a.old_lucene_jar}", "DurabilityLucene", "open_old", store, "1", "1", "0", "0"], capture_output=True, text=True, timeout=120)
        out = (p.stdout + p.stderr).strip().splitlines()
        rejected = p.returncode != 0 or any(l.startswith("rejected") for l in out)
        return {"pass": rejected, "output": (out[-1] if out else "")[:200], "method": "index written by the pinned release opened with the older lucene-core jar"}


class Writer:
    """A writer subprocess whose committed-batch announcements are captured as they appear."""

    def __init__(self, system, store, compact_every=0, revision=0, extra_env=None):
        env = system.env()
        env.update(extra_env or {})
        self.p = subprocess.Popen(system.writer(store, compact_every, revision), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        self.lines = []
        self.lock = threading.Lock()
        self.events = {"compacting": threading.Event(), "done": threading.Event()}
        self.t = threading.Thread(target=self.pump, daemon=True)
        self.t.start()

    def pump(self):
        for line in self.p.stdout:
            with self.lock:
                self.lines.append(line.strip())
            for k, e in self.events.items():
                if line.startswith(k):
                    e.set()

    def committed(self):
        with self.lock:
            return sum(1 for l in self.lines if l.startswith("committed"))

    def kill(self):
        try:
            self.p.send_signal(signal.SIGKILL)
        except ProcessLookupError:
            pass  # the process already exited; wait() below reaps it
        self.p.wait()
        self.t.join(timeout=5)


def kill_cell(system, a, store_root, points, rng, compaction=False, shim=None):
    rows = []
    for i in range(points):
        store = str(store_root / f"{system.name}-{'c' if compaction else 'w'}-{i}")
        shadow = None
        extra = None
        if shim:
            shadow = store + "-shadow"
            os.makedirs(shadow, exist_ok=True)
            extra = {"LD_PRELOAD": shim, "DURABILITY_STORE_ROOT": store, "DURABILITY_SHADOW_ROOT": shadow}
        w = Writer(system, store, compact_every=(5 if compaction else 0), extra_env=extra)
        if compaction:
            w.events["compacting"].wait(timeout=120)
            time.sleep(rng.uniform(0.0, 0.3))
        else:
            time.sleep(rng.uniform(0.02, a.write_window))
        before = dir_bytes(store) if os.path.isdir(store) else 0
        w.kill()
        committed = w.committed()
        if shim:
            restore_from_shadow(store, shadow)
        v = system.verify(store, committed)
        if committed == 0 and not v.get("open_ok"):
            # Killed before the first commit: an engine that has no index yet is not a loss.
            v = {"open_ok": True, "no_index_yet": True, "recovery_ms": v.get("recovery_ms"), "present": 0, "missing": 0, "mismatched": 0, "uncommitted_present": 0}
        after = dir_bytes(store) if os.path.isdir(store) else 0
        rows.append({"point": i, "committed": committed, "bytes_at_kill": before, "bytes_after": after, **v})
        shutil.rmtree(store, ignore_errors=True)
        if shadow:
            shutil.rmtree(shadow, ignore_errors=True)
    return summarize_kill(rows)


def restore_from_shadow(store, shadow):
    """Every file keeps only its content at its last sync (looked up by device and inode,
    which a rename keeps); never-synced files vanish."""
    for p in list(Path(store).rglob("*")):
        if not p.is_file():
            continue
        st = p.stat()
        s = Path(shadow) / f"{st.st_dev}-{st.st_ino}"
        if s.exists() and s.stat().st_size >= 0 and not s.name.endswith(".part"):
            if platform.system() == "Linux":
                subprocess.run(["cp", "--sparse=always", str(s), str(p)], check=True)
            else:
                shutil.copyfile(s, p)
        else:
            p.unlink()


def summarize_kill(rows):
    ok = [r for r in rows if r.get("open_ok")]
    lost = sum(r.get("missing", 0) + r.get("mismatched", 0) for r in ok)
    opens_failed = len(rows) - len(ok)
    rec = [r["recovery_ms"] for r in ok if r.get("recovery_ms") is not None]
    return {"points": len(rows), "pass": opens_failed == 0 and lost == 0, "open_failures": opens_failed,
            "committed_docs_lost": lost, "points_with_loss": sum(1 for r in ok if r.get("missing", 0) + r.get("mismatched", 0) > 0),
            "uncommitted_served": sum(r.get("uncommitted_present", 0) for r in ok),
            "recovery_ms_median": statistics.median(rec) if rec else None, "recovery_ms_max": max(rec) if rec else None,
            "bytes_lost_median": statistics.median(r["bytes_at_kill"] - r["bytes_after"] for r in ok) if ok else None,
            "mean_committed_batches": statistics.mean(r["committed"] for r in rows) if rows else 0,
            "failures": [r for r in rows if not r.get("open_ok") or r.get("missing", 0) + r.get("mismatched", 0) > 0][:5]}


def corruption_cell(system, a, store_root, trials, rng, torn):
    base = str(store_root / f"{system.name}-base")
    w = Writer(system, base)
    w.p.wait()
    w.t.join()
    committed = w.committed()
    counts = {"detected": 0, "benign": 0, "silent": 0, "trials": trials}
    for i in range(trials):
        store = str(store_root / f"{system.name}-{'torn' if torn else 'flip'}-{i}")
        copy_store(base, store)
        files = store_files(store)
        target = rng.choice(files)
        data = bytearray(target.read_bytes())
        extent = written_extent(target)
        if torn:
            end = extent
            start = max(0, end - 65536)
            sector = rng.randrange(start // 512, max(start // 512 + 1, end // 512)) * 512
            chunk = bytes(rng.getrandbits(8) for _ in range(max(1, min(512, end - sector))))
            data[sector:sector + len(chunk)] = chunk
        else:
            bit = rng.randrange(extent * 8)
            data[bit // 8] ^= 1 << (bit % 8)
        with open(target, "r+b") as f:
            f.write(data)
        v = system.verify(store, committed)
        if not v.get("open_ok") or v.get("detected"):
            counts["detected"] += 1
        elif v.get("missing", 0) + v.get("mismatched", 0) + v.get("read_errors", 0) > 0:
            counts["silent"] += 1
        else:
            counts["benign"] += 1
        shutil.rmtree(store, ignore_errors=True)
    shutil.rmtree(base, ignore_errors=True)
    counts["pass"] = counts["silent"] == 0
    return counts


def readers_cell(system, a, store_root):
    store = str(store_root / f"{system.name}-readers")
    w = Writer(system, store)
    w.p.wait()
    out = system.run(system.readers_cmd(store), timeout=900)
    line = [l for l in out.splitlines() if l.startswith("{")]
    r = json.loads(line[-1]) if line else {"error": out[-400:], "torn_reads": None}
    r["pass"] = r.get("torn_reads") == 0
    shutil.rmtree(store, ignore_errors=True)
    return r


def snapshot_cell(system, a, store_root):
    store = str(store_root / f"{system.name}-snap")
    w = Writer(system, store)
    w.p.wait()
    committed = w.committed()
    dst = store + "-restored"
    t = now_ms()
    system.snapshot(store, dst)
    snap_ms = now_ms() - t
    v = system.verify(dst, committed)
    v["pass"] = bool(v.get("open_ok")) and v.get("missing", 1) == 0 and v.get("mismatched", 1) == 0
    v["snapshot_ms"] = snap_ms
    v["snapshot_bytes"] = dir_bytes(dst)
    shutil.rmtree(store, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    return v


def old_binary_cell(system, a, store_root):
    store = str(store_root / f"{system.name}-old")
    w = Writer(system, store)
    w.p.wait()
    r = system.old_binary(store)
    shutil.rmtree(store, ignore_errors=True)
    shutil.rmtree(store + "-bumped", ignore_errors=True)
    return r


def build_shim(a, work):
    if platform.system() != "Linux":
        return None, "power-loss shim needs LD_PRELOAD (Linux); excluded on " + platform.system()
    src = work / "durability_shim.c"
    src.write_text(SHIM_SOURCE)
    so = work / "durability_shim.so"
    p = subprocess.run([a.shim_cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(src), "-ldl"], capture_output=True, text=True)
    if p.returncode != 0:
        return None, "shim failed to compile: " + p.stderr[-300:]
    return str(so), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--driver", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--systems", default="hms,sqlite,lucene")
    p.add_argument("--kill-points", type=int, default=200)
    p.add_argument("--flips", type=int, default=50)
    p.add_argument("--torn", type=int, default=50)
    p.add_argument("--power", type=int, default=50)
    p.add_argument("--docs", type=int, default=4000)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--write-window", type=float, default=3.0, help="latest kill time (s) during writes")
    p.add_argument("--java-home", default=os.environ.get("JAVA_HOME", ""))
    p.add_argument("--lucene-jar", default=str(Path.home() / ".cache/hms-bench/jars/lucene-core-9.12.3.jar"))
    p.add_argument("--old-lucene-jar", default="")
    p.add_argument("--old-hms-admin", default="")
    p.add_argument("--hms-admin", default="", help="current hms-admin for the integrity scan after corruption")
    p.add_argument("--old-hms-admin-args", default="stats")
    p.add_argument("--shim-cc", default="gcc")
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cells", default="kill_writes,kill_compaction,power_loss,torn_write,bit_flip,readers,snapshot_restore,old_binary")
    a = p.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    stores = work / "stores"
    stores.mkdir(exist_ok=True)
    shim, shim_note = build_shim(a, work)
    systems = {"hms": Hms, "sqlite": Sqlite, "lucene": Lucene}
    cells = a.cells.split(",")
    result = {"schema_version": 1, "host": platform.node(), "platform": platform.platform(), "docs": a.docs, "batch": a.batch,
              "versions": {"sqlite_library": sqlite3.sqlite_version, "lucene_jar": Path(a.lucene_jar).name, "java_home": a.java_home or "PATH"},
              "power_loss_method": shim_note or "LD_PRELOAD shim: fsync/fdatasync/msync copy the synced file under a shadow root keyed by device and inode (a rename keeps the inode, so atomic temp-and-rename publication is honoured without intercepting rename); after SIGKILL every store file is reset to its last synced copy and never-synced files are removed, i.e. every write after the last sync is lost; directory-entry durability (a rename not followed by a directory fsync) is taken as durable",
              "cells": {}, "started_unix": time.time()}
    rng = random.Random(a.seed)
    for name in a.systems.split(","):
        sysm = systems[name](a, work)
        r = {}
        if "kill_writes" in cells:
            r["kill_writes"] = kill_cell(sysm, a, stores, a.kill_points, rng)
        if "kill_compaction" in cells:
            r["kill_compaction"] = kill_cell(sysm, a, stores, a.kill_points, rng, compaction=True)
        if "power_loss" in cells:
            r["power_loss"] = kill_cell(sysm, a, stores, a.power, rng, shim=shim) if shim else {"excluded": shim_note}
        if "torn_write" in cells:
            r["torn_write"] = corruption_cell(sysm, a, stores, a.torn, rng, torn=True)
        if "bit_flip" in cells:
            r["bit_flip"] = corruption_cell(sysm, a, stores, a.flips, rng, torn=False)
        if "readers" in cells:
            r["readers"] = readers_cell(sysm, a, stores)
        if "snapshot_restore" in cells:
            r["snapshot_restore"] = snapshot_cell(sysm, a, stores)
        if "old_binary" in cells:
            r["old_binary"] = old_binary_cell(sysm, a, stores)
        result["cells"][name] = r
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
        print(name, json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    result["finished_unix"] = time.time()
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
