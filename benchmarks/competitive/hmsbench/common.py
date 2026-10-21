# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hardware class, statistics and subprocess measurement shared by the runner, workers and table."""

import contextlib
import fcntl
import json
import math
import os
import platform
import random
import resource
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("HMS_BENCH_RESULTS", ROOT / "benchmarks/results/competitive"))
CACHE = Path(os.environ.get("HMS_BENCH_DATA", Path.home() / ".cache" / "hms-bench"))


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def cpu_model():
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memory_bytes():
    if platform.system() == "Darwin":
        return int(_sysctl("hw.memsize") or 0)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def hardware():
    """Hardware description. `class` is the only key rows may be compared across."""
    cpu, cores, mem = cpu_model(), os.cpu_count() or 0, memory_bytes()
    slug = "".join(c if c.isalnum() else "-" for c in cpu.lower())
    slug = "-".join(part for part in slug.split("-") if part and part not in ("r", "tm", "cpu", "processor", "with", "core"))
    derived = f"{platform.system().lower()}-{platform.machine()}-{slug}-{cores}c-{round(mem / 2**30)}g"
    return {
        "class": os.environ.get("HMS_BENCH_HW_CLASS", derived),
        "cpu": cpu, "logical_cores": cores, "memory_bytes": mem,
        "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
        "hosted_runner": bool(os.environ.get("GITHUB_ACTIONS")),
        "load_1m": os.getloadavg()[0],
    }


def commit():
    if os.environ.get("HMS_BENCH_COMMIT"):  # a shipped source tree has no .git
        return os.environ["HMS_BENCH_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def pass_summary(ns):
    """Compact record of one timed pass over the queries (per-query nanoseconds in): what the
    table's QPS and latency intervals are computed from, with the same order statistics."""
    srt = sorted(ns)
    at = lambda q: srt[int((len(srt) - 1) * q)]  # noqa: E731
    return {"queries": len(srt), "total_ns": sum(srt), "p50_ns": srt[(len(srt) - 1) // 2], "p90_ns": at(0.90), "p99_ns": at(0.99), "max_ns": srt[-1]}


def bootstrap_ci(values, statistic=statistics.median, rounds=2000, seed=24731, level=0.95):
    """Percentile bootstrap interval of `statistic` over repeats; one repeat is degenerate."""
    values = list(values)
    if len(values) < 2:
        return [values[0], values[0]] if values else [math.nan, math.nan]
    rng = random.Random(seed)
    draws = sorted(statistic([rng.choice(values) for _ in values]) for _ in range(rounds))
    tail = (1 - level) / 2
    return [draws[int(tail * rounds)], draws[min(rounds - 1, int((1 - tail) * rounds))]]


def ratio_ci(numerator, denominator, rounds=2000, seed=24731, level=0.95):
    """Point ratio and bootstrap interval for median(numerator) / median(denominator)."""
    rng = random.Random(seed)
    num, den = list(numerator), list(denominator)
    point = statistics.median(num) / statistics.median(den)
    if len(num) < 2 and len(den) < 2:
        return point, [point, point]
    draws = sorted(
        statistics.median([rng.choice(num) for _ in num]) / statistics.median([rng.choice(den) for _ in den])
        for _ in range(rounds)
    )
    tail = (1 - level) / 2
    return point, [draws[int(tail * rounds)], draws[min(rounds - 1, int((1 - tail) * rounds))]]


def ndcg_at(found, relevant, k=10):
    gain = sum(relevant.get(key, 0) / math.log2(rank + 2) for rank, key in enumerate(found[:k]))
    ideal = sum(g / math.log2(rank + 2) for rank, g in enumerate(sorted(relevant.values(), reverse=True)[:k]))
    return gain / ideal if ideal else 0.0


def recall_at(found, relevant, k):
    positive = {key for key, grade in relevant.items() if grade > 0}
    return len(set(found[:k]) & positive) / len(positive) if positive else 0.0


def max_rss_bytes(ru):
    """`ru_maxrss` is bytes on macOS and kilobytes on Linux."""
    return ru.ru_maxrss if platform.system() == "Darwin" else ru.ru_maxrss * 1024


def self_peak_rss_bytes():
    return max_rss_bytes(resource.getrusage(resource.RUSAGE_SELF))


def current_rss_bytes(pid=None):
    pid = pid or os.getpid()
    try:
        if platform.system() == "Linux":
            return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True).stdout
        return int(out.strip()) * 1024
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


class _Spawner:
    """Client of `_spawner.py`: runs measured commands from a small process, so each child's
    `ru_maxrss` is its own rather than at least the orchestrator's RSS (see that module)."""

    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-I", "-S", str(Path(__file__).with_name("_spawner.py"))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.pending = b""

    def _line(self, deadline=None):
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.pending:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not select.select([fd], [], [], wait)[0]:
                return None
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("measurement helper exited")
            self.pending += chunk
        line, _, self.pending = self.pending.partition(b"\n")
        return line.decode()

    def run(self, job, timeout):
        deadline = time.monotonic() + timeout
        self.proc.stdin.write((json.dumps(job) + "\n").encode())
        self.proc.stdin.flush()
        pid = int(self._line().split()[1])
        line = self._line(deadline)
        if line is None:
            os.kill(pid, signal.SIGKILL)
            self._line()
            raise TimeoutError(f"{job['argv'][0]} exceeded {timeout}s")
        _, code, peak = line.split()
        return int(code), int(peak)


_SPAWNER = None


def run_measured(command, env=None, timeout=3600, cwd=None):
    """Run `command`; return (stdout, stderr, wall_s, peak_rss_bytes) of that child alone: its
    own wait4 rusage, from a small helper on Linux (see `_Spawner`) and directly elsewhere.
    Raises on a non-zero exit with the stderr tail."""
    global _SPAWNER
    with timing_lock(), tempfile.TemporaryDirectory() as tmp:
        out_path, err_path = Path(tmp) / "out", Path(tmp) / "err"
        full_env = {**os.environ, **(env or {})}
        start = time.perf_counter()
        if platform.system() == "Linux":
            if _SPAWNER is None or _SPAWNER.proc.poll() is not None:
                _SPAWNER = _Spawner()
            out_path.touch()
            err_path.touch()
            code, peak = _SPAWNER.run({"argv": [str(c) for c in command], "env": full_env, "cwd": str(cwd) if cwd else "",
                                       "stdout": str(out_path), "stderr": str(err_path)}, timeout)
        else:
            with out_path.open("w") as out, err_path.open("w") as err:
                proc = subprocess.Popen(command, stdout=out, stderr=err, env=full_env, cwd=cwd)
                deadline = time.monotonic() + timeout
                while True:
                    pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
                    if pid:
                        break
                    if time.monotonic() > deadline:
                        proc.kill()
                        os.wait4(proc.pid, 0)
                        raise TimeoutError(f"{command[0]} exceeded {timeout}s")
                    time.sleep(0.05)
            code, peak = os.waitstatus_to_exitcode(status), max_rss_bytes(ru)
        wall = time.perf_counter() - start
        stdout, stderr = out_path.read_text(errors="replace"), err_path.read_text(errors="replace")
    if code:
        raise RuntimeError(f"{command[0]} exited {code}: {stderr[-2000:]}")
    return stdout, stderr, wall, peak


@contextlib.contextmanager
def timing_lock():
    """Exclusive timing lock shared with the other harnesses on this host (one phase at a time,
    never held across a suite, so each hold stays far below the 14 minute rule). A no-op on CI
    and wherever the lock directory does not exist."""
    path = Path(os.environ.get("HMS_BENCH_LOCK", "./.hms-target/timing.lock"))
    if os.environ.get("GITHUB_ACTIONS") or not path.parent.exists():
        yield
        return
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


_BLOCK = 4096


def _data_bytes(path):
    """Bytes in 4 KiB blocks that hold a non-zero byte: what a sparse-capable filesystem would allocate."""
    zero = bytes(_BLOCK)
    total = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            total += sum(len(chunk[i:i + _BLOCK]) for i in range(0, len(chunk), _BLOCK) if chunk[i:i + _BLOCK] != zero)
    return total


def dir_bytes(path):
    """Allocated bytes (du semantics): sparse or preallocated files count what they occupy.
    Some sandboxes (gVisor) report every file as fully allocated, sparse ones included; for a
    file of 1 MiB or more reported that way, only its non-zero 4 KiB blocks are counted."""
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    total = 0
    for p in files:
        st = p.stat()
        allocated = min(st.st_size, st.st_blocks * 512)
        if allocated >= st.st_size >= 1 << 20:
            allocated = _data_bytes(p)
        total += allocated
    return total


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def log(message):
    print(message, file=sys.stderr, flush=True)
