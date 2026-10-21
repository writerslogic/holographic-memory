# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle of one server process tree for the server-mode rows (standard library only; the
server worker imports it inside its pinned environment).

A server starts in its own session, so its process group is the whole tree it forks (Postgres
backends, OpenSearch's JVM); stop() signals the group and waits, then kills the group. RSS of
the tree is sampled, never read from wait4: the client is not the server's parent. On Linux a
sample is the sum of PSS (smaps_rollup), so shared memory such as Postgres shared buffers counts
once; on macOS it is the sum of per-process RSS from ps, which counts shared pages once per
process. Sampling makes the reported peak a lower bound of the true peak.
"""

import os
import platform
import pwd
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

LINUX = platform.system() == "Linux"
# On a root container (Modal) servers that refuse root (Postgres, OpenSearch) run as this user.
RUN_AS = os.environ.get("HMS_BENCH_SERVER_USER", "hmsbench")


def free_ports(count):
    """`count` distinct loopback ports that were free at the time of the call. All sockets are
    held open together so the kernel cannot hand the same port out twice."""
    held = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            held.append(s)
        return [s.getsockname()[1] for s in held]
    finally:
        for s in held:
            s.close()


def _group_pids(pgid):
    if LINUX:
        pids = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                # Field 5 of /proc/<pid>/stat is the process group; the command name (field 2)
                # may contain spaces, so split after its closing parenthesis.
                stat = (entry / "stat").read_text()
                if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
                    pids.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
        return pids
    out = subprocess.run(["ps", "-ax", "-o", "pid=,pgid="], capture_output=True, text=True).stdout
    return [int(pid) for pid, group in (line.split() for line in out.splitlines() if line.strip()) if int(group) == pgid]


def tree_rss_bytes(pgid):
    """Memory of every process in group `pgid` (see the module note for the per-OS meaning)."""
    if LINUX:
        total = 0
        for pid in _group_pids(pgid):
            try:
                text = Path(f"/proc/{pid}/smaps_rollup").read_text()
                total += next(int(line.split()[1]) for line in text.splitlines() if line.startswith("Pss:")) * 1024
            except (OSError, StopIteration, ValueError):
                try:
                    total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                except (OSError, ValueError):
                    pass
        return total
    out = subprocess.run(["ps", "-ax", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    return sum(int(rss) * 1024 for group, rss in (line.split() for line in out.splitlines() if line.strip()) if int(group) == pgid)


RSS_METHOD = ("sum of PSS over the server process group (/proc/<pid>/smaps_rollup), sampled" if LINUX else
              "sum of RSS over the server process group (ps), sampled; shared pages count once per process")


def _drop_privileges():
    """preexec_fn for a root parent: become RUN_AS in a fresh session."""
    user = pwd.getpwnam(RUN_AS)
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)


def chown_for_server(*paths):
    """Hand data and log directories to RUN_AS when running as root (no-op otherwise)."""
    if os.geteuid() != 0:
        return
    user = pwd.getpwnam(RUN_AS)
    for path in paths:
        for root, dirs, files in os.walk(path):
            for name in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
                os.chown(name, user.pw_uid, user.pw_gid)


def as_server_user(argv, **kwargs):
    """Run a setup command (initdb, a config tool) as the user the server will run as."""
    if os.geteuid() == 0:
        kwargs["preexec_fn"] = _drop_privileges
    return subprocess.run(argv, **kwargs)


class Server:
    """One server process tree. `argv` runs in the foreground (no daemonizing); stdout and stderr
    go to `<log_dir>/<name>.log`. Use as a context manager so the tree is killed on every path."""

    def __init__(self, name, argv, log_dir, env=None, cwd=None):
        self.name, self.argv, self.env, self.cwd = name, [str(a) for a in argv], env, cwd
        self.log_path = Path(log_dir) / f"{name}.log"
        self.proc = None
        self.started_ns = None
        self.peak = 0
        self._sampling = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        kwargs = {"preexec_fn": _drop_privileges} if os.geteuid() == 0 else {}
        if os.geteuid() == 0:
            chown_for_server(self.log_path.parent)
        self.started_ns = time.time_ns()
        self.proc = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env={**os.environ, **(self.env or {})}, cwd=self.cwd, start_new_session=True, **kwargs)
        log.close()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.sampling(True)
        return self

    def _sample_loop(self):
        while not self._stop.is_set():
            if self._sampling.is_set():
                self.sample()
            self._stop.wait(0.25)

    def sampling(self, on):
        """Background sampling runs during ingest and startup only: a `ps` call in the client
        process during a timed query loop would be measured as server latency."""
        (self._sampling.set if on else self._sampling.clear)()

    def sample(self):
        if self.proc is None or self.proc.poll() is not None:
            return 0
        rss = tree_rss_bytes(self.proc.pid)
        self.peak = max(self.peak, rss)
        return rss

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n=3000):
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def wait_ready(self, probe, timeout=180.0, interval=0.05):
        """Poll `probe()` until it returns truthy. Raises if the process exits or `timeout`
        passes, with the log tail; probe exceptions count as not ready."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            if not self.alive():
                raise RuntimeError(f"{self.name} exited with {self.proc.returncode}: {self.log_tail()}")
            try:
                value = probe()
                if value:
                    return value
            except Exception as error:  # connection refused while starting is expected
                last = error
            time.sleep(interval)
        raise TimeoutError(f"{self.name} not ready after {timeout}s (last error {last!r}): {self.log_tail()}")

    def stop(self, grace=60.0):
        """SIGTERM the group (a clean shutdown flushes), wait up to `grace`, then SIGKILL it.
        Returns the exit code of the leader."""
        self._stop.set()
        if self.proc is None:
            return None
        pgid = self.proc.pid
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
            # The leader is gone; children it did not reap may still hold the group.
            deadline = time.monotonic() + wait
            while _group_pids(pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if not _group_pids(pgid):
                break
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.proc.poll() is None:
            self.proc.wait(timeout=10)
        return self.proc.returncode

    def __enter__(self):
        return self.start() if self.proc is None else self

    def __exit__(self, *exc):
        self.stop()
        return False
