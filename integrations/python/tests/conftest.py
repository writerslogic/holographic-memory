# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real warm-server fixture.

Set WARM_SERVER_BIN to a binary built with `cargo build --release --features server --bin warm-server`.
Each test gets its own tenant, so tests share one server process per configuration but no data.
"""

import itertools
import os
import subprocess
from pathlib import Path

import pytest

BIN = os.environ.get("WARM_SERVER_BIN")
_counter = itertools.count()


class Server:
    def __init__(self, url: str, dim: int):
        self.url, self.dim = url, dim

    def tenant(self) -> str:
        return f"t{os.getpid()}x{next(_counter)}"


@pytest.fixture(scope="session")
def server_factory(tmp_path_factory):
    procs, servers = [], {}

    def start(dim: int, **flags) -> Server:
        if not BIN or not Path(BIN).exists():
            pytest.skip("WARM_SERVER_BIN not set to an warm-server binary")
        key = (dim, tuple(sorted(flags.items())))
        if key not in servers:
            data = tmp_path_factory.mktemp(f"warm-data-{dim}")
            args = [BIN, "--bind", "127.0.0.1:0", "--no-auth", "--input-dim", str(dim), "--dim", "1024", "--data-dir", str(data)]
            for name, value in flags.items():
                args += [f"--{name.replace('_', '-')}", str(value)]
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
            procs.append(proc)
            line = proc.stdout.readline().strip()
            assert line.startswith("warm-server listening on "), line
            servers[key] = Server(f"http://{line.rsplit(' ', 1)[1]}", dim)
        return servers[key]

    yield start
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture
def server(server_factory):
    return server_factory(64)
