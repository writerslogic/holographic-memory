# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deploy the exact-S AML Cycle 2 memory API on one persistent Modal container."""

from __future__ import annotations

from contextlib import suppress
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/work/src")
SOURCE_SHA = "12c80decb094d57ee13ba187a01bf1de8788ca2f"
APP_NAME = "warm-aml-cycle2-text-s"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("warm-aml-cycle2-text-s", create_if_missing=True)
image = (
    modal.Image.from_registry("rust:1.96-bookworm", add_python="3.12")
    .apt_install("build-essential", "pkg-config", "libssl-dev")
    .pip_install("fastapi==0.162.0", "httpx==0.28.1", "uvicorn==0.54.0")
    .add_local_dir(
        ROOT,
        "/work/src",
        copy=True,
        ignore=[
            ".git", ".venv", "target", "node_modules", "benchmarks/results/**",
            "benchmarks/data/**", "docs/research/**", "figures/**", "*.env",
        ],
    )
    .run_commands("cd /work/src && cargo build --locked --release --features server --bin warm-server")
)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("warm-aml-cycle2-token")],
    cpu=(8, 8),
    memory=(32768, 32768),
    min_containers=1,
    max_containers=1,
    timeout=1800,
)
@modal.asgi_app()
def api():
    import httpx

    sys.path.insert(0, "/work/src/benchmarks/public/aml_cycle2")
    from adapter import create_app

    command = [
        "/work/src/target/release/warm-server",
        "--no-auth", "--bind", "127.0.0.1:8777",
        "--data-dir", "/data/stores",
        "--max-top-k", "100",
        "--max-tenants", "64",
        "--tenant-idle-seconds", "30",
        "--max-concurrent-work", "8",
        "--max-queue", "512",
        "--queue-timeout-ms", "600000",
        "--max-body-bytes", "67108864",
        "--log-level", "error",
    ]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with suppress(httpx.RequestError):
            if httpx.get("http://127.0.0.1:8777/readyz", timeout=2).status_code == 200:
                break
        time.sleep(0.2)
    else:
        raise RuntimeError("warm-server did not become ready")
    return create_app("http://127.0.0.1:8777", os.environ["AML_TOKEN"], volume.commit)
