# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Run mem0ai/memory-benchmarks' own runners against HMS, unmodified.

Starts hms-server (lexical memory API), the Mem0-OSS shim in front of it and the
OpenAI logging proxy, then runs ``python -m benchmarks.<bench>.run`` inside a clean
checkout of memory-benchmarks pinned to PINNED_COMMIT with the flags that reproduce
the published headline runs (gpt-5 answerer and judge, top-200, single pass). Writes
``manifest-<phase>-<UTC stamp>.json`` (HMS commit and binary hash, frozen retrieval config, harness
commit, exact command, dataset hash, model parameters and snapshots seen by the
proxy, spend, silent-failure audit) next to the harness output.

Example (3 LoCoMo questions, $1 cap):
    uv run benchmarks/public/mem0_protocol/run_mem0_protocol.py locomo \
        --clone /path/memory-benchmarks --python /path/venv/bin/python \
        --hms-bin target/release/hms-server --work /path/out/locomo-smoke \
        --max-spend 1 -- --conversations 0 --max-questions 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
HARNESS_REPO = "https://github.com/mem0ai/memory-benchmarks"
MODEL = "gpt-5"

# Flags matching the metadata of Mem0's published result files (results/platform/*.json):
# top_k_cutoffs ["top_200"], gpt-5 answerer and judge, LoCoMo categories 1-4,
# LongMemEval all 500 questions in answerer mode, BEAM 1M (700 q) and 10M (200 q).
PROTOCOL = {
    "locomo": ["-m", "benchmarks.locomo.run", "--categories", "1,2,3,4"],
    "longmemeval": ["-m", "benchmarks.longmemeval.run", "--mode", "answerer", "--all-questions"],
    "beam-1m": ["-m", "benchmarks.beam.run", "--chat-sizes", "1M"],
    "beam-10m": ["-m", "benchmarks.beam.run", "--chat-sizes", "10M"],
}
COMMON = ["--answerer-model", MODEL, "--judge-model", MODEL, "--provider", "openai",
          "--top-k", "200", "--top-k-cutoffs", "200", "--backend", "oss"]
DATASETS = {"locomo": "--dataset-path", "longmemeval": "--dataset-path", "beam-1m": "--dataset-cache-dir",
            "beam-10m": "--dataset-cache-dir"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retrieval_hash(out_dir: Path) -> dict:
    """Hash of the frozen retrieval only: question_id and each result's (id, memory, created_at).

    Answers and judgments that --evaluate-only adds to the same files, and the ingestion
    checkpoints, are excluded, so the hash before and after an evaluate-only run must match.
    """
    rows = []
    for p in out_dir.glob("predicted_*/*.json") if out_dir.exists() else []:
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "question_id" not in d or "retrieval" not in d:
            continue
        hits = [(r.get("id"), r.get("memory"), r.get("created_at")) for r in d["retrieval"].get("search_results", [])]
        rows.append((str(d["question_id"]), hits))
    rows.sort(key=lambda r: r[0])
    return {"questions": len(rows), "sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except urllib.error.HTTPError:
            return  # listening
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor()  # no sysctl or /proc/cpuinfo: fall back to the generic name
    return platform.processor()


def summarize_proxy(log: Path) -> dict:
    out = {"calls": 0, "errors": 0, "blocked": 0, "spend_usd": 0.0, "requested_params": {}, "resolved_models": {},
           "usage": {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
           "finish_reasons": {}}
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        r = json.loads(line)
        out["calls"] += 1
        if r.get("blocked"):
            out["blocked"] += 1
            continue
        if r.get("status") != 200:
            out["errors"] += 1
        key = json.dumps(r.get("request", {}), sort_keys=True)
        out["requested_params"][key] = out["requested_params"].get(key, 0) + 1
        if r.get("resolved_model"):
            out["resolved_models"][r["resolved_model"]] = out["resolved_models"].get(r["resolved_model"], 0) + 1
        u = r.get("usage") or {}
        out["usage"]["prompt_tokens"] += u.get("prompt_tokens", 0)
        out["usage"]["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        out["usage"]["completion_tokens"] += u.get("completion_tokens", 0)
        out["usage"]["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        for fr in r.get("finish_reason") or []:
            out["finish_reasons"][fr] = out["finish_reasons"].get(fr, 0) + 1
        out["spend_usd"] = max(out["spend_usd"], r.get("spend_usd", 0.0))
    return out


def audit_outputs(out_dir: Path, harness_log: Path) -> dict:
    """Failures the harness scores as WRONG without stopping: empty retrieval, empty answers, retries."""
    per_q = [p for p in out_dir.rglob("*.json") if not p.name.startswith("_") and "results" not in p.name]
    empty_answers, empty_retrieval, questions = [], [], 0
    for p in per_q:
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "question" not in d:
            continue
        questions += 1
        if (d.get("retrieval") or {}).get("total_results", 1) == 0:
            empty_retrieval.append(d.get("question_id"))
        for c in (d.get("cutoff_results") or {}).values():
            if not (c.get("generated_answer") or "").strip():
                empty_answers.append(d.get("question_id"))
    text = harness_log.read_text(errors="replace") if harness_log.exists() else ""
    return {"questions_on_disk": questions, "empty_generated_answers": sorted(set(empty_answers)),
            "empty_retrieval": sorted(set(empty_retrieval)),
            "failed_after_retries_lines": len(re.findall(r"failed after", text)),
            "retry_warning_lines": len(re.findall(r"attempt \d+/\d+", text))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("benchmark", choices=sorted(PROTOCOL))
    ap.add_argument("--clone", required=True, type=Path, help="memory-benchmarks checkout at PINNED_COMMIT")
    ap.add_argument("--python", required=True, help="interpreter with the harness requirements")
    ap.add_argument("--hms-bin", required=True, type=Path, help="hms-server built with --features server")
    ap.add_argument("--work", required=True, type=Path, help="output directory (server data, logs, results)")
    ap.add_argument("--dataset", type=Path, help="LoCoMo/LongMemEval JSON, or the BEAM cache dir")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="hms-server store (default WORK/hms-data); put it on a local SSD: every remember is synced")
    ap.add_argument("--config", type=Path, default=HERE / "retrieval_config.json")
    ap.add_argument("--project-name", default=None)
    ap.add_argument("--max-spend", type=float, required=True, help="USD hard stop enforced by the proxy")
    ap.add_argument("--max-workers", type=int, default=10, help="harness --max-workers (its default is 10)")
    ap.add_argument("extra", nargs="*", help="after --: harness flags such as --predict-only, --evaluate-only, "
                    "--conversations, --max-questions, --run-id")
    args = ap.parse_args()

    clone = args.clone.resolve()
    head = git(clone, "rev-parse", "HEAD")
    if head != PINNED_COMMIT:
        sys.exit(f"harness checkout is at {head}, expected {PINNED_COMMIT}")
    if git(clone, "status", "--porcelain", "--untracked-files=no"):
        sys.exit("harness checkout has modified tracked files; the protocol runs it unpatched")
    if (clone / ".env").exists():
        sys.exit("harness checkout has a .env; load_dotenv(override=True) would override the run environment")
    for flag in ("--answerer-model", "--judge-model", "--top-k", "--top-k-cutoffs", "--provider", "--categories",
                 "--chat-sizes", "--mode", "--dataset-path", "--dataset-cache-dir", "--with-evidence",
                 "--user-profile", "--per-type", "--question-types", "--backend", "--mem0-host"):
        if flag in args.extra:
            sys.exit(f"{flag} is fixed by the protocol and may not be passed")

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    hms_repo = HERE.parents[2]
    project = args.project_name or f"hms-{args.benchmark}"
    procs: list[subprocess.Popen] = []
    with open(work / "hms-server.log", "ab") as log_hms, open(work / "shim.log", "ab") as log_shim, \
            open(work / "proxy.log", "ab") as log_proxy, open(work / "harness.log", "ab") as log_harness:
        try:
            hms_cmd = [str(args.hms_bin), "--no-auth", "--bind", "127.0.0.1:0", "--data-dir", str((args.data_dir or work / "hms-data").resolve()),
                       "--input-dim", "384", "--max-top-k", str(config.get("max_k", 200)), "--max-tenants", "1024",
                       "--tenant-idle-seconds", "120", "--max-concurrent-work", "16", "--max-queue", "1024",
                       "--queue-timeout-ms", "60000"]
            hms = subprocess.Popen(hms_cmd, stdout=subprocess.PIPE, stderr=log_hms)
            procs.append(hms)
            line = hms.stdout.readline().decode()
            m = re.search(r"listening on (\S+)", line)
            if not m:
                raise RuntimeError(f"hms-server did not start: {line!r}")
            hms_url = f"http://{m.group(1)}"

            shim_port, proxy_port = free_port(), free_port()
            shim_cmd = [sys.executable, str(HERE / "shim.py"), "--hms-url", hms_url, "--config", str(args.config),
                        "--port", str(shim_port)]
            procs.append(subprocess.Popen(shim_cmd, stdout=log_shim, stderr=log_shim))
            proxy_log = work / "openai_calls.jsonl"
            proxy_cmd = [sys.executable, str(HERE / "openai_proxy.py"), "--port", str(proxy_port), "--log", str(proxy_log),
                         "--max-spend", str(args.max_spend)]
            procs.append(subprocess.Popen(proxy_cmd, stdout=log_proxy, stderr=log_proxy))
            wait_http(f"http://127.0.0.1:{shim_port}/_shim/stats")
            wait_http(f"http://127.0.0.1:{proxy_port}/_proxy/health")

            out_dir = work / "harness-out"
            cmd = [args.python, *PROTOCOL[args.benchmark], *COMMON, "--project-name", project,
                   "--output-dir", str(out_dir), "--max-workers", str(args.max_workers),
                   "--mem0-host", f"http://127.0.0.1:{shim_port}"]
            if args.dataset:
                cmd += [DATASETS[args.benchmark], str(args.dataset.resolve())]
            cmd += args.extra
            env = {k: v for k, v in os.environ.items() if not k.startswith(("MEM0_", "AZURE_"))}
            env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{proxy_port}/v1"
            retrieval_before = retrieval_hash(out_dir)
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            rc = subprocess.run(cmd, cwd=clone, env=env, stdout=log_harness, stderr=subprocess.STDOUT).returncode
            finished = dt.datetime.now(dt.timezone.utc).isoformat()
            with urllib.request.urlopen(f"http://127.0.0.1:{shim_port}/_shim/stats", timeout=10) as r:
                facts = json.loads(r.read())["facts_by_tenant"]
        finally:
            for p in reversed(procs):
                p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()

    phase = "evaluate" if "--evaluate-only" in args.extra else "predict" if "--predict-only" in args.extra else "full"
    retrieval_after = retrieval_hash(out_dir)
    audit = audit_outputs(out_dir, work / "harness.log") if out_dir.exists() else {}
    if phase == "evaluate":
        audit["retrieval_unchanged"] = retrieval_before == retrieval_after
    manifest = {
        "benchmark": args.benchmark,
        "started_utc": started, "finished_utc": finished, "harness_exit_code": rc,
        "hms": {"commit": git(hms_repo, "rev-parse", "HEAD"),
                "dirty": bool(git(hms_repo, "status", "--porcelain", "--untracked-files=no")),
                "server_binary_sha256": sha256_file(args.hms_bin), "server_command": hms_cmd[1:]},
        "retrieval_config": config, "retrieval_config_sha256": sha256_file(args.config),
        "adapter": {"shim_sha256": sha256_file(HERE / "shim.py"),
                    "ingest_unit": "one fact per harness add() call; text = '{role}: {content}' per message, newline-joined",
                    "created_at": "harness timestamp (session date) as naive UTC ISO-8601",
                    "tenant": "one hms-server tenant per harness user_id",
                    "model_calls_at_ingest": 0},
        "harness": {"repo": HARNESS_REPO, "commit": head, "patches": [], "command": cmd[1:], "cwd": "<clone>"},
        "dataset": {"path": str(args.dataset) if args.dataset else None,
                    "sha256": sha256_file(args.dataset) if args.dataset and args.dataset.is_file() else None},
        "llm": {"endpoint": "api.openai.com chat.completions (Mem0 published runs: Azure OpenAI)",
                **summarize_proxy(proxy_log)},
        "ingest": {"tenants": len(facts), "facts": sum(facts.values())},
        "phase": phase,
        "retrieval_before": retrieval_before, "retrieval_after": retrieval_after,
        "audit": audit,
        "machine": {"cpu": cpu_name(), "os": platform.system(), "load_avg": os.getloadavg()},
    }
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (work / f"manifest-{phase}-{stamp}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"exit": rc, "spend_usd": manifest["llm"]["spend_usd"], "audit": manifest["audit"]}, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
