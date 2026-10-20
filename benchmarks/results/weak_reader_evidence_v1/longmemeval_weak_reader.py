# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Run frozen local LongMemEval readers with resumable per-chunk artifacts."""

import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as output:
        output.write(canonical(value) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def label_preliminary(path):
    lines = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        if isinstance(value, dict):
            value = {**value, "preliminary": True, "timing_claim": False}
            line = canonical(value)
        lines.append(line)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as output:
        output.write("\n".join(lines) + ("\n" if lines else ""))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def make_request(reader_input, system):
    require(isinstance(reader_input, str) and reader_input, "missing frozen reader input")
    return {
        "model": system["reader_model"],
        "revision": system["revision"],
        "binary_sha256": system["binary_sha256"],
        "tokenizer_sha256": system["tokenizer_sha256"],
        "stage": "chat",
        "batch": system["batch"],
        "max_new_tokens": system["max_new_tokens"],
        "max_context": system["max_context"],
        "temperature": system["temperature"],
        "input": reader_input,
    }


def load_jobs(prepared, protocol):
    rows = prepared["rows"]
    require(isinstance(rows, list) and len(rows) == 100, "prepared input must cover 100 dev questions")
    require(len({row["qid"] for row in rows}) == 100, "duplicate prepared question")
    system = protocol["systems"]["weak_qwen4b"]
    require(system["batch"] == 1 and system["max_new_tokens"] == 1536
            and system["max_context"] == 12288 and system["temperature"] == 0,
            "unsupported frozen local generation parameters")
    jobs = []
    for row in rows:
        arms = row["systems"]["weak_qwen4b"]["arms"]
        require(set(arms) == set(protocol["arms"]), "frozen reader arm membership differs")
        for arm in protocol["arms"]:
            data = arms[arm]
            request = make_request(data["reader_input"], system)
            key = digest(request)
            jobs.append({"qid": row["qid"], "arm": arm, "request": request,
                         "request_sha256": key, "cache_key": key})
    random.Random(protocol["judge_blinding"]["shuffle_seed"]).shuffle(jobs)
    return jobs


def load_cache(path):
    rows, requests = {}, {}
    if not path.exists():
        return rows, requests
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            key = digest(record["request"])
            require(record["request_sha256"] == record["cache_key"] == key,
                    f"cache request fingerprint differs at line {line_number}")
            require(record["status"] in ("completed", "reader_failure")
                    and isinstance(record["raw_text"], str) and record["cost_usd"] == 0
                    and record["usage"] is None, "invalid frozen local cache record")
            identity = (record["qid"], record["arm"])
            require(identity not in rows, "duplicate cached reader arm")
            rows[identity] = record
            if key in requests:
                previous = requests[key]
                require(record["status"] == previous["status"]
                        and record["raw_text"] == previous["raw_text"],
                        "identical frozen request has inconsistent cached output")
            requests[key] = record
    return rows, requests


def append_record(output, job, text, status, reason=None):
    record = {**job, "raw_text": text, "response": {"text": text}, "status": status,
              "usage": None, "cost_usd": 0}
    if reason is not None:
        record["failure_reason"] = reason
    output.write(canonical(record) + "\n")
    output.flush()
    os.fsync(output.fileno())
    return record


def run(args):
    protocol = json.loads(args.protocol.read_text())
    system = protocol["systems"]["weak_qwen4b"]
    binary = Path(system["binary_path"])
    tokenizer = Path(system["tokenizer_path"])
    require(file_digest(binary) == system["binary_sha256"], "frozen reader binary fingerprint differs")
    require(file_digest(tokenizer) == system["tokenizer_sha256"], "frozen tokenizer fingerprint differs")
    prepared = json.loads(args.prepared.read_text())
    jobs = load_jobs(prepared, protocol)
    args.workdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.workdir / "weak_reader_outputs.jsonl"
    rows, requests = load_cache(cache_path)
    expected = {(job["qid"], job["arm"]): job for job in jobs}
    require(set(rows) <= set(expected), "cached arm outside frozen prepared inputs")
    for identity, record in rows.items():
        require(record["request"] == expected[identity]["request"], "cached input differs from frozen input")
    atomic_json(args.workdir / "run_manifest.json", {
        "prepared_sha256": file_digest(args.prepared),
        "protocol_sha256": file_digest(args.protocol),
        "binary_sha256": system["binary_sha256"],
        "tokenizer_sha256": system["tokenizer_sha256"],
        "jobs": [{key: job[key] for key in ("qid", "arm", "cache_key")} for job in jobs],
        "chunk_size": 10,
        "timing_claim": False,
    })
    pending = [job for job in jobs if (job["qid"], job["arm"]) not in rows]
    chunks = sorted(args.workdir.glob("chunk_*.manifest.json"))
    chunk_index = len(chunks)
    with cache_path.open("a") as output:
        while pending:
            fresh = []
            fresh_keys = set()
            remaining = []
            for job in pending:
                cached = requests.get(job["cache_key"])
                if cached is not None:
                    record = append_record(output, job, cached["raw_text"], cached["status"],
                                           cached.get("failure_reason"))
                    rows[(job["qid"], job["arm"])] = record
                elif len(fresh) < 10 and job["cache_key"] not in fresh_keys:
                    fresh.append(job)
                    fresh_keys.add(job["cache_key"])
                else:
                    remaining.append(job)
            pending = remaining
            if not fresh:
                continue
            chunk_index += 1
            prefix = args.workdir / f"chunk_{chunk_index:03d}"
            input_path = prefix.with_suffix(".input.json")
            output_path = prefix.with_suffix(".output.json")
            atomic_json(input_path, [job["request"]["input"] for job in fresh])
            command = [str(binary), "lme-model", "--stage", "chat", "--model", system["model_path"],
                       "--revision", system["revision"], "--batch", "1", "--input", str(input_path),
                       "--out", str(output_path)]
            atomic_json(prefix.with_suffix(".manifest.json"), {"jobs": fresh, "command": command})
            print(canonical({"chunk": chunk_index, "fresh_prompts": len(fresh),
                             "recorded_arms": len(rows), "total_arms": len(jobs)}), flush=True)
            with prefix.with_suffix(".stdout.log").open("wb") as stdout, \
                    prefix.with_suffix(".stderr.log").open("wb") as stderr:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            label_preliminary(prefix.with_suffix(".stderr.log"))
            status, reason, texts = "completed", None, None
            if completed.returncode != 0:
                status, reason = "reader_failure", f"native_process_exit_{completed.returncode}"
            else:
                try:
                    texts = json.loads(output_path.read_text())
                except (OSError, json.JSONDecodeError):
                    status, reason = "reader_failure", "native_output_missing_or_invalid_json"
                if status == "completed" and not (
                    isinstance(texts, list) and len(texts) == len(fresh)
                    and all(isinstance(text, str) for text in texts)
                ):
                    status, reason = "reader_failure", "native_output_shape_mismatch"
            if status != "completed":
                texts = [""] * len(fresh)
            for job, text in zip(fresh, texts, strict=True):
                record = append_record(output, job, text, status, reason)
                rows[(job["qid"], job["arm"])] = record
                requests[job["cache_key"]] = record
            atomic_json(prefix.with_suffix(".completion.json"), {
                "returncode": completed.returncode, "status": status, "failure_reason": reason,
                "recorded_arms": len(rows), "native_output_sha256": file_digest(output_path)
                if output_path.exists() else None,
            })
    require(len(rows) == 300, "weak reader must record all 300 frozen arms")
    print(canonical({"recorded_arms": len(rows), "completed": sum(
        row["status"] == "completed" for row in rows.values()),
        "failures": sum(row["status"] == "reader_failure" for row in rows.values()),
        "unique_requests": len(requests), "output": str(cache_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("target/superiority-validation/prepared_answers.json"))
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("reader_protocol_v1.json"))
    parser.add_argument("--workdir", type=Path, default=Path("target/superiority-validation/weak-reader"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
