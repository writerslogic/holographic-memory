# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Copy frozen development grades only when fresh official inputs match exactly."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from locomo_comparison import complete_grade
from mem0_protocol.run_mem0_protocol import COMMON, PINNED_COMMIT, retrieval_hash


def read_json(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("expected a regular JSON file of at most 16 MiB")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def binding(manifest: dict, expected: int, scored: bool) -> dict:
    audit = manifest["audit"]
    completion = audit["completion"]
    if (manifest["benchmark"] != "locomo" or manifest["harness"]["commit"] != PINNED_COMMIT
            or manifest["phase"] != ("evaluate" if scored else "predict")
            or manifest["warm"]["dirty"] or manifest["wrapper_exit_code"] or manifest["harness_exit_code"]
            or not completion["complete"] or completion["expected_questions"] != expected
            or completion["scored_questions"] != (expected if scored else 0)
            or any(audit.get(key) for key in ("ingestion_failures", "failed_after_retries_lines",
                                             "invalid_ingestion_checkpoints"))):
        raise ValueError("manifest does not establish complete successful protocol output")
    command = manifest["harness"]["command"]
    fixed = ["--categories", "1,2,3,4", *COMMON]
    for flag, value in zip(fixed[::2], fixed[1::2]):
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError("official protocol flags differ")
    if any(arg.split("=", 1)[0] in ("--judge-with-evidence", "--rejudge") for arg in command):
        raise ValueError("unsupported judge protocol override")
    operational = {"--dataset-path", "--output-dir", "--mem0-host", "--project-name", "--run-id", "--conversations"}
    effective, index = [], 0
    while index < len(command):
        argument = command[index]
        if argument in operational:
            if index + 1 == len(command):
                raise ValueError("missing operational argument")
            index += 2
            continue
        if argument not in ("--predict-only", "--evaluate-only"):
            effective.append(argument)
        index += 1
    llm = manifest["llm"]
    if llm["errors"] or llm["blocked"] or (not scored and llm["calls"]):
        raise ValueError("failed or unexpected model calls")
    if scored:
        allowed = [{"model": "gpt-5", "max_completion_tokens": 4096},
                   {"model": "gpt-5", "max_completion_tokens": 4096,
                    "response_format": {"type": "json_object"}}]
        params = llm["requested_params"]
        if (llm["calls"] != 2 * expected or len(params) != 2
                or {json.dumps(json.loads(key), sort_keys=True) for key in params}
                != {json.dumps(value, sort_keys=True) for value in allowed}
                or any(count != expected for count in params.values())
                or len(llm["resolved_models"]) != 1
                or sum(llm["resolved_models"].values()) != 2 * expected):
            raise ValueError("model, judge or completion budget is unproven")
    config = manifest["retrieval_config"]
    if config.get("max_context_tokens") != 7000 or config.get("max_k") != 200:
        raise ValueError("retrieval budget differs")
    return {"harness_commit": PINNED_COMMIT, "effective_command": effective, "retrieval_config": config,
            "retrieval_config_sha256": manifest["retrieval_config_sha256"],
            "shim_sha256": manifest["adapter"]["shim_sha256"],
            "server_binary_sha256": manifest["warm"]["server_binary_sha256"]}


def freeze_source(source: Path, manifest_path: Path, expected: int = 385) -> dict:
    manifest, manifest_hash = read_json(manifest_path)
    signature = binding(manifest, expected, True)
    if retrieval_hash(source.parent) != manifest["retrieval_after"]:
        raise ValueError("source retrieval differs from its evaluated manifest")
    paths = sorted(source.glob("conv*_q*.json"))
    if len(paths) != expected or any(not re.fullmatch(r"conv[012]_q\d+\.json", p.name) for p in paths):
        raise ValueError("source must contain only the established development scope")
    hashes = {}
    for path in paths:
        data, digest = read_json(path)
        if data.get("question_id") != path.stem:
            raise ValueError("question identity differs from filename")
        complete_grade(data, len(data["retrieval"]["search_results"]))
        hashes[path.name] = digest
    return {"source": str(source.resolve()), "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_hash, "expected_development_questions": expected,
            "binding": signature, "resolved_model": next(iter(manifest["llm"]["resolved_models"])),
            "source_sha256": hashes,
            "rule": "Reuse every complete grade with identical ranked memories and official answer/judge inputs; correctness is never a selection condition."}


def prepare(frozen: dict, fresh: Path, fresh_manifest_path: Path, destination: Path,
            render_prompt, expected_total: int = 1540) -> dict:
    source = Path(frozen["source"]).resolve(strict=True)
    current = freeze_source(source, Path(frozen["manifest"]), frozen["expected_development_questions"])
    if current != frozen:
        raise ValueError("frozen development source or protocol changed")
    manifest, manifest_hash = read_json(fresh_manifest_path)
    if binding(manifest, expected_total, False) != frozen["binding"]:
        raise ValueError("fresh prediction uses a different candidate or protocol")
    fresh = fresh.resolve(strict=True)
    destination = destination.resolve()
    for origin in (source, fresh):
        if destination == origin or destination in origin.parents or origin in destination.parents:
            raise ValueError("prediction directories must not overlap")
    if destination.exists():
        raise FileExistsError("destination exists; preserve prior results")
    entries = list(fresh.iterdir())
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > 16 * 1024 * 1024 for p in entries):
        raise ValueError("fresh predictions must contain only bounded regular files")
    if len(list(fresh.glob("conv*_q*.json"))) != expected_total:
        raise ValueError("fresh prediction scope is incomplete")
    if retrieval_hash(fresh.parent) != manifest["retrieval_after"]:
        raise ValueError("fresh retrieval differs from its prediction manifest")
    for path in fresh.glob("conv*_q*.json"):
        data, _ = read_json(path)
        if data.get("question_id") != path.stem or data.get("cutoff_results"):
            raise ValueError("fresh prediction identity is invalid or already scored")
    prepared, decisions = {}, {}
    fields = ("question", "category", "ground_truth_answer", "evidence", "reference_date", "user_profile")
    for name in frozen["source_sha256"]:
        old, old_hash = read_json(source / name)
        new, new_hash = read_json(fresh / name)
        if old["question_id"] != new.get("question_id") or new["question_id"] != Path(name).stem:
            raise ValueError("fresh question identity mismatch")
        if new.get("cutoff_results"):
            raise ValueError("fresh destination already contains scored development results")
        equal = (old["retrieval"]["search_results"] == new["retrieval"]["search_results"]
                 and all(old.get(field) == new.get(field) for field in fields))
        old_prompt, new_prompt = render_prompt(old), render_prompt(new)
        equal = equal and old_prompt == new_prompt
        decisions[name] = {"reused": equal, "source_sha256": old_hash, "fresh_sha256": new_hash,
                           "source_prompt_sha256": hashlib.sha256(old_prompt.encode()).hexdigest(),
                           "fresh_prompt_sha256": hashlib.sha256(new_prompt.encode()).hexdigest()}
        if equal:
            new["cutoff_results"] = copy.deepcopy(complete_grade(old, len(new["retrieval"]["search_results"])))
            prepared[name] = json.dumps(new, ensure_ascii=False, indent=2) + "\n"
    receipt = {"frozen_source": frozen, "fresh_predictions": str(fresh),
               "fresh_manifest_sha256": manifest_hash, "destination": str(destination),
               "expected_total_questions": expected_total, "reused_grades": len(prepared),
               "decisions": decisions, "required_final_resolved_model": frozen["resolved_model"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".locomo-reuse-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "predictions"
        stage.mkdir()
        for path in entries:
            shutil.copy2(path, stage / path.name)
        for name, text in prepared.items():
            (stage / name).write_text(text, encoding="utf-8")
        (stage / "_development_reuse_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage.rename(destination)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--fresh-manifest", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--clone", type=Path)
    args = parser.parse_args()
    if args.source:
        if args.freeze.exists() or not args.source_manifest:
            parser.error("freezing requires a source manifest and a new freeze path")
        args.freeze.write_text(json.dumps(freeze_source(args.source, args.source_manifest), indent=2) + "\n")
        return
    if not all((args.fresh, args.fresh_manifest, args.destination, args.clone)):
        parser.error("reuse requires fresh predictions/manifest, destination and pinned clone")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.clone, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.clone, text=True).strip()
    if actual != PINNED_COMMIT or dirty:
        raise ValueError("official harness must be the clean pinned commit")
    sys.path.insert(0, str(args.clone.resolve()))
    from benchmarks.locomo.prompts import get_answer_generation_prompt

    def render(data):
        return get_answer_generation_prompt(data["question"], data["retrieval"]["search_results"][:200],
                                            reference_date=data.get("reference_date"), user_profile=data.get("user_profile"))

    frozen, _ = read_json(args.freeze)
    receipt = prepare(frozen, args.fresh, args.fresh_manifest, args.destination, render)
    print(json.dumps({key: receipt[key] for key in ("expected_total_questions", "reused_grades", "required_final_resolved_model")}))


if __name__ == "__main__":
    main()
