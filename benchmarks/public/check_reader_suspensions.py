# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independently validate reader suspension journals and isolated native reports."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISOLATED = Path("/Volumes/A/hms-metal-sdpa")
BINARY = Path("/Volumes/A/.hms-target-gpubatch/release/public-bench")
BINARY_SHA256 = "d7cb53d980ea567c1f14ff48a8239b1e372bc6992aa43d4e5fa2b21bf20631ec"
PROTOCOL = ROOT / "benchmarks/public/reader_protocol_v1.json"
PROTOCOL_SHA256 = "66a5d33412117c4ef5c6d0a89b0a26d0194103dc515d5a891a2593b2afcc566d"
NATIVE_CHECKER = ISOLATED / "benchmarks/public/check_metal_sdpa_pairs.py"
ARCHIVE = ROOT / "benchmarks/results/metal_sdpa_draft_2026-10-09"
DEFAULT_JOURNALS = tuple(ROOT / f"benchmarks/results/weak_reader_suspension_{name}_2026-10-09.json"
                         for name in ("pilot", "functional"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class Archive:
    def __init__(self, manifest_path, overrides=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.overrides = {} if overrides is None else overrides
        self.mapping = {}
        manifest = self.manifest
        require(manifest["schema_version"] == 1 and manifest["timing_claim"] is False
                and manifest["all_elapsed_values_preliminary"] is True
                and manifest["native_completed_runs"] == 0 and manifest["source_bytes_unchanged"] is True,
                "archive claims an accepted native result")
        for record in manifest["files"]:
            path = (ROOT / record["archived_path"]).resolve()
            require(path.is_relative_to(ROOT / "benchmarks") and not Path(record["archived_path"]).is_absolute(),
                    "archived artifact escapes committed benchmarks")
            data = self.read(path)
            require(len(data) == record["archive_bytes"] == record["original_bytes"]
                    and hashlib.sha256(data).hexdigest() == record["archive_sha256"] == record["original_sha256"],
                    "archived artifact bytes differ")
            require(record["preliminary"] is True and record["timing_claim"] is False, "archive elapsed values lack scope")
            key = (record["original_path"], record["original_sha256"])
            require(key not in self.mapping, "duplicate archive identity")
            self.mapping[key] = path
        for record in manifest["external_resources"]:
            path = Path(record["path"]).resolve()
            require(path.parent in (BINARY.parent.resolve(), Path("/Volumes/A/.hms-target-metal-sdpa/release"))
                    and record["archived"] is False and record["timing_claim"] is False,
                    "unexpected external archive resource")
            require(path.stat().st_size == record["bytes"] and file_sha256(path) == record["sha256"],
                    "persistent binary resource changed")
            self.mapping[(record["path"], record["sha256"])] = path
        self.derived = {}
        for record in manifest["derived_views"]:
            path = (ROOT / record["path"]).resolve()
            require(path.is_relative_to(ARCHIVE / "verification_views")
                    and record["transformation"] == "metadata paths only; preregistration hash updated for the remapped view",
                    "invalid archived verification view")
            data = self.read(path)
            require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                    "archived verification view changed")
            self.derived[record["role"]] = path

    def read(self, path):
        return self.overrides.get(Path(path).resolve(), None) if Path(path).resolve() in self.overrides else Path(path).read_bytes()

    def resolve(self, original, expected):
        key = (original, expected)
        require(key in self.mapping, "historical artifact lacks durable archive mapping")
        return self.mapping[key]

    def remap(self, value):
        if isinstance(value, list):
            return [self.remap(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: self.remap(item) for key, item in value.items()}
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = self.resolve(value["path"], value["sha256"])
            result["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        return result

    def validation_report(self, summary):
        original = json.loads(self.read(Path(summary["archived_report_path"])))
        metadata = original["preregistration"]
        registration = json.loads(self.read(self.resolve(metadata["path"], metadata["sha256"])))
        registration_bytes = encoded(self.remap(registration))
        registration_path = self.derived["preregistration"]
        require(self.read(registration_path) == registration_bytes, "preregistration view changed more than metadata paths")
        transformed = self.remap(original)
        transformed["preregistration"] = {
            "path": str(registration_path.relative_to(ROOT)), "sha256": hashlib.sha256(registration_bytes).hexdigest(),
            "bytes": len(registration_bytes),
        }
        report_path = self.derived["functional_report"]
        require(self.read(report_path) == encoded(transformed), "native report view changed more than metadata paths")
        return report_path


def utc_time(value):
    require(isinstance(value, str), "missing UTC suspension timestamp")
    result = datetime.fromisoformat(value)
    require(result.utcoffset() == timedelta(0), "suspension timestamp is not UTC")
    return result


def checked_path(value):
    require(isinstance(value, str) and value, "missing report path")
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    require(path.is_relative_to(ROOT) or path.is_relative_to(ISOLATED), "report path escapes research worktrees")
    return path


def pid_set(values):
    require(isinstance(values, list) and len(values) == 2
            and all(type(pid) is int and 0 < pid < 2 ** 31 for pid in values)
            and len(set(values)) == 2, "invalid suspension PID set")
    return set(values)


def inspect_journal(journal, archive):
    require(journal["schema_version"] == 1 and journal["status"] == "pilot_stopped",
            "original reader has not reached a resumed terminal state")
    require(journal["frozen_binary_sha256"] == BINARY_SHA256
            and journal["frozen_protocol_sha256"] == PROTOCOL_SHA256,
            "original reader fingerprint changed")
    require(journal["timing_claim"] is False and journal["frozen_prompt_or_generation_parameters_changed"] is False,
            "suspension asserts a timing claim or changes the frozen reader")
    expected = pid_set([journal["native_pid"], journal["parent_pid"]])
    require(pid_set(journal["resumed_pids"]) == expected, "SIGCONT return PID set differs from the two workers")
    started, resumed = utc_time(journal["started_at_utc"]), utc_time(journal["resumed_at_utc"])
    wall_duration = (resumed - started).total_seconds()
    elapsed = journal["suspended_elapsed_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed)
            and elapsed >= 0 and wall_duration >= 0
            and abs(wall_duration - elapsed) <= 1, "UTC and monotonic suspension durations differ")
    checks = journal["checks"]
    require(isinstance(checks, list), "native checks must be a list")
    if "load gate" in journal.get("error", "").lower():
        require(not checks, "gated blocker started a native check")
    summaries = []
    for entry in checks:
        require(isinstance(entry, dict) and type(entry["returncode"]) is int
                and type(entry["output_agreement"]) is bool, "invalid native check outcome")
        checked_path(entry["report_path"])
        path = archive.resolve(entry["report_path"], entry["report_sha256"])
        require(hashlib.sha256(archive.read(path)).hexdigest() == entry["report_sha256"], "native report fingerprint mismatch")
        require(checked_path(entry["checker_path"]) == NATIVE_CHECKER.resolve()
                and file_sha256(archive.resolve(entry["checker_path"], entry["checker_sha256"])) == entry["checker_sha256"],
                "isolated checker fingerprint mismatch")
        report = json.loads(archive.read(path))
        require(report["binaries"]["control"]["sha256"] == BINARY_SHA256
                and Path(report["binaries"]["control"]["path"]).resolve() == BINARY.resolve(),
                "native report changes the original reader binary")
        require(report["functional_only"] is True and report["timing_claim"] is False
                and report["complete"] is False, "native suspension diagnostic claims measured speed")
        runs = report["runs"]
        require(isinstance(runs, list), "invalid native run records")
        agrees = bool(runs) and all(run["same_text"] is True for run in runs)
        require(entry["output_agreement"] == agrees, "journal prediction agreement differs from native report")
        if not runs:
            require(entry["returncode"] != 0 and isinstance(report["failure"], str) and report["failure"],
                    "zero native outputs recorded as a successful check")
        summaries.append({"name": entry["name"], "report_path": entry["report_path"], "archived_report_path": str(path),
                          "report_sha256": entry["report_sha256"], "native_runs": len(runs),
                          "checker_path": entry["checker_path"], "checker_sha256": entry["checker_sha256"],
                          "output_agreement": agrees, "returncode": entry["returncode"]})
    return {"parent_pid": journal["parent_pid"], "native_pid": journal["native_pid"],
            "expected_worker_pids": sorted(expected), "sigcont_return_pids": sorted(expected),
            "historical_os_states_observed": False, "preliminary": True, "timing_claim": False,
            "utc_elapsed_seconds": wall_duration, "monotonic_elapsed_seconds": elapsed,
            "duration_difference_seconds": abs(wall_duration - elapsed),
            "gated_blocker": "load gate" in journal.get("error", "").lower(), "native_checks": summaries}


def validate_native(summary, archive):
    report_path = archive.validation_report(summary)
    checker_path = archive.resolve(summary["checker_path"], summary["checker_sha256"])
    command = [sys.executable, str(checker_path), str(report_path), "--allow-incomplete"]
    if summary["native_runs"]:
        command.append("--tamper-tests")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    require(completed.returncode == 0, "independent isolated native checker rejected the report: "
            + completed.stderr[-2000:])
    require(len(completed.stdout) <= 65536, "native checker output exceeds provenance bound")
    checked = json.loads(completed.stdout)
    require(checked["report_sha256"] == file_sha256(report_path)
            and checked["checker_sha256"] == summary["checker_sha256"] == file_sha256(checker_path)
            and checked["validated_runs"] == summary["native_runs"], "native checker proof fingerprint differs")
    require((checked["validated_runs"] > 0 and checked["same_text_runs"] == checked["validated_runs"])
            == summary["output_agreement"], "native output agreement proof differs")
    if summary["native_runs"]:
        require(checked["tamper_tests_rejected"] == 4, "native checker did not reject all four mutations")
    return {"argv": command, "stdout": checked, "original_report_sha256": summary["report_sha256"],
            "archived_verification_view_sha256": file_sha256(report_path), "preliminary": True, "timing_claim": False,
            "scope": "native output and source provenance" if summary["native_runs"]
            else "failed zero-output attempt; no completed prediction to mutate"}


def check_live_states(pids):
    observed_at = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(["ps", "-o", "pid=,stat=", "-p", ",".join(map(str, sorted(pids)))],
                             capture_output=True, text=True, timeout=10, check=False)
    require(process.returncode in (0, 1), "cannot inspect original reader process states")
    states = {}
    for line in process.stdout.splitlines():
        pid, state = line.split()
        require(int(pid) in pids and not state.startswith("T"), "original reader remains suspended")
        states[pid] = state
    return {"observed_at_utc": observed_at, "observed_process_states": states,
            "no_longer_present_pids": sorted(pids - {int(pid) for pid in states}),
            "historical_suspension_attestation": False}


def tamper_tests(journals, archive):
    cases = []
    changed = copy.deepcopy(journals[0])
    changed["status"] = "running_unmodified_control"
    cases.append(("corrupted_state", changed))
    changed = copy.deepcopy(journals[0])
    changed["resumed_pids"] = [changed["native_pid"], changed["native_pid"]]
    cases.append(("corrupted_pid_balance", changed))
    changed = copy.deepcopy(journals[0])
    changed["suspended_elapsed_seconds"] += 2
    cases.append(("corrupted_duration", changed))
    changed = copy.deepcopy(journals[0])
    changed["frozen_binary_sha256"] = "0" * 64
    cases.append(("corrupted_binary_fingerprint", changed))
    changed = copy.deepcopy(journals[0])
    changed["frozen_protocol_sha256"] = "0" * 64
    cases.append(("corrupted_protocol_fingerprint", changed))
    changed = copy.deepcopy(next(journal for journal in journals if journal["checks"]))
    changed["checks"][0]["output_agreement"] = not changed["checks"][0]["output_agreement"]
    cases.append(("corrupted_prediction", changed))
    passed = []
    for name, changed in cases:
        try:
            inspect_journal(changed, archive)
        except (ValueError, KeyError, TypeError):
            passed.append(name)
        else:
            raise ValueError(f"accepted suspension mutation: {name}")
    target = archive.resolve(NATIVE_CHECKER.as_posix(), journals[1]["checks"][0]["checker_sha256"])
    try:
        Archive(archive.manifest_path, {target: target.read_bytes() + b"\nCORRUPTED ARCHIVE"})
    except (ValueError, KeyError, TypeError):
        passed.append("corrupted_archive_bytes")
    else:
        raise ValueError("accepted suspension mutation: corrupted_archive_bytes")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="*", type=Path, default=list(DEFAULT_JOURNALS))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/weak_reader_suspensions_check_2026-10-09.json")
    parser.add_argument("--tamper-tests", action="store_true")
    parser.add_argument("--archive-manifest", type=Path, default=ARCHIVE / "manifest.json")
    args = parser.parse_args()
    require(len(args.journals) == 2 and len({path.resolve() for path in args.journals}) == 2,
            "proof must contain both distinct suspension journals")
    require(file_sha256(BINARY) == BINARY_SHA256 and file_sha256(PROTOCOL) == PROTOCOL_SHA256,
            "original frozen reader artifacts changed")
    archive = Archive(args.archive_manifest)
    journals = [json.loads(path.read_text()) for path in args.journals]
    summaries = [inspect_journal(journal, archive) for journal in journals]
    require(utc_time(journals[0]["resumed_at_utc"]) <= utc_time(journals[1]["started_at_utc"]),
            "reader suspension intervals overlap")
    pids = {pid for summary in summaries for pid in summary["sigcont_return_pids"]}
    live_states = check_live_states(pids)
    for summary in summaries:
        for native in summary["native_checks"]:
            native["independent_check"] = validate_native(native, archive)
    result = {"schema_version": 1, "checker": "passed", "checker_sha256": file_sha256(__file__),
              "frozen_binary_sha256": BINARY_SHA256, "frozen_protocol_sha256": PROTOCOL_SHA256,
              "archive_manifest_path": str(archive.manifest_path), "archive_manifest_sha256": file_sha256(archive.manifest_path),
              "archived_artifact_identities": len(archive.manifest["files"]),
              "persistent_binary_resources": archive.manifest["external_resources"], "timing_claim": False,
              "evidence_limit": "Journals record successful SIGCONT return PID sets; historical OS suspension states were unobserved.",
              "journal_artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in args.journals],
              "journals": summaries, "live_states": live_states,
              "tamper_tests_rejected": tamper_tests(journals, archive) if args.tamper_tests else []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".writing")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
