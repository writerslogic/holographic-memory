"""Prepare or execute an explicitly featured local candidate wheel build."""

from contextlib import suppress
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import tomllib
import zipfile


QUERY_FEATURES = {"experimental-neighbor-admission", "experimental-query-x4"}
OPTIONAL_FEATURES = QUERY_FEATURES | {"experimental-build-search"}
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


class BuildCancelled(RuntimeError):
    pass


def profile_environment(environment):
    choices = {
        "OPT_LEVEL": {"0", "1", "2", "3", "s", "z"},
        "DEBUG": {"0", "1", "2", "true", "false", "none", "limited", "full", "line-tables-only", "line-directives-only"},
        "STRIP": {"none", "debuginfo", "symbols", "true", "false"},
        "LTO": {"true", "false", "fat", "thin", "off"},
        "PANIC": {"unwind", "abort"},
        "SPLIT_DEBUGINFO": {"off", "packed", "unpacked"},
        "DEBUG_ASSERTIONS": {"true", "false"},
        "OVERFLOW_CHECKS": {"true", "false"},
        "INCREMENTAL": {"true", "false"},
        "RPATH": {"true", "false"},
    }
    result = {}
    for key, value in environment.items():
        if not key.startswith("CARGO_PROFILE_"):
            continue
        match = re.fullmatch(r"CARGO_PROFILE_[A-Z][A-Z0-9_]*_(" + "|".join(sorted([*choices, "CODEGEN_UNITS"], key=len, reverse=True)) + ")", key)
        if match is None:
            raise ValueError("unsupported Cargo profile environment setting")
        option = match.group(1)
        if option == "CODEGEN_UNITS":
            valid = bool(re.fullmatch(r"[0-9]{1,4}", value)) and 1 <= int(value) <= 1024
        else:
            valid = value in choices[option]
        if not valid:
            raise ValueError("invalid Cargo profile environment value")
        result[key] = value
    return dict(sorted(result.items()))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_manifest(tree):
    tree = tree.resolve(strict=True)
    paths = [tree / name for name in ["Cargo.toml", "Cargo.lock", "pyproject.toml", "build.rs"]]
    paths.extend(sorted((tree / "src").rglob("*.rs")))
    result = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(tree) or path.is_symlink() or not path.is_file():
            raise ValueError("source must be a regular file inside the worktree")
        result[str(path.relative_to(tree))] = digest(path)
    return result


def feature_set(cargo, project, arm, extra):
    if arm not in {"disabled", "v3", "candidate"}:
        raise ValueError("arm must be disabled, v3, or candidate")
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ValueError("query features must be a list of names")
    if len(extra) != len(set(extra)) or not set(extra) <= OPTIONAL_FEATURES:
        raise ValueError("unknown or duplicate candidate feature")
    if extra and arm != "candidate":
        raise ValueError("extra features require the candidate arm")
    options = project.get("tool", {}).get("maturin", {})
    if options.get("all-features", False):
        raise ValueError("all-features prevents an isolated control package")
    configured = options.get("features", [])
    if not isinstance(configured, list) or not all(isinstance(x, str) for x in configured):
        raise ValueError("only explicit project feature names are supported")
    if set(configured) & (OPTIONAL_FEATURES | {"experimental-batch-pool"}):
        raise ValueError("project defaults already select a candidate mechanism")
    features = set(configured) | {"python"}
    if arm != "disabled":
        features.add("experimental-batch-pool")
    features.update(extra)
    declared = cargo.get("features", {})
    if not features <= set(declared):
        raise ValueError("requested feature is not declared by this source")
    closure, pending = set(), list(features)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        dependencies = declared.get(name, [])
        if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
            raise ValueError("invalid Cargo feature graph")
        pending.extend(x for x in dependencies if x in declared)
    mechanisms = OPTIONAL_FEATURES | {"experimental-batch-pool"}
    if closure & mechanisms != features & mechanisms:
        raise ValueError("feature dependencies enable an undeclared mechanism")
    return sorted(features)


def make_plan(tree, output, arm, extra, interpreter):
    tree = tree.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    cargo = tomllib.loads((tree / "Cargo.toml").read_text())
    project = tomllib.loads((tree / "pyproject.toml").read_text())
    features = feature_set(cargo, project, arm, extra)
    executable = Path(os.path.abspath(interpreter))
    if not executable.is_file():
        raise ValueError("Python interpreter is not a file")
    command = [str(executable), "-m", "maturin", "build", "--offline", "--locked", "--release", "--no-default-features", "--features", ",".join(features), "--interpreter", str(executable), "--out", str(output / "wheels")]
    return {
        "schema": "hms-vector-candidate-wheel-plan-v1",
        "scope": "Local development wheel only; no deployment, publication, or leaderboard evidence",
        "tree": str(tree), "output": str(output), "arm": arm,
        "requested_features": features, "candidate_features": sorted(extra),
        "no_default_features": True, "command": command,
        "source_sha256": source_manifest(tree),
        "planner_sha256": digest(Path(__file__).resolve()),
        "python_executable_sha256": digest(executable),
        "python_executable": str(executable),
        "limits": {"timeout_s": 600, "build_jobs": 1},
        "cargo_profile_environment": profile_environment(os.environ),
        "execution_requires": "Caller must own the declared timing lock and apply the declared compiler-memory guard; this tool does not infer admission",
        "build_executed": False,
    }


def native_members(wheel):
    if wheel.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("wheel exceeds 1 GiB")
    result = {}
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        if len(members) > 4096 or sum(info.file_size for info in members) > MAX_ARTIFACT_BYTES:
            raise ValueError("wheel exceeds unpacked limits")
        names = set()
        for info in members:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or info.filename in names:
                raise ValueError("invalid or duplicate wheel member")
            names.add(info.filename)
            if info.flag_bits & 1:
                raise ValueError("encrypted wheel member")
            if path.suffix not in {".so", ".pyd", ".dylib"}:
                continue
            value = hashlib.sha256()
            size = 0
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    if size > MAX_ARTIFACT_BYTES:
                        raise ValueError("native wheel member exceeds limit")
                    value.update(block)
            if size != info.file_size:
                raise ValueError("native wheel member size mismatch")
            result[info.filename] = {"bytes": size, "sha256": value.hexdigest()}
    if not result:
        raise ValueError("wheel has no native library")
    return result


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def stop_owned(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 1
    while True:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=1)


def run_build(command, tree, env, stdout, stderr, timeout=600):
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
        raise ValueError("build timeout must be positive and at most 600 seconds")
    cancellation = None
    previous = {}
    process = None

    def request_cancellation(signum, _frame):
        nonlocal cancellation
        cancellation = signum

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, request_cancellation)
        # Recording signals lets Popen finish assigning the owned child before unwinding.
        process = subprocess.Popen(command, cwd=tree, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = time.monotonic() + timeout
        while True:
            if cancellation is not None:
                raise BuildCancelled(f"local wheel build cancelled by {signal.Signals(cancellation).name}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local wheel build exceeded its declared timeout")
            try:
                code = process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            if cancellation is not None:
                raise BuildCancelled(f"local wheel build cancelled by {signal.Signals(cancellation).name}")
            return code
    finally:
        try:
            if process is not None:
                stop_owned(process)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def execute(plan):
    tree, output = Path(plan["tree"]), Path(plan["output"])
    if source_manifest(tree) != plan["source_sha256"]:
        raise ValueError("source differs from the prepared wheel plan")
    if digest(Path(__file__).resolve()) != plan["planner_sha256"]:
        raise ValueError("wheel planner differs from its prepared binding")
    if digest(Path(plan["python_executable"])) != plan["python_executable_sha256"]:
        raise ValueError("Python interpreter changed")
    env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
    env.update({"CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_TARGET_DIR": str(output / "target")})
    profiles = profile_environment(os.environ)
    if profiles != plan["cargo_profile_environment"]:
        raise ValueError("Cargo profile settings differ from the prepared plan")
    env.update(profiles)
    versions = {}
    for name, command in {
        "maturin": [plan["python_executable"], "-m", "maturin", "--version"],
        "rustc": ["rustc", "-vV"], "cargo": ["cargo", "--version"],
    }.items():
        versions[name] = subprocess.check_output(command, cwd=tree, env=env, text=True, timeout=10).strip()
    started = time.monotonic()
    with (output / "build.stdout").open("x") as stdout, (output / "build.stderr").open("x") as stderr:
        code = run_build(plan["command"], tree, env, stdout, stderr)
    if code:
        raise RuntimeError(f"local wheel build failed with exit status {code}")
    if source_manifest(tree) != plan["source_sha256"]:
        raise ValueError("source changed while building the wheel")
    wheels = list((output / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("build must produce exactly one wheel")
    wheel = wheels[0]
    return {
        "schema": "hms-vector-candidate-wheel-receipt-v1",
        "build_executed": True, "returncode": code,
        "elapsed_s": time.monotonic() - started,
        "plan_sha256": digest(output / "plan.json"),
        "command": plan["command"], "requested_features": plan["requested_features"],
        "tool_versions": versions,
        "cargo_profile_environment": profiles,
        "source_sha256": plan["source_sha256"],
        "wheel": {"path": str(wheel), "bytes": wheel.stat().st_size, "sha256": digest(wheel)},
        "native_members": native_members(wheel),
        "stdout_sha256": digest(output / "build.stdout"),
        "stderr_sha256": digest(output / "build.stderr"),
        "scope": "Command/source/feature/artifact provenance; no runtime correctness or performance claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["disabled", "v3", "candidate"], required=True)
    parser.add_argument("--feature", "--query-feature", dest="candidate_feature", action="append", default=[], choices=sorted(OPTIONAL_FEATURES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Run the local build under the caller's timing/memory guard")
    args = parser.parse_args()
    plan = make_plan(args.tree, args.output, args.arm, args.candidate_feature, args.python)
    output = Path(plan["output"])
    output.mkdir(parents=True)
    save_new(output / "plan.json", plan)
    if args.execute:
        try:
            save_new(output / "receipt.json", execute(plan))
        except BaseException as error:
            save_new(output / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
            raise
    print(json.dumps({"plan": str(output / "plan.json"), "executed": args.execute, "requested_features": plan["requested_features"]}))


if __name__ == "__main__":
    main()
