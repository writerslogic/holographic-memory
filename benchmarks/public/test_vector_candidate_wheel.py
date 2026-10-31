from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
import zipfile

import vector_candidate_wheel as wheel


class CandidateWheelTests(unittest.TestCase):
    def setUp(self):
        self.cargo = {"features": {
            "default": [], "python": [], "local-embed-download": [],
            "experimental": [], "experimental-batch-pool": ["experimental"],
            "experimental-neighbor-admission": ["experimental"],
            "experimental-query-x4": ["experimental"],
            "experimental-build-search": ["experimental"],
        }}
        self.project = {"tool": {"maturin": {"features": ["python", "local-embed-download"]}}}

    def test_disabled_and_candidate_features_preserve_project_configuration(self):
        self.assertEqual(wheel.feature_set(self.cargo, self.project, "disabled", []), ["local-embed-download", "python"])
        self.assertEqual(wheel.feature_set(self.cargo, self.project, "v3", []), ["experimental-batch-pool", "local-embed-download", "python"])
        self.assertEqual(wheel.feature_set(self.cargo, self.project, "candidate", ["experimental-query-x4"]), ["experimental-batch-pool", "experimental-query-x4", "local-embed-download", "python"])
        self.assertEqual(wheel.feature_set(self.cargo, self.project, "candidate", ["experimental-build-search"]), ["experimental-batch-pool", "experimental-build-search", "local-embed-download", "python"])
        self.assertEqual(self.project["tool"]["maturin"]["features"], ["python", "local-embed-download"])

    def test_unknown_duplicate_and_wrong_arm_feature_requests_fail(self):
        for arm, extra in [("unknown", []), ("disabled", ["experimental-query-x4"]), ("v3", ["experimental-query-x4"]), ("candidate", ["unknown"]), ("candidate", ["experimental-query-x4"] * 2)]:
            with self.subTest(arm=arm, extra=extra), self.assertRaises(ValueError):
                wheel.feature_set(self.cargo, self.project, arm, extra)
        del self.cargo["features"]["experimental-query-x4"]
        with self.assertRaises(ValueError):
            wheel.feature_set(self.cargo, self.project, "candidate", ["experimental-query-x4"])

    def test_project_defaults_cannot_contaminate_disabled_control(self):
        self.project["tool"]["maturin"]["all-features"] = True
        with self.assertRaises(ValueError):
            wheel.feature_set(self.cargo, self.project, "disabled", [])
        self.project["tool"]["maturin"]["all-features"] = False
        self.project["tool"]["maturin"]["features"].append("experimental-batch-pool")
        with self.assertRaises(ValueError):
            wheel.feature_set(self.cargo, self.project, "disabled", [])

    def test_transitive_feature_contamination_and_cycles_are_checked(self):
        self.cargo["features"]["local-embed-download"] = ["indirect"]
        self.cargo["features"]["indirect"] = ["local-embed-download", "experimental-batch-pool"]
        with self.assertRaises(ValueError):
            wheel.feature_set(self.cargo, self.project, "disabled", [])
        self.cargo["features"]["indirect"] = ["local-embed-download"]
        self.assertEqual(wheel.feature_set(self.cargo, self.project, "disabled", []), ["local-embed-download", "python"])

    def test_wheel_native_bytes_are_hashed_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("writerslogic_warm/writerslogic_warm.abi3.so", b"native fixture")
                archive.writestr("writerslogic_warm/__init__.py", b"pass")
            self.assertEqual(wheel.native_members(path), {"writerslogic_warm/writerslogic_warm.abi3.so": {"bytes": 14, "sha256": hashlib.sha256(b"native fixture").hexdigest()}})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_wheel_paths_duplicates_and_missing_native_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.whl"
            for name in ["../escape.so", "/absolute.so", "a/../escape.so", "a\\escape.so"]:
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(name, b"native")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    wheel.native_members(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("native.so", b"first")
                    archive.writestr("native.so", b"second")
            with self.assertRaises(ValueError):
                wheel.native_members(path)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("metadata.txt", b"not native")
            with self.assertRaises(ValueError):
                wheel.native_members(path)

    def test_oversized_wheel_is_rejected_before_zip_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.whl"
            with path.open("wb") as stream:
                stream.truncate(wheel.MAX_ARTIFACT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "exceeds 1 GiB"):
                wheel.native_members(path)

    def test_source_manifest_rejects_escape_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            (tree / "src").mkdir(parents=True)
            for name in ["Cargo.toml", "Cargo.lock", "pyproject.toml", "build.rs"]:
                (tree / name).write_text("fixture")
            (root / "outside.rs").write_text("outside")
            (tree / "src/lib.rs").symlink_to(root / "outside.rs")
            with self.assertRaises(ValueError):
                wheel.source_manifest(tree)

    def test_plan_is_explicit_offline_immutable_and_keeps_interpreter_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            (tree / "src").mkdir(parents=True)
            (tree / "src/lib.rs").write_text("pub fn fixture() {}")
            (tree / "Cargo.lock").write_text("version = 4\n")
            (tree / "build.rs").write_text("fn main() {}")
            (tree / "Cargo.toml").write_text('[features]\npython=[]\nexperimental=[]\nexperimental-batch-pool=["experimental"]\n')
            (tree / "pyproject.toml").write_text('[tool.maturin]\nfeatures=["python"]\n')
            executable = root / "environment-python"
            executable.symlink_to(sys.executable)
            before = wheel.source_manifest(tree)
            plan = wheel.make_plan(tree, root / "output", "v3", [], executable)
            self.assertEqual(plan["command"][0], str(executable))
            self.assertTrue(all(flag in plan["command"] for flag in ["--offline", "--locked", "--no-default-features"]))
            self.assertEqual(plan["requested_features"], ["experimental-batch-pool", "python"])
            self.assertFalse(plan["build_executed"])
            self.assertFalse((root / "output").exists())
            self.assertEqual(before, wheel.source_manifest(tree))
            (root / "output").mkdir()
            with self.assertRaises(ValueError):
                wheel.make_plan(tree, root / "output", "v3", [], executable)

    def test_cargo_profile_settings_are_preserved_without_other_environment(self):
        expected = {
            "CARGO_PROFILE_RELEASE_DEBUG": "1",
            "CARGO_PROFILE_RELEASE_STRIP": "none",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
            "CARGO_PROFILE_TEST_OPT_LEVEL": "1",
            "CARGO_PROFILE_DEV_DEBUG_ASSERTIONS": "false",
        }
        source = dict(expected, API_TOKEN="private-fixture-value")
        self.assertEqual(wheel.profile_environment(source), expected)
        for bad in [{"CARGO_PROFILE_RELEASE_DEBUG": "private-fixture-value"}, {"CARGO_PROFILE_SECRET": "private-fixture-value"}]:
            with self.assertRaises(ValueError) as caught:
                wheel.profile_environment(bad)
            self.assertNotIn("private-fixture-value", str(caught.exception))


@unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
class BuildCancellationTests(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, result = root / "ready.json", root / "result.json"
            fixture = root / "owned.py"
            fixture.write_text('''import json, os, signal, subprocess, sys, time
from pathlib import Path
child = None
if sys.argv[2] == "cancel":
    child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    def finish(_signum, _frame):
        child.wait(timeout=2)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, finish)
elif sys.argv[2] == "orphan":
    child_ready = Path(sys.argv[1] + ".child")
    child = subprocess.Popen([sys.executable, "-c", "import signal,sys; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).write_text('ready'); signal.pause()", str(child_ready)])
    while not child_ready.exists():
        time.sleep(0.01)
else:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(), "child": None if child is None else child.pid}))
while True:
    signal.pause()
''')
            supervisor = root / "supervisor.py"
            supervisor.write_text('''import json, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from vector_candidate_wheel import BuildCancelled, run_build
def previous(_signum, _frame):
    pass
signal.signal(signal.SIGTERM, previous)
try:
    with open(sys.argv[5], "w") as log:
        run_build([sys.executable, sys.argv[2], sys.argv[3], sys.argv[4]], Path.cwd(), dict(os.environ), log, log, timeout=0.5 if sys.argv[4] == "timeout" else 4)
except (BuildCancelled, TimeoutError) as error:
    Path(sys.argv[6]).write_text(json.dumps({"exception": type(error).__name__, "restored": signal.getsignal(signal.SIGTERM) is previous}))
else:
    raise RuntimeError("fixture unexpectedly completed")
''')
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            process = subprocess.Popen([sys.executable, str(supervisor), str(Path(wheel.__file__).parent), str(fixture), str(ready), mode, str(root / "child.log"), str(result)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            owned = None
            try:
                deadline = time.monotonic() + 3
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "child never became ready")
                owned = json.loads(ready.read_text())
                self.assertNotEqual(owned["pgid"], process.pid)
                self.assertNotEqual(owned["pgid"], unrelated.pid)
                if mode != "timeout":
                    os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=6)
                self.assertEqual(process.returncode, 0, (stdout, stderr))
                receipt = json.loads(result.read_text())
                self.assertEqual(receipt, {"exception": "TimeoutError" if mode == "timeout" else "BuildCancelled", "restored": True})
                self.assertIsNone(unrelated.poll())
                deadline = time.monotonic() + 3
                while True:
                    try:
                        os.killpg(owned["pgid"], 0)
                    except ProcessLookupError:
                        break
                    self.assertLess(time.monotonic(), deadline, "owned process group survived cancellation")
                    time.sleep(0.01)
                for pid in [owned["pid"], owned["child"]]:
                    if pid is not None:
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                print("PROCESS_OWNERSHIP_PROOF " + json.dumps({"mode": mode, "owned": owned, "supervisor_pid": process.pid, "unrelated_pid": unrelated.pid, "owned_group_gone": True, "unrelated_alive": True, "handlers_restored": True}), flush=True)
            finally:
                wheel.stop_owned(process)
                if owned is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(owned["pgid"], signal.SIGKILL)
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_sigterm_cleans_separate_child_group_without_touching_unrelated_process(self):
        self.exercise("cancel")

    def test_timeout_kills_term_ignoring_child_and_restores_handlers(self):
        self.exercise("timeout")

    def test_sigterm_kills_descendant_after_group_leader_exits(self):
        self.exercise("orphan")


if __name__ == "__main__":
    unittest.main()
