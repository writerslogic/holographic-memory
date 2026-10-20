# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from document_search_compare import peak_rss_bytes, run_timed


class TimingHarnessTests(unittest.TestCase):
    def test_rss_platform_units(self):
        self.assertEqual(peak_rss_bytes("12345 maximum resident set size", "Darwin"), 12345)
        self.assertEqual(peak_rss_bytes("Maximum resident set size (kbytes): 12345", "Linux"), 12345 * 1024)
        for system, text in (("Darwin", ""), ("Linux", ""), ("Darwin", "0 maximum resident set size"),
                             ("Linux", "Maximum resident set size (kbytes): 0")):
            with self.subTest(system=system, text=text), self.assertRaises(ValueError):
                peak_rss_bytes(text, system)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_timeout_stops_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            program = (
                'import subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
                'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(30)'
            )
            with (Path(directory) / "log").open("w") as log, self.assertRaises(subprocess.TimeoutExpired):
                run_timed([sys.executable, "-c", program, str(pidfile)], log, 1)
            pid = pidfile.read_text()
            time.sleep(0.05)
            process = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
            self.assertTrue(process.returncode or process.stdout.strip().startswith("Z"), process.stdout)

    def test_nonzero_exit_propagates(self):
        with tempfile.TemporaryFile(mode="w") as log, self.assertRaises(subprocess.CalledProcessError):
            run_timed([sys.executable, "-c", "raise SystemExit(17)"], log, 5)


if __name__ == "__main__":
    unittest.main()
