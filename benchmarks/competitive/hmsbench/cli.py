# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""`hms-bench` command line."""

import argparse
import sys

from . import envs, registry, runner, table
from .common import RESULTS


def main(argv=None):
    p = argparse.ArgumentParser(prog="hms-bench", description="Reproducible HMS comparison benchmark (see docs/benchmarks/README.md).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock", help="re-resolve the pinned comparator lock files (needs network)")
    ins = sub.add_parser("install", help="create the pinned comparator environments")
    ins.add_argument("systems", nargs="*")
    run = sub.add_parser("run", help="run a suite and write results under benchmarks/results/competitive/")
    run.add_argument("--suite", choices=list(runner.SUITES), default="light")
    run.add_argument("--only", choices=["ann", "text", "both"], default="both")
    run.add_argument("--systems", nargs="*", help="restrict to these system keys (hms is always included)")
    run.add_argument("--datasets", nargs="*", help="restrict to these dataset names")
    run.add_argument("--builds", nargs="*", metavar="SYSTEM:BUILD",
                     help="ANN only: restrict a comparator to these builds (systems not named keep all builds)")
    run.add_argument("--rounds", type=int)
    run.add_argument("--build-repeats", type=int)
    tab = sub.add_parser("table", help="regenerate docs/benchmarks/COMPARISON.md and the README block from committed results")
    tab.add_argument("--check", action="store_true", help="fail if the committed files differ from the regenerated ones")
    sub.add_parser("list", help="list systems and suites")
    a = p.parse_args(argv)

    if a.cmd == "lock":
        envs.lock()
    elif a.cmd == "install":
        for name in a.systems or [n for n in envs.PINS]:
            envs.ensure(name)
            print(name, envs.footprint(name))
    elif a.cmd == "list":
        for key, meta in registry.SYSTEMS.items():
            print(f"{key:14} {meta['kind']:5} {meta['display']}")
        for name, spec in runner.SUITES.items():
            print(f"suite {name}: {spec}")
        for key, why in registry.NOT_RUN.items():
            print(f"not run: {key}: {why}")
    elif a.cmd == "run":
        spec = {**runner.SUITES[a.suite], "name": a.suite}
        if a.rounds:
            spec["rounds"] = a.rounds
        if a.build_repeats:
            spec["build_repeats"] = a.build_repeats
        wanted = set(a.systems) if a.systems else None
        only_builds = None
        if a.builds:
            only_builds = {}
            for item in a.builds:
                system, sep, build = item.partition(":")
                if not sep or not build:
                    raise SystemExit(f"--builds expects SYSTEM:BUILD, got {item!r}")
                only_builds.setdefault(system, []).append(build)
        if a.only in ("ann", "both"):
            systems = [registry.HMS] + [s for s in registry.ANN_SYSTEMS if wanted is None or s in wanted]
            for dataset, n_train, n_test in spec["ann"]:
                if a.datasets and dataset not in a.datasets:
                    continue
                print(runner.run_ann(dataset, n_train, n_test, spec, systems, only_builds=only_builds))
        if a.only in ("text", "both"):
            systems = [registry.HMS] + [s for s in registry.TEXT_SYSTEMS if wanted is None or s in wanted]
            for dataset in spec["text"]:
                if a.datasets and dataset not in a.datasets:
                    continue
                print(runner.run_text(dataset, spec, systems))
    elif a.cmd == "table":
        return table.main(check=a.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
