# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Install-time smoke test for the writerslogic-warm-py wheel: run an associative search.

Run it with the interpreter of a clean virtualenv that has only the wheel installed:
    python scripts/ci/smoke_python.py [--expect-version X.Y.Z]

It stores 16 key/value pairs in one superposed memory, then for every key unbinds the
memory and must find the stored value among 64 distractors; it also factorizes a bound
pair with the resonator. Any wrong answer exits non-zero.
"""
import argparse
import importlib.metadata
import sys

import writerslogic_warm as hm

DIM = 2048
LEVELS = 64
PAIRS = 16
DISTRACTORS = 64


def search(memory, key, candidates):
    probe = memory.unbind(key)
    scored = [(probe.similarity(c), i) for i, c in enumerate(candidates)]
    return max(scored)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-version")
    args = parser.parse_args()

    version = importlib.metadata.version("writerslogic-warm-py")
    if args.expect_version and version != args.expect_version:
        print(f"installed writerslogic-warm-py {version}, expected {args.expect_version}", file=sys.stderr)
        return 1

    keys = [hm.PhaseHVec.random(DIM, LEVELS, 1000 + i) for i in range(PAIRS)]
    values = [hm.PhaseHVec.random(DIM, LEVELS, 2000 + i) for i in range(PAIRS)]
    distractors = [hm.PhaseHVec.random(DIM, LEVELS, 3000 + i) for i in range(DISTRACTORS)]
    memory = hm.PhaseHVec.bundle([k.bind(v) for k, v in zip(keys, values)])
    candidates = values + distractors
    wrong = [i for i, key in enumerate(keys) if search(memory, key, candidates) != i]
    if wrong:
        print(f"search returned the wrong value for keys {wrong}", file=sys.stderr)
        return 1

    codebooks = [keys[:8], values[:8]]
    composite = keys[3].bind(values[5])
    factors = [index for index, _ in hm.resonator_factorize(composite, codebooks)]
    if factors != [3, 5]:
        print(f"resonator factorized to {factors}, expected [3, 5]", file=sys.stderr)
        return 1

    print(f"writerslogic-warm-py {version}: {PAIRS}/{PAIRS} searches and resonator factorization ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
