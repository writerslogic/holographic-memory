# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lattice-estimator costs for the private-search LWE parameters.

Runs the lattice estimator (https://github.com/malb/lattice-estimator) on the
parameter set in `src/core/private_query.rs` and on the alternatives considered
for it. The estimator needs SageMath; the pip-installable `passagemath-standard`
distribution provides it. Reproduce with:

    git clone https://github.com/malb/lattice-estimator <dir>
    git -C <dir> checkout e4333d9bf9ef42fd8a29651b0c559fba7ce0d834
    uv venv --python 3.12 <venv>
    uv pip install --python <venv>/bin/python passagemath-standard==<version in the JSON>
    PYTHONPATH=<dir> <venv>/bin/python benchmarks/private_search/lattice_estimate.py > \
        benchmarks/results/private_query_lattice_estimate.json

Every set has q = 2^32 and a secret uniform mod q. The number of samples is
2^21, the most one query can publish (`MAX_DIM`). Costs are log2 of ring
operations ("rop"). `SKIPPED` lists the attacks left out. `estimate` is `LWE.estimate` with its default reduction
cost model (MATZOV); `rough` is `LWE.estimate.rough`, the core-SVP count
(2^(0.292 b) per BKZ block size b, one sieving call) used as a conservative
yardstick.
"""

import json
import math
import sys
from importlib.metadata import version

from estimator import LWE, ND

Q = 2**32
SAMPLES = 2**21
# Slow and not competitive for a Gaussian error and a 32-bit modulus.
SKIPPED = ("arora-gb", "bkw", "bdd_hybrid", "bdd_mitm_hybrid", "dual_mitm_hybrid")
CANDIDATES = [
    (1024, 6.4, "SimplePIR Section 4.2; the previous parameter set"),
    (1024, 11.0, "YPIR, s = 11*sqrt(2*pi)"),
    (1152, 6.4, "candidate"),
    (1280, 6.4, "candidate"),
    (1408, 6.4, "candidate"),
    (1536, 6.4, "candidate"),
    (2048, 6.4, "candidate"),
]


def bits(costs) -> dict:
    """log2(rop) per attack; attacks that do not apply (infinite cost) are dropped."""
    out = {}
    for name, cost in costs.items():
        rop = float(cost["rop"])
        if math.isfinite(rop):
            out[name] = round(math.log2(rop), 1)
    return out


def run(n: int, sigma: float) -> dict:
    params = LWE.Parameters(
        n=n, q=Q, Xs=ND.UniformMod(Q), Xe=ND.DiscreteGaussian(sigma), m=SAMPLES
    )
    full = LWE.estimate(params, jobs=8, quiet=True, deny_list=SKIPPED)
    rough = LWE.estimate.rough(params, jobs=8, quiet=True)
    out = {
        "estimate_bits": bits(full),
        "rough_bits": bits(rough),
    }
    if "usvp" in full:
        out["usvp_block_size"] = int(full["usvp"]["beta"])
    out["estimate_min_bits"] = min(out["estimate_bits"].values())
    out["rough_min_bits"] = min(out["rough_bits"].values())
    return out


def main() -> None:
    only = {int(a) for a in sys.argv[1:]}
    rows = []
    for n, sigma, source in CANDIDATES:
        if only and n not in only:
            continue
        row = {"n": n, "log_q": 32, "sigma": sigma, "samples": SAMPLES, "source": source}
        row.update(run(n, sigma))
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    json.dump(
        {
            "tool": "lattice-estimator",
            "commit": "e4333d9bf9ef42fd8a29651b0c559fba7ce0d834",
            "sage": "passagemath-standard " + version("passagemath-standard"),
            "secret": "uniform mod q",
            "error": "discrete Gaussian",
            "rows": rows,
        },
        sys.stdout,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
