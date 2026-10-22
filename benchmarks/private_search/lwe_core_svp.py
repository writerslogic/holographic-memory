# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Core-SVP cost of the primal uSVP attack on the private-search LWE parameters.

A sanity check, not a security proof: the 2016 estimate (Alkim, Ducas,
Poppelmann, Schwabe, "Post-quantum key exchange - a new hope", Section 6)
with sieving cost 2^(0.292 b) classical and 2^(0.265 b) quantum for BKZ block
size b. The secret is uniform mod q, so the attacker first moves to normal form
(secret drawn from the error distribution, n samples spent). The attacker picks
the number m of further samples, up to what one query publishes. Run with
`uv run --no-project python benchmarks/private_search/lwe_core_svp.py`.
"""

import json
import math
import sys

N = 1024
LOG_Q = 32


def delta(b: int) -> float:
    return ((math.pi * b) ** (1 / b) * b / (2 * math.pi * math.e)) ** (1 / (2 * (b - 1)))


def primal_block_size(n: int, log_q: int, sigma: float, max_m: int) -> tuple[int, int]:
    """Smallest b (and the m achieving it) with sigma*sqrt(b) <= delta^(2b-d-1) * q^(m/d)."""
    ln_q = log_q * math.log(2)
    for b in range(60, 2000):
        ln_d = math.log(delta(b))
        lhs = math.log(sigma) + 0.5 * math.log(b)
        for m in range(max(b - n, 1), max_m + 1, 8):
            d = m + n + 1
            if lhs <= (2 * b - d - 1) * ln_d + m / d * ln_q:
                return b, m
    raise ValueError("no block size below 2000")


def main() -> None:
    rows = []
    for sigma, label in [(6.4, "SimplePIR / this crate"), (11.0, "YPIR, s = 11*sqrt(2*pi)")]:
        for max_m in [4096, 1 << 21]:
            b, m = primal_block_size(N, LOG_Q, sigma, max_m)
            rows.append(
                {
                    "n": N,
                    "log_q": LOG_Q,
                    "sigma": sigma,
                    "parameter_source": label,
                    "samples_available": max_m,
                    "samples_used": m,
                    "block_size": b,
                    "core_svp_classical_bits": round(0.292 * b, 1),
                    "core_svp_quantum_bits": round(0.265 * b, 1),
                }
            )
    json.dump(
        {
            "method": "primal uSVP, 2016 estimate, core-SVP sieving cost; normal-form secret",
            "rows": rows,
        },
        sys.stdout,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
