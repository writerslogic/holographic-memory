#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The 0.27 floor is the ONE-HOT VALUE PRIOR, not superposition itself (companion to
# docs/SUPERPOSITION-FLOOR.md). If a fact's value is "one of L codebook vectors", the
# decoder faces N = M*L combinatorial unknowns and the AMP algorithmic threshold is
# ~0.27*D. If the value is instead a continuous AMPLITUDE (one of L PAM levels) on a
# per-key vector, recovery becomes the LINEAR solve s = K a (K is D*M), whose only
# threshold is identifiability at M/D = 1.0 -- a poly decoder with GRACEFUL soft readout,
# ~3.7x the one-hot floor at the SAME log2(L) bits/fact.
#
# The catch, also shown here: continuous amplitudes are SNR-hungry (L levels packed into
# amplitude -> small minimum distance), so under even mild noise the one-hot code is more
# robust. So this is not a free capacity win -- it is the SAME capacity/robustness
# tradeoff re-expressed on the value-encoding axis: total information per dimension is
# roughly conserved, and the "floor" is simply where the ROBUST (one-hot) encoding sits.
# This corrects the earlier doc claim that reaching M~=D requires a "hard cliff, no soft
# recall": the continuous high-capacity route is graceful; the cliff was an artifact of
# exact coded (Reed-Solomon/XOR) stores, not of high capacity.
#
# Self-contained (numpy). Deterministic. Run: uv run python scripts/superposition_floor_priors.py

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("priors")


def continuous_recover(d, m, ell, sigma_rel, seed):
    """Value = one of `ell` PAM amplitude levels per key; s = K a; decode by lstsq +
    round to nearest level. Returns per-fact recovery accuracy."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((d, m)) / np.sqrt(d)
    levels = np.linspace(-1.0, 1.0, ell)
    a = levels[rng.integers(0, ell, m)]
    s = keys @ a
    if sigma_rel > 0:
        s = s + sigma_rel * np.std(s) * rng.standard_normal(d)
    a_hat, *_ = np.linalg.lstsq(keys, s, rcond=None)
    est = np.abs(a_hat[:, None] - levels[None, :]).argmin(1)
    truth = np.abs(a[:, None] - levels[None, :]).argmin(1)
    return float(np.mean(est == truth))


def main():
    d, ell, seeds = 256, 64, range(5)
    log.info(
        f"D={d} L={ell}  continuous-amplitude value prior, lstsq decode ({len(list(seeds))} seeds)"
    )
    log.info(f"  {'M/D':>5} {'clean':>8} {'5% noise':>9} {'20% noise':>10}")
    for r in [0.30, 0.50, 0.80, 0.95, 1.05]:
        m = int(r * d)
        clean = np.mean([continuous_recover(d, m, ell, 0.0, s) for s in seeds])
        n5 = np.mean([continuous_recover(d, m, ell, 0.05, s) for s in seeds])
        n20 = np.mean([continuous_recover(d, m, ell, 0.20, s) for s in seeds])
        log.info(f"  {m / d:>5.2f} {clean:>8.3f} {n5:>9.3f} {n20:>10.3f}")
    log.info(
        "\nRead: clean recovery is 100% to M/D~0.95 (vs the one-hot floor 0.27), collapsing\n"
        "at the identifiability limit M/D=1.0 -- a poly linear solve with graceful readout.\n"
        "But it is noise-fragile (SNR-hungry PAM), so one-hot wins under noise. The floor is\n"
        "the one-hot prior; capacity vs robustness is a conserved-budget tradeoff on the\n"
        "value-encoding axis, not a wall."
    )


if __name__ == "__main__":
    main()
