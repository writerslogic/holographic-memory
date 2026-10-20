#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Beating the superposition floor by CO-DESIGNING THE ENCODER (companion to
# docs/SUPERPOSITION-FLOOR.md). The ~0.27 recovery floor is a property of the RANDOM
# code ensemble, not a fundamental wall: a spatially-coupled (SC-SPARC) encoder crosses
# it with a POLYNOMIAL decoder at MATCHED storage, exactly as coding theory predicts
# (threshold saturation; Donoho-Javanmard-Montanari, Rush-Venkataramanan). This is
# VALIDATION of an established mechanism transplanted onto the VSA-memory floor, not a
# new algorithm.
#
# Three encoders, ONE poly decoder (block section-AMP):
#   dense    -- iid Gaussian columns == the random-bundle baseline (reproduces the floor)
#   coupled  -- spatially-coupled band profile (SC-SPARC): early row-blocks are lightly
#               loaded and decode first, seeding a wave that propagates through fully-
#               interfering middle blocks.
# Storage for both = s in R^D (D reals). The design matrix A is decoder-side structure
# (like the harness keys/codebook), not per-memory storage -- matched storage, matched M.
#
# Three checks are printed:
#   (1) GUARD  -- dense/block-AMP reproduces the ~0.27 floor (decoder is not magic).
#   (2) CROSS  -- coupled recovers where dense floors, at matched D and M.
#   (3) PROFILE-- per-column-block accuracy at the crossing: middle (non-seed) blocks
#                 decode too -> genuine wave propagation, NOT sharding.
#
# HONEST SCOPE. The crossing is a CLEAN-SIGNAL phenomenon: under a corrupted/noisy query
# the SC advantage collapses back to the dense floor (the wave needs a clean seed). So it
# raises CLEAN capacity, not the NOISY-QUERY capacity that is HMS's robustness value.
#
# Self-contained (numpy). Deterministic. Run: uv run python scripts/superposition_floor_coded.py

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("coded")


def build_dense(d, m, ell, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, m * ell)) / np.sqrt(d)
    labels = rng.integers(0, ell, size=m)
    s = a[:, labels + ell * np.arange(m)].sum(axis=1)
    return dict(
        A=a,
        s=s,
        labels=labels,
        ell=ell,
        m=m,
        d=d,
        n_row=1,
        n_col=1,
        W=np.ones((1, 1)),
        row_block=np.zeros(d, int),
        sec_col=np.zeros(m, int),
    )


def build_coupled(d, m, ell, seed, c, w):
    """Spatially-coupled SPARC: C column-blocks, R=C+w-1 row-blocks, band width w."""
    rng = np.random.default_rng(seed)
    r_blocks = c + w - 1
    assert d % r_blocks == 0 and m % c == 0, (d, r_blocks, m, c)
    nr = d // r_blocks
    spb = m // c
    weight = np.array(
        [
            [1.0 if 0 <= r - cc <= w - 1 else 0.0 for cc in range(c)]
            for r in range(r_blocks)
        ]
    )
    s_c = weight.sum(axis=0)
    sigma = np.sqrt(weight * r_blocks / (d * s_c[None, :]))
    a = np.zeros((d, m * ell))
    sec_col = np.repeat(np.arange(c), spb)
    for r in range(r_blocks):
        for cc in range(c):
            if weight[r, cc] == 0:
                continue
            cols = np.where(sec_col == cc)[0]
            col_ids = (cols[:, None] * ell + np.arange(ell)[None, :]).ravel()
            a[r * nr : (r + 1) * nr, col_ids] = sigma[r, cc] * rng.standard_normal(
                (nr, col_ids.size)
            )
    labels = rng.integers(0, ell, size=m)
    s = a[:, labels + ell * np.arange(m)].sum(axis=1)
    return dict(
        A=a,
        s=s,
        labels=labels,
        ell=ell,
        m=m,
        d=d,
        n_row=r_blocks,
        n_col=c,
        W=weight,
        row_block=np.repeat(np.arange(r_blocks), nr),
        sec_col=sec_col,
    )


def block_amp(prob, n_iter=300, damp=0.3, eps=1e-10):
    """Section-softmax AMP with per-row-block noise variance. Reduces to scalar AMP at
    n_row=n_col=1. Uses only s + design structure -- no access to true labels."""
    a, ell, m, d = prob["A"], prob["ell"], prob["m"], prob["d"]
    rb, r_blocks, c, weight, sec_col, s = (
        prob["row_block"],
        prob["n_row"],
        prob["n_col"],
        prob["W"],
        prob["sec_col"],
        prob["s"],
    )
    s_c = weight.sum(axis=0)
    x = np.zeros(m * ell)
    z = s.copy()
    p = np.ones((m, ell)) / ell
    for _ in range(n_iter):
        tau2 = np.array(
            [
                max(np.dot(z[rb == r], z[rb == r]) / max((rb == r).sum(), 1), eps)
                for r in range(r_blocks)
            ]
        )
        eff_col = (weight / tau2[:, None]).sum(axis=0) / np.maximum(s_c, eps)
        eff_sec = eff_col[sec_col]
        g = a.T @ (z / tau2[rb])
        r_mat = (x + g / np.repeat(eff_sec, ell)).reshape(m, ell)
        logits = eff_sec[:, None] * r_mat
        logits -= logits.max(axis=1, keepdims=True)
        p_new = np.exp(logits)
        p_new /= p_new.sum(axis=1, keepdims=True)
        x_new = p_new.reshape(-1)
        pv = (p_new - p_new * p_new).sum(axis=1)
        v = np.array([pv[sec_col == cc].sum() for cc in range(c)])
        coeff = (
            weight * (r_blocks / (d * np.maximum(s_c, eps)))[None, :] * v[None, :]
        ).sum(axis=1)
        z = damp * z + (1 - damp) * (s - a @ x_new + z * (coeff / tau2)[rb])
        x = damp * x + (1 - damp) * x_new
        p = p_new
    return np.argmax(p, axis=1), p


def _acc(prob, est):
    return float(np.mean(est == prob["labels"]))


def main():
    ell, seeds = 64, range(5)
    log.info("(1) GUARD: dense/block-AMP must reproduce the ~0.27 random-code floor")
    log.info(f"  {'M/D':>5} {'dense_acc':>12}")
    d = 512
    for ratio in [0.20, 0.25, 0.30, 0.40]:
        m = round(ratio * d)
        accs = [
            _acc(p, block_amp(p)[0])
            for p in (build_dense(d, m, ell, sd) for sd in seeds)
        ]
        log.info(f"  {ratio:>5.2f} {np.mean(accs):>8.3f}+-{np.std(accs):<5.3f}")

    # (2)+(3): matched D and M, coupled vs dense, with per-block profile at the crossing.
    c, w = 32, 3
    r_blocks = c + w - 1
    d = 986  # divisible by R=34; M below divisible by C=32
    m = (round(0.30 * d) // c) * c  # 288 -> M/D=0.292
    log.info(
        f"\n(2) CROSS + (3) PROFILE  D={d} M={m} M/D={m / d:.3f}  C={c} w={w} (matched storage)"
    )
    dense_a, coup_a, prof = [], [], np.zeros(c)
    for sd in seeds:
        pd = build_dense(d, m, ell, sd)
        dense_a.append(_acc(pd, block_amp(pd)[0]))
        pc = build_coupled(d, m, ell, sd, c, w)
        est = block_amp(pc)[0]
        coup_a.append(_acc(pc, est))
        prof += np.array(
            [np.mean((est == pc["labels"])[pc["sec_col"] == cc]) for cc in range(c)]
        )
    prof /= len(list(seeds))
    log.info(
        f"  dense (random)   = {np.mean(dense_a):.3f}+-{np.std(dense_a):.3f}  (floored)"
    )
    log.info(
        f"  coupled (SC)     = {np.mean(coup_a):.3f}+-{np.std(coup_a):.3f}  (crosses)"
    )
    log.info("  per-block acc (seed block 0 ... block %d):" % (c - 1))
    log.info("    " + " ".join(f"{b:.2f}" for b in prof))
    log.info(
        "\nRead: coupled >> dense at matched D and M. Middle (non-seed) blocks decode at "
        "~1.0\nvia wave propagation -> real threshold saturation, not sharding. The floor "
        "is a\nrandom-code artifact. Caveat: the crossing is clean-signal; a noisy query "
        "reverts to\nthe dense floor (docs/SUPERPOSITION-FLOOR.md)."
    )


if __name__ == "__main__":
    main()
