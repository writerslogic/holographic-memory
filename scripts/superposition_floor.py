#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Superposition-recovery floor: computational, not information-theoretic.
#
# A VSA memory superposes M key-value pairs into ONE vector s = sum_i bind(k_i, v_i)
# (bind = circular convolution, keys UNITARY so unbind is the EXACT inverse -- the
# strong Plate/HRR baseline, not a strawman). Values are drawn from a SHARED L-entry
# codebook, reused across the M facts, so crosstalk is real (this is a genuine
# capacity test, not one-unique-symbol-per-fact). Given s and a key, recover the value.
#
# This script reproduces two facts that together characterise the "superposition
# floor":
#   (1) FLOOR. Blind recovery -- naive HRR unbind+cleanup, and Approximate Message
#       Passing (AMP, the Bayes-optimal iterative decoder) -- saturates near
#       M/D ~ 0.25-0.27, dimension-invariant. No blind polynomial decoder tested in
#       the cold-exploration sweep (AMP, SIC/peeling, modern-Hopfield, learned/
#       unrolled, structured codebooks, spatial coupling) crosses it; see
#       docs/SUPERPOSITION-FLOOR.md.
#   (2) THE GAP IS COMPUTATIONAL. Initialise a hard Gauss-Seidel solver AT the true
#       support ("genie") and it stays there (genie_stay = 1.0) all the way to
#       M/D = 0.75 -- the true solution is a stable, verifiable fixed point far past
#       the floor. Best-of-8 random restarts never find it. So the barrier is FINDING
#       the solution blindly (a statistical-to-computational hard phase), not its
#       existence or identifiability. This refutes a strict "info-theoretically
#       impossible past ~0.5D" reading and matches the SPARC/CDMA hard-phase picture.
#
# Soft/graceful readout (AMP posterior mass on the true entry) degrades continuously
# across the transition -- no hard cliff -- which an exact error-correcting code would
# destroy. That is the property the holographic layer preserves.
#
# Self-contained (numpy only). Deterministic (fixed seeds 0..4).
# Run: uv run python scripts/superposition_floor.py

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("floor")


def unitary_keys(m, d, rng):
    """M random unitary keys: unit-magnitude spectrum -> exact-inverse binding."""
    half = d // 2 + 1
    ph = rng.uniform(0, 2 * np.pi, size=(m, half))
    ph[:, 0] = 0.0
    if d % 2 == 0:
        ph[:, -1] = 0.0
    return np.fft.irfft(np.exp(1j * ph), n=d, axis=1)


def build(d, m, ell, seed):
    """Superpose M facts; return keys, codebook, true labels, s, and CS matrix Phi."""
    rng = np.random.default_rng(seed)
    keys = unitary_keys(m, d, rng)
    codebook = rng.standard_normal((ell, d))
    codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
    labels = rng.integers(0, ell, size=m)  # reused across facts -> real crosstalk
    phi = np.empty((d, m * ell))
    cb_f = np.fft.rfft(codebook, axis=1)
    for i in range(m):
        cols = np.fft.irfft(np.fft.rfft(keys[i])[None, :] * cb_f, n=d, axis=1)
        phi[:, i * ell : (i + 1) * ell] = cols.T
    s = phi[:, labels + ell * np.arange(m)].sum(axis=1)
    return keys, codebook, labels, s, phi


def naive_recover(keys, codebook, s):
    """Strong HRR baseline: exact-inverse unbind of each key, nearest codebook entry."""
    m, d = keys.shape
    s_f = np.fft.rfft(s)
    cbn = codebook / np.linalg.norm(codebook, axis=1, keepdims=True)
    est = np.empty(m, dtype=int)
    for i in range(m):
        kinv = np.roll(keys[i][::-1], 1)  # exact inverse for a unitary key
        v = np.fft.irfft(s_f * np.fft.rfft(kinv), n=d)
        est[i] = np.argmax(cbn @ (v / (np.linalg.norm(v) + 1e-12)))
    return est


def amp_recover(phi, s, m, ell, n_iter=200, damp=0.3):
    """Section-softmax AMP (Bayes-optimal one-hot denoiser). Returns est, posterior."""
    d, n = phi.shape
    x = np.zeros(n)
    z = s.copy()
    last_p = np.ones((m, ell)) / ell
    for _ in range(n_iter):
        tau2 = max(np.sum(z * z) / d, 1e-12)
        r = (phi.T @ z + x).reshape(m, ell) / tau2
        r -= r.max(axis=1, keepdims=True)
        p = np.exp(r)
        p /= p.sum(axis=1, keepdims=True)
        x_new = p.reshape(-1)
        div = np.sum(p - p * p) / tau2
        z_new = s - phi @ x_new + (1.0 / d) * div * z
        z = damp * z + (1 - damp) * z_new
        x = damp * x + (1 - damp) * x_new
        last_p = p
    return np.argmax(last_p, axis=1), last_p


def hard_solve(phi, s, m, ell, init, n_iter=300):
    """Gauss-Seidel hard interference cancellation from an initial support `init`."""
    est = init.copy()
    recon = phi[:, est + ell * np.arange(m)].sum(axis=1)
    order = np.arange(m)
    for it in range(n_iter):
        changed = 0
        np.random.default_rng(it).shuffle(order)
        for i in order:
            recon -= phi[:, i * ell + est[i]]
            sims = phi[:, i * ell : (i + 1) * ell].T @ (s - recon)
            best = int(np.argmax(sims))
            changed += best != est[i]
            est[i] = best
            recon += phi[:, i * ell + est[i]]
        if changed == 0:
            break
    return est


def floor_table(dims, ell, ratios, seeds):
    log.info("(1) BLIND RECOVERY FLOOR  |  naive HRR vs AMP, mean+-std over seeds")
    for d in dims:
        log.info(f"\n  D={d} L={ell}")
        log.info(f"  {'M/D':>5} {'naive':>14} {'amp':>14} {'amp_soft':>9}")
        for ratio in ratios:
            m = max(1, round(ratio * d))
            nv, ap, sf = [], [], []
            for sd in seeds:
                keys, cb, labels, s, phi = build(d, m, ell, sd)
                nv.append(np.mean(naive_recover(keys, cb, s) == labels))
                est, p = amp_recover(phi, s, m, ell)
                ap.append(np.mean(est == labels))
                sf.append(np.mean(p[np.arange(m), labels]))
            nv, ap, sf = map(np.array, (nv, ap, sf))
            log.info(
                f"  {ratio:>5.2f} {nv.mean():>6.3f}+-{nv.std():<5.3f} "
                f"{ap.mean():>6.3f}+-{ap.std():<5.3f} {sf.mean():>7.3f}"
            )


def gap_table(d, ell, ratios, seeds):
    log.info(
        "\n(2) THE GAP IS COMPUTATIONAL  |  genie-init stays; random-init never finds it"
    )
    log.info(f"\n  D={d} L={ell}")
    log.info(f"  {'M/D':>5} {'genie_stay':>12} {'rand_best_of_8':>16}")
    for ratio in ratios:
        m = max(1, round(ratio * d))
        gs, hb = [], []
        for sd in seeds:
            keys, cb, labels, s, phi = build(d, m, ell, sd)
            gs.append(np.mean(hard_solve(phi, s, m, ell, labels.copy()) == labels))
            rng = np.random.default_rng(1000 + sd)
            best_res, best_acc = np.inf, 0.0
            for _ in range(8):
                est = hard_solve(phi, s, m, ell, rng.integers(0, ell, size=m))
                res = np.linalg.norm(s - phi[:, est + ell * np.arange(m)].sum(axis=1))
                if res < best_res:
                    best_res, best_acc = res, np.mean(est == labels)
            hb.append(best_acc)
        gs, hb = np.array(gs), np.array(hb)
        log.info(
            f"  {ratio:>5.2f} {gs.mean():>12.3f} {hb.mean():>10.3f}+-{hb.std():<5.3f}"
        )


def main():
    ell = 64
    seeds = range(5)
    floor_table([256, 512], ell, [0.10, 0.20, 0.25, 0.30, 0.40], seeds)
    gap_table(256, ell, [0.20, 0.30, 0.40, 0.50, 0.60, 0.75], seeds)
    log.info(
        "\nRead: AMP holds ~100% to M/D~0.25 then falls (the floor); genie_stay=1.0 to "
        "0.75\nwhile random-init collapses -> the true solution is stable/verifiable far "
        "past the\nfloor but unfindable blindly. Computational hard phase, not an "
        "information wall."
    )


if __name__ == "__main__":
    main()
