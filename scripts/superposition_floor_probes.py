#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Attacking the assumptions behind the superposition floor (docs/SUPERPOSITION-FLOOR.md).
# The claim "no poly-time decoder beats M/D~0.27" rests on AMP being optimal and on a
# statistical-to-computational HARD-PHASE conjecture -- both are assumptions, not proven
# theorems, and the shared codebook VIOLATES AMP's iid state-evolution premise. This
# script tries to break 0.27 with stronger solvers, and maps the true identifiability
# ceiling (assumption #2). Every solver is validated in the easy regime (must be ~100%
# at M/D=0.1) before any past-floor number is trusted.
#
# Self-contained apart from the shared harness. Deterministic (seeds 0..4).
# Run: uv run python scripts/superposition_floor_probes.py

import logging

import numpy as np

from superposition_floor import amp_recover, build, hard_solve, naive_recover

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("probes")


def acc(est, labels):
    return float(np.mean(est == labels))


def resid(phi, s, est, ell, m):
    return float(np.linalg.norm(s - phi[:, est + ell * np.arange(m)].sum(axis=1)))


# --- Solver A: AMP followed by hard local-search refinement (best init we can cheaply get)
def amp_then_refine(phi, s, m, ell):
    est0, _ = amp_recover(phi, s, m, ell)
    return hard_solve(phi, s, m, ell, est0.copy())


# --- Solver B: many random restarts of hard local search, keep min-residual
def multistart(phi, s, m, ell, restarts, seed):
    rng = np.random.default_rng(7000 + seed)
    best, best_r = None, np.inf
    # seed one restart from AMP, the rest random
    cands = [amp_recover(phi, s, m, ell)[0]]
    cands += [rng.integers(0, ell, size=m) for _ in range(restarts - 1)]
    for init in cands:
        est = hard_solve(phi, s, m, ell, init.copy())
        r = resid(phi, s, est, ell, m)
        if r < best_r:
            best_r, best = r, est
    return best


# --- Solver C: annealed AMP (temperature schedule) with more iterations
def amp_annealed(phi, s, m, ell, n_iter=400, damp=0.3):
    d, n = phi.shape
    x = np.zeros(n)
    z = s.copy()
    p = np.ones((m, ell)) / ell
    for t in range(n_iter):
        beta = min(1.0, 0.2 + t / (n_iter * 0.5))  # anneal temperature 0.2 -> 1.0
        tau2 = max(np.sum(z * z) / d, 1e-12) / beta
        r = (phi.T @ z + x).reshape(m, ell) / tau2
        r -= r.max(axis=1, keepdims=True)
        p = np.exp(r)
        p /= p.sum(axis=1, keepdims=True)
        x_new = p.reshape(-1)
        div = np.sum(p - p * p) / tau2
        z_new = s - phi @ x_new + (1.0 / d) * div * z
        z = damp * z + (1 - damp) * z_new
        x = damp * x + (1 - damp) * x_new
    return np.argmax(p, axis=1)


# --- Solver D: VAMP / OAMP (LMMSE linear stage + section-softmax denoiser).
# AMP assumes iid Gaussian Phi; VAMP is exact for right-rotationally-invariant Phi and
# is the natural fix if the shared-codebook column correlation is what caps AMP.
def _sec_softmax(r, gam, m, ell):
    rr = (r * gam).reshape(m, ell)
    rr -= rr.max(axis=1, keepdims=True)
    p = np.exp(rr)
    p /= p.sum(axis=1, keepdims=True)
    x = p.reshape(-1)
    alpha = float(np.mean(gam * (p - p * p)))  # avg divergence d xhat / d r
    return x, max(alpha, 1e-9)


def vamp_recover(phi, s, m, ell, n_iter=150, damp=0.6, wvar=1e-6):
    d, n = phi.shape
    u, sv, vt = np.linalg.svd(phi, full_matrices=False)  # thin: sv length min(d,n)=d
    uts = u.T @ s
    sv2 = sv**2
    r1 = np.zeros(n)
    gam1 = 1.0
    p = np.ones((m, ell)) / ell
    for _ in range(n_iter):
        # denoiser (nonlinear) stage
        x1, a1 = _sec_softmax(r1, gam1, m, ell)
        p = x1.reshape(m, ell)
        eta1 = gam1 / a1
        gam2 = max(eta1 - gam1, 1e-8)
        r2 = (eta1 * x1 - gam1 * r1) / gam2
        # LMMSE (linear) stage:  x2 = argmin (1/wvar)||s-phi x||^2 + gam2||x-r2||^2
        vtr2 = vt @ r2
        coeff = (sv / wvar) / (gam2 + sv2 / wvar)  # length d
        x2 = r2 + vt.T @ (coeff * (uts - sv * vtr2))
        # avg divergence: (gam2/n)[ sum_k 1/(gam2+sv_k^2/wvar) + (n-d)/gam2 ]
        a2 = (gam2 / n) * (np.sum(1.0 / (gam2 + sv2 / wvar)) + (n - d) / gam2)
        a2 = min(max(a2, 1e-9), 1 - 1e-9)
        eta2 = gam2 / a2
        gam1_new = max(eta2 - gam2, 1e-8)
        r1_new = (eta2 * x2 - gam2 * r2) / gam1_new
        r1 = damp * r1 + (1 - damp) * r1_new
        gam1 = damp * gam1 + (1 - damp) * gam1_new
    return np.argmax(p, axis=1)


def solver_table(d, ell, ratios, seeds):
    solvers = {
        "naive": lambda phi, s, m, keys, cb: naive_recover(keys, cb, s),
        "AMP": lambda phi, s, m, keys, cb: amp_recover(phi, s, m, ell)[0],
        "AMP+refine": lambda phi, s, m, keys, cb: amp_then_refine(phi, s, m, ell),
        "multistart32": lambda phi, s, m, keys, cb: multistart(phi, s, m, ell, 32, m),
        "AMP-anneal": lambda phi, s, m, keys, cb: amp_annealed(phi, s, m, ell),
        "VAMP": lambda phi, s, m, keys, cb: vamp_recover(phi, s, m, ell),
    }
    log.info(
        f"SOLVER SHOOTOUT  D={d} L={ell}  (mean acc over {len(list(seeds))} seeds)"
    )
    log.info("  M/D  " + "".join(f"{name:>13}" for name in solvers))
    for ratio in ratios:
        m = max(1, round(ratio * d))
        row = {name: [] for name in solvers}
        for sd in seeds:
            keys, cb, labels, s, phi = build(d, m, ell, sd)
            for name, fn in solvers.items():
                row[name].append(acc(fn(phi, s, m, keys, cb), labels))
        log.info(
            f"  {ratio:>4.2f} " + "".join(f"{np.mean(row[n]):>13.3f}" for n in solvers)
        )


def identifiability_ceiling(d, ell, ratios, seeds):
    # Assumption #2: is 0.75 really the identifiability ceiling? Genie-init stability +
    # is truth the UNIQUE min residual (vs the best random-restart's residual)?
    log.info(f"\nIDENTIFIABILITY CEILING  D={d} L={ell}")
    log.info(
        f"  {'M/D':>5} {'genie_stay':>11} {'truth_res':>10} {'rand_res':>10} {'unique?':>8}"
    )
    for ratio in ratios:
        m = max(1, round(ratio * d))
        gs, tr, rr, uq = [], [], [], []
        for sd in seeds:
            keys, cb, labels, s, phi = build(d, m, ell, sd)
            est_g = hard_solve(phi, s, m, ell, labels.copy())
            gs.append(acc(est_g, labels))
            tr.append(resid(phi, s, labels, ell, m))
            rng = np.random.default_rng(2000 + sd)
            best_r = min(
                resid(
                    phi,
                    s,
                    hard_solve(phi, s, m, ell, rng.integers(0, ell, size=m)),
                    ell,
                    m,
                )
                for _ in range(8)
            )
            rr.append(best_r)
            uq.append(
                best_r > 1e-6
            )  # any decoy reaching ~0 residual would break uniqueness
        log.info(
            f"  {ratio:>5.2f} {np.mean(gs):>11.3f} {np.mean(tr):>10.4f} "
            f"{np.mean(rr):>10.3f} {float(np.mean(uq)):>8.2f}"
        )


def main():
    ell = 64
    seeds = range(5)
    # Past-floor solver shootout: does ANYTHING cross ~0.27? Validate at 0.10 first.
    solver_table(256, ell, [0.10, 0.25, 0.30, 0.35, 0.40, 0.50], seeds)
    # Map the identifiability ceiling past the 0.75 genie stability radius.
    identifiability_ceiling(256, ell, [0.5, 0.75, 1.0, 1.25, 1.5], seeds)
    log.info(
        "\nRead: at M/D=0.10 every solver must be ~1.0 (validation). Past 0.27, if any "
        "column\nstays high the hard-phase assumption is FALSE for that solver. "
        "genie_stay dropping\nbelow 1.0 marks where truth stops being a stable fixed "
        "point (the real IT ceiling)."
    )


if __name__ == "__main__":
    main()
