"""10-minute ANN proxies on held-out train vectors (the test set is never read). Each proxy
writes one entry into benchmarks/results/proxies_2026-10.json under `ann.<name>` with the
question it answers, the numbers and the decision rule. Usage:
  uv run --with numpy python ann_proxies.py <set> <graph-cache> <proxy>[,<proxy>...] [--queries N]
Proxies: oracle_stop, adaptive_ef, entry_point, rerank_ceiling, quantizers, pool_usage, screen."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qgraph_sim import Graph, VGraphSim, load_dataset, quantize, recall  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmarks/results/proxies_2026-10.json"


def write(section: str, name: str, payload: dict):
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"note": "10-minute proxies on dev/held-out data only; counts and recalls, never timings"}
    doc.setdefault(section, {})[name] = payload
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {section}.{name}")


def boot_ci(x, seed=20261007, n=1000):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def oracle_stop(sim, q, truth, ef, rr, nq):
    """How many evaluations after the last true top-10 neighbour entered the pool are wasted,
    and what fixed stop rules (k-th best unchanged for t expansions) recover of that."""
    ratios, total, orac, found = [], [], [], []
    rule_evals = {t: [] for t in (16, 32, 64, 96, 128, 192, 256, 384, 512)}
    rule_rec = {t: [] for t in rule_evals}
    slack_evals = {e: [] for e in (0.01, 0.02, 0.05, 0.1, 0.2)}
    slack_rec = {e: [] for e in slack_evals}
    recs = []
    for i in range(nq):
        ids, ev, (steps, seen, pool, ins) = sim.search(q[i], ef, rerank=rr, trace=True)
        recs.append(recall(ids, truth[i]))
        T = truth[i][:10]
        # `seen` lists ids in insertion order (the entry first); steps[s][5] is the number
        # inserted after step s, so id at insertion index j (>= 1) entered at the first step
        # whose inserted count is >= j
        at = {int(u): j for j, u in enumerate(seen.tolist())}
        ins_cum = np.array([s[5] for s in steps])
        last = 0
        for t in T.tolist():
            if t in at:
                j = at[t]
                st = int(np.searchsorted(ins_cum, j, side="left")) if j > 0 else 0
                last = max(last, min(st, len(steps) - 1))
        found.append(sum(int(t in at) for t in T.tolist()) / 10)
        oe = steps[min(last, len(steps) - 1)][2] if steps else ev
        orac.append(oe)
        total.append(ev)
        ratios.append(oe / ev)
        # relative-slack rule (admissible-bound mapping): stop when the popped candidate's
        # estimate is worse than the 10th best by more than eps * |10th best|
        for e in slack_evals:
            stop = len(steps) - 1
            for j, st_ in enumerate(steps):
                if j > 0 and st_[1] > st_[4] + e * abs(st_[4]):
                    stop = j
                    break
            slack_evals[e].append(steps[stop][2] + min(rr, ef))
            seen_by = set(seen[: 1 + int(steps[stop][5])].tolist())
            slack_rec[e].append(sum(int(t_ in seen_by) for t_ in T.tolist()) / 10)
        # fixed rule: stop when the 10th-best estimate has not improved for t expansions
        kth = [s[4] for s in steps]
        for t in rule_evals:
            stop = len(steps) - 1
            since = 0
            best = np.inf
            for j, v in enumerate(kth):
                if v < best - 1e-12:
                    best, since = v, 0
                else:
                    since += 1
                if since >= t:
                    stop = j
                    break
            rule_evals[t].append(steps[stop][2] + min(rr, ef))
            # recall under the rule: true neighbours inserted by `stop` (the pool holds every
            # inserted id while fewer than ef were inserted, which is the case at these stops)
            seen_by = set(seen[: 1 + int(steps[stop][5])].tolist())
            rule_rec[t].append(sum(int(t_ in seen_by) for t_ in T.tolist()) / 10)
    out = {
        "question": "If the search stopped the moment the last true top-10 neighbour entered the pool, what fraction of the evaluations would remain? (upper bound of any early-termination policy)",
        "ef": ef, "rerank": rr, "queries": nq,
        "recall_fixed_ef": float(np.mean(recs)),
        "evals_fixed_ef_mean": float(np.mean(total)),
        "oracle_evals_mean": float(np.mean(orac)),
        "oracle_over_fixed_ratio_mean": float(np.mean(ratios)),
        "oracle_ratio_ci95": boot_ci(ratios),
        "oracle_ratio_percentiles": {p: float(np.percentile(ratios, p)) for p in (10, 25, 50, 75, 90)},
        "true_neighbours_ever_seen": float(np.mean(found)),
        "stop_rule_kth_unchanged": {str(t): {"evals_mean": float(np.mean(rule_evals[t])), "recall_upper_bound": float(np.mean(rule_rec[t]))} for t in rule_evals},
        "stop_rule_relative_slack": {str(e): {"evals_mean": float(np.mean(slack_evals[e])), "recall_upper_bound": float(np.mean(slack_rec[e]))} for e in slack_evals},
        "oracle_evals_percentiles": {p: float(np.percentile(orac, p)) for p in (50, 75, 90, 95, 99)},
        "decision_rule": "early termination is worth a Rust implementation only if a fixed rule keeps recall within 0.005 of the fixed-ef row at >= 15% fewer evaluations (the A1 kill test), or the oracle ratio is <= 0.6 so a learned policy has room",
    }
    return out


def adaptive_ef(sim, q, truth, nq):
    efs = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    need = []
    rec_at = {ef: [] for ef in efs}
    for i in range(nq):
        got = None
        for ef in efs:
            ids, ev, _ = sim.search(q[i], ef, rerank=64)
            r = recall(ids, truth[i])
            rec_at[ef].append(r)
            if got is None and r >= 1.0:
                got = ef
        need.append(got if got is not None else 2048)
    need = np.array(need)
    return {
        "question": "Is query difficulty heavy-tailed, i.e. does one ef per query fit all queries?",
        "queries": nq,
        "min_ef_for_full_recall_percentiles": {p: int(np.percentile(need, p)) for p in (10, 25, 50, 75, 90, 95, 99)},
        "share_needing_over_512": float((need > 512).mean()),
        "share_satisfied_by_64": float((need <= 64).mean()),
        "recall_by_ef": {str(ef): float(np.mean(rec_at[ef])) for ef in efs},
        "decision_rule": "if the median query is satisfied by ef <= 64 while recall 0.95 needs ef >= 384, the fixed ef pays for the tail and a per-query budget (adaptive stop) is the lever",
    }


def entry_point(sim, keep, q, truth, nq, seed=20261007):
    rng = np.random.default_rng(seed)
    out = {"question": "Does the entry vertex matter? Compare the upper-layer descent (current) with the nearest of C k-means centroids' representative vertices and with the true nearest neighbour as entry (oracle).", "queries": nq}
    # k-means entries (5 Lloyd iterations on a 50k sample, assignment of all rows)
    results = {}
    for C in (256, 1024, 4096):
        samp = keep[rng.choice(keep.shape[0], 50000, replace=False)]
        cent = samp[rng.choice(50000, C, replace=False)].copy()
        for _ in range(5):
            a = np.argmax(samp @ cent.T, axis=1)
            for c in range(C):
                m = samp[a == c]
                if len(m):
                    cent[c] = m.mean(axis=0)
            cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
        # representative vertex: nearest index row to each centroid
        rep = np.empty(C, dtype=np.int64)
        for s in range(0, C, 64):
            rep[s:s + 64] = np.argmax(cent[s:s + 64] @ keep.T, axis=1)
        for ef, rr in ((128, 64), (384, 64)):
            recs, evs = [], []
            for i in range(nq):
                c = int(np.argmax(cent @ q[i]))
                ids, ev, _ = sim.search(q[i], ef, rerank=rr, entry=int(rep[c]))
                recs.append(recall(ids, truth[i]))
                evs.append(ev + C)  # C centroid dots charged
            results[f"kmeans{C}_ef{ef}"] = {"recall": float(np.mean(recs)), "evals_incl_centroids": float(np.mean(evs))}
    for ef, rr in ((128, 64), (384, 64)):
        recs, evs, desc = [], [], []
        for i in range(nq):
            qc = sim.encode_query(q[i])
            e, de = sim.descend(qc)
            desc.append(de)
            ids, ev, _ = sim.search(q[i], ef, rerank=rr)
            recs.append(recall(ids, truth[i]))
            evs.append(ev)
        results[f"current_descent_ef{ef}"] = {"recall": float(np.mean(recs)), "evals": float(np.mean(evs)), "descent_evals": float(np.mean(desc))}
        recs, evs = [], []
        for i in range(nq):
            ids, ev, _ = sim.search(q[i], ef, rerank=rr, entry=int(truth[i][0]))
            recs.append(recall(ids, truth[i]))
            evs.append(ev)
        results[f"oracle_entry_ef{ef}"] = {"recall": float(np.mean(recs)), "evals": float(np.mean(evs))}
    out["results"] = results
    out["decision_rule"] = "the entry point is a lever only if the oracle entry cuts evaluations by >= 20% at equal recall; k-means entries are adopted only if they beat the descent on evals at equal or better recall"
    return out


def rerank_ceiling(sim, keep, q, truth, nq):
    out = {"question": "How much recall does the 8-bit code + 8-bit residual re-rank lose against exact float re-rank over the same pool, and how does recall depend on the re-rank count?", "queries": nq}
    res = {}
    for ef in (128, 384):
        for rr in (10, 16, 32, 64, 128):
            recs, ex = [], []
            for i in range(nq):
                ids, ev, (_, _, pool, _) = sim.search(q[i], ef, rerank=rr)
                recs.append(recall(ids, truth[i]))
                take = pool[:max(rr, 10)]
                exact = take[np.argsort(-(keep[take] @ q[i]), kind="stable")][:10]
                ex.append(recall(exact, truth[i]))
            res[f"ef{ef}_rr{rr}"] = {"recall_code_residual": float(np.mean(recs)), "recall_exact_float": float(np.mean(ex))}
        recs = []
        for i in range(nq):
            _, _, (_, _, pool, _) = sim.search(q[i], ef, rerank=0)
            recs.append(len(set(pool.tolist()) & set(truth[i][:10].tolist())) / 10)
        res[f"ef{ef}_pool_ceiling"] = float(np.mean(recs))
    out["results"] = res
    out["decision_rule"] = "a better re-rank code is a lever only if exact float re-rank beats code+residual by >= 0.005 at the chosen rerank count; otherwise the pool (ef) bounds recall"
    return out


def quantizers(keep, q, truth, nq, seed=20261007):
    """Estimator fidelity without the graph: among the true top-512 of each query, does the
    estimator keep the true top-10 inside its top-r (r = 16, 64)? Each variant is (code, scale,
    offset) so that est = scale * (code . q) + offset * sum(q)."""
    rng = np.random.default_rng(seed)
    dim = keep.shape[1]
    mean = keep.mean(axis=0)
    x = (keep - mean).astype(np.float32)
    R = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    xr = x @ R
    cand = truth[:nq, :512]
    rows = np.unique(cand)  # only these rows are ever scored
    pos = {int(r): i for i, r in enumerate(rows.tolist())}

    def sym(xx, levels):
        c, s = quantize(xx[rows], levels)
        return c.astype(np.float32), s, np.zeros_like(s)

    def asym(xx, levels):
        v = xx[rows]
        lo, hi = v.min(axis=1), v.max(axis=1)
        s = np.where(hi > lo, (hi - lo) / levels, 1.0).astype(np.float32)
        c = np.clip(np.rint((v - lo[:, None]) / s[:, None]), 0, levels).astype(np.float32)
        return c, s, lo.astype(np.float32)

    def q8(v):
        c, s = quantize(v[None], 127)
        return c[0].astype(np.float32) * s[0]

    specs = {
        "sym8_q8 (current)": (sym, 127, False, q8),
        "sym8_qfloat": (sym, 127, False, lambda v: v),
        "asym8_lvq_q8": (asym, 255, False, q8),
        "sym8_rotated_q8": (sym, 127, True, q8),
        "asym8_lvq_rotated_q8": (asym, 255, True, q8),
        "sym4_q8": (sym, 7, False, q8),
        "sym4_rotated_q8": (sym, 7, True, q8),
        "asym4_lvq_q8": (asym, 15, False, q8),
        "asym4_lvq_rotated_q8": (asym, 15, True, q8),
        "sym6_q8": (sym, 31, False, q8),
    }
    variants = {}
    for name, (mk, levels, rot, qf) in specs.items():
        c, s, off = mk(xr if rot else x, levels)
        keep10 = {16: [], 64: []}
        for i in range(nq):
            ids = cand[i]
            qv = qf((q[i] @ R if rot else q[i]).astype(np.float32))
            sub = np.array([pos[int(t)] for t in ids.tolist()])
            e = s[sub] * (c[sub] @ qv) + off[sub] * qv.sum()
            order = ids[np.argsort(-e, kind="stable")]
            T = set(ids[:10].tolist())
            for r in keep10:
                keep10[r].append(len(T & set(order[:r].tolist())) / 10)
        variants[name] = {"true_top10_within_top16_by_est": float(np.mean(keep10[16])), "within_top64": float(np.mean(keep10[64]))}
        print(name, variants[name])
    return {"question": "Which code (symmetric max-abs 8-bit as now; LVQ min/max; rotation; 4/6-bit) orders the true neighbours best at equal bytes, measured on the true top-512 of each query?", "queries": nq, "results": variants,
            "decision_rule": "adopt a code only if it keeps >= 0.005 more of the true top-10 inside the top-16 by estimate than the current code at the same bits; a 4-bit code is a candidate default only if it is within 0.005 of 8-bit"}


def screen(sim, keep, q, truth, nq):
    from qgraph_sim import SignScreen
    rows = {}
    cfgs = [(ef, rr) for ef, rr in ((128, 64), (384, 64))] if sim.g.dim >= 200 else [(256, 16), (768, 16)]
    for bits in (1, 4):
        for m in (0.0, 1.0, 2.0, 3.0):
            sc = SignScreen(sim, keep, m, bits=bits)
            for ef, rr in cfgs:
                recs, ev8, evc = [], [], []
                for i in range(nq):
                    ids, ev, _ = sim.search(q[i], ef, rerank=rr, screen=sc)
                    recs.append(recall(ids, truth[i]))
                    ev8.append(ev)
                    evc.append(sim.last["cheap_evals"])
                rows[f"{bits}bit_m{m}_ef{ef}"] = {"recall": float(np.mean(recs)), "evals_8bit": float(np.mean(ev8)), "evals_cheap": float(np.mean(evc)), "sigma": sc.sigma}
                print(rows[f"{bits}bit_m{m}_ef{ef}"])
    for ef, rr in cfgs:
        recs, evs = [], []
        for i in range(nq):
            ids, ev, _ = sim.search(q[i], ef, rerank=rr)
            recs.append(recall(ids, truth[i]))
            evs.append(ev)
        rows[f"no_screen_ef{ef}"] = {"recall": float(np.mean(recs)), "evals_8bit": float(np.mean(evs))}
    return {"question": "If every fresh neighbour is first scored with a 1-bit (32 B) or 4-bit (128 B) code and only those whose optimistic estimate could enter the pool get the 8-bit (256 B) estimate, how many 8-bit evaluations are skipped at what recall cost?", "queries": nq, "results": rows,
            "decision_rule": "a two-level estimator is a lever if some margin skips >= 60% of the 8-bit evaluations at a recall loss <= 0.003; the byte saving is then (skipped x (256 - cheap_bytes)) / total"}


def pool_usage(sim, q, truth, nq):
    rows = {}
    for ef in (128, 384):
        fresh_frac, ins_frac, steps_n = [], [], []
        for i in range(nq):
            ids, ev, (steps, seen, pool, ins) = sim.search(q[i], ef, rerank=64, trace=True)
            nfresh = sum(s[3] for s in steps)
            fresh_frac.append(nfresh / (len(steps) * sim.g.degree))
            ins_frac.append(ins / max(nfresh, 1))
            steps_n.append(len(steps))
        rows[f"ef{ef}"] = {"expansions_mean": float(np.mean(steps_n)), "fresh_share_of_degree": float(np.mean(fresh_frac)), "inserted_share_of_evaluated": float(np.mean(ins_frac))}
    return {"question": "Of the neighbours scanned per expansion, how many are new (not visited) and how many enter the pool? Low shares mean a cheaper screen (4-bit or 1-bit first pass) could skip most 8-bit evaluations.", "queries": nq, "results": rows,
            "decision_rule": "a two-level estimator is a lever only if <= 30% of evaluated neighbours enter the pool"}


def main():
    a = sys.argv[1:]
    name, cache, proxies = a[0], Path(a[1]), a[2].split(",")
    nq = int(a[a.index("--queries") + 1]) if "--queries" in a else 500
    t = time.time()
    keep, q, truth = load_dataset(name)
    g = Graph(cache)
    assert g.n == keep.shape[0]
    sim = VGraphSim(g, keep)
    print(f"loaded {name} n={g.n} degree={g.degree} in {time.time() - t:.0f}s")
    for p in proxies:
        t = time.time()
        if p == "oracle_stop":
            for ef, rr in ((128, 64), (384, 64)) if name.startswith("nytimes") else ((256, 16), (768, 16)):
                r = oracle_stop(sim, q, truth, ef, rr, nq)
                r["minutes"] = round((time.time() - t) / 60, 1)
                r["dataset"] = name
                write("ann", f"oracle_stop_{name}_ef{ef}", r)
        elif p == "adaptive_ef":
            r = adaptive_ef(sim, q, truth, min(nq, 300))
        elif p == "entry_point":
            r = entry_point(sim, keep, q, truth, nq)
        elif p == "rerank_ceiling":
            r = rerank_ceiling(sim, keep, q, truth, nq)
        elif p == "quantizers":
            r = quantizers(keep, q, truth, nq)
        elif p == "pool_usage":
            r = pool_usage(sim, q, truth, nq)
        elif p == "screen":
            r = screen(sim, keep, q, truth, nq)
        else:
            raise SystemExit(f"unknown proxy {p}")
        if p != "oracle_stop":
            r["minutes"] = round((time.time() - t) / 60, 1)
            r["dataset"] = name
            write("ann", f"{p}_{name}", r)
        print(f"{p}: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
