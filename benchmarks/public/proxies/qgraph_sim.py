"""numpy model of the VGraph search (`src/core/qgraph/vertex.rs`) over a graph read from the
HMSGRF02 cache, for 10-minute proxies on held-out train vectors. It reproduces the encoding
(mean-centred 8-bit symmetric codes, one scale per vertex, 8-bit residual), the quantized query,
the upper-layer descent and the best-first loop with the ef-pool termination rule, and records
every expansion so that oracle questions (when were the true neighbours found?) can be asked.
Timings from this module mean nothing; only counts and recalls are used."""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np

NONE = np.uint32(0xFFFF_FFFF)
HOLDOUT = 2000


def load_dataset(name: str, holdout: int = HOLDOUT, cache=Path.home() / ".cache/hms-bench"):
    """Held-out split exactly as `public-bench ann-qgraph --holdout N` makes it (every
    (n // N)-th train row is a query, the rest are the index; rows L2-normalized). Returns the
    normalized index rows, the normalized queries and the exact top-100 ids by inner product."""
    import json
    d = cache / name
    meta = json.loads((d / "meta.json").read_text())
    dim = int(meta["dim"])
    rows = np.fromfile(d / "train.f32", dtype=np.float32).reshape(-1, dim)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1
    rows = rows / norms
    n = rows.shape[0]
    step = n // holdout
    idx = np.arange(n)
    is_q = (idx % step == 0)
    q_pos = np.flatnonzero(is_q)[:holdout]
    is_q[:] = False
    is_q[q_pos] = True
    queries = rows[is_q]
    keep = rows[~is_q]
    truth = exact_topk(keep, queries, 100)
    return keep, queries, truth


def exact_topk(keep: np.ndarray, queries: np.ndarray, k: int, block: int = 256) -> np.ndarray:
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for s in range(0, queries.shape[0], block):
        sc = queries[s:s + block] @ keep.T
        part = np.argpartition(-sc, k, axis=1)[:, :k]
        ps = np.take_along_axis(sc, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        out[s:s + block] = np.take_along_axis(part, order, axis=1)
    return out


class Graph:
    def __init__(self, path: Path):
        data = np.fromfile(path, dtype=np.uint8)
        assert data[:8].tobytes() == b"HMSGRF02", "not an HMSGRF02 cache"
        hdr = np.frombuffer(data[8:72].tobytes(), dtype="<u8")
        self.n, self.dim, self.degree = int(hdr[0]), int(hdr[1]), int(hdr[2])
        self.build_ef, self.alpha = int(hdr[3]), float(np.frombuffer(np.uint32(hdr[4]).tobytes(), "<f4")[0])
        self.seed, self.entry, self.fingerprint = int(hdr[5]), int(hdr[6]), int(hdr[7])
        off = 72

        def u32s(off):
            ln = int(np.frombuffer(data[off:off + 8].tobytes(), "<u8")[0])
            off += 8
            arr = np.frombuffer(data[off:off + 4 * ln].tobytes(), "<u4")
            return arr, off + 4 * ln

        lens, off = u32s(off)
        flat, off = u32s(off)
        self.adj = np.full((self.n, self.degree), NONE, dtype=np.uint32)
        starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        rows = np.repeat(np.arange(self.n), lens.astype(np.int64))
        cols = np.arange(len(flat)) - starts[rows]
        self.adj[rows, cols] = flat
        self.upper_ids, off = u32s(off)
        nl = int(np.frombuffer(data[off:off + 8].tobytes(), "<u8")[0])
        off += 8
        self.layers = []
        for _ in range(nl):
            l2, off = u32s(off)
            f2, off = u32s(off)
            st = np.concatenate([[0], np.cumsum(l2)]).astype(np.int64)
            self.layers.append([f2[st[i]:st[i + 1]] for i in range(len(l2))])
        self.deg = lens.astype(np.int64)


def quantize(x: np.ndarray, levels: int = 127):
    """Rows -> int8 codes and per-row scales (`quantize` in vertex.rs)."""
    m = np.abs(x).max(axis=1)
    scale = np.where(m == 0, 1.0, m / levels).astype(np.float32)
    code = np.clip(np.rint(x / scale[:, None]), -levels, levels).astype(np.int8)
    scale = np.where(m == 0, 0.0, scale).astype(np.float32)
    return code, scale


class VGraphSim:
    def __init__(self, g: Graph, keep: np.ndarray, residual: bool = True, bits: int = 8):
        self.g = g
        assert keep.shape[0] == g.n
        self.mean = keep.mean(axis=0).astype(np.float32)
        x = keep - self.mean
        levels = 127 if bits == 8 else 7
        self.code, self.scale = quantize(x, levels)
        self.code32 = self.code.astype(np.int32)
        self.residual = None
        if residual:
            r = x - self.scale[:, None] * self.code.astype(np.float32)
            self.rcode, self.rscale = quantize(r, 127)
        self.levels = levels

    def encode_query(self, q: np.ndarray):
        qc, _ = quantize(q[None], self.levels)
        return qc[0].astype(np.int32)

    def est(self, qc: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return -(self.scale[ids] * (self.code32[ids] @ qc).astype(np.float32))

    def fine(self, qf: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Re-rank score: the raw float query against code + residual (as `fine` in vertex.rs)."""
        ip = self.scale[ids] * (self.code[ids].astype(np.float32) @ qf)
        if self.residual is not False and hasattr(self, "rcode"):
            ip = ip + self.rscale[ids] * (self.rcode[ids].astype(np.float32) @ qf)
        return -ip

    def descend(self, qc):
        g = self.g
        if not g.layers:
            return g.entry, 0
        p = 0
        best = float(self.est(qc, np.array([g.upper_ids[0]]))[0])
        evals = 1
        for layer in reversed(g.layers):
            while True:
                nb = layer[p]
                if len(nb) == 0:
                    break
                d = self.est(qc, g.upper_ids[nb])
                evals += len(nb)
                moved = False
                for u, dd in zip(nb, d):  # sequential update as in Rust
                    if dd < best:
                        best, p, moved = float(dd), int(u), True
                if not moved:
                    break
        return int(g.upper_ids[p]), evals

    def search(self, q: np.ndarray, ef: int, k: int = 10, rerank: int = 0, trace: bool = False,
               entry: int | None = None, screen=None):
        """Returns (ids, evals, (steps, seen_order, pool_ids, inserted)): `steps` holds one
        (expanded_id, est, evals_after, fresh, kth_best_est, inserted_so_far) per expansion when `trace`;
        `seen_order` lists every id in the order it entered the pool; `inserted` counts them."""
        g = self.g
        qc = self.encode_query(q)
        qf = q.astype(np.float32)  # raw query: <q, x - mean> differs from <q, x> by a constant
        visited = np.zeros(g.n, dtype=bool)
        if entry is None:
            entry, evals = self.descend(qc)
        else:
            evals = 0
        visited[entry] = True
        e0 = float(self.est(qc, np.array([entry]))[0])
        evals += 1
        cand = [(e0, entry)]
        best = [(-e0, entry)]  # max-heap on est via negation: top is the farthest
        steps = []
        seen_order = [entry]
        inserted = 0
        screened = 0
        cheap_evals = 0
        qs = screen.encode(q) if screen is not None else None
        while cand:
            c = heapq.heappop(cand)
            if len(best) >= ef and c[0] > -best[0][0]:
                break
            nb = g.adj[c[1]]
            nb = nb[nb != NONE]
            fresh = nb[~visited[nb]]
            visited[fresh] = True
            if screen is not None and len(fresh) and len(best) >= ef:
                cheap_evals += len(fresh)
                lo = screen.est(qs, fresh) - screen.margin
                keep_mask = lo < -best[0][0]
                screened += int((~keep_mask).sum())
                fresh = fresh[keep_mask]
            ests = self.est(qc, fresh) if len(fresh) else np.zeros(0, dtype=np.float32)
            evals += len(fresh)
            for u, e in zip(fresh.tolist(), ests.tolist()):
                if len(best) < ef:
                    heapq.heappush(best, (-e, u))
                else:
                    if e >= -best[0][0]:
                        continue
                    heapq.heapreplace(best, (-e, u))
                heapq.heappush(cand, (e, u))
                seen_order.append(u)
                inserted += 1
            if trace:
                kth = sorted(-x[0] for x in best)[min(k, len(best)) - 1]
                steps.append((c[1], c[0], evals, len(fresh), kth, inserted))
        pool = sorted((-e, u) for e, u in best)
        pool_ids = np.array([u for _, u in pool], dtype=np.int64)
        take = min(max(rerank, k), len(pool_ids))
        if rerank > 0:
            f = self.fine(qf, pool_ids[:take])
            evals += take
            order = np.lexsort((pool_ids[:take], f))
            ids = pool_ids[:take][order][:k]
        else:
            ids = pool_ids[:k]
        self.last = {"screened": screened, "cheap_evals": cheap_evals}
        return ids, evals, (steps, np.array(seen_order, dtype=np.int64), pool_ids, inserted)


def recall(ids: np.ndarray, truth_row: np.ndarray, k: int = 10) -> float:
    return len(set(ids[:k].tolist()) & set(truth_row[:k].tolist())) / k


class SignScreen:
    """RaBitQ-style 1-bit screen: sign code of the rotated centred vector with a per-vertex
    factor so that est1 ~ <q, x - mean>; `margin` (in the same units) is subtracted before the
    comparison with the pool's worst entry, so a neighbour is skipped only if even its
    optimistic estimate cannot enter the pool."""

    def __init__(self, sim: VGraphSim, keep: np.ndarray, margin_sigmas: float, seed: int = 20261007, bits: int = 1):
        rng = np.random.default_rng(seed)
        d = keep.shape[1]
        self.R = np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32)
        x = (keep - sim.mean).astype(np.float32) @ self.R
        self.bits = bits
        if bits == 1:
            norm = np.linalg.norm(x, axis=1)
            u = x / np.maximum(norm, 1e-12)[:, None]
            self.code = np.where(u >= 0, 1, -1).astype(np.int8)
            uo = (np.abs(u).sum(axis=1) / np.sqrt(d))  # <u, sign(u)/sqrt(d)>
            self.f = (norm / np.maximum(uo, 1e-12) / np.sqrt(d)).astype(np.float32)
        else:
            levels = 7 if bits == 4 else 127
            self.code, self.f = quantize(x, levels)
        self.sim = sim
        # error scale: std of (est1 - est8) over random (query, vertex) pairs
        qs_idx = rng.choice(keep.shape[0], 200, replace=False)
        vs = rng.choice(keep.shape[0], 2000, replace=False)
        diffs = []
        for i in qs_idx:
            qv = keep[i]
            e1 = self.est(self.encode(qv), vs)
            e8 = sim.est(sim.encode_query(qv), vs)
            diffs.append(e1 - e8)
        self.sigma = float(np.std(np.concatenate(diffs)))
        self.margin = margin_sigmas * self.sigma

    def encode(self, q):
        # the 8-bit path drops the query scale (est = -scale_v * <code_v, qc>), so the screen
        # estimate is put in the same units: divide the rotated query by s_q = max|q| / 127
        _, sq = quantize(q[None], 127)
        return (q.astype(np.float32) @ self.R) / max(float(sq[0]), 1e-12)

    def est(self, qr, ids):
        return -(self.f[ids] * (self.code[ids].astype(np.float32) @ qr))
