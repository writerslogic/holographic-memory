# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inline SVG charts and the fallback comparison table, built from committed results files.

Nothing here carries a number of its own: values come from files under benchmarks/results/, and
a system that did not reach a recall target is drawn as "not reached" rather than dropped.
Colours are CSS classes (see docs/stylesheets/home.css) so the charts follow the light and dark
palettes.
"""

import json
from html import escape


def _fmt_qps(value):
    return f"{value:,.0f}"


def headline_bars(rows, metric_label):
    """Horizontal bars on a zero-based 0..1 axis."""
    label_w, bar_x, bar_w, row_h = 250, 262, 360, 40
    height = len(rows) * row_h + 30
    parts = [
        f'<svg class="hc" viewBox="0 0 680 {height}" role="img" '
        f'aria-label="{escape(metric_label)} by method">',
        f'<title>{escape(metric_label)} by method</title>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = bar_x + bar_w * tick
        parts.append(f'<line class="hc-grid" x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{height - 24}"/>')
        parts.append(f'<text class="hc-tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">{tick:g}</text>')
    for i, row in enumerate(rows):
        y = 10 + i * row_h
        cls = "hc-hms" if row["hms"] else "hc-other"
        w = bar_w * row["value"]
        parts.append(f'<text class="hc-label{" hc-strong" if row["hms"] else ""}" x="{label_w}" y="{y + 18}" text-anchor="end">{escape(row["label"])}</text>')
        parts.append(f'<rect class="{cls}" x="{bar_x}" y="{y + 4}" width="{w:.1f}" height="22" rx="4"/>')
        parts.append(f'<text class="hc-value" x="{bar_x + w + 8:.1f}" y="{y + 20}">{row["value"]:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _clean(name):
    return name.replace("(rabitqlib, ", "(").replace("faiss", "FAISS")


def vector_chart(root, spec):
    path = root / spec["file"]
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = spec["recall_targets"]
    table = data["qps_at_recall"]
    systems = []
    for name, per in table.items():
        values = []
        for t in targets:
            cell = per.get(t, {})
            values.append(cell.get("qps"))
        note = (spec.get("annotations") or {}).get(name)
        systems.append({"name": _clean(name) + (" \u2020" if note else ""), "values": values, "hms": name.startswith(spec["hms_prefix"]), "note": note})
    systems.sort(key=lambda s: (s["values"][0] is None, -(s["values"][0] or 0)))
    peak = max(v for s in systems for v in s["values"] if v) or 1

    configs = list(data.get("competitors", []))
    for h in data.get("hms", []):
        configs.extend(h.get("sweep", []))
    gate_total = len(configs)
    gate_met = sum(1 for c in configs if c.get("load_gate_met"))

    label_w, col_x, col_w, gap, row_h = 232, 244, 220, 24, 30
    width = col_x + len(targets) * (col_w + gap)
    height = 46 + len(systems) * row_h
    parts = [
        f'<svg class="hc" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Queries per second at fixed recall, by system">',
        "<title>Queries per second at fixed recall@10, by system</title>",
    ]
    for ci, t in enumerate(targets):
        x0 = col_x + ci * (col_w + gap)
        parts.append(f'<text class="hc-head" x="{x0}" y="20">recall@10 &#8805; {escape(t)}</text>')
    for ri, s in enumerate(systems):
        y = 34 + ri * row_h
        parts.append(f'<text class="hc-label{" hc-strong" if s["hms"] else ""}" x="{label_w}" y="{y + 17}" text-anchor="end">{escape(s["name"])}</text>')
        for ci, v in enumerate(s["values"]):
            x0 = col_x + ci * (col_w + gap)
            if v is None:
                parts.append(f'<text class="hc-miss" x="{x0}" y="{y + 17}">not reached</text>')
                continue
            w = max(2.0, (col_w - 56) * v / peak)
            parts.append(f'<rect class="{"hc-hms" if s["hms"] else "hc-other"}" x="{x0}" y="{y + 3}" width="{w:.1f}" height="18" rx="3"/>')
            parts.append(f'<text class="hc-value" x="{x0 + w + 6:.1f}" y="{y + 17}">{_fmt_qps(v)}</text>')
    parts.append("</svg>")
    summary = []
    for ci, t in enumerate(targets):
        mine = [(s["values"][ci], s["name"]) for s in systems if s["hms"] and not s["note"] and s["values"][ci]]
        other = [(s["values"][ci], s["name"]) for s in systems if not s["hms"] and s["values"][ci]]
        missed = [s["name"] for s in systems if not s["hms"] and not s["values"][ci]]
        if mine and other:
            (mv, mn), (ov, on) = max(mine), max(other)
            verdict = "faster than" if mv > ov else "slower than"
            summary.append(
                f"At recall@10 of {t}: the fastest HMS configuration ({mn}) ran at {_fmt_qps(mv)} QPS, "
                f"{verdict} the fastest other system ({on}, {_fmt_qps(ov)} QPS)."
            )
        elif other:
            names = ", ".join(sorted({n for _, n in other}))
            summary.append(
                f"At recall@10 of {t}: no HMS configuration reached it. Reached by: {names}."
            )
        if missed and mine:
            summary.append(f"Not reached at {t} by: {', '.join(missed)}.")
    for s in systems:
        if s["note"]:
            summary.append(f"† {s['name'][:-2]}: {s['note']}")
    env = data.get("environment", {})
    if isinstance(env, list):
        env = next((e for e in env if isinstance(e, dict) and "cpu" in e), {})
    ds = data.get("dataset", {})
    return {
        "summary": summary,
        "title": spec["title"], "svg": "".join(parts), "file": spec["file"],
        "link": spec.get("link"), "targets": targets, "systems": systems,
        "dataset": ds.get("name", ""), "n_train": ds.get("n_train"), "n_test": ds.get("n_test"),
        "machine": env.get("cpu", ""), "gate_met": gate_met, "gate_total": gate_total,
        "max_load": data.get("max_load"),
    }


METHODS = [
    ("hms_hybrid", "HMS hybrid (BM25 + exact cosine)"),
    ("hms_lexical", "HMS lexical (stemmed BM25)"),
    ("hms_dense_exact", "HMS dense (exact cosine)"),
    ("numpy_dense_exact_reference", "NumPy exact cosine, same embeddings"),
    ("hms_sparse", "HMS sparse vector path"),
]


def fallback_comparison(root, site):
    """A table derived from the committed SciFact result when no generated one exists."""
    rel = site["comparison_fallback"][0]
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    metrics = data["metrics"]
    rows = []
    for key, label in METHODS:
        if key in metrics:
            m = metrics[key]
            rows.append({"system": label, "values": [f'{m["ndcg_at_10"]:.3f}', f'{m["recall_at_100"]:.3f}'],
                         "hms": key.startswith("hms"), "source": rel})
    return {
        "title": "Retrieval quality on SciFact",
        "columns": ["nDCG@10", "Recall@100"],
        "rows": rows, "generated": False, "source": rel,
        "note": "Same embeddings for every dense row. The benchmark pages carry the full comparison, including where HMS loses.",
    }
