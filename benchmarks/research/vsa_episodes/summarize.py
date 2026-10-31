"""Apply frozen gates without tuning the evaluated configuration."""
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'benchmarks/results/vsa_episodes_v1'
cases = json.loads((OUT / 'evaluation/results.json').read_text())
assert len(cases) == 18
assert {(c['seed'], c['documents'], c['facts_per_document']) for c in cases} == {
    (s, n, f) for s in (101, 211, 307) for n in (256, 1024) for f in (2, 8, 32)
}
rows = []
for m in range(5):
    points = [c[p][m] for c in cases for p in ('initial', 'mutated')]
    query_ratios = [c[p][m]['p50_us'] / c[p][0]['p50_us'] for c in cases for p in ('initial', 'mutated')]
    memory_ratios = [(c[base] + c[idx][m]) / (c[base] + c[idx][0])
                     for c in cases for base, idx in [('canonical_bytes', 'index_bytes'), ('mutated_canonical_bytes', 'mutated_index_bytes')]]
    ratios = {}
    for field, common in [('build_us', 'canonical_create_us'), ('update_us', 'canonical_update_us'), ('reopen_build_us', 'snapshot_read_us')]:
        ratios[field] = [(c[field][m] + c[common]) / (c[field][0] + c[common]) for c in cases]
    quality = all(c[p][m]['recall'] >= 0.99 and c[p][m]['recall'] >= c[p][0]['recall'] - 0.01 for c in cases for p in ('initial', 'mutated'))
    safety = all(x['safety_errors'] == 0 for x in points) and all(c['reopen_parity'] for c in cases)
    no_regress = all(r <= 1.1 for r in query_ratios) and all(r <= 1.1 for rs in ratios.values() for r in rs)
    advantage = all(min(q, mem) <= 0.8 for q, mem in zip(query_ratios, memory_ratios))
    span = lambda xs: {'min': min(xs), 'median': statistics.median(xs), 'max': max(xs)}
    rows.append({
        'method': points[0]['method'], 'quality_pass': quality, 'safety_pass': safety,
        'no_regression_pass': no_regress, 'advantage_pass': advantage,
        'promote': m >= 2 and quality and safety and no_regress and advantage,
        'recall': span([x['recall'] for x in points]),
        'query_p50_us': span([x['p50_us'] for x in points]),
        'query_p50_ratio_vs_exact': span(query_ratios),
        'total_retained_bytes_ratio_vs_exact': span(memory_ratios),
        'lifecycle_ratio_vs_exact': {k: span(v) for k, v in ratios.items()},
        'fallback_fraction': span([x['fallback_fraction'] for x in points]),
        'safety_errors': sum(x['safety_errors'] for x in points),
    })
result = {
    'decision': 'GO' if any(r['promote'] for r in rows) else 'NO_GO',
    'cases': len(cases), 'queries_per_phase': 96,
    'quality_query_executions': 18 * 96 * 2 * 5,
    'reopen_parity_query_executions': 18 * 96 * 5 * 2,
    'arms': rows,
    'interpretation': 'Admission test of this fixed structured signature design, not a universal result about VSA. Gate ratios use shared canonical creation/update/read costs plus per-index costs. Advantage must hold per case/phase, with no >10% query/build/update/reopen regression anywhere.',
}
(OUT / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

fmt = lambda value: f'{value:.2f}'
table = ['| Method | Worst case recall | Median p50 (µs) | Median latency / exact | Median total heap / exact | Promote |',
         '|---|---:|---:|---:|---:|---|']
for r in rows:
    label = 'Reference' if r['method'] == 'exact_role_postings' else 'Control' if r['method'] == 'bm25_verified' else 'Yes' if r['promote'] else 'No'
    table.append(f"| {r['method']} | {r['recall']['min']:.1%} | {r['query_p50_us']['median']:.2f} | {r['query_p50_ratio_vs_exact']['median']:.2f}× | {r['total_retained_bytes_ratio_vs_exact']['median']:.2f}× | {label} |")
density = ['| Facts / episode | Exact recall | BM25 recall | VSA 32 recall | VSA 128 recall | VSA fallback recall |', '|---|---:|---:|---:|---:|---:|']
for nf in (2, 8, 32):
    vals = [statistics.mean(c[p][m]['recall'] for c in cases if c['facts_per_document'] == nf for p in ('initial', 'mutated')) for m in range(5)]
    density.append('| ' + str(nf) + ' | ' + ' | '.join(f'{v:.1%}' for v in vals) + ' |')
lifecycle = ['| Method | Ingest ratio | Update ratio | Reopen ratio |', '|---|---:|---:|---:|']
for r in rows:
    lifecycle.append('| ' + r['method'] + ' | ' + ' | '.join(f"{r['lifecycle_ratio_vs_exact'][k]['median']:.2f}×" for k in ('build_us', 'update_us', 'reopen_build_us')) + ' |')
manifest = json.loads((OUT / 'evaluation/manifest.json').read_text())
rss_match = __import__('re').search(r'(\d+)\s+maximum resident set size', (OUT / 'evaluation/execution.log').read_text())
rss_mib = int(rss_match.group(1)) / 1048576 if rss_match else None
source_paths = [p for p in manifest['sha256'] if not p.startswith('target/')]
unchanged = all(__import__('hashlib').sha256((ROOT / p).read_bytes()).hexdigest() == manifest['sha256'][p] for p in source_paths)
assert unchanged, 'Frozen evaluation inputs changed'
assert manifest['exit_code'] == 0
assert all(r['safety_pass'] for r in rows)
assert rows[0]['recall']['min'] == 1
lower_bound = [(c['canonical_bytes'] + c['documents'] * 7 * 256) / (c['canonical_bytes'] + c['index_bytes'][0]) for c in cases]
report = f'''# Practical VSA admission experiment — 30 October 2026

**Decision: {result['decision']}.** {'No VSA arm meets the frozen promotion gates. Keep this design in research; do not promote it into the production retrieval path.' if result['decision'] == 'NO_GO' else 'At least one arm passes the local admission gates; application validation is still required.'}

This answers a bounded question: can role-bound episode signatures beat an exact structural index when both receive the same canonical facts? It does **not** establish that every VSA architecture or application is unhelpful. It also does not measure natural-language understanding, learned extraction, dense search, hardware acceleration, or production HMS APIs.

## Evidence and decision

Eighteen held-out cases: three seeds × 256/1,024 documents × 2/8/32 facts per document. Each case has 96 queries in six classes, measured before and after 16 replacements and 16 deletions. All indexes are reconstructed from a canonical snapshot and checked for result parity. There were {result['quality_query_executions']:,} diagnostic method/query executions across initial/mutated states, in addition to timed sweeps and reopen comparisons. These are repeated synthetic queries, not independent real-world examples.

{chr(10).join(table)}

“Worst case” is the lowest phase-level average over the 36 initial/mutated case-phases, not minimum individual-query recall. Other table entries are medians across those case-phases. Heap includes canonical source records, signatures/postings, codebook, and both indexes for fallback. A ratio above 1 favors exact indexing.

All five methods produced **zero observed unsupported, wrong-version, wrong-namespace, invalid-time, deleted, or incorrectly quoted outputs**. Exact retrieval achieved 100% recall in every case; all reopen comparisons passed. Safety comes from exact canonical verification shared by every arm, not from hypervector similarity. The same verifier rejects role confusion and cross-fact joins.

The frozen gate requires recall ≥99% and within one percentage point of exact in every case; a ≥20% latency or retained-memory advantage at matched quality; and no >10% query/ingest/update/reopen regression anywhere. [Machine-readable decisions](decision.json) show each gate separately. No evaluation parameter tuning was performed.

## Interference and lifecycle

{chr(10).join(density)}

Recall here averages the nonempty-query phase averages at each interference level. Empty queries are excluded from recall and checked for false output. Raw per-class results remain in [results.json](evaluation/results.json).

{chr(10).join(lifecycle)}

Lifecycle entries are median ratios to exact, including shared canonical preparation or snapshot reading. These are single measurements per case; small differences would require controlled replication. BM25 rebuilds on mutation because the reused reference implementation exposes no incremental update API. This limitation does not affect the decisive comparison with the incrementally maintained exact index.

The fallback invoked exact retrieval for a median {rows[4]['fallback_fraction']['median']:.1%} of queries. It retains and maintains both indexes. It therefore tests whether approximate candidates add value to the exact path rather than assuming that a fallback is free.

## Why the design fails to earn promotion

1. **Canonical structure already solves the target ambiguity.** Role/filler postings narrow candidates directly, and fact-level verification preserves subject/predicate/object association. Role-sensitive retrieval is useful; that alone does not establish a need for VSA.
2. **Partial matching has a material representation cost.** Seven projections cost 1,792 signature bytes per episode before the codebook or canonical sources. Even an unrealistically free codebook leaves an initial total-heap ratio of {min(lower_bound):.2f}–{max(lower_bound):.2f}× exact for this layout. This is an arithmetic lower bound, not a benchmark of an optimized implementation.
3. **There is no quality dividend over exact retrieval here.** VSA matched exact recall and outperformed the roleless BM25 control at 32 facts per episode: the representation worked. It nevertheless added cost without improving the decisive baseline. The fallback triggered for almost every query because most had fewer than five valid answers; it provided no observed recall gain in this workload.
4. **The tested query path scans eligible signatures.** It also creates native phase vectors for query binding. Packed codebooks, direct bitwise binding, or another candidate index could reduce overhead, but their gains were not measured. They would be a new hypothesis, not a reason to reinterpret this failure as success.

The experiment uses the actual production phase primitives and existing research popcount/BM25 implementations. It does not pretend that this small harness is a fully optimized VSA system or that 1,024 documents establish scaling behavior.

## Practical engineering decision

* **Proceed with exact, evidence-backed structural retrieval as the reference design.** Before product work, validate the actual application's relation/query contract, annotation provenance, and end-to-end source fidelity. This experiment supports that baseline, not an unqualified production-readiness claim.
* **Keep VSA opt-in and outside authoritative storage.** There is no measured justification here for making it a default retrieval dependency or replacing canonical facts.
* **Do not spend paid model/API credits or broaden the benchmark to rescue this arm.** No arm passed the agreed local admission gate. The appropriate action is to stop promotion of this design.
* **Reopen VSA productization only for a materially different workload and explicit advantage.** Require a concrete user need, a strong exact/non-VSA baseline, a fixed resource budget, and falsifiable held-out success criteria. Potential improvements remain research until independently measured.

This work changed only research harness/artifact files. It did not implement production features or repairs. Other workspace work may proceed independently.

## Reproduction and limitations

See the [harness README](../../research/vsa_episodes/README.md), [frozen plan](PLAN.json), [final test log](tests-final.log), [source/binary hashes and environment](evaluation/manifest.json), and [run/RSS log](evaluation/execution.log). Fixtures and post-mutation snapshots are preserved per case. All 24 tests passed. Evaluation completed in {manifest['elapsed_seconds']:.2f} seconds on {manifest['hardware']}, {manifest['machine']}; whole-process peak RSS was {rss_mib:.2f} MiB. All recorded source/plan inputs still match the pre-run hashes.

The plan references audit commit `2e4c91d`; execution HEAD was `{manifest['execution_head']}` because unrelated work advanced the repository. The three reused phase, popcount, and BM25 files had no diff from the audit commit when checked before evaluation; the manifest pins the actual consumed files and binary.

Annotations are oracle inputs shared equally; no extraction accuracy/cost is included. Synthetic entities, exact literals, small corpora, a single thread, co-resident indexes, allocator instrumentation, and uncontrolled host activity limit external validity. Retained requested heap bytes exclude fragmentation, allocator metadata, stack, and transient peaks. Whole-process RSS is reported separately and cannot be assigned to an individual arm. Snapshot reopen is a harness check, not evidence of production crash recovery or durability. The quality/safety findings are stronger than microsecond timing precision; this is an admission decision for the tested design, not a universal VSA verdict.
'''
(OUT / 'REPORT.md').write_text(report)
