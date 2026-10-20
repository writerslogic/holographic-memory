# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6"]
# ///
"""Descriptive report and frozen primary contrasts from independently checked rows."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def mean(values):
    return sum(values) / len(values) if values else None


def report(attempt):
    validation = json.loads((attempt / 'independent-check.json').read_text())
    assert validation['complete']
    aggregates = defaultdict(lambda: defaultdict(list))
    unique = defaultdict(dict)
    quality = defaultdict(lambda: defaultdict(list))
    perturb = defaultdict(lambda: defaultdict(list))
    times = defaultdict(list)
    with gzip.open(attempt / 'checked-metrics.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            meta = row['input']
            dim = row['dim']
            corpus = meta['corpus']
            if corpus in ('uniform', 'repeated') and meta['variant'] == 'base':
                times[(dim, meta['length'])].append(row['timings']['production_ns'])
            for metric in row['metrics']:
                key = (dim, metric['arm'], metric['budget'])
                rare_auc = metric['rare_auc']
                if metric['arm'] == 'U' and meta['rare_id'] is not None:
                    density = metric['active'] / dim
                    corrected = [max(0.0, (score-density)/(1-density)) if density < 1 else 0.0 for score in metric['rare_scores']]
                    positive = corrected[meta['rare_id']]
                    rare_auc = sum((positive > value) + .5*(positive == value) for i,value in enumerate(corrected) if i != meta['rare_id']) / 63
                if corpus in ('uniform', 'repeated') and meta['variant'] == 'base':
                    aggregates[key]['base_inputs'].append(1)
                    if 'active' in metric:
                        aggregates[key]['active'].append(metric['active'])
                        aggregates[key]['delta_bytes'].append(metric['delta_bytes'])
                        aggregates[key]['json_bytes'].append(metric['json_bytes'])
                        aggregates[key]['json_lz4_bytes'].append(metric['json_lz4_bytes'])
                        unique[key][(meta['seed'], meta['text'])] = metric['signature']
                        if metric['selection_ns'] is not None:
                            aggregates[key]['selection_ns'].append(metric['selection_ns'])
                if corpus in ('uniform', 'repeated') and meta['variant'] in ('rare', 'rare_other'):
                    unit = (meta['seed'], meta['length'], meta['family'])
                    quality[key][unit].append(rare_auc)
                if corpus == 'dedicated' and meta['variant'] == 'rare':
                    aggregates[key]['dedicated_rare_auc'].append(rare_auc)
                if corpus in ('template', 'varied') and 'active' in metric:
                    aggregates[key][corpus+'_active'].append(metric['active'])
                    aggregates[key][corpus+'_signature'].append(metric['signature'])
    per_variant = defaultdict(list)
    with gzip.open(attempt / 'checked-pairs.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['dim'], row['arm'], row['budget'])
            unit = (row['seed'], row['length'], row['family'])
            perturb[key][unit].append(row['ordering'])
            per_variant[(*key, row['variant'])].append((row['ordering'], 1/row['rank'], row['rank']==1, row['degenerate_query']))
    rows = []
    for key in sorted(set(aggregates) | set(quality)):
        dim, arm, budget = key
        a = aggregates[key]
        active = a['active']
        u = unique[key]
        multiplicities = [n for seed in (11,23,47,89,179) for n in Counter(signature for (s,_),signature in u.items() if s==seed).values()]
        rows.append({'dim': dim, 'arm': arm, 'budget': budget,
                     'base_count': len(active), 'base_inputs': len(a['base_inputs']), 'base_empty': sum(n==0 for n in active),
                     'unique_base_texts': len(u), 'unique_vectors': len(multiplicities),
                     'distinctness_scope':'sum of separate fixed-seed counts; no cross-seed distinctness claim',
                     'collision_multiplicity_histogram':dict(sorted(Counter(multiplicities).items())),
                     'mean_active': mean(active), 'mean_delta_payload_bytes': mean(a['delta_bytes']),
                     'mean_json_bytes': mean(a['json_bytes']), 'mean_json_lz4_bytes_producer_only': mean(a['json_lz4_bytes']),
                     'median_selection_ns': statistics.median(a['selection_ns']) if a['selection_ns'] else None,
                     'rare_auc': mean([mean(x) for x in quality[key].values()]),
                     'rare_score': 'corrected containment' if arm == 'U' else 'native membership z-score' if arm == 'CB' else 'raw containment',
                     'dedicated_rare_auc': mean(a['dedicated_rare_auc']),
                     'perturbation_ordering': mean([mean(x) for x in perturb[key].values()]),
                     'parent_varied_empty': sum(n==0 for n in a['varied_active']),
                     'parent_varied_unique': len(set(a['varied_signature'])) if a['varied_signature'] else None,
                     'parent_template_empty': sum(n==0 for n in a['template_active'])})
    contrasts = []
    for dim in (1024,4096,16384,65536):
        budget = dim//256
        controls = ('F','FN','H','UC')
        arms = (*controls,'R25','R50')
        units = sorted(quality[(dim,'FN',budget)])
        assert len(units) == 480
        assert all(sorted(quality[(dim,arm,budget)]) == units for arm in arms)
        strata = sorted({unit[:2] for unit in units})
        assert len(strata)==60
        q = np.array([[mean(quality[(dim,arm,budget)][unit]) for unit in units] for arm in arms]).reshape(6,60,8)
        p = np.array([[mean(perturb[(dim,arm,budget)][unit]) for unit in units] for arm in arms]).reshape(6,60,8)
        rng = np.random.default_rng(20261012)
        draw = rng.integers(0,8,size=(10000,60,8))
        qboot = np.array([np.take_along_axis(np.broadcast_to(values,(10000,60,8)),draw,axis=2).mean(axis=(1,2)) for values in q])
        pboot = np.array([np.take_along_axis(np.broadcast_to(values,(10000,60,8)),draw,axis=2).mean(axis=(1,2)) for values in p])
        for i, arm in enumerate(('R25','R50'),4):
            qdiff = qboot[i] - qboot[:4].max(axis=0)
            pdiff = pboot[i] - pboot[:4].max(axis=0)
            point = float(q[i].mean() - q[:4].mean(axis=(1,2)).max())
            interval = list(map(float,np.quantile(qdiff,[.0125,.9875])))
            ordering_interval = list(map(float,np.quantile(pdiff,[.0125,.9875])))
            contrasts.append({'dim':dim,'budget':budget,'candidate':arm,'rare_auc_gain':point,
                              'rare_auc_gain_97_5_interval':interval,
                              'perturbation_gain_97_5_interval':ordering_interval,
                              'quality_signal_pass':point>=.02 and interval[0]>0 and ordering_interval[0]>=-.01})
    variants = [{'dim':key[0],'arm':key[1],'budget':key[2],'variant':key[3],
                 'pairs':len(v),'ordering':mean([x[0] for x in v]),'mrr':mean([x[1] for x in v]),
                 'recall1':mean([x[2] for x in v]),'degenerate_queries':sum(x[3] for x in v)}
                for key,v in sorted(per_variant.items())]
    summary = {'scope':'bounded synthetic ARM diagnostic; production encoder unchanged',
               'independent_check':validation,'rows':rows,'contrasts':contrasts,'variants':variants,
               'production_timing': [{'dim':d,'length':n,'median_ns':statistics.median(v)} for (d,n),v in sorted(times.items())],
               'adoption_allowed':False,'missing_adoption_requirements':['gated end-to-end arm costs','versioned migration/crash laboratory','isolated working/serving memory'],
               'reporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (attempt/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Sparse composition diagnostic','',
           'Synthetic inputs only. Production encoding is unchanged. No competitor or novelty claim.',
           '',f"Independent checks: {validation['rows_checked']:,} input/dimension rows, {validation['parent_vectors_checked']:,} parent text vectors, and {validation['operator_check']['rows']} operator-capacity rows.",
           '', '## D=16384, native budget64', '',
           '| Arm | Empty / bases | Unique vectors / unique texts | Rare AUC | Perturbation ordering | Mean raw varint bytes |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['dim'] != 16384 or r['budget'] not in (0, 64):
            continue
        def fmt(x):
            return 'N/A' if x is None else f'{x:.4f}'
        empties = f"{r['base_empty']}/{r['base_count']}" if r['base_count'] else 'N/A'
        distinct = f"{r['unique_vectors']}/{r['unique_base_texts']}" if r['base_count'] else 'N/A'
        lines.append(f"| {r['arm']} | {empties} | {distinct} | {fmt(r['rare_auc'])} | {fmt(r['perturbation_ordering'])} | {fmt(r['mean_delta_payload_bytes'])} |")
    lines+=['','Distinctness sums counts from separate fixed-seed corpora; it does not credit seed changes as text information. Raw varint payload excludes descriptors, framing, source text and indexes; it is not total-store memory. CB uses a 4D-byte counter payload plus metadata. U is uncapped and uses density-corrected containment for rare-feature AUC; CB uses its native membership score and fixed-density arms use primitive containment.','',
            '## Frozen candidate contrasts','', '| Dimension | Candidate | Rare AUC gain over strongest control | 97.5% interval | Perturbation interval | Quality signal |',
            '|---:|---|---:|---|---|---|']
    for c in contrasts:
        lines.append(f"| {c['dim']} | {c['candidate']} | {c['rare_auc_gain']:.4f} | {c['rare_auc_gain_97_5_interval']} | {c['perturbation_gain_97_5_interval']} | {c['quality_signal_pass']} |")
    lines+=['',f"Top-k projection disagrees after partitioning in {validation['operator_check']['lossy_partition_disagreements']} of {validation['operator_check']['rows']} operator cases; exact count merges pass. Sparse output alone is not a lossless merge state.",
            '', '## Limits','',
            'The host was above the timing gate. Selection/component timings are descriptive and exclude some common work; they cannot establish an end-to-end speed improvement. Actual JSON/raw-varint lengths are checked; LZ4 JSON lengths are producer-reported, not independently recompressed. The RSS observer measures the research producer, not HMS serving RSS.',
            '', 'Independent reconstruction covers primitive counts, candidate selection and counting-Bloom scores. Production-majority ties are checked for valid threshold/rank eligibility; exact production identity is additionally checked on the parent golden vectors. Native XOR/permutation/JSON inverse checks are assertions in the producer, while the separate capacity checker verifies count merges, union and projection behavior.',
            '', 'Position-free and level-drop ablations, exclusive-coordinate rare attribution, equal-total-byte frontiers and the separate five-round timing campaign were not implemented; these remain gaps against the original design.',
            '', 'Existing-version document restart tests passed. A versioned migration/crash implementation and equal-total-byte accounting remain unverified. Neither reserve candidate is eligible for production adoption, even if its quality-only contrast passes. Preserve every dimension, budget, empty vector, tie and loss in summary.json and the raw witnesses.',
            '', 'Reproduce: uv run --script benchmarks/public/check_sparse_composition.py <attempt>; uv run --script benchmarks/public/report_sparse_composition.py <attempt>. The checker creates fresh outputs and refuses to overwrite an earlier check; archive/check into a separate copy when reproducing.']
    (attempt/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'contrasts':contrasts,'adoption_allowed':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('attempt',type=Path)
    report(parser.parse_args().attempt)
