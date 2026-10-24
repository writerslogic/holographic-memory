import json
from pathlib import Path
from statistics import median
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'hms-sbg-positions'
pairs = [(0, 1), (3, 2), (4, 5)]
paths = sorted((root / 'paired-results').glob('*.json'))
assert len(paths) == 6, paths
runs = {r['run']: r for r in map(lambda p: json.loads(p.read_text()), paths)}
keys = lambda row: (row['query'], row['k'], row['count'])
reference = {keys(row): (row['matches'], row['hash']) for row in runs[0]['results']}
for run in runs.values():
    assert {keys(row): (row['matches'], row['hash']) for row in run['results']} == reference
rows = []
for group in dict.fromkeys(row['group'] for row in runs[0]['results']):
    for k, count in [(10, False), (100, True), (0, True)]:
        elapsed = {}
        for n, run in runs.items():
            selected = [row for row in run['results'] if (row['group'], row['k'], row['count']) == (group, k, count)]
            elapsed[n] = sum(row['ns'] / row['repeats'] for row in selected) / len(selected) / 1000
        base = [elapsed[a] for a, _ in pairs]
        candidate = [elapsed[b] for _, b in pairs]
        ratios = [elapsed[b] / elapsed[a] for a, b in pairs]
        rows.append(dict(group=group, k=k, count=count, base_us=base, candidate_us=candidate, paired_ratios=ratios, median_ratio=median(ratios)))
summary = dict(development_only=True, standing=False, machine='Intel Core Ultra 9 185H', parent_sha=runs[0]['parent_sha'], candidate_sha=runs[0]['candidate_sha'], rounds=3, query_threads=1, repeats=100, queries=len(reference)//3, mismatches=0, rows=rows, rss_kib=[dict(run=n, **run['rss_kib']) for n,run in runs.items()], load=[dict(run=n, start=run['load'], end=run['end_load']) for n,run in runs.items()])
ingest_paths = sorted((root / 'ingest-results').glob('*.json'))
if ingest_paths:
    assert len(ingest_paths) == 6
    ingest = {r['run']: r for r in map(lambda p: json.loads(p.read_text()), ingest_paths)}
    assert len({row['input_hash'] for row in ingest.values()}) == 1
    assert all(row['documents'] == 10000 for row in ingest.values())
    summary['ingest'] = {}
    for metric in ['ingest_ns', 'replace_ns', 'delete_ns', 'rss_ingested_kib', 'rss_after_kib']:
        ratios = [ingest[b][metric] / ingest[a][metric] for a,b in pairs]
        summary['ingest'][metric] = dict(base=[ingest[a][metric] for a,b in pairs], candidate=[ingest[b][metric] for a,b in pairs], paired_ratios=ratios, median_ratio=median(ratios))
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in rows:
    print(row['group'], row['k'], row['count'], round(median(row['base_us']), 1), round(median(row['candidate_us']), 1), round(row['median_ratio'], 3))
if 'ingest' in summary:
    print('ingest', json.dumps(summary['ingest']))
