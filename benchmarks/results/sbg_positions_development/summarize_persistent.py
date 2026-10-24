import json
from pathlib import Path
from statistics import median
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'hms-sbg-positions'
raw = json.loads((root / 'persistent-results.json').read_text())
assert len(raw['runs']) == 5
rows = []
for group in dict.fromkeys(row['group'] for row in raw['runs'][0]['results']):
    for k, count in [(10, False), (100, True), (0, True)]:
        values = {variant: [] for variant in ('base', 'candidate')}
        for run in raw['runs']:
            for variant in values:
                selected = [r for r in run['results'] if (r['group'], r['k'], r['count'], r['variant']) == (group, k, count, variant)]
                values[variant].append(sum(r['ns'] / r['repeats'] for r in selected) / len(selected) / 1000)
        ratios = [b/a for a,b in zip(values['base'], values['candidate'])]
        rows.append(dict(group=group, k=k, count=count, base_us=values['base'], candidate_us=values['candidate'], paired_ratios=ratios, median_ratio=median(ratios)))
summary = dict(development_only=True, standing=False, metadata=raw['start'], rounds=5, queries=82, commands=3, mismatches=0, rows=rows, load=[dict(run=r['run'], start=r['load'], end=r['end_load']) for r in raw['runs']])
(root / 'persistent-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for row in rows:
    print(row['group'], row['k'], row['count'], round(median(row['base_us']), 2), round(median(row['candidate_us']), 2), round(row['median_ratio'], 3), [round(x,3) for x in row['paired_ratios']])
