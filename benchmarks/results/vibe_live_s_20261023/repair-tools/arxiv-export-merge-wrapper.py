import json, runpy, shutil, sys
from pathlib import Path
root = Path('/vol/vibe/run-s-20261023')
dataset = 'arxiv-nomic-768-normalized'
results = Path(sys.argv[sys.argv.index('--results') + 1])
ledger = root / 'ledger' / dataset / 'glass.json'
if ledger.exists():
    record = json.loads(ledger.read_text())
    if record['state'] == 'done':
        assert record['hms_rev'] == '12c80decb094d57ee13ba187a01bf1de8788ca2f'
        assert record['validation']['done'] and record['validation']['valid'] == record['validation']['expected']
        source = root / 'results' / dataset / 'glass' / 'results' / dataset / '100' / 'glass'
        target = results / dataset / '100' / 'glass'
        if not target.exists():
            shutil.copytree(source, target)
runpy.run_path('/opt/vibe/export_results.official.py', run_name='__main__')
