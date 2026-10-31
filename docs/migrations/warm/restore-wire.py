from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'scripts'))
from rename_warm import PROTECTED,rename
m=json.loads((root/'docs/migrations/warm/rename-applied.json').read_text())
plain=[p for p in PROTECTED if p not in {'hms.','hms_','hms_namespace','hms-data'}]
plain += ['HMS-TXN-','urn:hms:','$hms_positions','https://writerslogic.com/ns/hms/']
for c in m['changes']:
 p=root/c['new']
 if not p.exists():continue
 if c['old'].startswith(('src/core/provenance/testdata/','tests/fixtures/')):
  p.write_bytes((root/'.git/warm-migration-backup'/c['old']).read_bytes());continue
 try:s=p.read_text()
 except UnicodeError:continue
 t=s
 for old in sorted(plain,key=len,reverse=True):t=t.replace(rename(old),old)
 for old in ['hms.','hms_','hms-data']:
  for quote in ['"',"'"]:
   t=t.replace(quote+rename(old)+quote,quote+old+quote)
 if t!=s:p.write_text(t)
