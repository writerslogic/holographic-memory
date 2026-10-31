"""One-shot WARM source migration; immutable compatibility bytes stay unchanged."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTECTED = {
    'HMS_ARENA', '$hms_positions', 'hms-embedding-v1:', 'hms_signing.key',
    'hms_provenance.key', 'hms-snapshot.json', '.hms.lock', 'hms:meaning:',
    'hms:chunk:', 'hms-vector-mask-v1', 'hms.sigstore.message.v1', 'hms-batch-root',
    'HMSFactCredential', 'urn:hms:fact:', 'urn:hms:access:', 'urn:hms:store:',
    'urn:hms:manifest:', 'hms.store.metadata', 'application/x-hms-store',
    'https://docs.holographic-memory.com/ns/hms/v1.jsonld',
    'https://writerslogic.com/ns/hms/v1', 'holographic-reduced-representation',
    'hms.created','hms.updated','hms.title','hms.tags','hms.lead','hms.trail',
    'hms.blank','hms.tag.','hms.','hms:read','hms:write','hms_namespace',
    'hms://memory/','hms://stats','hms://namespace/','hms_', './hms-data',
    'hms-data', 'holographic-memory/embed',
}
SKIP = ('docs/research/', 'docs/migrations/warm/', 'benchmarks/results/', 'assets/', '.git/')
BINARY = {'.png','.jpg','.jpeg','.gif','.ico','.webp','.woff','.woff2','.ttf','.pdf','.wasm','.node','.bin','.hmsemb','.npz','.npy','.gz','.zip','.tgz','.parquet','.sqlite','.db'}
SELF = {'scripts/rename_warm.py','scripts/rename-warm.zsh'}

def rename(value):
    value=value.replace('https://github.com/writerslogic/holographic-memory','https://github.com/writerslogic/warm')
    value=value.replace('git@github.com:writerslogic/holographic-memory','git@github.com:writerslogic/warm')
    value=value.replace('writerslogic/holographic-memory','writerslogic/warm')
    value=value.replace('https://docs.holographic-memory.com','https://writerslogic.com/warm/docs')
    value=value.replace('https://www.holographic-memory.com','https://writerslogic.com/warm')
    value=value.replace('https://holographic-memory.com','https://writerslogic.com/warm')
    value=value.replace('HolographicMemorySystem','WarmMemorySystem')
    value=value.replace('HolographicMemory','WarmMemory')
    value=value.replace('Holographic Memory System','WritersLogic Adaptive Retention & Memory')
    value=value.replace('Holographic Memory','WARM')
    value=value.replace('@holographic-memory/','@writerslogic/warm-')
    value=value.replace('holographic-memory','writerslogic-warm')
    value=value.replace('holographic_memory','writerslogic_warm')
    value=re.sub(r'(?<![A-Za-z])HMS(?=$|[^a-z])','WARM',value)
    value=re.sub(r'(?<![A-Za-z])Hms(?=$|[^a-z])','Warm',value)
    value=re.sub(r'(?<![A-Za-z])hms(?=$|[^a-z0-9]|[A-Z])','warm',value)
    return value

# Protect stable wire vocabulary before changing source identifiers.
def transform(text):
    placeholders=[]
    def save(value):
        placeholders.append(value)
        return f"__WARM_PRESERVED_{len(placeholders)-1}__"
    def byte(m):
        value=m.group(0)
        return save(value) if re.search(r"hms|HMS|holographic",value) else value
    text=re.sub(r'''(?<![\w"'])b"(?:\\.|[^"\\])*"''',byte,text)
    specific=PROTECTED-{'hms.','hms_','hms-data'}
    for value in sorted(specific,key=len,reverse=True):
        if value in text:text=text.replace(value,save(value))
    for value in ['hms.','hms_','hms-data']:
        for quote in ['"',"'"]:
            literal=quote+value+quote
            if literal in text:text=text.replace(literal,save(literal))
    text=re.sub(r'''(["'])(?:urn:hms:|hms://|hms:|hms\.|hms_\{\}_|HMS-TXN-)[^"'\n]*\1''',lambda m:save(m.group(0)),text)
    text=rename(text)
    for i,value in enumerate(placeholders):text=text.replace(f'__WARM_PRESERVED_{i}__',value)
    return text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if args.apply and (ROOT/'docs/migrations/warm/rename-applied.json').exists():
        raise SystemExit('Migration already applied; preserve subsequent compatibility aliases.')
    listed=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    changes=[]
    for relative in sorted(filter(None,listed)):
        p=ROOT/relative
        if relative in SELF or relative.startswith(SKIP) or p.suffix.lower() in BINARY or not p.is_file() or p.is_symlink():continue
        if '.env' in p.name or p.suffix in {'.key','.pem','.secret'}:continue
        try:before=p.read_bytes();text=before.decode('utf-8')
        except (UnicodeError,OSError):continue
        if '\0' in text:continue
        after=transform(text).encode()
        # Files are renamed; public package directories retain their existing layout.
        new_relative=rename(relative)
        if relative.startswith('tests/fixtures/'):new_relative=relative
        target=ROOT/new_relative
        if after==before and target==p:continue
        changes.append((relative,str(target.relative_to(ROOT)),before,after))
    destinations=[c[1] for c in changes]
    if len(destinations)!=len(set(destinations)):raise SystemExit('Rename collision')
    sources={c[0] for c in changes}
    for old,new,_,_ in changes:
        if old!=new and (ROOT/new).exists() and new not in sources:raise SystemExit(f'Target exists: {new}')
    manifest={'mode':'apply' if args.apply else 'dry-run','files':len(changes),'renames':sum(a!=b for a,b,_,_ in changes),'changes':[{'old':a,'new':b,'before_sha256':hashlib.sha256(c).hexdigest(),'after_sha256':hashlib.sha256(d).hexdigest()} for a,b,c,d in changes]}
    destination=ROOT/'docs/migrations/warm'/('rename-applied.json' if args.apply else 'rename-plan.json')
    if args.apply:
        backup=ROOT/'.git/warm-migration-backup'
        for old,new,before,after in changes:
            save=backup/old;save.parent.mkdir(parents=True,exist_ok=True)
            if not save.exists():save.write_bytes(before)
            source=ROOT/old;target=ROOT/new
            if source!=target:
                target.parent.mkdir(parents=True,exist_ok=True)
                source.rename(target)
            target.write_bytes(after)
    destination.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':manifest['files'],'renames':manifest['renames'],'manifest':str(destination)}))
if __name__=='__main__':main()
