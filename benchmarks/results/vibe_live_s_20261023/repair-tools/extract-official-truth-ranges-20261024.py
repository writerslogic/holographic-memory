"""Extract only official distance arrays via bounded HTTP ranges for metric export."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import h5py

base = Path('/Volumes/A/.win-vibe-live/benchmarks/results/vibe_live_s_20261023')
root = Path('/Volumes/A/hms-coordination/public-bench/vibe-results/workspace-disabled-20261024T1435Z')
plan = json.loads((base / 'plan.json').read_text())['datasets']
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=plan)
args = parser.parse_args()
ds = args.dataset
meta = plan[ds]
url = f'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/{ds}.hdf5?download=true'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

try:
    response = urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, method='HEAD'), timeout=60)
except urllib.error.HTTPError as exc:
    if exc.code not in (301, 302, 303, 307, 308):
        raise
    response = exc
advertised = response.headers.get('x-linked-etag', '').strip('"')
assert advertised == meta['sha256'], 'Advertised full-file SHA differs from verified run plan'
output = root / 'truth-distance-extracts' / ds
output.mkdir(parents=True, exist_ok=True)

class RangeReader(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.cache = {}
        self.records = []
        self.transferred = 0
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else meta['size_bytes'] + offset
        assert 0 <= position <= meta['size_bytes']
        self.position = position
        return position
    def read(self, size=-1):
        if size < 0:
            size = meta['size_bytes'] - self.position
        assert 0 <= size <= 8 << 20, 'Single-read allocation limit'
        end = min(self.position + size, meta['size_bytes'])
        chunks = []
        while self.position < end:
            block = self.position // 65536
            start = block * 65536
            stop = min(start + 65536, meta['size_bytes'])
            if block not in self.cache:
                assert self.transferred + stop - start <= 8 << 20, 'Per-dataset transfer limit'
                for attempt in range(2):
                    try:
                        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{stop-1}', 'Accept-Encoding': 'identity'})
                        with urllib.request.urlopen(req, timeout=60) as response:
                            assert response.status == 206, 'Server did not honor byte range'
                            assert response.headers['Content-Range'] == f'bytes {start}-{stop-1}/{meta["size_bytes"]}', 'Unexpected range/size'
                            data = response.read(stop - start + 1)
                        assert len(data) == stop - start
                        break
                    except Exception:
                        if attempt:
                            raise
                        time.sleep(2)
                self.transferred += len(data)
                self.cache[block] = data
                (output / f'range-{start}-{stop-1}.bin').write_bytes(data)
                self.records.append({'start': start, 'end': stop-1, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            offset = self.position - start
            take = min(end - self.position, stop - self.position)
            chunks.append(self.cache[block][offset:offset+take])
            self.position += take
        return b''.join(chunks)
    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

reader = RangeReader()
with h5py.File(reader, 'r') as source:
    truth = source['distances'][:]
assert truth.ndim == 2 and truth.shape[1] >= 100 and truth.shape[0] > 0
path = output / f'{ds}.hdf5'
with h5py.File(path, 'w') as target:
    target['distances'] = truth
receipt = {'dataset': ds, 'source_url': url, 'source_full_file_sha256': meta['sha256'], 'source_full_file_bytes': meta['size_bytes'],
           'advertised_sha_matches_fully_verified_volume_plan': True, 'extracted_only': 'distances', 'shape': list(truth.shape), 'dtype': str(truth.dtype),
           'local_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'transferred_bytes': reader.transferred, 'ranges': reader.records,
           'verification_scope': 'Source full hash verified by original Modal download; this extraction verifies matching HF advertised SHA and preserves exact requested byte ranges, not a second full-file hash',
           'purpose': 'Official metric export only; no fitting, query execution, tuning or dataset bulk download', 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'ranges'}))
