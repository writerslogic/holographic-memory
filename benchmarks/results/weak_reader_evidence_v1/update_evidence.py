# /// script
# requires-python = ">=3.13"
# dependencies = ["tokenizers==0.23.3"]
# ///
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tokenizers


ROOT = Path('/Volumes/A/holographic-memory')
SOURCE = ROOT / 'target/superiority-validation/weak-reader'
DEST = ROOT / 'benchmarks/results/weak_reader_evidence_v1'
REPORT = ROOT / 'benchmarks/results/weak_reader_evidence_v1.json'
PROTOCOL = ROOT / 'benchmarks/public/reader_protocol_v1.json'
RUNNER = ROOT / 'benchmarks/public/longmemeval_weak_reader.py'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (canonical(value) + '\n').encode())


def artifact(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
            'bytes': path.stat().st_size}


def copy_immutable(source, name=None):
    data = source.read_bytes()
    target = DEST / (name or source.name)
    if target.exists():
        require(target.read_bytes() == data, f'immutable evidence changed: {source}')
    else:
        atomic_bytes(target, data)
    return artifact(target)


def model_manifest(system):
    path = DEST / 'model_manifest.json'
    model = Path(system['model_path'])
    revision_path = model / '.cache/huggingface/download/config.json.metadata'
    require(revision_path.read_text().splitlines()[0] == system['revision'],
            'model revision differs from frozen reader protocol')
    names = ['config.json', 'generation_config.json', 'merges.txt',
             'model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json']
    names.extend(p.name for p in sorted(model.glob('*.safetensors')))
    require(any(name.endswith('.safetensors') for name in names), 'model has no weights')
    if path.exists():
        value = json.loads(path.read_text())
        require(value['model'] == system['reader_model']
                and value['revision'] == system['revision']
                and {entry['name'] for entry in value['files']} == set(names),
                'cached model manifest differs from frozen reader')
        for entry in value['files']:
            require((model / entry['name']).stat().st_size == entry['bytes'],
                    f'model file size changed: {entry["name"]}')
        return value, artifact(path)
    files = [{'name': name, 'path': str(model / name),
              'bytes': (model / name).stat().st_size,
              'sha256': file_hash(model / name)} for name in sorted(names)]
    value = {'model': system['reader_model'], 'revision': system['revision'],
             'model_path': str(model), 'revision_metadata': str(revision_path),
             'revision_metadata_sha256': file_hash(revision_path), 'files': files,
             'total_bytes': sum(entry['bytes'] for entry in files)}
    atomic_json(path, value)
    return value, artifact(path)


def update():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    system = protocol['systems']['weak_qwen4b']
    binary = Path(system['binary_path'])
    require(file_hash(binary) == system['binary_sha256'], 'frozen binary changed')
    require(file_hash(Path(system['tokenizer_path'])) == system['tokenizer_sha256'],
            'frozen tokenizer changed')
    model, model_artifact = model_manifest(system)
    run = json.loads((SOURCE / 'run_manifest.json').read_text())
    require(run['protocol_sha256'] == file_hash(PROTOCOL), 'frozen protocol changed')
    require(run['binary_sha256'] == system['binary_sha256']
            and run['tokenizer_sha256'] == system['tokenizer_sha256'],
            'run fingerprints differ from protocol')
    order = run['jobs']
    require(len(order) == 300 and len({(j['qid'], j['arm']) for j in order}) == 300,
            'run job coverage differs')
    expected = {(j['qid'], j['arm']): j['cache_key'] for j in order}
    fresh_outputs = {}
    chunks = []
    for completed_path in sorted(SOURCE.glob('chunk_*.completion.json')):
        prefix = completed_path.name.removesuffix('.completion.json')
        manifest_path = SOURCE / f'{prefix}.manifest.json'
        input_path = SOURCE / f'{prefix}.input.json'
        output_path = SOURCE / f'{prefix}.output.json'
        stderr_path = SOURCE / f'{prefix}.stderr.log'
        completion = json.loads(completed_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest['jobs']
        inputs = json.loads(input_path.read_text())
        require(inputs == [job['request']['input'] for job in jobs],
                f'chunk inputs differ: {prefix}')
        require(1 <= len(jobs) <= 10, f'invalid chunk size: {prefix}')
        for job in jobs:
            key = digest(canonical(job['request']).encode())
            require(key == job['request_sha256'] == job['cache_key']
                    and expected[(job['qid'], job['arm'])] == key,
                    f'chunk frozen request differs: {prefix}')
            require(key not in fresh_outputs, 'request generated in multiple chunks')
        telemetry = []
        for line in stderr_path.read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                require(value.get('preliminary') is True
                        and value.get('timing_claim') is False,
                        f'completed chunk telemetry lacks preliminary label: {prefix}')
                telemetry.append(value)
        output = None
        if output_path.exists():
            require(file_hash(output_path) == completion['native_output_sha256'],
                    f'native output fingerprint differs: {prefix}')
        if completion['status'] == 'completed':
            output = json.loads(output_path.read_text())
            require(completion['returncode'] == 0 and isinstance(output, list)
                    and len(output) == len(jobs)
                    and all(isinstance(text, str) for text in output),
                    f'native output shape differs: {prefix}')
            texts = output
        else:
            require(completion['status'] == 'reader_failure', 'invalid native status')
            texts = [''] * len(jobs)
        for job, text in zip(jobs, texts, strict=True):
            fresh_outputs[job['cache_key']] = (completion['status'], text)
        files = {'manifest': copy_immutable(manifest_path),
                 'input': copy_immutable(input_path),
                 'completion': copy_immutable(completed_path),
                 'stderr': copy_immutable(stderr_path),
                 'stdout': copy_immutable(SOURCE / f'{prefix}.stdout.log')}
        if output_path.exists():
            files['native_output'] = copy_immutable(output_path)
        chunks.append({'chunk': int(prefix.removeprefix('chunk_')),
                       'fresh_requests': len(jobs), 'files': files,
                       'completion': completion, 'telemetry': telemetry,
                       'preliminary': True, 'timing_claim': False})
    raw_cache = (SOURCE / 'weak_reader_outputs.jsonl').read_bytes()
    end = raw_cache.rfind(b'\n') + 1
    raw_cache = raw_cache[:end]
    records = [json.loads(line) for line in raw_cache.splitlines()]
    identities, requests = set(), set()
    for record in records:
        identity = (record['qid'], record['arm'])
        key = digest(canonical(record['request']).encode())
        require(identity not in identities and expected[identity] == key
                == record['request_sha256'] == record['cache_key'],
                'cache identity or frozen request fingerprint differs')
        require(key in fresh_outputs, 'cache advanced before native completion snapshot')
        require((record['status'], record['raw_text']) == fresh_outputs[key]
                and record['response'] == {'text': record['raw_text']}
                and record['usage'] is None and record['cost_usd'] == 0,
                'cached generation differs from native output')
        identities.add(identity)
        requests.add(key)
    completed = sum(r['status'] == 'completed' for r in records)
    failures = sum(r['status'] == 'reader_failure' for r in records)
    require(completed + failures == len(records), 'invalid cache status')
    atomic_bytes(DEST / 'weak_reader_outputs.jsonl', raw_cache)
    require(tokenizers.__version__ == '0.23.3', 'local output tokenizer package differs')
    tokenizer = tokenizers.Tokenizer.from_file(system['tokenizer_path'])
    visible_counts = [{'cache_key': key,
                      'raw_text_sha256': digest(text.encode()),
                      'visible_output_tokens': len(tokenizer.encode(text, add_special_tokens=False).ids)}
                     for key, (status, text) in sorted(fresh_outputs.items())]
    atomic_bytes(DEST / 'update_evidence.py', Path(__file__).read_bytes())
    fixed = {'run_manifest': copy_immutable(SOURCE / 'run_manifest.json'),
             'reader_protocol': copy_immutable(PROTOCOL),
             'runner_source': copy_immutable(RUNNER),
             'bundle_updater_source': artifact(DEST / 'update_evidence.py'),
             'model_manifest': model_artifact}
    value = {'schema': 'hms.longmemeval.weak_reader_evidence.v1',
             'updated_at': datetime.now(timezone.utc).isoformat(),
             'status': 'complete' if len(records) == 300 else 'partial',
             'dataset': 'LongMemEval S dev 100', 'systems': ['weak_qwen4b'],
             'arms': protocol['arms'], 'frozen_system': system,
             'binary': {'path': str(binary), 'bytes': binary.stat().st_size,
                        'sha256': system['binary_sha256']},
             'model': model, 'fixed_artifacts': fixed,
             'prepared_sha256': run['prepared_sha256'],
             'cache': artifact(DEST / 'weak_reader_outputs.jsonl'),
             'total_questions': 100, 'total_arms': 300,
             'total_distinct_requests': len({j['cache_key'] for j in order}),
             'recorded_arms': len(records), 'completed_arms': completed,
             'failed_arms': failures, 'generated_distinct_requests': len(fresh_outputs),
             'cached_distinct_requests': len(requests), 'completed_chunks': len(chunks),
             'output_tokenizer': {'package': 'tokenizers==0.23.3',
                                  'tokenizer_sha256': system['tokenizer_sha256'],
                                  'add_special_tokens': False,
                                  'definition': 'Visible decoded raw_text encoded verbatim, excluding additional special tokens; distinct from native generated_tokens/EOS telemetry.'},
             'per_request_output_tokens': visible_counts,
             'chunks': chunks, 'cost_usd': 0, 'timing_claim': False,
             'preliminary_native_telemetry': True,
             'telemetry_note': 'Ungated functional reader diagnostics; all original numeric fields are preserved and labeled preliminary.'}
    atomic_json(REPORT, value)
    print(canonical({key: value[key] for key in ('status', 'recorded_arms',
          'completed_arms', 'failed_arms', 'completed_chunks', 'generated_distinct_requests')}),
          flush=True)


if __name__ == '__main__':
    update()
