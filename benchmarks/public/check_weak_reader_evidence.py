# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.13"
# dependencies = ["tokenizers==0.23.3"]
# ///
"""Independently check frozen local reader files and answer-report equality."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import tokenizers

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
MODEL_DIR = Path('/Volumes/A/hms-models/Qwen3-4B-Instruct-2507')
BINARY = Path('/Volumes/A/.hms-target-gpubatch/release/public-bench')
BINARY_SHA256 = 'd7cb53d980ea567c1f14ff48a8239b1e372bc6992aa43d4e5fa2b21bf20631ec'
TOKENIZER_SHA256 = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
PROTOCOL_SHA256 = '66a5d33412117c4ef5c6d0a89b0a26d0194103dc515d5a891a2593b2afcc566d'
ARMS = {'first_five', 'knapsack', 'operand'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
            'invalid SHA256 format')


def root_path(value):
    require(isinstance(value, str) and value, 'missing artifact path')
    relative = Path(value)
    require(not relative.is_absolute(), 'committed artifact path must be relative')
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), 'artifact path escapes repository')
    return path


def plain_filename(value):
    require(isinstance(value, str) and value not in ('', '.', '..')
            and '/' not in value and '\\' not in value and '\x00' not in value,
            'model shard name must be a plain filename')
    return value


class Files:
    def __init__(self):
        self.byte_cache = {}
        self.hash_cache = {}

    def read(self, path, overrides):
        path = Path(path).resolve()
        if path in overrides:
            return overrides[path]
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        cached = self.byte_cache.get(path)
        if cached is None or cached[0] != state:
            cached = (state, path.read_bytes())
            self.byte_cache[path] = cached
        return cached[1]

    def verify(self, metadata, overrides):
        require(isinstance(metadata, dict) and set(metadata) == {'path', 'sha256', 'bytes'},
                'artifact metadata fields differ')
        fingerprint(metadata['sha256'])
        require(type(metadata['bytes']) is int and metadata['bytes'] >= 0,
                'invalid artifact size')
        data = self.read(root_path(metadata['path']), overrides)
        require(len(data) == metadata['bytes'] and sha(data) == metadata['sha256'],
                f'artifact bytes or fingerprint differ: {metadata["path"]}')
        return data

    def external_hash(self, path):
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        cached = self.hash_cache.get(path)
        if cached is None or cached[0] != state:
            hasher = hashlib.sha256()
            with path.open('rb') as source:
                while data := source.read(4 * 1024 * 1024):
                    hasher.update(data)
            cached = (state, hasher.hexdigest())
            self.hash_cache[path] = cached
        return cached[1]


def check_model(bundle, files, overrides):
    model = bundle['model']
    require(model['model'] == MODEL and model['revision'] == REVISION
            and model['model_path'] == str(MODEL_DIR), 'model pin differs')
    manifest = json.loads(files.verify(bundle['fixed_artifacts']['model_manifest'], overrides))
    require(manifest == model, 'model manifest and bundle disagree')
    revision_path = MODEL_DIR / '.cache/huggingface/download/config.json.metadata'
    require(model['revision_metadata'] == str(revision_path)
            and revision_path.read_text().splitlines()[0] == REVISION
            and files.external_hash(revision_path) == model['revision_metadata_sha256'],
            'recorded model revision or metadata fingerprint differs')
    index = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text())
    require(isinstance(index.get('weight_map'), dict), 'model weight index lacks a map')
    shard_names = {plain_filename(name) for name in index['weight_map'].values()}
    names = {'config.json', 'generation_config.json', 'merges.txt',
             'model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json'} | shard_names
    for item in model['files']:
        plain_filename(item['name'])
    require(len(model['files']) == len(names)
            and {item['name'] for item in model['files']} == names,
            'model/config/tokenizer/weight coverage differs')
    for item in model['files']:
        path = MODEL_DIR / item['name']
        require(item['path'] == str(path) and type(item['bytes']) is int
                and item['bytes'] == path.stat().st_size,
                'model file path or size differs')
        fingerprint(item['sha256'])
        require(files.external_hash(path) == item['sha256'],
                f'model file fingerprint differs: {item["name"]}')
    require(model['total_bytes'] == sum(item['bytes'] for item in model['files']),
            'model total bytes differ')
    tokenizer = next(item for item in model['files'] if item['name'] == 'tokenizer.json')
    require(tokenizer['sha256'] == TOKENIZER_SHA256, 'model tokenizer pin differs')


def check(bundle, files, allow_incomplete=False, answers=None, overrides=None):
    overrides = overrides or {}
    require(bundle['schema'] == 'hms.longmemeval.weak_reader_evidence.v1'
            and bundle['dataset'] == 'LongMemEval S dev 100'
            and bundle['systems'] == ['weak_qwen4b'] and set(bundle['arms']) == ARMS,
            'bundle scope differs')
    require(bundle['timing_claim'] is False and bundle['cost_usd'] == 0
            and bundle['preliminary_native_telemetry'] is True,
            'functional reader labeled as timing or paid claim')
    fixed = bundle['fixed_artifacts']
    protocol_bytes = files.verify(fixed['reader_protocol'], overrides)
    require(sha(protocol_bytes) == PROTOCOL_SHA256, 'frozen reader protocol differs')
    protocol = json.loads(protocol_bytes)
    system = protocol['systems']['weak_qwen4b']
    require(bundle['frozen_system'] == system and system['revision'] == REVISION
            and system['reader_model'] == MODEL
            and system['binary_path'] == str(BINARY)
            and system['binary_sha256'] == BINARY_SHA256
            and system['tokenizer_sha256'] == TOKENIZER_SHA256
            and system['model_path'] == str(MODEL_DIR)
            and system['tokenizer_path'] == str(MODEL_DIR / 'tokenizer.json')
            and system['batch'] == 1 and system['max_new_tokens'] == 1536
            and system['max_context'] == 12288 and system['temperature'] == 0,
            'frozen local reader parameters differ')
    require(bundle['binary'] == {'path': str(BINARY), 'sha256': BINARY_SHA256,
                                 'bytes': BINARY.stat().st_size}
            and files.external_hash(BINARY) == BINARY_SHA256, 'frozen reader binary differs')
    check_model(bundle, files, overrides)
    require(files.verify(fixed['runner_source'], overrides)
            == (ROOT / 'benchmarks/public/longmemeval_weak_reader.py').read_bytes(),
            'frozen runner source differs')
    files.verify(fixed['bundle_updater_source'], overrides)
    run = json.loads(files.verify(fixed['run_manifest'], overrides))
    require(run['protocol_sha256'] == PROTOCOL_SHA256
            and run['binary_sha256'] == BINARY_SHA256
            and run['tokenizer_sha256'] == TOKENIZER_SHA256
            and run['chunk_size'] == 10 and run['timing_claim'] is False
            and bundle['prepared_sha256'] == run['prepared_sha256'], 'run pins differ')
    expected = {(job['qid'], job['arm']): job['cache_key'] for job in run['jobs']}
    require(len(run['jobs']) == len(expected) == 300
            and len({qid for qid, _ in expected}) == 100
            and all({arm for qid, arm in expected if qid == question} == ARMS
                    for question in {qid for qid, _ in expected}), 'frozen arm coverage differs')
    expected_request_keys = set(expected.values())
    generated = {}
    chunk_numbers = []
    for chunk in bundle['chunks']:
        require(type(chunk['chunk']) is int and chunk['chunk'] > 0
                and chunk['preliminary'] is True and chunk['timing_claim'] is False,
                'invalid chunk number or telemetry status')
        chunk_numbers.append(chunk['chunk'])
        artifacts = chunk['files']
        manifest = json.loads(files.verify(artifacts['manifest'], overrides))
        inputs = json.loads(files.verify(artifacts['input'], overrides))
        completion = json.loads(files.verify(artifacts['completion'], overrides))
        require(chunk['completion'] == completion, 'chunk completion differs')
        jobs = manifest['jobs']
        require(1 <= len(jobs) <= 10 and chunk['fresh_requests'] == len(jobs)
                and inputs == [job['request']['input'] for job in jobs],
                'chunk size or serialized inputs differ')
        prefix = f'chunk_{chunk["chunk"]:03d}'
        command = [str(BINARY), 'lme-model', '--stage', 'chat', '--model', str(MODEL_DIR),
                   '--revision', REVISION, '--batch', '1', '--input',
                   f'target/superiority-validation/weak-reader/{prefix}.input.json',
                   '--out', f'target/superiority-validation/weak-reader/{prefix}.output.json']
        require(manifest['command'] == command, 'native reader command differs')
        stderr = files.verify(artifacts['stderr'], overrides).decode()
        files.verify(artifacts['stdout'], overrides)
        telemetry = []
        for line in stderr.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                require(item.get('preliminary') is True and item.get('timing_claim') is False,
                        'ungated native telemetry lacks preliminary label')
                telemetry.append(item)
        require(telemetry == chunk['telemetry'], 'native telemetry fields changed')
        raw_output = files.verify(artifacts['native_output'], overrides) if 'native_output' in artifacts else None
        require(completion['native_output_sha256'] == (sha(raw_output) if raw_output is not None else None),
                'native output and completion fingerprints differ')
        if completion['status'] == 'completed':
            output = json.loads(raw_output)
            require(completion['returncode'] == 0 and completion['failure_reason'] is None
                    and isinstance(output, list) and len(output) == len(jobs)
                    and all(isinstance(text, str) for text in output), 'native output shape differs')
        else:
            require(completion['status'] == 'reader_failure'
                    and isinstance(completion['failure_reason'], str), 'unrecorded native failure')
            output = [''] * len(jobs)
        for job, text in zip(jobs, output, strict=True):
            identity = (job['qid'], job['arm'])
            request = job['request']
            key = sha(encode(request))
            require(identity in expected and expected[identity] == key
                    == job['request_sha256'] == job['cache_key']
                    and key not in generated, 'native request duplicate or fingerprint differs')
            require(request == {'model': MODEL, 'revision': REVISION,
                    'binary_sha256': BINARY_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
                    'stage': 'chat', 'batch': 1, 'max_new_tokens': 1536,
                    'max_context': 12288, 'temperature': 0, 'input': request['input']}
                    and isinstance(request['input'], str) and request['input'],
                    'native request differs from frozen generation parameters')
            generated[key] = {'status': completion['status'], 'raw_text': text,
                              'request': request}
    require(chunk_numbers == list(range(1, len(chunk_numbers) + 1)), 'chunk coverage is not contiguous')
    cache_bytes = files.verify(bundle['cache'], overrides)
    require(cache_bytes.endswith(b'\n') or not cache_bytes, 'cache has incomplete JSONL record')
    records = [json.loads(line) for line in cache_bytes.splitlines()]
    cached = {}
    for record in records:
        identity = (record['qid'], record['arm'])
        key = sha(encode(record['request']))
        require(identity in expected and identity not in cached
                and expected[identity] == key == record['request_sha256'] == record['cache_key']
                and key in generated, 'cached arm identity or request fingerprint differs')
        native = generated[key]
        require(record['request'] == native['request'] and record['status'] == native['status']
                and record['raw_text'] == native['raw_text']
                and record['response'] == {'text': native['raw_text']}
                and record['usage'] is None and record['cost_usd'] == 0,
                'cached predictions or identical-request reuse differ from native output')
        cached[identity] = record
    require(type(bundle['recorded_arms']) is int and bundle['recorded_arms'] == len(cached)
            and bundle['completed_arms'] == sum(r['status'] == 'completed' for r in records)
            and bundle['failed_arms'] == sum(r['status'] == 'reader_failure' for r in records)
            and bundle['total_questions'] == 100 and bundle['total_arms'] == 300
            and bundle['total_distinct_requests'] == len(expected_request_keys)
            and bundle['generated_distinct_requests'] == len(generated)
            and bundle['cached_distinct_requests'] == len({r['cache_key'] for r in records})
            and bundle['completed_chunks'] == len(chunk_numbers), 'bundle counts differ')
    complete = len(cached) == 300
    require(bundle['status'] == ('complete' if complete else 'partial'), 'completion status differs')
    require(allow_incomplete or complete, 'bundle is incomplete')
    require(not complete or (set(cached) == set(expected) and set(generated) == expected_request_keys),
            'completed reader lacks required arms or distinct requests')
    output_tokenizer = bundle['output_tokenizer']
    require(tokenizers.__version__ == '0.23.3'
            and output_tokenizer['package'] == 'tokenizers==0.23.3'
            and output_tokenizer['tokenizer_sha256'] == TOKENIZER_SHA256
            and output_tokenizer['add_special_tokens'] is False, 'visible output tokenizer differs')
    tokenizer = tokenizers.Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
    counts = bundle['per_request_output_tokens']
    require(len(counts) == len(generated)
            and {item['cache_key'] for item in counts} == set(generated), 'visible token count coverage differs')
    for item in counts:
        text = generated[item['cache_key']]['raw_text']
        require(item['raw_text_sha256'] == sha(text.encode())
                and type(item['visible_output_tokens']) is int
                and item['visible_output_tokens'] == len(tokenizer.encode(text, add_special_tokens=False).ids),
                'visible output token count or fingerprint differs')
    if answers is not None:
        require(complete and answers['complete'] is True and answers['dev_only'] is True
                and answers['heldout_run'] is False and len(answers['rows']) == 100,
                'final answer report is incomplete or outside dev scope')
        answered = {}
        for row in answers['rows']:
            arms = row['systems']['weak_qwen4b']['arms']
            require(set(arms) == ARMS, 'answer report arm coverage differs')
            for arm, data in arms.items():
                identity = (row['qid'], arm)
                require(identity not in answered and identity in cached,
                        'answer report has duplicate or unknown arm identity')
                record = cached[identity]
                require(data['reader']['call'] == record
                        and data['reader']['raw_text'] == record['raw_text']
                        and data['reader_input'] == record['request']['input'],
                        'answer report weak reader call/input/prediction differs from evidence')
                answered[identity] = True
        require(set(answered) == set(cached), 'answer report weak reader coverage differs')
    return {'recorded_arms': len(cached), 'distinct_requests': len(generated),
            'completed_chunks': len(chunk_numbers), 'complete': complete,
            'final_answer_report_checked': answers is not None}


def tamper_tests(bundle, files, allow_incomplete, answers):
    cases = []
    changed = copy.deepcopy(bundle)
    changed['cache']['bytes'] += 1
    cases.append(('corrupted_bundle_bytes', changed, {}))
    changed = copy.deepcopy(bundle)
    changed['model']['revision'] = '0' * 40
    cases.append(('model_revision', changed, {}))
    changed = copy.deepcopy(bundle)
    changed['model']['files'][0]['name'] = '../model-00001-of-00003.safetensors'
    replacement = encode(changed['model']) + b'\n'
    manifest = changed['fixed_artifacts']['model_manifest']
    manifest['sha256'] = sha(replacement)
    manifest['bytes'] = len(replacement)
    cases.append(('model_filename_escape', changed,
                  {root_path(manifest['path']): replacement}))
    changed = copy.deepcopy(bundle)
    changed['binary']['sha256'] = '0' * 64
    cases.append(('frozen_fingerprint', changed, {}))
    changed = copy.deepcopy(bundle)
    changed['per_request_output_tokens'][0]['visible_output_tokens'] += 1
    cases.append(('visible_output_count', changed, {}))
    chunk = next(c for c in bundle['chunks'] if c['completion']['status'] == 'completed')
    metadata = chunk['files']['native_output']
    path = root_path(metadata['path'])
    original = files.read(path, {})
    cases.append(('corrupted_output_bytes', copy.deepcopy(bundle), {path: original + b' '}))
    changed = copy.deepcopy(bundle)
    target = changed['chunks'][chunk['chunk'] - 1]
    output = json.loads(original)
    output[0] += ' corrupted prediction'
    replacement = encode(output) + b'\n'
    target['files']['native_output']['sha256'] = sha(replacement)
    target['files']['native_output']['bytes'] = len(replacement)
    target['completion']['native_output_sha256'] = sha(replacement)
    completion = encode(target['completion']) + b'\n'
    target['files']['completion']['sha256'] = sha(completion)
    target['files']['completion']['bytes'] = len(completion)
    cases.append(('prediction_with_updated_hashes', changed,
                  {path: replacement, root_path(target['files']['completion']['path']): completion}))
    changed = copy.deepcopy(bundle)
    cache_path = root_path(changed['cache']['path'])
    records = [json.loads(line) for line in files.read(cache_path, {}).splitlines()]
    records[0]['raw_text'] += ' corrupted reused prediction'
    records[0]['response']['text'] = records[0]['raw_text']
    replacement = b''.join(encode(row) + b'\n' for row in records)
    changed['cache']['bytes'] = len(replacement)
    changed['cache']['sha256'] = sha(replacement)
    cases.append(('cache_prediction_with_updated_hash', changed, {cache_path: replacement}))
    passed = []
    for name, changed, overrides in cases:
        try:
            check(changed, files, allow_incomplete, answers, overrides)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            passed.append(name)
        else:
            raise ValueError(f'tamper test accepted: {name}')
    if answers is not None:
        changed_answers = copy.deepcopy(answers)
        changed_answers['rows'][0]['systems']['weak_qwen4b']['arms']['first_five']['reader']['raw_text'] += ' corrupted'
        try:
            check(bundle, files, allow_incomplete, changed_answers)
        except (ValueError, KeyError, TypeError):
            passed.append('answer_report_prediction')
        else:
            raise ValueError('tamper test accepted: answer_report_prediction')
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', nargs='?', type=Path,
                        default=ROOT / 'benchmarks/results/weak_reader_evidence_v1.json')
    parser.add_argument('--answers', type=Path,
                        default=ROOT / 'benchmarks/results/longmemeval_dev_answers_v1.json')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--tamper-tests', action='store_true')
    args = parser.parse_args()
    bundle_bytes = args.report.read_bytes()
    bundle = json.loads(bundle_bytes)
    answer_bytes = None if args.allow_incomplete else args.answers.read_bytes()
    answers = None if answer_bytes is None else json.loads(answer_bytes)
    files = Files()
    result = check(bundle, files, args.allow_incomplete, answers)
    result['tamper_tests'] = tamper_tests(bundle, files, args.allow_incomplete, answers) if args.tamper_tests else []
    result['checker'] = 'passed'
    result['bundle_sha256'] = sha(bundle_bytes)
    result['checker_sha256'] = sha(Path(__file__).read_bytes())
    if answer_bytes is not None:
        result['answer_report_sha256'] = sha(answer_bytes)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
