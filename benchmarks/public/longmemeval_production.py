# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Frozen dev-only experiment over native conversation recall; source-only inputs."""

import argparse
import copy
import json
import math
import os
import selectors
import subprocess
import time
from collections import Counter
from datetime import timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import tiktoken

import longmemeval_verifier as paid
from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_protocol, digest, file_digest, recompute_scores, require, same, wilson,
)
from longmemeval_answer_evidence import _observation_key, reader_payload
from longmemeval_dev_conversations import load_dev_conversations

SYSTEM, ARM = 'sol61_reader', 'production_recall'
RESULTS = ROOT / 'benchmarks/results'
DEFAULT_WORK = RESULTS / 'longmemeval_production_v1'
SPLIT = ROOT / 'benchmarks/public/longmemeval_split.json'
SOURCE = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
BASE_PROTOCOL = ROOT / 'benchmarks/public/reader_protocol_v1.json'
PREREGISTRATION = RESULTS / 'preregistration_2026-10.json'
OPTIONS = {'roles': ['user', 'assistant'], 'k': 64, 'candidateLimit': 256, 'maxSources': 128, 'maxBytes': 65536}
SOURCE_POLICY = 'native ConversationMemory recall over complete original turns; bounded direct premises; no prior answers or rankings'
RPC_TIMEOUT_SECONDS = 1200
MAX_REQUEST_BYTES, MAX_REPLY_BYTES = 8 * 1024 * 1024, 16 * 1024 * 1024
SOURCE_GATE = {'metadata_ids_verified': 500, 'dev_records_decoded': 100, 'heldout_records_decoded': 0}
MODEL_PINS = {'Qwen/Qwen3-Embedding-0.6B': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
              'Qwen/Qwen3-Reranker-0.6B': 'e61197ed45024b0ed8a2d74b80b4d909f1255473',
              'Qwen/Qwen3-4B-Instruct-2507': 'cdbee75f17c01a7cc42f958dc650907174af0554'}
STAGE_PARAMS = {'k0': 60.0, 'origWeight': 1.0, 'rewriteWeight': 1.0, 'subqueryWeight': 1.0,
                'lexicalWeight': 0.5, 'semanticWeight': 1.0, 'rerankWeight': 8.0, 'rerankN': 20}


def membership(split):
    dev, heldout = split['dev'], split['heldout']
    require(len(dev) == len(set(dev)) == 100 and len(heldout) == len(set(heldout)) == 400
            and not set(dev).intersection(heldout), 'invalid frozen dev100/heldout400 membership')
    return set(dev)


def index_rows(rows, dev):
    require(len(rows) == 100 and len({row['qid'] for row in rows}) == 100
            and {row['qid'] for row in rows} == dev, 'all 100 dev questions required')
    return {row['qid']: row for row in rows}


def export_history(raw):
    sessions, dates = raw['haystack_sessions'], raw['haystack_dates']
    require(len(sessions) == len(dates), 'unaligned conversation dates')
    conversations, originals = [], {}
    for session_index, (session, date) in enumerate(zip(sessions, dates, strict=True)):
        observed = int(_observation_key(date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        require(all(turn['role'] in ('user', 'assistant') and isinstance(turn['content'], str) for turn in session),
                'unsupported source role or text')
        sid = digest(canonical(['lme-production-session-v1', session_index, date,
                                [[turn['role'], turn['content']] for turn in session]]))
        turns = []
        for turn_index, turn in enumerate(session):
            source = digest(canonical(['lme-production-source-v1', sid, turn_index, turn['role'], turn['content'], date]))
            require(source not in originals, 'duplicate original source identity')
            turns.append({'sourceId': source, 'role': turn['role'], 'text': turn['content'], 'observationTime': observed})
            originals[source] = {'source_id': source, 'session_id': sid, 'session_index': session_index,
                                 'turn_index': turn_index, 'observation_time': date,
                                 'role': turn['role'], 'text': turn['content'], 'native_observation_time': observed}
        if turns:
            conversations.append({'sessionId': sid, 'version': '1', 'turns': turns})
    return conversations, originals


def source_row(qid, raw):
    conversations, originals = export_history(raw)
    question, date = raw['question'], raw['question_date']
    require(isinstance(question, str) and isinstance(date, str), 'invalid source question')
    return {'qid': qid, 'question': question, 'question_date': date, 'original_sources': originals,
            'native_input': {'conversations': conversations, 'query': f'As of {date}, {question}',
                             'options': copy.deepcopy(OPTIONS)}}


def make_protocol(base, preregistration):
    check_protocol(base, preregistration)
    return {
        'schema_version': 1, 'dev_only': True, 'heldout_run': False,
        'reader_prompt': base['reader_prompt'], 'judge_source': copy.deepcopy(base['judge_source']),
        'systems': {SYSTEM: {'reader_model': 'gpt-6.1-sol', 'judge_model': 'gpt-5.4-2026-03-05',
                            'reader_reasoning_effort': 'high', 'judge_reasoning_effort': 'medium',
                            'reader_max_output_tokens': 4096, 'judge_max_output_tokens': 2048}},
        'token_cap': 32768, 'exact_preflight_cap': 32256, 'framing_reserve_tokens': 512,
        'source_policy': SOURCE_POLICY, 'source_gate': SOURCE_GATE.copy(),
        'native_options': copy.deepcopy(OPTIONS), 'native_query': 'As of {question_date}, {question}',
        'reader_serialization': 'official chronological source whitelist; retain every native source without clipping',
        'native_profile': {'embedding': 'Qwen/Qwen3-Embedding-0.6B', 'reranker': 'Qwen/Qwen3-Reranker-0.6B',
                           'query_generator': 'Qwen/Qwen3-4B-Instruct-2507', 'extract_facts': False,
                           'embedding_query_task': 'conversation', 'original_weight': 1.0,
                           'rewrite_weight': 1.0, 'total_subquery_weight': 1.0, 'rrf_k0': 60,
                           'embedding_batch': 4, 'rerank_batch': 4, 'generator_batch': 1,
                           'model_pins': MODEL_PINS.copy(), 'stage_params': STAGE_PARAMS.copy()},
        'native_execution': {'timeout_seconds': RPC_TIMEOUT_SECONDS, 'max_request_bytes': MAX_REQUEST_BYTES,
                             'max_reply_bytes': MAX_REPLY_BYTES,
                             'question_attempts': 1, 'reset_before_each_question': True},
        'input_count': {'endpoint': '/v1/responses/input_tokens', 'request_fields': ['model', 'input'],
                        'settings_framing_reserve': 512},
        'service_tier': 'default', 'reader_prompt_cache_options': {'mode': 'explicit'},
        'rates_usd_per_million': copy.deepcopy(paid.RATES), 'new_spend_cap_usd': 14.0,
        'global_spend_caps_usd': copy.deepcopy(preregistration['spend_caps_usd']),
        'automatic_transport_retries': 2, 'transport_retry_policy': copy.deepcopy(paid.TRANSPORT_RETRY_POLICY),
        'reader_judge_concurrency': 1, 'paid_identity_namespace': 'sol61-production',
        'paid_cache_namespace': 'sol61-verifier-call-v1', 'input_count_namespace': 'sol61-input-count-v1',
        'limitations': ['Repeated-use dev profile; this is not a generalization measurement.',
                        'Native local computation is measured separately; its dollar cost is unknown, not zero.',
                        'The reader model is an alias, not an immutable dated snapshot.',
                        'Native byte caps and local o200k counts are not model token caps; exact endpoint count gates generation.',
                        'Native model-stage token/call counters are unavailable; conservative bounds and measured times are retained.'],
    }


def artifact(path):
    path = Path(path).expanduser().resolve()
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'sha256': file_digest(path)}


def source_artifacts():
    public = ROOT / 'benchmarks/public'
    return [artifact(path) for path in (
        Path(__file__), SOURCE, SPLIT, BASE_PROTOCOL, PREREGISTRATION,
        artifact_path(json.loads(BASE_PROTOCOL.read_text())['judge_source']['path']),
        public / 'longmemeval_dev_conversations.py', public / 'longmemeval_verifier.py',
        public / 'longmemeval_reader_api.py', public / 'longmemeval_answer_evidence.py',
        public / 'check_longmemeval_answers.py', public / 'longmemeval_verifier_sources.py',
        public / 'longmemeval_role_reader.py', public / 'longmemeval_role_retrieval.py',
        public / 'longmemeval_full_history.py', public / 'longmemeval_user_first.py',
        public / 'longmemeval_dev_sources.py', public / 'longmemeval_reader_budget.py',
        public / 'proxies/beir_proxies.py',
    )]


def verify_artifacts(records):
    require(isinstance(records, list) and records, 'frozen artifacts missing')
    for record in records:
        require(file_digest(artifact_path(record['path'])) == record['sha256'], 'frozen artifact changed')


def native_source_artifacts(source_root):
    source_root = Path(source_root).expanduser().resolve()
    require((source_root / 'src/bin/conversation-eval.rs').is_file(), 'native source root lacks the bridge')
    paths = sorted((source_root / 'src').rglob('*.rs'))
    paths.extend(source_root / name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'))
    require(paths and len(paths) <= 4096 and all(path.is_file() for path in paths), 'invalid native source inventory')
    return [artifact(path) for path in paths]


def prepare_source_only(work):
    require(not (work / 'source_inputs.json').exists(), 'source-only input already exists; resume instead')
    dev = membership(json.loads(SPLIT.read_text()))
    raw = load_dev_conversations(SOURCE, SPLIT)
    require(set(raw) == dev, 'gated dev source membership changed')
    base, preregistration = json.loads(BASE_PROTOCOL.read_text()), json.loads(PREREGISTRATION.read_text())
    protocol = make_protocol(base, preregistration)
    rows = [source_row(qid, raw[qid]) for qid in sorted(dev)]
    value = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'source_gate': SOURCE_GATE.copy(),
             'rows': rows, 'protocol': protocol, 'artifacts': source_artifacts()}
    paid.reader.write(work / 'source_inputs.json', value)


def native_record_id(original):
    return 'conversation:source:' + digest(canonical([1, 'source', [original['session_id'], '1',
                                                                  original['turn_index'], original['source_id']]]))


def native_premises(row, original):
    session = sorted((source for source in row['original_sources'].values()
                      if source['session_id'] == original['session_id']), key=lambda source: source['turn_index'])
    index = original['turn_index']
    require([source['turn_index'] for source in session] == list(range(len(session))), 'original source order differs')
    prior_users = [source['turn_index'] for source in session[:index] if source['role'] == 'user']
    selected = prior_users[:2]
    user = index if original['role'] == 'user' else (prior_users[-1] if prior_users else None)
    if user is not None:
        if user != index:
            selected.append(user)
        if user > 0 and session[user - 1]['role'] == 'assistant':
            selected.append(user - 1)
    return [native_record_id(session[position]) for position in sorted(set(selected))]


def native_sources(row, evidence):
    require(isinstance(evidence, dict) and set(evidence) == {'sources', 'anchors', 'dropped', 'serializedBytes'},
            'invalid native evidence shape')
    sources = evidence['sources']
    require(isinstance(sources, list) and len(sources) <= OPTIONS['maxSources'], 'native source cap exceeded')
    size = len(json.dumps(sources, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
    require(type(evidence['serializedBytes']) is int and evidence['serializedBytes'] == size <= OPTIONS['maxBytes'],
            'native source byte count differs or exceeds cap')
    seen, records = set(), {}
    for source in sources:
        require(set(source) == {'recordId', 'sourceId', 'sessionId', 'version', 'turnIndex', 'role',
                                'observationTime', 'text', 'premiseRecordIds'}, 'unexpected native source fields')
        sid = source['sourceId']
        require(sid in row['original_sources'] and sid not in seen, 'native source identity differs')
        original = row['original_sources'][sid]
        same({key: source[key] for key in ('sourceId', 'sessionId', 'version', 'turnIndex', 'role', 'observationTime', 'text')},
             {'sourceId': sid, 'sessionId': original['session_id'], 'version': '1', 'turnIndex': original['turn_index'],
              'role': original['role'], 'observationTime': original['native_observation_time'], 'text': original['text']},
             'native source differs from complete original turn')
        require(source['recordId'] == native_record_id(original) and source['recordId'] not in records,
                'duplicate native canonical source')
        same(source['premiseRecordIds'], native_premises(row, original), 'native direct premise links differ')
        seen.add(sid)
        records[source['recordId']] = source
    covered, anchors = set(), set()
    for anchor in evidence['anchors']:
        require(set(anchor) == {'recordId', 'sourceId', 'sessionId', 'version', 'turnIndex', 'score'}
                and type(anchor['score']) in (int, float) and math.isfinite(anchor['score']) and anchor['score'] > 0,
                'invalid native anchor')
        require(anchor['recordId'] in records, 'native anchor not retained')
        require(anchor['recordId'] not in anchors, 'duplicate native anchor')
        anchors.add(anchor['recordId'])
        covered.add(anchor['recordId'])
        original = records[anchor['recordId']]
        same({key: anchor[key] for key in ('sourceId', 'sessionId', 'version', 'turnIndex')},
             {key: original[key] for key in ('sourceId', 'sessionId', 'version', 'turnIndex')}, 'native anchor scope differs')
        for premise in original['premiseRecordIds']:
            require(premise in records and records[premise]['sessionId'] == original['sessionId']
                    and records[premise]['version'] == original['version']
                    and records[premise]['turnIndex'] < original['turnIndex'], 'native required premise missing or outside scope')
            covered.add(premise)
    require(covered == set(records), 'native packet includes sources outside admitted anchor closures')
    return [{key: row['original_sources'][source['sourceId']][key] for key in (
        'source_id', 'session_id', 'session_index', 'turn_index', 'observation_time', 'role', 'text')} for source in sources]


def arm_for_native(row, journal, protocol):
    failure = None
    if journal['status'] != 'ok':
        sources, failure = [], 'native_bridge_failed'
    else:
        sources = native_sources(row, journal['exchanges'][-1]['response']['evidence'])
        if not sources:
            failure = 'native_empty_evidence'
    groups = [{'group_id': source['source_id'], 'selection_rank': rank, 'sources': [source]}
              for rank, source in enumerate(sources)]
    prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': groups}))
    arm = {'source_groups': groups, 'retained_sources': paid.reader.retained(groups),
           'reader_input': prompt, 'reader_input_sha256': digest(prompt), 'input_tokens': 0,
           'local_proxy_input_tokens': len(tiktoken.get_encoding('o200k_base').encode(prompt)), 'framing_checks': [],
           'reader_cap': {'cap': 32768, 'exact_preflight_cap': 32256, 'input_tokens': 0,
                          'selection_failure': failure is not None, 'failure_reason': failure, 'source_policy': ARM},
           'reader': {'status': 'selection_failure' if failure else 'pending', 'raw_text': '', 'parsed': None},
           'judge': {'status': 'pending', 'raw_text': '', 'score': False}}
    arm['scores'] = recompute_scores(arm, allow_incomplete=True)
    return arm


class NativeBridge:
    def __init__(self, command, stderr_path):
        self.command = list(command)
        self.stderr = Path(stderr_path).open('ab')
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr)
        except Exception:
            self.stderr.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.buffer = bytearray()

    def exchange(self, request):
        request_line = canonical(request) + '\n'
        outgoing = request_line.encode()
        require(len(outgoing) <= MAX_REQUEST_BYTES, 'native request exceeds JSONL byte bound')
        deadline, sent = time.monotonic() + RPC_TIMEOUT_SECONDS, 0
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            while sent < len(outgoing):
                require(time.monotonic() < deadline, 'native RPC timed out while sending')
                if selector.select(max(0, deadline - time.monotonic())):
                    sent += os.write(self.process.stdin.fileno(), outgoing[sent:sent + 65536])
            selector.unregister(self.process.stdin)
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b'\n' not in self.buffer:
                require(time.monotonic() < deadline, 'native RPC timed out while reading')
                if selector.select(max(0, deadline - time.monotonic())):
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    require(chunk, 'native bridge exited before replying')
                    self.buffer.extend(chunk)
                    require(len(self.buffer) <= MAX_REPLY_BYTES, 'native reply exceeds JSONL byte bound')
        raw, _, remaining = self.buffer.partition(b'\n')
        self.buffer = bytearray(remaining)
        try:
            raw_line = raw.decode('utf-8') + '\n'
        except UnicodeDecodeError:
            return {'request': copy.deepcopy(request), 'request_jsonl': request_line,
                    'request_sha256': digest(canonical(request)), 'response': None, 'response_jsonl': None,
                    'response_bytes_hex': (raw + b'\n').hex(), 'error_type': 'UnicodeDecodeError'}
        try:
            response = json.loads(raw_line)
        except ValueError:
            response = None
        return {'request': copy.deepcopy(request), 'request_jsonl': request_line,
                'request_sha256': digest(canonical(request)), 'response': response, 'response_jsonl': raw_line}

    def close(self):
        failed = False
        try:
            if not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                except Exception:
                    failed = True
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception:
            failed = True
        finally:
            for handle in (self.process.stdin, self.process.stdout, self.stderr):
                try:
                    if not handle.closed:
                        handle.close()
                except BrokenPipeError:
                    pass
                except Exception:
                    failed = True
        if failed:
            raise RuntimeError('native bridge shutdown failed') from None


def write_journal(path, value):
    paid.reader.write(path, value)
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def record_exchange(bridge, request, journal, path):
    value = {'request': copy.deepcopy(request), 'request_jsonl': canonical(request) + '\n',
             'request_sha256': digest(canonical(request)), 'response': None, 'response_jsonl': None}
    journal['exchanges'].append(value)
    write_journal(path, journal)
    try:
        value.update(bridge.exchange(request))
    except Exception as error:
        value['error_type'] = type(error).__name__
        partial = getattr(bridge, 'buffer', b'')
        if partial:
            value['response_partial_bytes_hex'] = bytes(partial).hex()
        journal.update(status='error', error_type=type(error).__name__)
        write_journal(path, journal)
        raise
    write_journal(path, journal)
    require(isinstance(value['response'], dict), 'native bridge returned invalid JSON object')
    require(value['response'].get('command') == request['command'], 'native reply command differs')
    require(value['response'].get('status') == 'ok', 'native bridge returned an error')
    return value['response']


def question_native(bridge, row, path, initialization):
    if path.exists():
        previous = json.loads(path.read_text())
        same(previous['source_input_sha256'], digest(canonical(row['native_input'])), 'native input changed on resume')
        if previous['status'] == 'running':
            previous.update(status='error', error_type='InterruptedNativeAttempt', elapsed_seconds=None, timing_unknown=True)
            write_journal(path, previous)
        return previous
    journal = {'schema_version': 1, 'qid': row['qid'], 'status': 'running', 'initialization': initialization,
               'source_input_sha256': digest(canonical(row['native_input'])), 'exchanges': []}
    start = time.monotonic()
    try:
        reset = record_exchange(bridge, {'command': 'reset'}, journal, path)
        epoch = reset['epoch']
        require(reset['sourceCount'] == reset['chunkCount'] == reset['sessionCount'] == 0, 'native reset retained history')
        sources, sessions = 0, 0
        for conversation in row['native_input']['conversations']:
            response = record_exchange(bridge, {'command': 'ingest', 'conversation': conversation}, journal, path)
            require(response['epoch'] == epoch, 'native ingest changed epoch')
            sources += len(conversation['turns'])
            sessions += 1
            require(response['sourceCount'] == sources and response['sessionCount'] == sessions
                    and response['ingestAttempts'] == sessions, 'native whole-session ingestion count differs')
        response = record_exchange(bridge, {'command': 'recall', 'query': row['native_input']['query'],
                                             'options': row['native_input']['options']}, journal, path)
        require(response['epoch'] == epoch, 'native recall changed epoch')
        require(response['sourceCount'] == sources and response['sessionCount'] == sessions
                and response['recallAttempts'] == 1, 'native recall history count differs')
        native_sources(row, response['evidence'])
        journal['status'] = 'ok'
    except Exception as error:
        journal.update(status='error', error_type=type(error).__name__)
    journal['elapsed_seconds'] = time.monotonic() - start
    write_journal(path, journal)
    return journal


def model_artifacts(model_root):
    paths = []
    for name in MODEL_PINS:
        directory = Path(model_root).expanduser().resolve() / name.split('/')[-1]
        require(directory.is_dir(), 'pinned model directory missing')
        paths.extend(path for path in directory.iterdir() if path.is_file() and
                     (path.name in ('config.json', 'tokenizer.json', 'generation_config.json', 'REVISION',
                                    'model.safetensors.index.json') or path.suffix == '.safetensors'))
        revision_metadata = directory / '.cache/huggingface/download/config.json.metadata'
        if revision_metadata.is_file():
            paths.append(revision_metadata)
    paths.sort()
    require(paths and len(paths) <= 4096, 'model root must contain bounded pinned local model files')
    return [artifact(path) for path in paths]


def validate_native_profile(response, identity):
    profile = response['profile']
    require(profile['mode'] == 'models' and profile['extractFacts'] is False and profile['rewriteQueries'] is True
            and profile['embeddingQueryTask'] == 'conversation', 'native model profile differs')
    same(profile['stageParams'], STAGE_PARAMS, 'native stage weights differ')
    require(profile['embeddingBatch'] == profile['rerankBatch'] == 4 and profile['generatorBatch'] == 1,
            'native model batches differ')
    require([model['model'] for model in profile['models']] == list(MODEL_PINS), 'native model identities differ')
    files = {artifact_path(value['path']).resolve(): value['sha256'] for value in identity['model_artifacts']}
    for model in profile['models']:
        same(model['revision'], MODEL_PINS[model['model']], 'native model revision differs')
        require(model['files'], 'native model file proof missing')
        for receipt in model['files']:
            path = Path(identity['model_root']) / receipt['path']
            require(path.resolve() in files and files[path.resolve()] == receipt['sha256']
                    and path.stat().st_size == receipt['sizeBytes'], 'native model file differs from frozen artifact')
    same(response['transportPolicy']['defaultRecallOptions'], OPTIONS, 'native recall defaults differ')
    require(response['modelUsage']['remoteApiCalls'] == 0 and response['modelUsage']['localComputeUsd'] is None
            and response['modelUsage']['exactCountersAvailable'] is False, 'native accounting claims differ')


def summarize(report):
    require(len(report['rows']) == len({row['qid'] for row in report['rows']}) == 100, 'accuracy requires all 100 questions')
    arms = [row['systems'][SYSTEM]['arms'][ARM] for row in report['rows']]
    correct = sum(arm['scores']['correct'] for arm in arms)
    tokens = [arm['input_tokens'] for arm in arms]
    return {'n': 100, 'correct': correct, 'answer_accuracy': correct / 100, 'answer_accuracy_ci95': wilson(correct, 100),
            'native_status': dict(Counter(row['native_status'] for row in report['rows'])),
            'reader_status': dict(Counter(arm['reader']['status'] for arm in arms)),
            'judge_status': dict(Counter(arm['judge']['status'] for arm in arms)),
            'input_tokens': {'median': sorted(tokens)[49] / 2 + sorted(tokens)[50] / 2, 'max': max(tokens)},
            'abstentions': sum(arm['scores']['abstain'] for arm in arms),
            'exact_quote_claims': sum(arm['scores']['citation_valid_claims'] for arm in arms),
            'total_claims': sum(arm['scores']['citation_total_claims'] for arm in arms),
            'user_occurrences': sum(source['role'] == 'user' for arm in arms for source in arm['retained_sources']),
            'assistant_occurrences': sum(source['role'] == 'assistant' for arm in arms for source in arm['retained_sources'])}


def call_id(work, stage, qid):
    return digest(f'sol61-production:{work.name}:{stage}:{qid}')


def all_call_ids(work, rows):
    return {paid.retry_call_id(call_id(work, stage, row['qid']), attempt)
            for row in rows for stage in ('reader', 'judge') for attempt in range(3)}


def save(report, work):
    report['summary'] = summarize(report)
    ledger = json.loads((work / 'spend.json').read_text())
    report['spend_ledger_sha256'] = file_digest(work / 'spend.json')
    report['incremental_spend_usd'] = paid.variant_spend(ledger, all_call_ids(work, report['rows']))
    report['global_recorded_spend_usd'] = ledger['total_usd']
    paid.reader.write(work / 'answers.json', report)


def progress(stage, done, status, cost):
    print(json.dumps({'stage': stage, 'done': done, 'total': 100, 'status': status,
                      'cost_usd': cost}, allow_nan=False), flush=True)


def native_retrieve(work, binary, models, native_source_root=ROOT):
    started = time.monotonic()
    require(not (work / 'answers.json').exists(), 'native report exists; do not retrieve a second answer')
    source_path = work / 'source_inputs.json'
    source = json.loads(source_path.read_text())
    verify_artifacts(source['artifacts'])
    dev = membership(json.loads(SPLIT.read_text()))
    index_rows(source['rows'], dev)
    protocol = copy.deepcopy(source['protocol'])
    frozen_models = model_artifacts(models)
    native_identity = {'binary': artifact(binary), 'model_artifacts': frozen_models,
                       'model_root': str(Path(models).expanduser().resolve()),
                       'native_source_root': str(Path(native_source_root).expanduser().resolve()),
                       'native_source_artifacts': native_source_artifacts(native_source_root),
                       'argv': [str(Path(binary).resolve()), '--models', str(Path(models).expanduser().resolve())]}
    identity_path = work / 'native_identity.json'
    if identity_path.exists():
        same(json.loads(identity_path.read_text()), native_identity, 'native identity changed after initial freeze')
    else:
        paid.reader.write(identity_path, native_identity)
    initializations, journals, rows, bridge = [], [], [], None
    work.mkdir(parents=True, exist_ok=True)
    try:
        for row in source['rows']:
            path = work / 'native' / (digest(row['qid']) + '.json')
            if not path.exists() and bridge is None:
                bridge = NativeBridge(native_identity['argv'], work / 'native_stderr.log')
                init_path = work / 'native_initializations' / f'{len(list((work / "native_initializations").glob("*.json"))):04d}.json'
                init = {'schema_version': 1, 'native_identity': native_identity, 'status': 'running', 'exchanges': []}
                init_started = time.monotonic()
                try:
                    response = record_exchange(bridge, {'command': 'initialize'}, init, init_path)
                    require(response.get('epoch') == 1, 'native initialization epoch differs')
                    validate_native_profile(response, native_identity)
                except Exception as error:
                    init.update(status='error', error_type=type(error).__name__, elapsed_seconds=time.monotonic() - init_started)
                    paid.reader.write(init_path, init)
                    raise
                init['status'] = 'ok'
                init['elapsed_seconds'] = time.monotonic() - init_started
                paid.reader.write(init_path, init)
                initializations.append(artifact(init_path))
            init_artifact = initializations[-1] if initializations else {'path': '', 'sha256': ''}
            journal = question_native(bridge, row, path, init_artifact)
            journals.append(artifact(path))
            arm = arm_for_native(row, journal, protocol)
            rows.append({'qid': row['qid'], 'question': row['question'], 'question_date': row['question_date'],
                         'native_status': journal['status'], 'native_journal': artifact(path),
                         'systems': {SYSTEM: {'arms': {ARM: arm}}}})
            progress('native', len(rows), journal['status'], None)
            if bridge is not None and (bridge.process.poll() is not None
                                      or any(exchange.get('response') is None for exchange in journal['exchanges'])):
                bridge.close()
                bridge = None
    finally:
        if bridge is not None:
            bridge.close()
    raw = load_dev_conversations(SOURCE, SPLIT)
    for row in rows:
        row['qtype'] = raw[row['qid']]['question_type']
    protocol['native_identity'] = native_identity
    protocol['native_initializations'] = [artifact(path) for path in sorted((work / 'native_initializations').glob('*.json'))]
    protocol['prepared_native_inputs'] = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                                          'native_journal': row['native_journal']} for row in rows]
    paid.reader.write(work / 'protocol.json', protocol)
    report = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'reader_systems': [SYSTEM],
              'source_gate': SOURCE_GATE.copy(), 'complete': False, 'framing_frozen': False, 'rows': rows,
              'protocol_sha256': file_digest(work / 'protocol.json'),
              'artifacts': source['artifacts'] + [artifact(source_path), artifact(identity_path), native_identity['binary']] + frozen_models
                           + native_identity['native_source_artifacts']
                           + journals + protocol['native_initializations'],
              'native_accounting': {'remote_api_calls': 0, 'local_compute_usd': None,
                                    'producer_elapsed_seconds': time.monotonic() - started,
                                    'journal_paths': [item['path'] for item in journals],
                                    'initialization_paths': [item['path'] for item in protocol['native_initializations']]}}
    with paid.ledger_lock(work):
        paid.refresh_ledger(work, all_call_ids(work, rows), report['protocol_sha256'])
    save(report, work)


def provider_provenance(client):
    endpoint = urlsplit(str(client.base_url))
    require(endpoint.scheme in ('https', 'http') and endpoint.hostname, 'invalid provider endpoint')
    return {'scheme': endpoint.scheme, 'hostname': endpoint.hostname, 'sdk_version': version('openai')}


def prepared_manifest(rows):
    return [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
             'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens'],
             'input_count_cache_key': (row['systems'][SYSTEM]['arms'][ARM]['framing_checks'][0]['cache_key']
                                       if row['systems'][SYSTEM]['arms'][ARM]['framing_checks'] else None),
             'selection_failure': row['systems'][SYSTEM]['arms'][ARM]['reader_cap']['selection_failure']}
            for row in rows]


def validate_inputs(report, work, protocol):
    dev = membership(json.loads(SPLIT.read_text()))
    index_rows(report['rows'], dev)
    require(report['dev_only'] is True and report['heldout_run'] is False
            and report['reader_systems'] == [SYSTEM], 'production report scope differs')
    verify_artifacts(report['artifacts'])
    same(file_digest(work / 'protocol.json'), report['protocol_sha256'], 'production protocol changed')
    source = json.loads((work / 'source_inputs.json').read_text())
    originals = index_rows(source['rows'], dev)
    for row in report['rows']:
        arm = row['systems'][SYSTEM]['arms'][ARM]
        same({key: row[key] for key in ('question', 'question_date')},
             {key: originals[row['qid']][key] for key in ('question', 'question_date')}, 'original question changed')
        journal = json.loads(artifact_path(row['native_journal']['path']).read_text())
        same(journal['qid'], row['qid'], 'native journal question binding changed')
        same(journal['source_input_sha256'], digest(canonical(originals[row['qid']]['native_input'])), 'native journal input binding changed')
        same(row['native_status'], journal['status'], 'native terminal status changed')
        rebuilt = arm_for_native(originals[row['qid']], journal, protocol)
        for field in ('source_groups', 'retained_sources', 'reader_input', 'reader_input_sha256', 'local_proxy_input_tokens'):
            same(arm[field], rebuilt[field], 'native-to-reader source input changed')
        expected_cap = rebuilt['reader_cap'].copy()
        if report['framing_frozen'] and not expected_cap['selection_failure']:
            require(len(arm['framing_checks']) == 1, 'missing exact preflight')
            check = arm['framing_checks'][0]
            same(check, json.loads((work / 'input_counts' / (check['cache_key'] + '.json')).read_text()), 'input-count cache changed')
            same(check['request'], {'model': 'gpt-6.1-sol', 'input': arm['reader_input']}, 'exact-count request differs')
            same(check['request_sha256'], digest(canonical(check['request'])), 'exact-count request hash differs')
            same(check['cache_key'], digest(canonical({'namespace': 'sol61-input-count-v1', 'request': check['request']})),
                 'exact-count cache identity differs')
            require(check['status'] in ('ok', 'error'), 'invalid input-count status')
            if check['status'] == 'ok':
                tokens = check['response']['input_tokens']
                require(type(tokens) is int and tokens >= 0, 'invalid exact endpoint token count')
                failure = 'exact_input_count_exceeds_cap' if tokens > 32256 else None
            else:
                require(check['response'] is None, 'failed input-count record has a response')
                tokens, failure = 0, 'exact_input_count_failed'
            expected_cap.update(input_tokens=tokens, selection_failure=failure is not None, failure_reason=failure)
        else:
            require(not arm['framing_checks'], 'unissued or failed native packet has input-count records')
        same(arm['reader_cap'], expected_cap, 'reader preflight cap changed')
        same(arm['input_tokens'], expected_cap['input_tokens'], 'exact prepared count changed')
        require((arm['reader']['status'] == 'selection_failure') == expected_cap['selection_failure'],
                'reader selection status differs from fixed source or exact-count gate')
        if expected_cap['selection_failure']:
            require(arm['reader'].get('call') is None and not arm['reader'].get('transport_attempts'),
                    'failed selection issued a reader call')
    if report['framing_frozen']:
        same(protocol['prepared_inputs'], prepared_manifest(report['rows']), 'frozen prepared inputs changed')


def count_inputs(report, work):
    require(not report['framing_frozen'] and not list((work / 'cache').glob('*.json')),
            'framing is already frozen or a generation call was issued')
    protocol = json.loads((work / 'protocol.json').read_text())
    validate_inputs(report, work, protocol)
    client = paid.reader.client()
    protocol['provider'] = provider_provenance(client)
    provider_path = work / 'provider.json'
    if provider_path.exists():
        same(json.loads(provider_path.read_text()), protocol['provider'], 'provider changed during input counting')
    else:
        paid.reader.write(provider_path, protocol['provider'])
    protocol['input_count_provider'] = protocol['provider'].copy()
    protocol['cost_provenance'] = 'predeclared quoted-rate estimate, not a billing receipt'
    records = []
    for done, row in enumerate(report['rows'], 1):
        arm = row['systems'][SYSTEM]['arms'][ARM]
        if arm['reader_cap']['selection_failure']:
            progress('count', done, arm['reader']['status'], report.get('incremental_spend_usd', 0))
            continue
        check = paid.exact_input_count(client, arm['reader_input'], work)
        arm['framing_checks'] = [check]
        records.append(artifact(work / 'input_counts' / (check['cache_key'] + '.json')))
        failure = None if check['status'] == 'ok' else 'exact_input_count_failed'
        tokens = check['response']['input_tokens'] if failure is None else 0
        if tokens > 32256:
            failure = 'exact_input_count_exceeds_cap'
        arm['input_tokens'] = arm['reader_cap']['input_tokens'] = tokens
        arm['reader_cap'].update(selection_failure=failure is not None, failure_reason=failure)
        arm['reader']['status'] = 'selection_failure' if failure else 'pending'
        arm['scores'] = recompute_scores(arm, allow_incomplete=True)
        progress('count', done, arm['reader']['status'], report.get('incremental_spend_usd', 0))
    protocol['prepared_inputs'] = prepared_manifest(report['rows'])
    paid.reader.write(work / 'protocol.json', protocol)
    report.update(framing_frozen=True, protocol_sha256=file_digest(work / 'protocol.json'))
    report['artifacts'].extend(records + [artifact(provider_path)])
    with paid.ledger_lock(work):
        paid.refresh_ledger(work, all_call_ids(work, report['rows']), report['protocol_sha256'])
    save(report, work)


def judge_references(dev):
    raw = load_dev_conversations(SOURCE, SPLIT)
    require(set(raw) == dev, 'judge reference membership changed')
    require(all(isinstance(row['answer'], str) for row in raw.values()), 'invalid official judge reference')
    return {qid: row['answer'] for qid, row in raw.items()}


def run_stage(report, work, stage):
    require(stage in ('reader', 'judge') and report['framing_frozen'], 'freeze exact inputs before generation')
    require(stage != 'judge' or all(row['systems'][SYSTEM]['arms'][ARM]['reader']['status'] != 'pending'
                                     for row in report['rows']), 'finish all readers before loading judge references')
    protocol = json.loads((work / 'protocol.json').read_text())
    validate_inputs(report, work, protocol)
    tasks = [row for row in report['rows'] if row['systems'][SYSTEM]['arms'][ARM][stage]['status'] == 'pending']
    require(report['complete'] is False or not tasks, 'completed experiment cannot be resampled')
    references = judge_references(membership(json.loads(SPLIT.read_text()))) if stage == 'judge' and tasks else None
    client = paid.reader.client() if tasks else None
    if client is not None:
        same(provider_provenance(client), protocol['provider'], 'provider endpoint or SDK changed after freeze')
    identities = all_call_ids(work, report['rows'])
    count = lambda value: len(tiktoken.get_encoding('o200k_base').encode(value))
    for row in tasks:
        arm = row['systems'][SYSTEM]['arms'][ARM]
        if stage == 'judge' and arm['reader']['status'] != 'ok':
            arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
        else:
            prompt = arm['reader_input'] if stage == 'reader' else paid.reader.judge_prompt(
                protocol, row, arm['reader']['parsed']['answer'], references[row['qid']])
            tokens = arm['input_tokens'] if stage == 'reader' else count(prompt)
            output, earlier = paid.call_with_transport_retries(client, protocol, paid.request(protocol, prompt, stage),
                call_id(work, stage, row['qid']), stage, work, identities, tokens)
            if stage == 'reader':
                parsed = paid.reader.parse_answer(output['raw_text']) if output['status'] == 'ok' else None
                arm['reader'] = {'status': 'ok' if parsed is not None else 'error', 'raw_text': output['raw_text'],
                                 'parsed': parsed, 'call': output, 'transport_attempts': earlier}
            else:
                text = output['raw_text'].strip().lower()
                arm['judge'] = {'status': 'ok' if output['status'] == 'ok' and text in ('yes', 'no') else 'error',
                                'raw_text': output['raw_text'], 'score': output['status'] == 'ok' and text == 'yes',
                                'call': output, 'transport_attempts': earlier}
        arm['scores'] = recompute_scores(arm, allow_incomplete=True)
        save(report, work)
        done = sum(item['systems'][SYSTEM]['arms'][ARM][stage]['status'] != 'pending' for item in report['rows'])
        progress(stage, done, arm[stage]['status'], report['incremental_spend_usd'])
    report['complete'] = all(row['systems'][SYSTEM]['arms'][ARM][part]['status'] != 'pending'
                             for row in report['rows'] for part in ('reader', 'judge'))
    save(report, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare-source-only', 'native-retrieve', 'count-inputs', 'reader', 'judge', 'run'))
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--bridge', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--native-source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'run from the repository root')
    work = args.work.expanduser().resolve()
    require(work.is_relative_to(RESULTS), 'experiment must live under benchmark results')
    if args.stage == 'prepare-source-only':
        prepare_source_only(work)
    elif args.stage == 'native-retrieve':
        require(args.bridge is not None and args.models is not None, 'native retrieval requires bridge and local model paths')
        native_retrieve(work, args.bridge, args.models, args.native_source_root)
    else:
        report = json.loads((work / 'answers.json').read_text())
        if args.stage == 'count-inputs':
            count_inputs(report, work)
        else:
            for stage in (('reader', 'judge') if args.stage == 'run' else (args.stage,)):
                run_stage(report, work, stage)
        print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
