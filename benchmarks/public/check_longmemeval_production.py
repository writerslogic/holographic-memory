# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Check source-only native conversation recall and failure-inclusive dev outcomes offline."""

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from datetime import timezone
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_protocol, digest, file_digest, recompute_scores,
    require, same, sorted_sources, wilson,
)
from check_longmemeval_user_first import dev_membership, index_rows, judge_prompt, verify_artifacts
from check_longmemeval_verifier import RATES, TRANSPORT_RETRY_POLICY, check_api_call, transport_failure
from longmemeval_answer_evidence import _observation_key, reader_payload
from longmemeval_dev_conversations import SOURCE_SHA256, SPLIT_SHA256, load_dev_conversations

SYSTEM, ARM = 'sol61_reader', 'production_recall'
SOURCE_GATE = {'metadata_ids_verified': 500, 'dev_records_decoded': 100, 'heldout_records_decoded': 0}
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_production_v1'
SOURCE = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
PINNED = {
    'Qwen3-Embedding-0.6B': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
    'Qwen3-Reranker-0.6B': 'e61197ed45024b0ed8a2d74b80b4d909f1255473',
    'Qwen3-4B-Instruct-2507': 'cdbee75f17c01a7cc42f958dc650907174af0554',
}
NATIVE_OPTIONS = {'roles': ['user', 'assistant'], 'k': 64, 'candidateLimit': 256,
                  'maxSources': 128, 'maxBytes': 65536}
STAGE_PARAMS = {'k0': 60.0, 'origWeight': 1.0, 'rewriteWeight': 1.0, 'subqueryWeight': 1.0,
                'lexicalWeight': 0.5, 'semanticWeight': 1.0, 'rerankWeight': 8.0, 'rerankN': 20}
SOURCE_POLICY = 'native ConversationMemory recall over complete original turns; bounded direct premises; no prior answers or rankings'
TIMING_FIELDS = ('modelLoadSeconds', 'ingestSeconds', 'querySeconds', 'resetSeconds')


def check_production_protocol(protocol, base, preregistration):
    check_protocol(base, preregistration)
    expected = {
        'schema_version': 1, 'dev_only': True, 'heldout_run': False,
        'reader_prompt': base['reader_prompt'], 'judge_source': base['judge_source'],
        'systems': {SYSTEM: {'reader_model': 'gpt-6.1-sol', 'judge_model': 'gpt-5.4-2026-03-05',
                            'reader_reasoning_effort': 'high', 'judge_reasoning_effort': 'medium',
                            'reader_max_output_tokens': 4096, 'judge_max_output_tokens': 2048}},
        'token_cap': 32768, 'exact_preflight_cap': 32256, 'framing_reserve_tokens': 512,
        'source_policy': SOURCE_POLICY, 'source_gate': SOURCE_GATE,
        'native_options': NATIVE_OPTIONS, 'native_query': 'As of {question_date}, {question}',
        'reader_serialization': 'official chronological source whitelist; retain every native source without clipping',
        'native_profile': {'embedding': 'Qwen/Qwen3-Embedding-0.6B', 'reranker': 'Qwen/Qwen3-Reranker-0.6B',
                           'query_generator': 'Qwen/Qwen3-4B-Instruct-2507', 'extract_facts': False,
                           'embedding_query_task': 'conversation', 'original_weight': 1.0,
                           'rewrite_weight': 1.0, 'total_subquery_weight': 1.0, 'rrf_k0': 60,
                           'embedding_batch': 4, 'rerank_batch': 4, 'generator_batch': 1,
                           'model_pins': {'Qwen/' + name: revision for name, revision in PINNED.items()},
                           'stage_params': STAGE_PARAMS},
        'native_execution': {'timeout_seconds': 1200, 'max_request_bytes': 8 * 1024 * 1024,
                             'max_reply_bytes': 16 * 1024 * 1024,
                             'question_attempts': 1, 'reset_before_each_question': True},
        'input_count': {'endpoint': '/v1/responses/input_tokens', 'request_fields': ['model', 'input'],
                        'settings_framing_reserve': 512},
        'service_tier': 'default', 'reader_prompt_cache_options': {'mode': 'explicit'},
        'rates_usd_per_million': RATES, 'new_spend_cap_usd': 14.0,
        'global_spend_caps_usd': preregistration['spend_caps_usd'],
        'automatic_transport_retries': 2, 'transport_retry_policy': TRANSPORT_RETRY_POLICY,
        'reader_judge_concurrency': 1, 'paid_identity_namespace': 'sol61-production',
        'paid_cache_namespace': 'sol61-verifier-call-v1', 'input_count_namespace': 'sol61-input-count-v1',
    }
    require(set(expected) | {'limitations'} <= set(protocol)
            and set(protocol) <= set(expected) | {'limitations', 'native_identity', 'prepared_inputs',
                'native_initializations', 'prepared_native_inputs', 'provider', 'input_count_provider', 'cost_provenance'},
            'production protocol has missing or forbidden fields')
    for field, value in expected.items():
        same(protocol[field], value, f'production protocol {field} changed')
    limitations = protocol['limitations']
    require(isinstance(limitations, list) and all(isinstance(item, str) for item in limitations)
            and any('dev' in item and 'generalization' in item for item in limitations)
            and any('alias' in item and 'immutable' in item for item in limitations)
            and any('cost' in item and 'unknown' in item for item in limitations), 'production limitations missing')


def check_source_row(row, raw):
    conversations, originals, metadata = source_inputs(raw)
    same(row['question'], raw['question'], 'source question changed')
    same(row['question_date'], raw['question_date'], 'source question date changed')
    expected = {'conversations': conversations, 'query': f"As of {raw['question_date']}, {raw['question']}",
                'options': NATIVE_OPTIONS}
    same(row['native_input'], expected, 'native input differs from complete original observations')
    table = {}
    for source in originals.values():
        sid = source['sourceId']
        table[sid] = {'source_id': sid, 'session_id': source['sessionId'], **metadata[sid],
                      'role': source['role'], 'text': source['text'],
                      'native_observation_time': source['observationTime']}
    same(row['original_sources'], table, 'original source table changed')
    require(set(row) == {'qid', 'question', 'question_date', 'original_sources', 'native_input'},
            'source-only input row contains forbidden fields')
    return originals, metadata


def independent_summary(rows, scores):
    require(len(rows) == len(scores) == 100 and len({row['qid'] for row in rows}) == 100,
            'accuracy requires all 100 dev rows')
    arms = [row['systems'][SYSTEM]['arms'][ARM] for row in rows]
    counts = [arm['input_tokens'] for arm in arms]
    correct = sum(score['correct'] for score in scores)
    return {'n': 100, 'correct': correct, 'answer_accuracy': correct / 100,
            'answer_accuracy_ci95': wilson(correct, 100),
            'reader_status': dict(Counter(arm['reader']['status'] for arm in arms)),
            'judge_status': dict(Counter(arm['judge']['status'] for arm in arms)),
            'native_status': dict(Counter(row['native_status'] for row in rows)),
            'input_tokens': {'median': statistics.median(counts), 'max': max(counts)},
            'abstentions': sum(score['abstain'] for score in scores),
            'exact_quote_claims': sum(score['citation_valid_claims'] for score in scores),
            'total_claims': sum(score['citation_total_claims'] for score in scores),
            'user_occurrences': sum(source['role'] == 'user' for arm in arms for source in arm['retained_sources']),
            'assistant_occurrences': sum(source['role'] == 'assistant' for arm in arms for source in arm['retained_sources'])}


def attempt_identity(work_name, qid, stage, attempt):
    base = digest(f'sol61-production:{work_name}:{stage}:{qid}')
    return base if attempt == 0 else base + f':transport-retry:{attempt}'


def check_ledger(report, ledger, calls, preregistration, work_name, protocol_sha256):
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'global spend caps changed')
    require(ledger['caps_usd']['reader_judge_dev'] == 120 and ledger['caps_usd']['total'] == 250
            and ledger['caps_usd']['heldout400'] == 0, 'spend authorization changed')
    require(type(ledger.get('schema_version')) is int and ledger['schema_version'] == 1, 'invalid ledger schema')
    totals = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    own_ids = {attempt_identity(work_name, row['qid'], stage, attempt)
               for row in report['rows'] for stage in ('reader', 'judge') for attempt in range(3)}
    entries, own = {}, {}
    for entry in ledger['entries']:
        cost, key = entry['cost_usd'], entry['cache_key']
        require(isinstance(key, str) and key and key not in entries, 'duplicate ledger cache identity')
        require(entry['category'] in totals and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                'invalid ledger charge')
        entries[key] = entry
        totals[entry['category']] += cost
        if entry.get('call_id') in own_ids:
            require(entry['call_id'] not in own, 'duplicate or resampled production call')
            own[entry['call_id']] = entry
    total = sum(totals.values())
    incremental = sum(entry['cost_usd'] for entry in own.values())
    same(ledger['by_category_usd'], totals, 'ledger category totals mismatch')
    same(ledger['total_usd'], total, 'global ledger total mismatch')
    same(report['global_recorded_spend_usd'], total, 'reported global spend mismatch')
    same(report['incremental_spend_usd'], incremental, 'reported production spend mismatch')
    require(total <= 250 and totals['reader_judge_dev'] <= 120 and incremental <= 14
            and all(value <= ledger['caps_usd'][key] for key, value in totals.items()), 'production or global spend cap exceeded')
    seen, issued = set(), set()
    for qid, stage, attempt, call, reserve in calls:
        identity = attempt_identity(work_name, qid, stage, attempt)
        key = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': call['request'],
                                'call_id': identity, 'protocol_sha256': protocol_sha256}))
        require(call['call_id'] == identity and call['protocol_sha256'] == protocol_sha256
                and call['cache_key'] == key and key not in seen, 'production call provenance mismatch')
        seen.add(key)
        if call['status'] == 'budget_exhausted':
            require(key not in entries and call['cost_usd'] == 0, 'unissued production call charged')
            continue
        require(key in entries, 'production call missing from ledger')
        entry = entries[key]
        require(entry['call_id'] == identity and entry['status'] == call['status']
                and entry['category'] == 'reader_judge_dev', 'production ledger status or identity mismatch')
        same(entry['cost_usd'], call['cost_usd'], 'production ledger charge mismatch')
        same(entry['usage'], call['usage'], 'production ledger usage mismatch')
        same(entry['reserved_usd'], reserve, 'production ledger reservation mismatch')
        require(call['cost_usd'] <= reserve + 1e-12, 'production charge exceeded worst-case reservation')
        issued.add(identity)
    require(set(own) == issued, 'unreported or resampled production ledger call')
    return total, incremental


def native_record_id(kind, parts):
    return f'conversation:{kind}:' + digest(canonical([1, kind, parts]))


def source_inputs(raw):
    """Identity depends only on observable ordinal, date, roles and original text."""
    inputs, sources, metadata = [], {}, {}
    for session_index, (date, turns) in enumerate(zip(raw['haystack_dates'], raw['haystack_sessions'], strict=True)):
        sid = digest(canonical(['lme-production-session-v1', session_index, date,
                                [[turn['role'], turn['content']] for turn in turns]]))
        at = int(_observation_key(date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        native_turns = []
        for turn_index, turn in enumerate(turns):
            source_id = digest(canonical(['lme-production-source-v1', sid, turn_index,
                                         turn['role'], turn['content'], date]))
            native_turns.append({'sourceId': source_id, 'role': turn['role'], 'text': turn['content'], 'observationTime': at})
            rid = native_record_id('source', [sid, '1', turn_index, source_id])
            require(rid not in sources, 'duplicate original native source identity')
            sources[rid] = {'recordId': rid, 'sourceId': source_id, 'sessionId': sid, 'version': '1',
                            'turnIndex': turn_index, 'role': turn['role'], 'observationTime': at,
                            'text': turn['content'], 'premiseRecordIds': []}
            metadata[source_id] = {'session_index': session_index, 'turn_index': turn_index, 'observation_time': date}
        if not turns:
            continue
        inputs.append({'sessionId': sid, 'version': '1', 'turns': native_turns})
        members = [sources[native_record_id('source', [sid, '1', index, turn['sourceId']])]
                   for index, turn in enumerate(native_turns)]
        for index, source in enumerate(members):
            users = [item for item in members[:index + 1] if item['role'] == 'user'][:2]
            premises = [item['recordId'] for item in users if item['turnIndex'] != index]
            if source['role'] == 'assistant':
                previous = [item for item in members[:index] if item['role'] == 'user']
                if previous and previous[-1]['recordId'] not in premises:
                    premises.append(previous[-1]['recordId'])
            source['premiseRecordIds'] = premises
    return inputs, sources, metadata


def check_evidence(evidence, originals, options):
    require(isinstance(evidence, dict) and set(evidence) == {'sources', 'anchors', 'dropped', 'serializedBytes'},
            'invalid native evidence fields')
    sources, anchors, dropped = evidence['sources'], evidence['anchors'], evidence['dropped']
    require(all(isinstance(items, list) for items in (sources, anchors, dropped)), 'invalid native evidence lists')
    selected = {}
    for source in sources:
        rid = source.get('recordId')
        require(rid in originals and rid not in selected, 'unknown or repeated native source')
        same(source, originals[rid], 'native source bytes, role or provenance changed')
        selected[rid] = source
    same(sources, sorted(sources, key=lambda source: (source['sessionId'], source['version'], source['turnIndex'])),
         'native source ordering changed')
    anchor_ids, closure = set(), set()
    rank_slots = set()
    for anchor in anchors:
        require(set(anchor) == {'recordId', 'sourceId', 'sessionId', 'version', 'turnIndex', 'score'},
                'invalid native anchor fields')
        rid, score = anchor['recordId'], anchor['score']
        require(rid in selected and rid not in anchor_ids, 'unknown or repeated native anchor')
        source = selected[rid]
        require(source['role'] in options['roles'], 'native anchor role is excluded')
        for field in ('recordId', 'sourceId', 'sessionId', 'version', 'turnIndex'):
            same(anchor[field], source[field], 'anchor scope differs from canonical source')
        require(type(score) in (int, float) and math.isfinite(score) and score > 0, 'invalid native anchor score')
        rank = round(1 / score - 60)
        require(1 <= rank <= options['k'] and math.isclose(score, 1 / (60 + rank), rel_tol=1e-12),
                'native anchor score is outside role RRF bounds')
        slot = (source['role'], rank)
        require(slot not in rank_slots, 'duplicate role rank')
        rank_slots.add(slot)
        anchor_ids.add(rid)
        closure.update([rid, *source['premiseRecordIds']])
    require(set(selected) == closure, 'native packet omits whole premises or adds unanchored sources')
    for source in sources:
        require(set(source['premiseRecordIds']) <= set(selected), 'native premise closure is incomplete')
    dropped_ids = set()
    for item in dropped:
        require(set(item) == {'recordId', 'sourceId', 'sessionId', 'version', 'reason'}, 'invalid native drop fields')
        rid = item['recordId']
        require(rid in originals and rid not in dropped_ids and rid not in anchor_ids, 'invalid native dropped identity')
        for field in ('recordId', 'sourceId', 'sessionId', 'version'):
            same(item[field], originals[rid][field], 'native drop scope changed')
        require(item['reason'] in ('max_sources', 'max_bytes'), 'immutable complete input unexpectedly lost a source or premise')
        dropped_ids.add(rid)
    size = len(canonical(sources).encode())
    same(evidence['serializedBytes'], size, 'native source JSON byte accounting changed')
    require(len(sources) <= options['maxSources'] and size <= options['maxBytes'], 'native packet exceeds source bounds')
    require(len(anchors) + len(dropped) <= options['k'] * len(options['roles']), 'native source candidate bound exceeded')
    return selected


def reader_sources(evidence, metadata):
    sources = []
    for native in evidence['sources']:
        meta = metadata[native['sourceId']]
        sources.append({'source_id': native['sourceId'], 'session_id': native['sessionId'],
                        'session_index': meta['session_index'], 'turn_index': meta['turn_index'],
                        'observation_time': meta['observation_time'], 'role': native['role'], 'text': native['text']})
    return sorted(sources, key=lambda source: (_observation_key(source['observation_time']),
                                              source['session_index'], source['turn_index']))


def chunk_count(text):
    spans, start, offset, words, in_word = [], 0, 0, 0, False
    for char in text:
        whitespace = char.isspace() and char not in '\x1c\x1d\x1e\x1f'
        width = len(char.encode())
        begins = not whitespace and not in_word
        if offset + width - start > 1024 or begins and words == 256:
            spans.append((start, offset))
            start, words, in_word = offset, 0, False
        if not whitespace and not in_word:
            words += 1
        in_word = not whitespace
        offset += width
    if start < offset:
        spans.append((start, offset))
    encoded = text.encode()
    return sum(any(not (char.isspace() and char not in '\x1c\x1d\x1e\x1f')
                   for char in encoded[start:end].decode()) for start, end in spans)


def check_profile(profile, identity):
    require(isinstance(profile, dict) and set(profile) == {'mode', 'models', 'device', 'extractFacts',
            'rewriteQueries', 'embeddingQueryTask', 'stageParams', 'embeddingBatch', 'rerankBatch',
            'generatorBatch', 'modelLoadIncludesFileFingerprinting'}, 'invalid native profile fields')
    same({key: profile[key] for key in ('mode', 'extractFacts', 'rewriteQueries', 'embeddingQueryTask', 'stageParams')},
         {'mode': 'models', 'extractFacts': False, 'rewriteQueries': True, 'embeddingQueryTask': 'conversation',
          'stageParams': STAGE_PARAMS},
         'native model policy changed')
    same({key: profile[key] for key in ('embeddingBatch', 'rerankBatch', 'generatorBatch', 'modelLoadIncludesFileFingerprinting')},
         {'embeddingBatch': 4, 'rerankBatch': 4, 'generatorBatch': 1, 'modelLoadIncludesFileFingerprinting': True},
         'native batching or fingerprinting policy changed')
    require(isinstance(profile['device'], str) and profile['device'], 'native device missing')
    root = Path(identity['argv'][2]).resolve()
    frozen = {artifact_path(item['path']).resolve(): item['sha256'] for item in identity['model_artifacts']}
    require(len(frozen) == len(identity['model_artifacts']), 'duplicate frozen model file')
    require(isinstance(profile['models'], list) and len(profile['models']) == 3, 'three pinned model receipts required')
    receipts = set()
    for model, (name, revision) in zip(profile['models'], PINNED.items(), strict=True):
        require(set(model) == {'model', 'revision', 'files'}, 'invalid native model receipt')
        same(model['model'], 'Qwen/' + name, 'native model changed')
        same(model['revision'], revision, 'native model revision changed')
        names = set()
        require(isinstance(model['files'], list) and model['files'], 'native model files missing')
        for item in model['files']:
            require(set(item) == {'path', 'sizeBytes', 'sha256'} and isinstance(item['path'], str), 'invalid model file receipt')
            relative = Path(item['path'])
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == name,
                    'model receipt escapes its model root')
            path = (root / relative).resolve()
            require(path.is_relative_to(root) and path in frozen and path not in receipts, 'unfrozen or repeated native model receipt')
            same(item['sha256'], frozen[path], 'native receipt differs from frozen model hash')
            require(type(item['sizeBytes']) is int and item['sizeBytes'] > 0, 'invalid native model file size')
            receipts.add(path)
            names.add(str(relative.relative_to(name)))
        require({'config.json', 'tokenizer.json'} <= names and any(value.endswith('.safetensors') for value in names)
                and ('REVISION' in names or '.cache/huggingface/download/config.json.metadata' in names),
                'native model receipt omits weights, tokenizer, configuration or revision')


def check_exchange(exchange, expected, interrupted=False):
    require({'request', 'request_jsonl', 'request_sha256', 'response', 'response_jsonl'} <= set(exchange)
            and set(exchange) <= {'request', 'request_jsonl', 'request_sha256', 'response', 'response_jsonl',
                                  'error_type', 'response_bytes_hex', 'response_partial_bytes_hex'},
            'invalid native exchange journal fields')
    same(exchange['request'], expected, 'native request contains changed or non-source material')
    same(exchange['request_jsonl'], canonical(expected) + '\n', 'native raw request differs')
    same(exchange['request_sha256'], digest(canonical(expected)), 'native request hash differs')
    require(len(exchange['request_jsonl'].encode()) - 1 <= 8 * 1024 * 1024, 'native request exceeds transport bound')
    response, raw = exchange['response'], exchange['response_jsonl']
    if response is None:
        if raw is not None:
            require(isinstance(raw, str) and raw.endswith('\n') and len(raw.encode()) - 1 <= 16 * 1024 * 1024,
                    'invalid failed native response frame')
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            require(parsed is None, 'discarded valid native JSON response')
        else:
            require(interrupted or isinstance(exchange.get('error_type'), str) and exchange['error_type'],
                    'unreturned native request lacks failure record')
        for field in ('response_bytes_hex', 'response_partial_bytes_hex'):
            if field in exchange:
                value = exchange[field]
                require(raw is None and isinstance(value, str) and 0 < len(value) <= 32 * 1024 * 1024,
                        'invalid failed native byte frame')
                try:
                    bytes.fromhex(value)
                except ValueError as error:
                    raise ValueError('invalid failed native byte encoding') from error
        return None
    require(isinstance(raw, str) and raw.endswith('\n') and len(raw.encode()) - 1 <= 16 * 1024 * 1024,
            'invalid raw native response frame')
    same(json.loads(raw), response, 'native raw response differs from parsed journal')
    if not isinstance(response, dict):
        return None
    require(response.get('status') in ('ok', 'error'), 'invalid native response status')
    require(response.get('command') == expected['command']
            or response.get('status') == 'error' and response.get('command') is None, 'native response command differs')
    return response


def check_common_response(response, previous_timings):
    cumulative = {}
    for field in TIMING_FIELDS:
        value = response[field]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid native operation timing')
        cumulative[field] = previous_timings[field] + value
    same(response['cumulativeTimings'], cumulative, 'native cumulative timing accounting changed')
    same(response['transportPolicy'], {
        'schemaVersion': 1, 'maxLineBytes': 8 * 1024 * 1024, 'maxReplyBytes': 16 * 1024 * 1024,
        'requestTimeoutSeconds': None, 'timeoutPolicy': 'caller-enforced process termination; no automatic retry',
        'epochScope': 'process; each successful reset creates an empty isolated history',
        'defaultRecallOptions': NATIVE_OPTIONS}, 'native transport policy changed')
    same(response['modelUsage'], {
        'exactCountersAvailable': False, 'reason': 'production model stages do not expose token and pair counters',
        'remoteApiCalls': 0, 'localComputeUsd': None, 'wallTimeIncludesAllModelStages': True,
        'maxEmbeddingTokensPerInput': 2048, 'maxQueryVariantsPerSearch': 6,
        'maxGeneratedPlanTokensPerSearch': 1536, 'maxRerankPairsPerSearch': 20,
        'maxIngestEmbeddingInputsPerAttempt': 4096}, 'native usage bounds or local price accounting changed')
    return cumulative


def check_native_journal(journal, source_row, originals, identity, initialization, previous_epoch):
    require(journal['schema_version'] == 1 and journal['qid'] == source_row['qid']
            and journal['status'] in ('ok', 'error'), 'invalid terminal native question journal')
    same(journal['source_input_sha256'], digest(canonical(source_row['native_input'])), 'native source input hash changed')
    same(journal['initialization'], initialization, 'native initialization link changed')
    elapsed = journal.get('elapsed_seconds')
    unknown_time = (journal['status'] == 'error' and journal.get('error_type') == 'InterruptedNativeAttempt'
                    and journal.get('timing_unknown') is True and elapsed is None)
    require(unknown_time or type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            'invalid native elapsed time')
    native = source_row['native_input']
    expected = [{'command': 'reset'}] + [{'command': 'ingest', 'conversation': item} for item in native['conversations']]
    expected.append({'command': 'recall', 'query': native['query'], 'options': native['options']})
    exchanges = journal['exchanges']
    require(isinstance(exchanges, list) and 1 <= len(exchanges) <= len(expected), 'native attempt was resampled or omitted')
    if journal['status'] == 'ok':
        require(len(exchanges) == len(expected), 'successful native journal omitted original ingestion')
    counts = {'sessionCount': 0, 'sourceCount': 0, 'chunkCount': 0, 'ingestAttempts': 0, 'recallAttempts': 0}
    epoch, evidence = previous_epoch, None
    for index, (exchange, request) in enumerate(zip(exchanges, expected)):
        last = index == len(exchanges) - 1
        response = check_exchange(exchange, request, interrupted=unknown_time and last)
        if response is None or response['status'] == 'error':
            require(last and journal['status'] == 'error', 'native failure retried or scored as success')
            continue
        check_profile(response['profile'], identity)
        if request['command'] == 'reset':
            epoch += 1
        elif request['command'] == 'ingest':
            conversation = request['conversation']
            count = sum(chunk_count(turn['text']) for turn in conversation['turns'])
            remembered = {'sessionId': conversation['sessionId'], 'version': '1',
                          'sourceCount': len(conversation['turns']), 'chunkCount': count, 'created': True}
            same(response['remembered'], remembered, 'native ingestion is incomplete, repeated or not original')
            counts['sessionCount'] += 1
            counts['sourceCount'] += len(conversation['turns'])
            counts['chunkCount'] += count
            counts['ingestAttempts'] += 1
        else:
            counts['recallAttempts'] += 1
            same(response['recallOptions'], NATIVE_OPTIONS, 'native recall policy changed')
            same(response['modelUsageBounds'], {'maxNativeSearches': 8, 'maxQueryPlanCalls': 8,
                  'maxQueryEmbeddingInputs': 48, 'maxGeneratedPlanTokens': 12288, 'maxRerankPairs': 160},
                 'native model usage bound changed')
            evidence = response['evidence']
            check_evidence(evidence, originals, NATIVE_OPTIONS)
        same(response['epoch'], epoch, 'native isolation epoch changed')
        for field, count in counts.items():
            same(response[field], count, 'native source or attempt count differs')
        if last and journal['status'] == 'error':
            require(index + 1 < len(expected) and journal.get('error_type') == 'InterruptedNativeAttempt',
                    'valid completed native response recorded as failure')
    return evidence if journal['status'] == 'ok' else None, epoch


def validate(report, split, protocol, base_protocol, preregistration, raw_questions, source_document,
             native_journals, initializations, ledger, work_name, protocol_sha256, cached_calls,
             input_counts, inputs_only=False):
    require(report.get('schema_version') == 1 and report.get('dev_only') is True
            and report.get('heldout_run') is False and report.get('reader_systems') == [SYSTEM]
            and type(report.get('framing_frozen')) is bool and type(report.get('complete')) is bool
            and (report['complete'] or inputs_only), 'expected a terminal source-only dev report')
    require(inputs_only or report['framing_frozen'], 'paid outputs lack frozen exact framing')
    require(not inputs_only or not report['complete'], 'inputs-only report claims completed generation')
    require('acceptance_correct' not in report and 'acceptance_met' not in report.get('summary', {}),
            'development result declares a generalization acceptance target')
    dev = dev_membership(split)
    rows = index_rows(report['rows'], dev, 'production report')
    require(set(raw_questions) == set(native_journals) == dev, 'original dev or native journal membership differs')
    check_production_protocol(protocol, base_protocol, preregistration)
    same(report['source_gate'], SOURCE_GATE, 'source gate changed')
    require(source_document['schema_version'] == 1 and source_document['dev_only'] is True
            and source_document['heldout_run'] is False, 'invalid source-only preparation scope')
    same(source_document['source_gate'], SOURCE_GATE, 'prepared source gate changed')
    source_rows = index_rows(source_document['rows'], dev, 'source-only preparation')
    same([row['qid'] for row in report['rows']], [row['qid'] for row in source_document['rows']], 'native question order changed')
    runtime_keys = {'native_identity', 'native_initializations', 'prepared_native_inputs', 'prepared_inputs',
                    'provider', 'input_count_provider', 'cost_provenance'}
    same({key: value for key, value in protocol.items() if key not in runtime_keys},
         source_document['protocol'], 'source-only protocol changed after retrieval')
    identity = protocol['native_identity']
    require(isinstance(identity['argv'], list) and len(identity['argv']) == 3
            and identity['argv'][1] == '--models'
            and artifact_path(identity['binary']['path']).resolve() == Path(identity['argv'][0]).resolve(),
            'native binary or models invocation differs')
    same(str(Path(identity['model_root']).resolve()), str(Path(identity['argv'][2]).resolve()), 'native model root differs from invocation')
    if report['framing_frozen']:
        provider = protocol['provider']
        require(isinstance(provider, dict) and set(provider) == {'scheme', 'hostname', 'sdk_version'}
                and provider['scheme'] in ('http', 'https') and isinstance(provider['hostname'], str)
                and 0 < len(provider['hostname']) <= 253 and not any(char.isspace() or ord(char) < 32 for char in provider['hostname'])
                and provider['sdk_version'] == '3.27.0', 'invalid frozen provider metadata')
        same(protocol['input_count_provider'], provider, 'input count and generation providers differ')
        same(protocol['cost_provenance'], 'predeclared quoted-rate estimate, not a billing receipt', 'cost provenance changed')
    init_artifacts = protocol['native_initializations']
    require(isinstance(init_artifacts, list) and init_artifacts
            and len({item['path'] for item in init_artifacts}) == len(init_artifacts)
            and set(initializations) == {item['path'] for item in init_artifacts}, 'native initialization membership differs')
    epochs, process_ids, timing_totals = {}, {}, {}
    native_calls, native_errors, local_seconds, unknown_timings = 0, 0, 0.0, 0
    for item in init_artifacts:
        initial = initializations[item['path']]
        require(initial['schema_version'] == 1 and initial['status'] == 'ok' and len(initial['exchanges']) == 1,
                'invalid native model initialization journal')
        same(initial['native_identity'], identity, 'native initialization identity changed')
        response = check_exchange(initial['exchanges'][0], {'command': 'initialize'})
        require(response is not None and response['status'] == 'ok' and response['epoch'] == 1,
                'native model initialization failed')
        check_profile(response['profile'], identity)
        timing_totals[item['path']] = check_common_response(response, dict.fromkeys(TIMING_FIELDS, 0.0))
        for field in ('sourceCount', 'chunkCount', 'sessionCount', 'ingestAttempts', 'recallAttempts'):
            same(response[field], 0, 'native initialization retained old history')
        pid = response['processId']
        require(type(pid) is int and pid > 0, 'invalid native process identity')
        epochs[item['path']], process_ids[item['path']] = 1, pid
        native_calls += 1
        local_seconds += response['modelLoadSeconds'] + response['resetSeconds']
    prepared_native, prepared_exact = [], []
    scores, calls, cache_keys, count_keys = [], [], set(), set()
    tokenizer = tiktoken.get_encoding('o200k_base')
    count = lambda text: len(tokenizer.encode(text))
    all_readers_terminal = all(row['systems'][SYSTEM]['arms'][ARM]['reader']['status'] != 'pending' for row in report['rows'])
    for row in report['rows']:
        qid, raw, source_row = row['qid'], raw_questions[row['qid']], source_rows[row['qid']]
        originals, metadata = check_source_row(source_row, raw)
        for field, original in (('question', 'question'), ('question_date', 'question_date'), ('qtype', 'question_type')):
            same(row[field], raw[original], f'original dev {field} changed')
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM}, 'unexpected system or arm')
        arm, journal = row['systems'][SYSTEM]['arms'][ARM], native_journals[qid]
        same(row['native_status'], journal['status'], 'native status differs from raw journal')
        init_key = journal['initialization']['path']
        require(init_key in initializations, 'unknown native initialization link')
        init_artifact = next(item for item in init_artifacts if item['path'] == init_key)
        evidence, epoch = check_native_journal(journal, source_row, originals, identity, init_artifact, epochs[init_key])
        epochs[init_key] = epoch
        for exchange in journal['exchanges']:
            response = exchange['response']
            if isinstance(response, dict) and response.get('command') is not None:
                same(response['processId'], process_ids[init_key], 'native process changed within one initialization')
                timing_totals[init_key] = check_common_response(response, timing_totals[init_key])
            native_calls += 1
            native_errors += not isinstance(response, dict) or response.get('status') != 'ok'
        if journal['elapsed_seconds'] is None:
            unknown_timings += 1
        else:
            local_seconds += journal['elapsed_seconds']
        native_sources = [] if evidence is None else reader_sources(evidence, metadata)
        by_sid = {source['source_id']: source for source in native_sources}
        native_order = [] if evidence is None else [source['sourceId'] for source in evidence['sources']]
        groups = [{'group_id': sid, 'selection_rank': rank, 'sources': [by_sid[sid]]} for rank, sid in enumerate(native_order)]
        same(arm['source_groups'], groups, 'native source membership, bytes or order changed before reader')
        same(arm['retained_sources'], sorted_sources(groups), 'retained source scope or chronological order changed')
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': groups}))
        same(arm['reader_input'], prompt, 'reader prompt contains changed or non-source material')
        same(arm['reader_input_sha256'], digest(prompt), 'reader input hash changed')
        same(arm['local_proxy_input_tokens'], count(prompt), 'diagnostic local token count changed')
        failure = 'native_bridge_failed' if evidence is None else ('native_empty_evidence' if not native_sources else None)
        checks, exact = arm['framing_checks'], 0
        require(isinstance(checks, list), 'invalid exact framing checks')
        if report['framing_frozen'] and failure is None:
            require(len(checks) == 1, 'exactly one unchanged-packet input count required')
            check = checks[0]
            request = {'model': 'gpt-6.1-sol', 'input': prompt}
            same(check['request'], request, 'input-count request differs from reader source packet')
            same(check['request_sha256'], digest(canonical(request)), 'input-count request fingerprint changed')
            key = digest(canonical({'namespace': 'sol61-input-count-v1', 'request': request}))
            same(check['cache_key'], key, 'input-count identity changed')
            require(key in input_counts, 'frozen input-count cache missing')
            same(check, input_counts[key], 'input-count record differs from raw RPC cache')
            count_keys.add(key)
            if check['status'] == 'ok':
                exact = check['response'].get('input_tokens')
                require(type(exact) is int and exact >= 0, 'invalid exact endpoint token count')
                if exact > 32256:
                    failure = 'exact_input_count_exceeds_cap'
            else:
                require(check['status'] == 'error' and check['response'] is None, 'invalid failed input-count record')
                failure = 'exact_input_count_failed'
        else:
            require(not checks, 'unexpected native-failure or unfrozen count RPC')
        cap = {'cap': 32768, 'exact_preflight_cap': 32256, 'input_tokens': exact,
               'selection_failure': failure is not None, 'failure_reason': failure, 'source_policy': ARM}
        same(arm['reader_cap'], cap, 'reader preflight or selection failure changed')
        same(arm['input_tokens'], exact, 'exact input count differs from report')
        require((arm['reader']['status'] == 'selection_failure') == (failure is not None), 'reader failure status differs from evidence/preflight')
        if inputs_only:
            require(all(arm[stage]['status'] in ('pending', 'selection_failure') and arm[stage].get('call') is None
                        and not arm[stage].get('transport_attempts') for stage in ('reader', 'judge')),
                    'inputs-only check requires unissued generation calls')
        else:
            require(all(arm[stage]['status'] != 'pending' for stage in ('reader', 'judge')), 'pending stage in terminal report')
        actual = recompute_scores(arm, allow_incomplete=inputs_only)
        same(arm['scores'], actual, 'answer or citation score changed')
        scores.append(actual)
        prepared_native.append({'qid': qid, 'reader_input_sha256': digest(prompt), 'native_journal': row['native_journal']})
        prepared_exact.append({'qid': qid, 'reader_input_sha256': digest(prompt), 'input_tokens': exact,
                               'input_count_cache_key': checks[0]['cache_key'] if checks else None,
                               'selection_failure': failure is not None})
        for stage in ('reader', 'judge'):
            output = arm[stage]
            require(not output.get('prior_attempts'), 'semantic resampling recorded')
            attempts, call = output.get('transport_attempts', []), output.get('call')
            require(isinstance(attempts, list) and len(attempts) <= 2, 'transport retry limit exceeded')
            if call is None:
                require(not attempts and output['status'] != 'ok', 'successful or retried stage lacks raw call')
                if stage == 'judge' and not inputs_only and arm['reader']['status'] != 'ok':
                    require(output['status'] == 'reader_failure', 'failed reader has invalid judge status')
                continue
            require(report['framing_frozen'] and failure is None, 'paid call issued before successful exact preflight')
            if stage == 'reader':
                expected_prompt, reserve_input = prompt, exact
            else:
                require(all_readers_terminal and arm['reader']['status'] == 'ok', 'judge reference accessed before readers were terminal')
                expected_prompt = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], raw['answer'])
                reserve_input = count(expected_prompt)
            require(not transport_failure(call) or len(attempts) == 2, 'retryable transport failure stopped before frozen retry limit')
            for attempt, raw_call in enumerate(attempts + [call]):
                previous = attempt < len(attempts)
                if previous:
                    require(transport_failure(raw_call) and raw_call.get('raw_text') == '', 'retry after output or disallowed error')
                recorded = {'status': 'error', 'raw_text': ''} if previous else output
                reserve = check_api_call(raw_call, recorded, protocol, expected_prompt, stage, reserve_input)
                if raw_call['status'] != 'budget_exhausted':
                    key = raw_call['cache_key']
                    require(key in cached_calls, 'paid API cache missing')
                    same(raw_call, cached_calls[key], 'raw paid call differs from durable cache')
                    same(raw_call['reserved_usd'], reserve, 'paid reservation changed')
                    cache_keys.add(key)
                calls.append((qid, stage, attempt, raw_call, reserve))
    same(protocol['prepared_native_inputs'], prepared_native, 'prepared native input manifest changed')
    if report['framing_frozen']:
        same(protocol['prepared_inputs'], prepared_exact, 'prepared exact framing manifest changed')
    else:
        require(not protocol.get('prepared_inputs'), 'unfrozen report has prepared exact inputs')
    require(cache_keys == set(cached_calls) and count_keys == set(input_counts), 'unreported or resampled API cache records')
    summary = independent_summary(report['rows'], scores)
    same(report['summary'], summary, 'failure-inclusive production summary changed')
    total, incremental = check_ledger(report, ledger, calls, preregistration, work_name, protocol_sha256)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'new_model_or_judge_calls': 0,
            'source_gate': SOURCE_GATE.copy(), 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'input_count_records_verified': len(count_keys),
            'paid_calls_verified': sum(call['status'] != 'budget_exhausted' for _, _, _, call, _ in calls),
            'native_requests_verified': native_calls, 'native_failed_requests': native_errors,
            'native_known_wall_seconds': local_seconds, 'native_unknown_duration_questions': unknown_timings,
            'native_local_compute_usd': None,
            'global_recorded_spend_usd': total, 'incremental_spend_usd': incremental, 'summary': summary,
            'limitations': ['Candle retrieval ranks are not independently reexecuted.',
                            'Frozen binary/source hashes do not independently reproduce their build relationship.',
                            'This repeatedly used development set does not estimate unseen-set generalization.',
                            'Recorded native stage counters are conservative bounds; local compute price is unknown.']}


def check_work(work, inputs_only=False):
    work = Path(work).expanduser().resolve()
    public = ROOT / 'benchmarks/public'
    paths = {'split': public / 'longmemeval_split.json', 'base': public / 'reader_protocol_v1.json',
             'preregistration': ROOT / 'benchmarks/results/preregistration_2026-10.json'}
    report = json.loads((work / 'answers.json').read_text())
    protocol = json.loads((work / 'protocol.json').read_text())
    source_document = json.loads((work / 'source_inputs.json').read_text())
    same(file_digest(paths['split']), SPLIT_SHA256, 'frozen dev split changed')
    same(file_digest(SOURCE), SOURCE_SHA256, 'pinned source release changed')
    preregistration = json.loads(paths['preregistration'].read_text())
    base = json.loads(paths['base'].read_text())
    required = [SOURCE, *paths.values(), work / 'source_inputs.json', work / 'native_identity.json',
                artifact_path(base['judge_source']['path'])]
    required.extend(public / name for name in ('longmemeval_production.py', 'longmemeval_dev_conversations.py',
                    'longmemeval_verifier.py', 'longmemeval_reader_api.py', 'longmemeval_answer_evidence.py',
                    'check_longmemeval_answers.py'))
    identity = protocol['native_identity']
    native_root = Path(identity['native_source_root']).expanduser().resolve()
    require(native_root != ROOT and (native_root / 'src/bin/conversation-eval.rs').is_file(), 'native source root is not an isolated frozen snapshot')
    inventory = {path.resolve() for path in (native_root / 'src').rglob('*.rs')}
    inventory.update((native_root / name).resolve() for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'))
    source_artifacts = identity['native_source_artifacts']
    frozen_sources = {artifact_path(item['path']).resolve(): item['sha256'] for item in source_artifacts}
    require(len(frozen_sources) == len(source_artifacts) and set(frozen_sources) == inventory
            and all(path.is_relative_to(native_root) and path.is_file() for path in inventory),
            'native snapshot source inventory is incomplete or escapes root')
    for path, expected_hash in frozen_sources.items():
        same(file_digest(path), expected_hash, 'native snapshot source hash changed')
    required.extend(inventory)
    required.extend(artifact_path(item['path']) for item in [identity['binary'], *identity['model_artifacts']])
    recorded = verify_artifacts({**report, 'protocol_path': str(work / 'protocol.json')}, work, required)
    same(json.loads((work / 'native_identity.json').read_text()), identity, 'saved native identity changed')
    require(all(file_digest(artifact_path(item['path'])) == item['sha256'] for item in source_document['artifacts']),
            'source preparation artifact changed')
    journals, initializations = {}, {}
    journal_paths = set()
    for row in report['rows']:
        item = row['native_journal']
        path = artifact_path(item['path']).resolve()
        require(path == (work / 'native' / (digest(row['qid']) + '.json')).resolve()
                and path in recorded and path not in journal_paths, 'native journal is not uniquely frozen')
        same(file_digest(path), item['sha256'], 'native journal hash changed')
        journal_paths.add(path)
        journals[row['qid']] = json.loads(path.read_text())
    require(journal_paths == {path.resolve() for path in (work / 'native').glob('*.json')}, 'unreported native attempt journal')
    for item in protocol['native_initializations']:
        path = artifact_path(item['path']).resolve()
        require(path.is_relative_to((work / 'native_initializations').resolve()) and path in recorded,
                'native initialization is not frozen within work')
        same(file_digest(path), item['sha256'], 'native initialization hash changed')
        initializations[item['path']] = json.loads(path.read_text())
    require({artifact_path(path).resolve() for path in initializations}
            == {path.resolve() for path in (work / 'native_initializations').glob('*.json')}, 'unreported native initialization journal')
    model_root = Path(identity['argv'][2]).resolve()
    for initial in initializations.values():
        for model in initial['exchanges'][0]['response']['profile']['models']:
            for receipt in model['files']:
                path = (model_root / receipt['path']).resolve()
                require(path.is_relative_to(model_root) and path in recorded
                        and path.stat().st_size == receipt['sizeBytes'], 'native model file size or frozen path differs')
            directory = model_root / model['model'].split('/')[-1]
            revision_file = directory / 'REVISION'
            if not revision_file.exists():
                revision_file = directory / '.cache/huggingface/download/config.json.metadata'
            require(revision_file.resolve() in recorded, 'model revision gate file was not frozen')
            with revision_file.open('rb') as stream:
                revision = stream.readline(256).decode().strip()
            same(revision, PINNED[directory.name], 'recorded model revision differs from pinned model')
    counts = {}
    for path in (work / 'input_counts').glob('*.json'):
        require(path.resolve() in recorded, 'input-count RPC was not frozen')
        counts[path.stem] = json.loads(path.read_text())
    if report['framing_frozen']:
        provider_path = work / 'provider.json'
        require(provider_path.resolve() in recorded, 'provider metadata is not frozen')
        same(json.loads(provider_path.read_text()), protocol['provider'], 'provider metadata changed')
    caches = {path.stem: json.loads(path.read_text()) for path in (work / 'cache').glob('*.json')}
    spend_path = work / 'spend.json'
    same(file_digest(spend_path), report['spend_ledger_sha256'], 'spend ledger hash changed')
    result = validate(report, json.loads(paths['split'].read_text()), protocol, base, preregistration,
                      load_dev_conversations(SOURCE, paths['split']), source_document, journals, initializations,
                      json.loads(spend_path.read_text()), work.name, report['protocol_sha256'], caches, counts, inputs_only)
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'),
                        'protocol_sha256': report['protocol_sha256'], 'checker_sha256': file_digest(__file__),
                        'spend_sha256': file_digest(spend_path)}
    output = work / ('inputs_validation.json' if inputs_only else 'validation.json')
    temporary = output.with_suffix('.json.writing')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--inputs-only', action='store_true', help='validate native packets and, when frozen, exact counts before paid generation')
    args = parser.parse_args()
    print(json.dumps(check_work(args.work, args.inputs_only), allow_nan=False))


if __name__ == '__main__':
    main()
