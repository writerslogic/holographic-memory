# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Independently validate frozen original-source verifier packets and paid outcomes."""

import argparse
import ast
import copy
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_call, check_protocol, digest, file_digest, parse_answer, recompute_scores,
    require, response_text, same, sorted_sources, wilson,
)
from check_longmemeval_role_reader import SOURCE_GATE, original_groups, replay_retrieval
from check_longmemeval_user_first import (
    check_historical_policy, check_references, dev_membership, index_rows, judge_prompt, verify_artifacts,
)
from longmemeval_answer_evidence import _observation_key, reader_payload
from longmemeval_dev_conversations import load_dev_conversations

SYSTEM, ARM = 'sol61_reader', 'citation_union'
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_verifier_v1'
SOURCE = Path.home() / '.cache/warm-bench/downloads/longmemeval_s_cleaned.json'
PRIOR_NAMES = ('v3_operand', 'full_history', 'role_retrieval', 'complete_history')
RATES = {'reader': {'input': 2.0, 'cached_input': 0.1, 'cache_write_30m': 2.5, 'cache_write_12h': 5.0, 'output': 10.0},
         'judge': {'input': 2.5, 'cached_input': 0.25, 'output': 15.0}}
SOURCE_POLICY = 'uniform four-prior citation union, source premises, then question-only RRF counterevidence; complete original turns'
TRANSPORT_RETRY_POLICY = {
    'max_retries': 2, 'http_statuses': [429], 'http_status_range': [500, 599],
    'error_types': ['APIConnectionError', 'APITimeoutError'], 'requires_status': 'error',
    'requires_response_none': True, 'requires_usage_none': True, 'backoff_seconds': [2, 8],
    'retry_call_id_suffix': ':transport-retry:{retry_number}', 'completed_response_is_final': True,
    'semantic_retries': False,
}


def identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def source_valid(source):
    if not (isinstance(source, dict) and identity(source.get('source_id')) and identity(source.get('session_id'))
            and all(type(source.get(field)) is int and 0 <= source[field] < 2 ** 31 for field in ('session_index', 'turn_index'))
            and source.get('role') in ('user', 'assistant') and isinstance(source.get('text'), str)
            and len(source['text'].encode()) <= 16 * 1024 * 1024 and isinstance(source.get('observation_time'), str)):
        return False
    try:
        _observation_key(source['observation_time'])
    except ValueError:
        return False
    return True


def replay_sources(groups, seeds, counter_order, serialize, count, cap=24000):
    require(all(isinstance(seed, dict) and set(seed) == {'source_groups', 'cited_source_ids'} for seed in seeds),
            'prior seeds contain forbidden fields')
    sources, owners, aliases, by_group, sessions = {}, {}, {}, {}, {}
    for group in groups:
        gid = group['group_id']
        by_group[gid] = group
        for source in group['sources']:
            require(source_valid(source), 'invalid original source metadata')
            sid = source['source_id']
            if sid in sources:
                require(sources[sid] == source, 'conflicting original source identity')
            else:
                sources[sid], owners[sid] = source, gid
                sessions.setdefault(source['session_index'], []).append(sid)
            if group.get('frozen_selected_id_sha256') is not None:
                aliases.setdefault(group['frozen_selected_id_sha256'], []).append(sid)
    for members in sessions.values():
        members.sort(key=lambda sid: sources[sid]['turn_index'])
    require(len(set(counter_order)) == len(counter_order) and set(counter_order) <= set(by_group), 'invalid counterevidence order')
    diagnostics = {'mapped_citations': [], 'unmapped_citations': [], 'ambiguous_citations': [],
                   'dropped_cited_source_ids': [], 'dropped_premise_source_ids': [], 'dropped_counterevidence_source_ids': []}
    cited = []
    for prior_index, seed in enumerate(seeds):
        previous = {}
        for group in seed['source_groups']:
            for source in group['sources']:
                if isinstance(source, dict) and identity(source.get('source_id')):
                    previous.setdefault(source['source_id'], []).append((source, group.get('frozen_selected_id_sha256')))
        citations = list(dict.fromkeys(sid for sid in seed['cited_source_ids'] if identity(sid)))
        for citation in citations:
            matches, reason = [], 'citation_not_in_prior_packet'
            records = previous.get(citation, [])
            if records and any(source != records[0][0] for source, _ in records):
                records, reason = [], 'conflicting_prior_source_identity'
            for old, alias in records:
                if not source_valid(old):
                    reason = 'invalid_prior_source_metadata'
                    continue
                if citation in sources:
                    fields = ('role', 'text', 'observation_time', 'session_index', 'turn_index', 'session_id')
                    if all(old[field] == sources[citation][field] for field in fields):
                        matches.append(citation)
                    else:
                        reason = 'prior_source_conflicts_with_original'
                else:
                    reason = 'no_original_alias_content_match'
                    if identity(alias):
                        matches.extend(sid for sid in aliases.get(alias, []) if all(
                            old[field] == sources[sid][field] for field in ('role', 'text', 'observation_time')))
            matches = list(dict.fromkeys(matches))
            item = {'prior_index': prior_index, 'cited_source_id': citation}
            if matches:
                item['original_source_ids'] = matches
                diagnostics['mapped_citations'].append(item)
                if len(matches) > 1:
                    diagnostics['ambiguous_citations'].append(copy.deepcopy(item))
                cited.extend(matches)
            else:
                diagnostics['unmapped_citations'].append({**item, 'reason': reason})
        diagnostics['unmapped_citations'].extend({'prior_index': prior_index, 'cited_source_id': None,
                                                 'reason': 'invalid_citation_identity'}
                                                for citation in seed['cited_source_ids'] if not identity(citation))
    cited = list(dict.fromkeys(cited))
    premise = []
    for sid in cited:
        source = sources[sid]
        before = [candidate for candidate in sessions[source['session_index']] if sources[candidate]['role'] == 'user'
                  and sources[candidate]['turn_index'] < source['turn_index']]
        if before:
            premise.append(before[-1])
    for session in dict.fromkeys(sources[sid]['session_index'] for sid in cited):
        premise.extend([sid for sid in sessions[session] if sources[sid]['role'] == 'user'][:2])
    counter = [source['source_id'] for gid in counter_order for source in by_group[gid]['sources']]

    def packet(selected):
        selected_set = set(selected)
        order = list(dict.fromkeys(owners[sid] for sid in selected))
        result = []
        for rank, gid in enumerate(order):
            members = [copy.deepcopy(sources[sid]) for sid in dict.fromkeys(source['source_id'] for source in by_group[gid]['sources'])
                       if sid in selected_set and owners[sid] == gid]
            result.append({**copy.deepcopy(by_group[gid]), 'selection_rank': rank, 'sources': members})
        return result

    tokens = count(serialize(packet([])))
    overflow = tokens > cap
    selected, tried = [], set()
    for category, candidates in (('cited', cited), ('premise', premise), ('counterevidence', counter)):
        for sid in candidates:
            if sid in tried:
                continue
            tried.add(sid)
            proposed = cap + 1 if overflow else count(serialize(packet(selected + [sid])))
            if proposed <= cap:
                selected.append(sid)
                tokens = proposed
            else:
                diagnostics['dropped_' + category + '_source_ids'].append(sid)
    kept = set(selected)
    diagnostics['dropped_prior_citations'] = [
        {**item, 'dropped_original_source_ids': [sid for sid in item['original_source_ids'] if sid not in kept]}
        for item in diagnostics['mapped_citations'] if any(sid not in kept for sid in item['original_source_ids'])]
    return {'source_groups': packet(selected), 'input_tokens': tokens, 'cap': cap,
            'selection_failure': overflow or not selected,
            'failure_reason': 'question_exceeds_cap' if overflow else ('empty_evidence_packet' if not selected else None),
            'diagnostics': diagnostics,
            'source_policy': 'complete cited original turns; preceding/opening user premises; question-ranked counterevidence'}


def literal(path, symbol):
    nodes = ast.parse(path.read_text()).body
    values = [node.value for node in nodes if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets)]
    require(len(values) == 1, 'missing or ambiguous frozen prompt rule')
    value = ast.literal_eval(values[0])
    require(isinstance(value, str) and value, 'invalid frozen prompt rule')
    return value


def check_verifier_protocol(protocol, base, preregistration, rules):
    check_protocol(base, preregistration)
    expected = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'acceptance_correct': 100,
                'token_cap': 32768, 'exact_preflight_cap': 32256, 'framing_reserve_tokens': 512,
                'local_proxy_cap': 24000, 'source_policy': SOURCE_POLICY, 'source_gate': SOURCE_GATE,
                'service_tier': 'default', 'reader_prompt_cache_options': {'mode': 'explicit'},
                'new_spend_cap_usd': 14.0, 'automatic_transport_retries': 2,
                'transport_retry_policy': TRANSPORT_RETRY_POLICY, 'reader_judge_concurrency': 1,
                'rates_usd_per_million': RATES,
                'systems': {SYSTEM: {'reader_model': 'gpt-6.1-sol', 'judge_model': 'gpt-5.4-2026-03-05',
                                     'reader_reasoning_effort': 'high', 'judge_reasoning_effort': 'medium',
                                     'reader_max_output_tokens': 4096, 'judge_max_output_tokens': 2048}},
                'source_selection': {'prior_names': list(PRIOR_NAMES), 'prior_answers_visible': False,
                                     'question_only_counterevidence': 'reserved user prior then equal-weight RRF60 fused ranking',
                                     'whole_turns_only': True, 'cap_overflow': 'selection_failure; no source clipping'},
                'input_count': {'endpoint': '/v1/responses/input_tokens', 'request_fields': ['model', 'input'],
                                'settings_framing_reserve': 512}}
    for field, value in expected.items():
        same(protocol[field], value, f'verifier protocol {field} changed')
    same(protocol['judge_source'], base['judge_source'], 'official judge policy changed')
    same(protocol['reader_prompt'], base['reader_prompt'] + ''.join(rules), 'verifier prompt rules changed')
    same(protocol['additional_reasoning_rules'], rules[-1], 'verifier additional rules changed')
    same(protocol['global_spend_caps_usd'], preregistration['spend_caps_usd'], 'global spend caps changed')
    require(any('alias' in item and 'immutable' in item for item in protocol['limitations']), 'reader alias limitation missing')


def snapshot_rows(snapshots, dev):
    require(isinstance(snapshots, list) and [snapshot.get('name') for snapshot in snapshots] == list(PRIOR_NAMES),
            'uniform four-prior snapshots required')
    indexed = []
    for index, snapshot in enumerate(snapshots):
        require(set(snapshot) == {'schema_version', 'dev_only', 'heldout_run', 'name', 'source_path',
                                  'source_sha256_at_snapshot', 'system', 'arm', 'rows'}, 'prior snapshot contains forbidden fields')
        require(snapshot['schema_version'] == 1 and snapshot['dev_only'] is True and snapshot['heldout_run'] is False
                and snapshot['system'] == 'matched_gpt54' and snapshot['arm'] == ('operand' if index == 0 else 'full_history'),
                'invalid prior snapshot scope')
        require(isinstance(snapshot['source_path'], str) and snapshot['source_path']
                and isinstance(snapshot['source_sha256_at_snapshot'], str) and len(snapshot['source_sha256_at_snapshot']) == 64
                and all(char in '0123456789abcdef' for char in snapshot['source_sha256_at_snapshot']), 'invalid prior source fingerprint')
        rows = index_rows(snapshot['rows'], dev, 'stripped prior')
        for row in rows.values():
            require(set(row) == {'qid', 'question', 'question_date', 'reader_status', 'source_groups', 'cited_source_ids'},
                    'prior row contains answer, judgment, or forbidden metadata')
            require(row['reader_status'] in ('ok', 'error', 'selection_failure') and isinstance(row['source_groups'], list)
                    and isinstance(row['cited_source_ids'], list) and all(isinstance(value, str) and value for value in row['cited_source_ids']),
                    'invalid or unfinished stripped prior')
            for group in row['source_groups']:
                require(isinstance(group, dict) and set(group) <= {'group_id', 'frozen_selected_id_sha256', 'selection_rank', 'sources'}
                        and isinstance(group.get('sources'), list), 'prior group contains forbidden metadata')
                for source in group['sources']:
                    require(isinstance(source, dict) and set(source) == {'source_id', 'session_id', 'session_index', 'turn_index',
                                                                       'observation_time', 'role', 'text'}, 'prior source contains forbidden metadata')
        indexed.append(rows)
    return indexed


def independent_summary(rows, scores, complete):
    require(len(rows) == len(scores) == 100 and len({row['qid'] for row in rows}) == 100, 'accuracy requires all 100 dev rows')
    arms = [row['systems'][SYSTEM]['arms'][ARM] for row in rows]
    correct = sum(score['correct'] for score in scores)
    terminal = all(arm[stage]['status'] != 'pending' for arm in arms for stage in ('reader', 'judge'))
    counts = [arm['input_tokens'] for arm in arms]
    return {'n': 100, 'correct': correct, 'answer_accuracy': correct / 100, 'answer_accuracy_ci95': wilson(correct, 100),
            'acceptance_correct': 100, 'acceptance_met': correct == 100 and complete and terminal,
            'reader_status': dict(Counter(arm['reader']['status'] for arm in arms)),
            'judge_status': dict(Counter(arm['judge']['status'] for arm in arms)),
            'input_tokens': {'median': statistics.median(counts), 'max': max(counts)},
            'abstentions': sum(score['abstain'] for score in scores),
            'exact_quote_claims': sum(score['citation_valid_claims'] for score in scores),
            'total_claims': sum(score['citation_total_claims'] for score in scores),
            'user_occurrences': sum(source['role'] == 'user' for arm in arms for source in arm['retained_sources']),
            'assistant_occurrences': sum(source['role'] == 'assistant' for arm in arms for source in arm['retained_sources'])}


def fee(usage, stage):
    require(isinstance(usage, dict), 'invalid API usage')
    incoming, outgoing = usage.get('input_tokens'), usage.get('output_tokens')
    details = usage.get('input_tokens_details') or {}
    require(isinstance(details, dict), 'invalid input token details')
    cached, writes = details.get('cached_tokens', 0), details.get('cache_write_tokens', 0)
    require(all(type(value) is int and value >= 0 for value in (incoming, outgoing, cached, writes))
            and cached + writes <= incoming, 'invalid disjoint cache token counts')
    conservative = False
    if stage == 'reader':
        classes = details.get('cache_write_tokens_details')
        if classes is None:
            short, long = 0, writes
            conservative = writes > 0
        else:
            require(isinstance(classes, dict) and set(classes) <= {'ttl_30m_tokens', 'ttl_12h_tokens'}, 'unknown cache write class')
            short, long = classes.get('ttl_30m_tokens', 0), classes.get('ttl_12h_tokens', 0)
            require(all(type(value) is int and value >= 0 for value in (short, long)) and short + long == writes,
                    'cache write classes do not sum to writes')
        cost = ((incoming - cached - writes) * 2.0 + cached * 0.1 + short * 2.5 + long * 5.0 + outgoing * 10.0) / 1e6
    else:
        require(writes == 0, 'unexpected judge cache writes')
        cost = ((incoming - cached) * 2.5 + cached * 0.25 + outgoing * 15.0) / 1e6
    return cost, conservative, writes > 0


def check_api_call(call, output, protocol, prompt, stage, input_tokens):
    settings = protocol['systems'][SYSTEM]
    request = {'model': settings[stage + '_model'], 'input': prompt,
               'reasoning': {'effort': settings[stage + '_reasoning_effort']},
               'max_output_tokens': settings[stage + '_max_output_tokens'], 'store': False,
               'truncation': 'disabled', 'service_tier': 'default'}
    if stage == 'reader':
        request['prompt_cache_options'] = {'mode': 'explicit'}
    same(call['request'], request, f'{stage} request differs from frozen verifier policy')
    same(call['request_sha256'], digest(canonical(request)), f'{stage} request fingerprint mismatch')
    same(call['raw_text'], output['raw_text'], 'raw API text differs from stage output')
    response, usage = call['response'], call['usage']
    if response is not None:
        same(response_text(response), output['raw_text'], 'raw response text differs')
        same(usage, response.get('usage'), 'usage differs from raw response')
        require(response.get('service_tier') in (None, 'default'), 'response service tier differs from standard pricing')
    else:
        require(usage is None, 'API usage has no raw response')
    reserve = ((input_tokens + 512) * (5.0 if stage == 'reader' else 2.5)
               + request['max_output_tokens'] * (10.0 if stage == 'reader' else 15.0)) / 1e6
    if call['status'] == 'budget_exhausted':
        require(output['status'] == 'error' and response is None and usage is None, 'issued output recorded as budget failure')
        same(call['cost_usd'], 0.0, 'unissued call has paid cost')
    elif call.get('cost_is_conservative_reservation') is True:
        require(call['status'] == 'error' and output['status'] == 'error', 'successful output retains unmeasured reservation')
        same(call['cost_usd'], reserve, 'conservative reservation differs from frozen worst-case rates')
    else:
        require(isinstance(response, dict), 'measured call lacks raw response')
        cost, conservative, writes = fee(usage, stage)
        same(call['cost_usd'], cost, 'verifier API usage cost mismatch')
        same(call['cost_is_conservative'], conservative, 'conservative write-price flag mismatch')
        same(call['unexpected_cache_writes'], writes, 'unexpected cache-write flag mismatch')
        same(call['cost_is_conservative_reservation'], False, 'measured API call marked as reservation')
        problems = []
        if response.get('model') != request['model']:
            problems.append('response_model_differs_from_requested_alias')
        if stage == 'reader' and usage['input_tokens'] > 32768:
            problems.append('actual_input_exceeds_cap')
        if writes:
            problems.append('unexpected_explicit_mode_cache_writes')
        same(call['validation_failures'], problems, 'API validation failure reasons mismatch')
        expected_status = 'validation_failure' if problems else ('ok' if response.get('status') == 'completed' else 'incomplete')
        same(call['status'], expected_status, 'API transport or validation status mismatch')
        if problems or expected_status != 'ok':
            require(output['status'] == 'error', 'invalid API response scored as successful')
    return reserve


def transport_failure(call):
    if call.get('status') != 'error' or call.get('response') is not None or call.get('usage') is not None:
        return False
    status = call.get('http_status')
    return ((type(status) is int and (status == 429 or 500 <= status <= 599))
            or call.get('error_type') in ('APIConnectionError', 'APITimeoutError'))


def attempt_identity(work_name, qid, stage, attempt):
    base = digest(f'sol61-verifier:{work_name}:{stage}:{qid}')
    return base if attempt == 0 else base + f':transport-retry:{attempt}'


def check_ledger(report, ledger, calls, protocol, preregistration, work_name, protocol_sha256):
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'global spend caps changed')
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
            require(entry['call_id'] not in own, 'duplicate or resampled verifier call')
            own[entry['call_id']] = entry
    total = sum(totals.values())
    same(ledger['by_category_usd'], totals, 'ledger category totals mismatch')
    same(ledger['total_usd'], total, 'global ledger total mismatch')
    same(report['global_recorded_spend_usd'], total, 'reported global spend mismatch')
    incremental = sum(entry['cost_usd'] for entry in own.values())
    same(report['incremental_spend_usd'], incremental, 'reported verifier spend mismatch')
    require(total <= 250 and totals['reader_judge_dev'] <= 120 and incremental <= 14
            and all(value <= ledger['caps_usd'][key] for key, value in totals.items()), 'verifier or global spend cap exceeded')
    seen, issued = set(), set()
    for qid, stage, attempt, call, reserve in calls:
        call_id = attempt_identity(work_name, qid, stage, attempt)
        expected_key = digest(canonical({'namespace': 'sol61-verifier-call-v1', 'request': call['request'],
                                         'call_id': call_id, 'protocol_sha256': protocol_sha256}))
        require(call['call_id'] == call_id and call['protocol_sha256'] == protocol_sha256
                and call['cache_key'] == expected_key and expected_key not in seen, 'verifier call provenance mismatch')
        seen.add(expected_key)
        if call['status'] == 'budget_exhausted':
            require(expected_key not in entries and call['cost_usd'] == 0, 'unissued verifier call charged')
            continue
        require(expected_key in entries, 'verifier call missing from ledger')
        entry = entries[expected_key]
        require(entry['call_id'] == call_id and entry['status'] == call['status'] and entry['category'] == 'reader_judge_dev',
                'verifier ledger status or identity mismatch')
        same(entry['cost_usd'], call['cost_usd'], 'verifier ledger charge mismatch')
        same(entry['usage'], call['usage'], 'verifier ledger usage mismatch')
        same(entry['reserved_usd'], reserve, 'verifier ledger reservation mismatch')
        require(call['cost_usd'] <= reserve + 1e-12, 'verifier charge exceeded worst-case reservation')
        issued.add(call_id)
    require(set(own) == issued, 'unreported or resampled verifier ledger call')
    return total, incremental


def verify_prior_proofs(snapshots, proofs, indexed, dev, base, rules):
    require([proof.get('name') for proof in proofs] == list(PRIOR_NAMES), 'four frozen reader proofs required')
    missing = 0
    for index, (snapshot, proof, rows) in enumerate(zip(snapshots, proofs, indexed, strict=True)):
        same({key: value for key, value in proof.items() if key != 'rows'},
             {key: value for key, value in snapshot.items() if key != 'rows'}, 'prior reader proof metadata differs from seed snapshot')
        previous = index_rows(proof['rows'], dev, 'prior reader-only proof')
        prefix = base['reader_prompt'] + (rules[0] if index else '') + (rules[1] if index == 3 else '')
        for qid, seed in rows.items():
            item = previous[qid]
            require(set(item) == {'qid', 'reader'} and isinstance(item['reader'], dict), 'reader proof contains forbidden metadata')
            reader = item['reader']
            require({'status', 'raw_text', 'parsed'} <= set(reader) <= {'status', 'raw_text', 'parsed', 'call'},
                    'reader-only proof contains forbidden fields')
            same(reader['status'], seed['reader_status'], 'prior reader status differs from frozen proof')
            require(isinstance(reader['raw_text'], str), 'invalid prior raw reader text')
            if reader['status'] == 'ok':
                parsed = parse_answer(reader['raw_text'])
                same(reader['parsed'], parsed, 'prior parsed response differs from strict raw JSON')
            else:
                require(reader['parsed'] in (None, {'answer': '', 'abstain': False, 'claims': []}), 'failed prior retains asserted claims')
                parsed = {'claims': []}
            citations = list(dict.fromkeys(citation['source_id'] for claim in parsed['claims'] for citation in claim['citations']))
            same(seed['cited_source_ids'], citations, 'seed citations do not derive from frozen raw reader proof')
            call = reader.get('call')
            if call is None:
                missing += 1
                continue
            prompt = prefix + canonical(reader_payload(seed, {'source_groups': seed['source_groups']}))
            if index == 3:
                from check_longmemeval_complete_history import check_stage_call

                policy = {'systems': {'matched_gpt54': {'reader_model': 'gpt-5.4-2026-03-05',
                    'reader_reasoning_effort': 'high', 'reader_max_output_tokens': 8192}}}
                check_stage_call(call, reader, policy, base, prompt, 'reader', tiktoken.get_encoding('o200k_base'))
            else:
                check_call(call, reader['raw_text'], reader['status'], base, prompt,
                           base['systems']['matched_gpt54']['reader_model'], 'reader')
            if reader['status'] != 'ok' and call['status'] == 'ok':
                try:
                    parse_answer(reader['raw_text'])
                except ValueError:
                    pass
                else:
                    raise ValueError('valid completed prior response recorded as reader failure')
    return {'reader_rows_verified': 400, 'missing_original_call_records': missing}


def validate(report, split, protocol, base_protocol, preregistration, raw_questions, history_rows,
             snapshots, reader_proofs, references, ledger, work_name, protocol_sha256, rules,
             cached_calls, input_counts, inputs_only=False):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('dev_only') is True and report.get('heldout_run') is False
            and report.get('reader_systems') == [SYSTEM] and (report.get('complete') is True or inputs_only),
            'expected a completed dev-only verifier report')
    dev = dev_membership(split)
    index_rows(report['rows'], dev, 'verifier experiment')
    history = index_rows(history_rows, dev, 'historical ranking')
    require(set(raw_questions) == set(references) == dev, 'dev source or reference membership mismatch')
    check_verifier_protocol(protocol, base_protocol, preregistration, rules)
    priors = snapshot_rows(snapshots, dev)
    proof_limits = verify_prior_proofs(snapshots, reader_proofs, priors, dev, base_protocol, rules)
    prepared = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                 'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens'],
                 'input_count_cache_key': row['systems'][SYSTEM]['arms'][ARM]['framing_checks'][0]['cache_key']}
                for row in report['rows']]
    same(protocol['prepared_inputs'], prepared, 'frozen verifier input manifest mismatch')
    tokenizer = tiktoken.get_encoding('o200k_base')
    count = lambda text: len(tokenizer.encode(text))
    scores, calls, cache_keys, count_keys = [], [], set(), set()
    for row in report['rows']:
        qid, raw = row['qid'], raw_questions[row['qid']]
        same(raw['question_id'], qid, 'original dev question identity mismatch')
        for field, original in (('question', 'question'), ('question_date', 'question_date'), ('qtype', 'question_type')):
            same(row[field], raw[original], f'original dev {field} mismatch')
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM}, 'unexpected verifier system or arm')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        pool, by_alias, orphans = original_groups(raw, set(history[qid]['ids']))
        aliases = list(dict.fromkeys(history[qid]['baseline_ranking']))
        require(set(aliases) == set(by_alias), 'historical counterevidence prior membership mismatch')
        ranked = replay_retrieval(row['question'], pool, [by_alias[alias] for alias in aliases])
        order = list(dict.fromkeys(ranked['reserved_user_prior'] + ranked['fused_ranking']))
        counter = [pool[index]['group_id'] for index in order]
        same(row['counterevidence_group_ids'], counter, 'question-only counterevidence order mismatch')
        same(row['conversation_groups'], len(pool), 'original conversation group count mismatch')
        same(row['orphan_assistant_groups'], orphans, 'orphan assistant group count mismatch')
        seeds = []
        for previous in priors:
            same(previous[qid]['question'], row['question'], 'prior question mismatch')
            same(previous[qid]['question_date'], row['question_date'], 'prior observation date mismatch')
            seeds.append({key: previous[qid][key] for key in ('source_groups', 'cited_source_ids')})
        serialize = lambda groups: protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': groups}))
        selected = replay_sources(pool, seeds, counter, serialize, count)
        same(arm['source_groups'], selected['source_groups'], 'verifier source selection or original bytes changed')
        same(arm['source_selection'], {key: value for key, value in selected.items() if key != 'source_groups'}, 'source selection diagnostics changed')
        same(arm['retained_sources'], sorted_sources(arm['source_groups']), 'retained verifier source metadata changed')
        prompt = serialize(selected['source_groups'])
        same(arm['reader_input'], prompt, 'verifier prompt contains changed or non-source material')
        same(arm['reader_input_sha256'], digest(prompt), 'verifier input fingerprint mismatch')
        same(arm['local_proxy_input_tokens'], selected['input_tokens'], 'local selection proxy count changed')
        require(isinstance(arm['framing_checks'], list) and len(arm['framing_checks']) == 1, 'exactly one frozen preflight required')
        check = arm['framing_checks'][0]
        request = {'model': 'gpt-6.1-sol', 'input': prompt}
        same(check['request'], request, 'input-count request differs from source-only reader input')
        same(check['request_sha256'], digest(canonical(request)), 'input-count request hash mismatch')
        key = digest(canonical({'namespace': 'sol61-input-count-v1', 'request': request}))
        same(check['cache_key'], key, 'input-count cache identity mismatch')
        require(key in input_counts, 'frozen input-count record missing')
        same(check, input_counts[key], 'input-count record differs from frozen RPC cache')
        count_keys.add(key)
        require(check['status'] in ('ok', 'error'), 'invalid input-count status')
        if check['status'] == 'ok':
            exact = check['response'].get('input_tokens')
            require(type(exact) is int and exact >= 0, 'invalid exact endpoint token count')
        else:
            require(check['response'] is None, 'failed input-count RPC has scored response')
            exact = 0
        failure = selected['failure_reason'] if selected['selection_failure'] else None
        if check['status'] != 'ok':
            failure = 'exact_input_count_failed'
        elif exact > 32256:
            failure = 'exact_input_count_exceeds_cap'
        expected_cap = {'cap': 32768, 'exact_preflight_cap': 32256, 'input_tokens': exact,
                        'selection_failure': failure is not None, 'failure_reason': failure, 'source_policy': 'citation_union'}
        same(arm['reader_cap'], expected_cap, 'verifier preflight or input cap changed')
        same(arm['input_tokens'], exact, 'exact endpoint count differs from prepared count')
        require((arm['reader']['status'] == 'selection_failure') == (failure is not None), 'reader selection failure status mismatch')
        if inputs_only:
            require(all(arm[stage]['status'] in ('pending', 'selection_failure') and arm[stage].get('call') is None
                        and not arm[stage].get('transport_attempts') for stage in ('reader', 'judge')),
                    'inputs-only validation requires unissued generation calls')
        else:
            require(all(arm[stage]['status'] != 'pending' for stage in ('reader', 'judge')), 'pending generation stage in completed report')
        actual = recompute_scores(arm, allow_incomplete=inputs_only)
        same(arm['scores'], actual, 'answer or citation score mismatch')
        scores.append(actual)
        for stage in ('reader', 'judge'):
            output = arm[stage]
            require(not output.get('prior_attempts'), 'verifier semantic resampling recorded')
            attempts = output.get('transport_attempts', [])
            require(isinstance(attempts, list) and len(attempts) <= 2, 'transport retry limit exceeded')
            if stage == 'judge' and arm['reader']['status'] != 'ok' and not inputs_only:
                require(output['status'] == 'reader_failure', 'failed verifier reader has invalid judge status')
            call = output.get('call')
            if call is None:
                require(not attempts, 'transport attempts lack final call')
                require(output['status'] != 'ok', 'successful verifier stage lacks call record')
                continue
            if stage == 'reader':
                require(failure is None, 'failed preflight issued a paid reader call')
                expected_prompt, reserve_input = prompt, exact
            else:
                require(arm['reader']['status'] == 'ok', 'failed reader issued official judge call')
                expected_prompt = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], references[qid])
                reserve_input = count(expected_prompt)
            require(not transport_failure(call) or len(attempts) == 2, 'retryable transport error stopped before frozen retry limit')
            for attempt, raw_call in enumerate(attempts + [call]):
                previous = attempt < len(attempts)
                if previous:
                    require(isinstance(raw_call, dict) and transport_failure(raw_call)
                            and raw_call.get('raw_text') == '', 'retry after a returned response or disallowed transport failure')
                recorded = {'status': 'error', 'raw_text': ''} if previous else output
                reserve = check_api_call(raw_call, recorded, protocol, expected_prompt, stage, reserve_input)
                if raw_call['status'] != 'budget_exhausted':
                    require(raw_call['cache_key'] in cached_calls, 'paid verifier API cache missing')
                    require(canonical(raw_call) == canonical(cached_calls[raw_call['cache_key']]), 'paid cache differs from raw call record')
                    same(raw_call['reserved_usd'], reserve, 'durable cached reservation differs from frozen rates')
                    cache_keys.add(raw_call['cache_key'])
                calls.append((qid, stage, attempt, raw_call, reserve))
    require(cache_keys == set(cached_calls), 'unreported or resampled paid API cache')
    require(count_keys == set(input_counts), 'unexpected input-count RPC cache')
    summary = independent_summary(report['rows'], scores, report['complete'])
    same(report['summary'], summary, 'verifier summary mismatch')
    total, incremental = check_ledger(report, ledger, calls, protocol, preregistration, work_name, protocol_sha256)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'source_gate': SOURCE_GATE.copy(),
            'new_model_or_judge_calls': 0, 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'input_count_records_verified': len(count_keys),
            'paid_calls_verified': sum(call['status'] != 'budget_exhausted' for _, _, _, call, _ in calls),
            'source_selection_replayed': True, 'prior_reader_proofs': proof_limits,
            'reader_identity': {'requested_alias': 'gpt-6.1-sol', 'immutable_snapshot': False},
            'global_recorded_spend_usd': total, 'incremental_spend_usd': incremental, 'summary': summary}


def check_work(work, inputs_only=False):
    work = Path(work).expanduser().resolve()
    report = json.loads((work / 'answers.json').read_text())
    require(report.get('complete') is True or inputs_only, 'verifier experiment is not complete')
    public, results = ROOT / 'benchmarks/public', ROOT / 'benchmarks/results'
    paths = {'split': public / 'longmemeval_split.json', 'baseline': results / 'longmemeval_dev_answers_v3.json',
             'history': results / 'evidence_completeness_dev.json', 'official': public / 'reader_protocol_v1.json',
             'preregistration': results / 'preregistration_2026-10.json',
             'export': artifact_path('~/.cache/warm-bench/holo_dev/export_dev.json')}
    required = list(paths.values()) + [SOURCE, work / 'priors/manifest.json'] + [public / name for name in (
        'longmemeval_verifier.py', 'longmemeval_verifier_sources.py', 'longmemeval_role_reader.py',
        'longmemeval_reader_api.py', 'longmemeval_user_first.py', 'longmemeval_role_retrieval.py',
        'longmemeval_dev_conversations.py', 'longmemeval_answer_evidence.py', 'longmemeval_dev_sources.py',
        'longmemeval_reader_budget.py', 'check_longmemeval_answers.py', 'proxies/beir_proxies.py',
        'longmemeval_full_history.py', 'longmemeval_complete_history.py')]
    recorded = verify_artifacts(report, work, required)
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    protocol = json.loads((work / 'protocol.json').read_text())
    manifest = json.loads((work / 'priors/manifest.json').read_text())
    same(report['prior_snapshots'], manifest, 'report prior manifest changed')
    same(protocol['prior_snapshots'], manifest, 'protocol prior manifest changed')
    require([item['name'] for item in manifest] == list(PRIOR_NAMES), 'four uniform frozen prior names required')
    snapshots, proofs = [], []
    for item in manifest:
        require(set(item) == {'name', 'path', 'sha256', 'reader_proof_path', 'reader_proof_sha256'}, 'invalid frozen prior manifest')
        for field, hash_field, target in (('path', 'sha256', snapshots), ('reader_proof_path', 'reader_proof_sha256', proofs)):
            path = artifact_path(item[field])
            require(path.resolve().is_relative_to((work / 'priors').resolve()) and path.resolve() in recorded,
                    'prior snapshot or proof missing from frozen artifacts')
            same(file_digest(path), item[hash_field], 'prior snapshot or proof fingerprint mismatch')
            target.append(json.loads(path.read_text()))
    rule_specs = [('longmemeval_full_history.py', 'REASONING_RULES'), ('longmemeval_complete_history.py', 'COMPLETE_REASONING_RULES')]
    rules = []
    expected_sources = []
    for filename, symbol in rule_specs:
        path = public / filename
        value = literal(path, symbol)
        rules.append(value)
        expected_sources.append({'path': str(path.relative_to(ROOT)), 'sha256': file_digest(path), 'symbol': symbol, 'rule_sha256': digest(value)})
    same(protocol['prompt_rule_sources'], expected_sources, 'frozen generic rule sources changed')
    rules.append(literal(public / 'longmemeval_verifier.py', 'VERIFIER_RULES'))
    require(artifact_path(data['official']['judge_source']['path']).resolve() in recorded, 'official judge source fingerprint missing')
    same(file_digest(paths['split']), data['history']['split_sha256'], 'historical split fingerprint mismatch')
    dev = dev_membership(data['split'])
    baseline = index_rows(data['baseline']['rows'], dev, 'baseline')
    references = check_references(data['export'], baseline, dev, data['official'])
    historical = check_historical_policy(data['official'], data['preregistration'], data['baseline'], references)
    caches = {path.stem: json.loads(path.read_text()) for path in (work / 'cache').glob('*.json')}
    counts = {}
    for path in (work / 'input_counts').glob('*.json'):
        require(path.resolve() in recorded, 'input-count RPC record is not frozen')
        counts[path.stem] = json.loads(path.read_text())
    ledger_path = work / 'spend.json'
    same(file_digest(ledger_path), report['spend_ledger_sha256'], 'spend ledger fingerprint mismatch')
    result = validate(report, data['split'], protocol, data['official'], data['preregistration'],
                      load_dev_conversations(SOURCE, paths['split']), data['history']['rows'], snapshots, proofs,
                      references, json.loads(ledger_path.read_text()), work.name, report['protocol_sha256'], rules,
                      caches, counts, inputs_only)
    result['historical_policy'] = historical
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'), 'spend_sha256': file_digest(ledger_path),
                        'checker_sha256': file_digest(__file__), 'protocol_sha256': report['protocol_sha256']}
    output = work / ('inputs_validation.json' if inputs_only else 'validation.json')
    temporary = output.with_suffix('.json.writing')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--inputs-only', action='store_true', help='validate frozen packets before any generation call')
    args = parser.parse_args()
    print(json.dumps(check_work(args.work, args.inputs_only), allow_nan=False))


if __name__ == '__main__':
    main()
