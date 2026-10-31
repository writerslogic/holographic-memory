# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Validate complete original dev histories, stage-specific requests, and official scores."""

import argparse
import copy
import json
import math
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_protocol, digest, file_digest, recompute_scores,
    require, response_text, same, sorted_sources,
)
from check_longmemeval_full_history import independent_summary, reasoning_rules
from check_longmemeval_role_reader import SOURCE_GATE, original_groups
from check_longmemeval_user_first import (
    check_historical_policy, check_references, dev_membership, index_rows, judge_prompt, verify_artifacts,
)
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_conversations import load_dev_conversations

SYSTEM, ARM = 'matched_gpt54', 'full_history'
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_complete_history_v1'
SOURCE = Path.home() / '.cache/warm-bench/downloads/longmemeval_s_cleaned.json'
SOURCE_POLICY = 'all complete original dev user and assistant turns including unpaired assistant blocks; no retrieval, filtering, or truncation'


def check_complete_protocol(protocol, base, preregistration, full_rules, complete_rules):
    check_protocol(base, preregistration)
    require(type(protocol.get('schema_version')) is int and protocol['schema_version'] == 1
            and protocol['dev_only'] is True and protocol['heldout_run'] is False
            and protocol['acceptance_correct'] == 100, 'invalid complete-history protocol scope')
    settings = copy.deepcopy(base['systems'][SYSTEM])
    del settings['reasoning_effort']
    settings.update(reader_reasoning_effort='high', judge_reasoning_effort='medium', reader_max_output_tokens=8192,
                    framing_verification='complete local prompt <=196096; actual API input <=196608')
    same(protocol['systems'], {SYSTEM: settings}, 'complete-history stage settings changed')
    same(protocol['judge_source'], base['judge_source'], 'official judge policy changed')
    same(protocol['reader_prompt'], base['reader_prompt'] + full_rules + complete_rules, 'complete-history reasoning policy changed')
    same(protocol['additional_reasoning_rules'], complete_rules, 'complete-history additional rules changed')
    same(protocol['source_policy'], SOURCE_POLICY, 'complete-history source policy changed')
    same(protocol['source_gate'], SOURCE_GATE, 'complete-history dev-only source gate changed')
    require('retrieval' not in protocol, 'complete history declares a retrieval policy')
    for field, value in (('token_cap', 196608), ('local_token_cap', 196096), ('framing_reserve_tokens', 512),
                         ('automatic_transport_retries', 0), ('reader_judge_concurrency', 1), ('new_spend_cap_usd', 50.0)):
        same(protocol[field], value, f'complete-history {field} changed')
    same(protocol['global_spend_caps_usd'], preregistration['spend_caps_usd'], 'global spend policy changed')


def check_stage_call(call, output, protocol, base, prompt, stage, tokenizer):
    if call is None:
        require(output['status'] != 'ok', 'successful stage lacks an API call')
        return 0.0
    settings = protocol['systems'][SYSTEM]
    expected = {'model': settings[stage + '_model'], 'input': prompt,
                'reasoning': {'effort': settings[stage + '_reasoning_effort']},
                'max_output_tokens': settings[stage + '_max_output_tokens'], 'store': False, 'truncation': 'disabled'}
    same(call['request'], expected, f'{stage} request differs from frozen stage policy')
    same(call['request_sha256'], digest(canonical(expected)), f'{stage} request fingerprint mismatch')
    require(isinstance(call['cache_key'], str) and call['cache_key'], 'missing API cache identity')
    require(call['status'] in ('ok', 'incomplete', 'error', 'budget_exhausted'), 'invalid API transport status')
    same(call['raw_text'], output['raw_text'], f'{stage} raw API text differs')
    response = call['response']
    if response is not None:
        same(response_text(response), output['raw_text'], f'{stage} response text differs')
        same(response.get('model'), expected['model'], f'{stage} response model differs')
    if output['status'] == 'ok':
        require(call['status'] == 'ok' and isinstance(response, dict) and response.get('status') == 'completed',
                'incomplete API call scored as successful')
    usage = call['usage']
    require(usage is None or isinstance(usage, dict), 'invalid API usage type')
    same(usage, response.get('usage') if isinstance(response, dict) else None, 'API usage differs from raw response')
    rates = base['cost']
    if usage:
        inputs, outputs = usage['input_tokens'], usage['output_tokens']
        cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
        require(all(type(value) is int and value >= 0 for value in (inputs, outputs, cached))
                and cached <= inputs, 'invalid API token usage')
        if stage == 'reader':
            require(inputs <= 196608, 'actual complete-history reader context cap exceeded')
        cost = ((inputs - cached) * rates['input_usd_per_million'] + cached * rates['cached_input_usd_per_million']
                + outputs * rates['output_usd_per_million']) / 1_000_000
    else:
        require(output['status'] != 'ok', 'successful API call lacks usage')
        if call.get('cost_is_conservative_reservation') is True:
            cost = ((len(tokenizer.encode(prompt)) + 512) * rates['input_usd_per_million']
                    + expected['max_output_tokens'] * rates['output_usd_per_million']) / 1_000_000
        else:
            require(call['status'] == 'budget_exhausted', 'unmeasured API call lacks conservative reservation')
            cost = 0.0
    same(call['cost_usd'], cost, f'{stage} API cost mismatch')
    return cost


def check_spend(report, ledger, calls, protocol, base, preregistration, work_name, official_sha256, tokenizer):
    require(type(ledger.get('schema_version')) is int and ledger['schema_version'] == 1, 'invalid spend ledger schema')
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'global spend caps changed')
    totals = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    expected_ids = {digest(f"full-history:{work_name}:{stage}:{row['qid']}") for row in report['rows'] for stage in ('reader', 'judge')}
    entries, own = {}, {}
    for entry in ledger['entries']:
        key, cost = entry['cache_key'], entry['cost_usd']
        require(isinstance(key, str) and key and key not in entries, 'duplicate or invalid spend cache identity')
        require(entry['category'] in totals and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                'invalid spend category or cost')
        entries[key] = entry
        totals[entry['category']] += cost
        if entry.get('call_id') in expected_ids:
            require(entry['call_id'] not in own, 'duplicate or resampled experiment call')
            own[entry['call_id']] = entry
    total = sum(totals.values())
    same(ledger['by_category_usd'], totals, 'spend category totals mismatch')
    same(ledger['total_usd'], total, 'spend total mismatch')
    same(report['global_recorded_spend_usd'], total, 'reported global spend mismatch')
    require(total <= 250 and totals['reader_judge_dev'] <= 120
            and all(totals[category] <= ledger['caps_usd'][category] for category in totals), 'global spend cap exceeded')
    incremental = sum(entry['cost_usd'] for entry in own.values())
    same(report['incremental_spend_usd'], incremental, 'complete-history own-call spend mismatch')
    require(incremental <= protocol['new_spend_cap_usd'] == 50.0, 'complete-history experiment spend cap exceeded')
    seen, issued = set(), set()
    for qid, stage, call, cost in calls:
        call_id = digest(f'full-history:{work_name}:{stage}:{qid}')
        key = digest(canonical({'request': call['request'], 'call_id': call_id, 'protocol_sha256': official_sha256}))
        require(call['cache_key'] == key and key not in seen, 'complete-history call cache identity mismatch')
        seen.add(key)
        if call['status'] == 'budget_exhausted':
            require(key not in entries and cost == 0, 'unissued API call charged')
            continue
        require(key in entries, 'API call missing from spend ledger')
        entry = entries[key]
        require(entry['call_id'] == call_id and entry['status'] == call['status']
                and entry['category'] == 'reader_judge_dev', 'API spend identity or status mismatch')
        same(entry['cost_usd'], cost, 'API spend charge mismatch')
        same(entry['usage'], call['usage'], 'API spend usage mismatch')
        reserve = ((len(tokenizer.encode(call['request']['input'])) + 512) * base['cost']['input_usd_per_million']
                   + call['request']['max_output_tokens'] * base['cost']['output_usd_per_million']) / 1_000_000
        same(entry['reserved_usd'], reserve, 'API spend reservation mismatch')
        require(cost <= reserve + 1e-12, 'API charge exceeded reservation')
        issued.add(call_id)
    require(set(own) == issued, 'unreported or resampled complete-history call')
    return total, incremental


def validate(report, split, protocol, base_protocol, preregistration, raw_questions, baseline_rows, history_rows,
             references, ledger, work_name, official_sha256, full_rules, complete_rules):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('complete') is True and report.get('dev_only') is True
            and report.get('heldout_run') is False and report.get('reader_systems') == [SYSTEM],
            'expected a completed dev-only complete-history report')
    dev = dev_membership(split)
    index_rows(report['rows'], dev, 'complete-history experiment')
    baseline = index_rows(baseline_rows, dev, 'baseline')
    history = index_rows(history_rows, dev, 'historical identities')
    require(set(raw_questions) == set(references) == dev, 'decoded dev source or reference membership mismatch')
    check_complete_protocol(protocol, base_protocol, preregistration, full_rules, complete_rules)
    prepared = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                 'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    same(protocol['prepared_inputs'], prepared, 'frozen prepared input manifest mismatch')
    tokenizer = tiktoken.get_encoding(protocol['systems'][SYSTEM]['tokenizer'])
    scores, calls = [], []
    for row in report['rows']:
        qid, raw = row['qid'], raw_questions[row['qid']]
        require(raw['question_id'] == qid, 'original dev question identity mismatch')
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM}, 'unexpected complete-history reader or arm')
        for field, raw_field in (('question', 'question'), ('question_date', 'question_date'), ('qtype', 'question_type')):
            same(row[field], baseline[qid][field], f'baseline {field} mismatch')
            same(row[field], raw[raw_field], f'original dev {field} mismatch')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        require(arm['reader']['status'] != 'pending' and arm['judge']['status'] != 'pending', 'pending stage in completed history')
        pool, _, orphans = original_groups(raw, set(history[qid]['ids']))
        expected = pool
        same(arm['source_groups'], expected, 'complete original exchange or orphan history changed')
        same(arm['retained_sources'], sorted_sources(expected), 'complete original source bytes or metadata changed')
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': expected}))
        count = len(tokenizer.encode(prompt))
        same(arm['reader_input'], prompt, 'complete-history reader input differs from original sources')
        same(arm['reader_input_sha256'], digest(prompt), 'complete-history input fingerprint mismatch')
        same(arm['input_tokens'], count, 'complete-history input token count mismatch')
        failed = count > 196096
        expected_cap = {'cap': 196608, 'local_cap': 196096, 'input_tokens': count, 'uncapped_input_tokens': count,
                        'dropped_group_ids': [], 'selection_failure': failed,
                        'failure_reason': 'complete_history_input_exceeds_cap' if failed else None,
                        'source_policy': 'complete_original_history'}
        same(arm['reader_cap'], expected_cap, 'complete-history cap or truncation policy changed')
        require((arm['reader']['status'] == 'selection_failure') == failed, 'complete-history reader selection status mismatch')
        same(row['candidate_groups'], len(pool), 'complete-history group count mismatch')
        same(row['conversation_groups'], len(pool), 'complete-history conversation count mismatch')
        same(row['orphan_assistant_groups'], orphans, 'complete-history orphan count mismatch')
        actual = recompute_scores(arm)
        same(arm['scores'], actual, 'answer or citation score mismatch')
        scores.append(actual)
        for stage in ('reader', 'judge'):
            output = arm[stage]
            require(not output.get('prior_attempts'), 'complete-history resampling or retries recorded')
            if stage == 'reader':
                request_input = prompt
                if failed:
                    require(output.get('call') is None, 'selection failure issued reader call')
            elif arm['reader']['status'] == 'ok':
                request_input = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], references[qid])
            else:
                require(output['status'] == 'reader_failure' and output.get('call') is None, 'failed reader issued judge call')
                request_input = None
            call = output.get('call')
            cost = check_stage_call(call, output, protocol, base_protocol, request_input, stage, tokenizer)
            if call is not None:
                calls.append((qid, stage, call, cost))
        require(type(history[qid]['scored']) is bool, 'invalid historical scoring label')
        gold = {digest(alias) for alias in history[qid]['ids'] if 'answer' in alias}
        retained = {group['frozen_selected_id_sha256'] for group in expected}
        same(row['scored_evidence'], history[qid]['scored'], 'historical scoring label mismatch')
        same(row['actual_complete'], not failed and gold <= retained, 'complete-history labeled coverage mismatch')
    summary = independent_summary(report['rows'], scores, True)
    same(report['summary'], summary, 'complete-history summary mismatch')
    total, incremental = check_spend(report, ledger, calls, protocol, base_protocol, preregistration,
                                     work_name, official_sha256, tokenizer)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'new_model_or_judge_calls': 0,
            'source_gate': SOURCE_GATE.copy(), 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'paid_calls_verified': len(calls), 'global_recorded_spend_usd': total,
            'incremental_spend_usd': incremental, 'summary': summary}


def check_work(work):
    work = Path(work).expanduser().resolve()
    report = json.loads((work / 'answers.json').read_text())
    require(report.get('complete') is True, 'complete-history experiment is not complete')
    public, results = ROOT / 'benchmarks/public', ROOT / 'benchmarks/results'
    paths = {'split': public / 'longmemeval_split.json', 'baseline': results / 'longmemeval_dev_answers_v3.json',
             'history': results / 'evidence_completeness_dev.json',
             'official': public / 'reader_protocol_v1.json', 'preregistration': results / 'preregistration_2026-10.json',
             'export': artifact_path('~/.cache/warm-bench/holo_dev/export_dev.json')}
    required = list(paths.values()) + [SOURCE] + [public / name for name in (
        'longmemeval_complete_history.py', 'longmemeval_role_reader.py', 'longmemeval_full_history.py',
        'longmemeval_user_first.py', 'longmemeval_reader_api.py', 'longmemeval_role_retrieval.py',
        'longmemeval_dev_conversations.py', 'longmemeval_answer_evidence.py', 'check_longmemeval_answers.py', 'proxies/beir_proxies.py')]
    recorded = verify_artifacts(report, work, required)
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    require(artifact_path(data['official']['judge_source']['path']).resolve() in recorded, 'official judge source fingerprint missing')
    same(file_digest(paths['split']), data['history']['split_sha256'], 'historical split fingerprint mismatch')
    dev = dev_membership(data['split'])
    baseline = index_rows(data['baseline']['rows'], dev, 'baseline')
    references = check_references(data['export'], baseline, dev, data['official'])
    historical = check_historical_policy(data['official'], data['preregistration'], data['baseline'], references)
    ledger_path = work / 'spend.json'
    same(file_digest(ledger_path), report['spend_ledger_sha256'], 'spend ledger fingerprint mismatch')
    result = validate(report, data['split'], json.loads((work / 'protocol.json').read_text()), data['official'],
                      data['preregistration'], load_dev_conversations(SOURCE, paths['split']), data['baseline']['rows'],
                      data['history']['rows'], references, json.loads(ledger_path.read_text()), work.name,
                      file_digest(paths['official']), reasoning_rules(public / 'longmemeval_full_history.py'),
                      complete_reasoning_rules(public / 'longmemeval_complete_history.py'))
    result['historical_policy'] = historical
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'), 'spend_sha256': file_digest(ledger_path),
                        'checker_sha256': file_digest(__file__),
                        'source_checker_sha256': file_digest(public / 'check_longmemeval_role_reader.py'),
                        'summary_checker_sha256': file_digest(public / 'check_longmemeval_full_history.py'),
                        'reference_checker_sha256': file_digest(public / 'check_longmemeval_user_first.py'),
                        'protocol_sha256': report['protocol_sha256']}
    output = work / 'validation.json'
    temporary = output.with_suffix('.json.writing')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return result


def complete_reasoning_rules(path):
    import ast

    module = ast.parse(path.read_text())
    values = [node.value for node in module.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'COMPLETE_REASONING_RULES' for target in node.targets)]
    require(len(values) == 1, 'missing or ambiguous complete-history reasoning policy')
    value = ast.literal_eval(values[0])
    require(isinstance(value, str) and value, 'invalid complete-history reasoning policy')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    print(json.dumps(check_work(args.work), allow_nan=False))


if __name__ == '__main__':
    main()
