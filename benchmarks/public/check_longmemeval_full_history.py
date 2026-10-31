# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Validate a completed full-history dev experiment without model or corpus access."""

import argparse
import ast
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_call, check_protocol, digest, file_digest,
    recompute_scores, require, same, wilson,
)
from check_longmemeval_user_first import (
    check_historical_policy, check_references, check_sources, dev_membership, index_rows, judge_prompt, verify_artifacts,
)
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_sources import load_dev_sources

SYSTEM = 'matched_gpt54'
ARM = 'full_history'
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_full_history_v1'


def reasoning_rules(path):
    module = ast.parse(path.read_text())
    values = [node.value for node in module.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'REASONING_RULES' for target in node.targets)]
    require(len(values) == 1, 'missing or ambiguous frozen reasoning policy')
    value = ast.literal_eval(values[0])
    require(isinstance(value, str) and value, 'invalid frozen reasoning policy')
    return value


def check_full_protocol(protocol, base, preregistration, rules):
    check_protocol(base, preregistration)
    require(type(protocol.get('schema_version')) is int and protocol['schema_version'] == 1
            and protocol['dev_only'] is True and protocol['heldout_run'] is False
            and protocol['acceptance_correct'] == 100, 'invalid full-history protocol scope')
    same(protocol['systems'], {SYSTEM: base['systems'][SYSTEM]}, 'full-history reader settings mismatch')
    same(protocol['judge_source'], base['judge_source'], 'official judge policy changed')
    same(protocol['reader_prompt'], base['reader_prompt'] + rules, 'full-history reasoning policy changed')
    expected = {'token_cap': 131072, 'local_token_cap': 130560, 'framing_reserve_tokens': 512,
                'automatic_transport_retries': 0, 'reader_judge_concurrency': 1, 'new_spend_cap_usd': 40.0}
    for field, value in expected.items():
        same(protocol[field], value, f'full-history {field} changed')
    same(protocol['global_spend_caps_usd'], preregistration['spend_caps_usd'], 'global spend policy changed')
    same(protocol['source_policy'], 'all complete dev user observations and all available saved v1/v3 assistant turns',
         'full-history source policy changed')


def independent_summary(rows, scores, complete):
    require(len(rows) == len(scores) == 100 and len({row['qid'] for row in rows}) == 100,
            'accuracy requires all 100 dev questions')
    arms = [row['systems'][SYSTEM]['arms'][ARM] for row in rows]
    correct = sum(score['correct'] for score in scores)
    terminal = all(arm['reader']['status'] != 'pending' and arm['judge']['status'] != 'pending' for arm in arms)
    by_type = {}
    for row, score in zip(rows, scores):
        group = by_type.setdefault(row['qtype'], {'n': 0, 'correct': 0})
        group['n'] += 1
        group['correct'] += score['correct']
    counts = [arm['input_tokens'] for arm in arms]
    return {'n': 100, 'correct': correct, 'answer_accuracy': correct / 100,
            'answer_accuracy_ci95': wilson(correct, 100), 'acceptance_correct': 100,
            'acceptance_met': correct == 100 and complete and terminal,
            'reader_status': dict(Counter(arm['reader']['status'] for arm in arms)),
            'judge_status': dict(Counter(arm['judge']['status'] for arm in arms)), 'by_type': by_type,
            'input_tokens': {'median': statistics.median(counts), 'max': max(counts)},
            'user_occurrences': sum(source['role'] == 'user' for arm in arms for source in arm['retained_sources']),
            'assistant_occurrences': sum(source['role'] == 'assistant' for arm in arms for source in arm['retained_sources']),
            'coverage': {'scored': sum(row['scored_evidence'] for row in rows),
                         'actual_complete': sum(row['scored_evidence'] and row['actual_complete'] for row in rows)},
            'abstentions': sum(score['abstain'] for score in scores),
            'exact_quote_claims': sum(score['citation_valid_claims'] for score in scores),
            'total_claims': sum(score['citation_total_claims'] for score in scores)}


def check_spend(report, ledger, calls, protocol, base, preregistration, work_name, official_sha256, tokenizer):
    require(type(ledger.get('schema_version')) is int and ledger['schema_version'] == 1, 'invalid spend ledger schema')
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'global spend caps changed')
    totals = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    entries = {}
    call_ids = {digest(f"full-history:{work_name}:{stage}:{row['qid']}") for row in report['rows'] for stage in ('reader', 'judge')}
    variant_entries = {}
    for entry in ledger['entries']:
        key, cost = entry['cache_key'], entry['cost_usd']
        require(isinstance(key, str) and key and key not in entries, 'duplicate or invalid spend cache identity')
        require(entry['category'] in totals and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                'invalid spend amount or category')
        entries[key] = entry
        totals[entry['category']] += cost
        if entry.get('call_id') in call_ids:
            require(entry['call_id'] not in variant_entries, 'duplicate or resampled experiment call')
            variant_entries[entry['call_id']] = entry
    same(ledger['by_category_usd'], totals, 'spend category totals mismatch')
    total = sum(totals.values())
    same(ledger['total_usd'], total, 'spend total mismatch')
    same(report['global_recorded_spend_usd'], total, 'reported global spend mismatch')
    require(total <= 250 and totals['reader_judge_dev'] <= 120
            and all(totals[category] <= ledger['caps_usd'][category] for category in totals), 'global spend cap exceeded')
    variant = sum(entry['cost_usd'] for entry in variant_entries.values())
    same(report['incremental_spend_usd'], variant, 'experiment call spend mismatch')
    require(variant <= protocol['new_spend_cap_usd'] == 40.0, 'experiment spend cap exceeded')
    seen, issued = set(), set()
    for qid, stage, call, cost in calls:
        call_id = digest(f'full-history:{work_name}:{stage}:{qid}')
        key = digest(canonical({'request': call['request'], 'call_id': call_id, 'protocol_sha256': official_sha256}))
        require(call['cache_key'] == key and key not in seen, 'experiment call cache identity mismatch')
        seen.add(key)
        require(call['status'] in ('ok', 'incomplete', 'error', 'budget_exhausted'), 'invalid experiment API status')
        if call['status'] == 'budget_exhausted':
            require(key not in entries and cost == 0, 'unissued experiment call charged')
            continue
        require(key in entries, 'experiment call missing from spend ledger')
        entry = entries[key]
        require(entry['call_id'] == call_id and entry['status'] == call['status']
                and entry['category'] == 'reader_judge_dev', 'experiment spend identity or status mismatch')
        same(entry['cost_usd'], cost, 'experiment call charge mismatch')
        same(entry['usage'], call['usage'], 'experiment call usage mismatch')
        reserve = ((len(tokenizer.encode(call['request']['input'])) + 512) * base['cost']['input_usd_per_million']
                   + call['request']['max_output_tokens'] * base['cost']['output_usd_per_million']) / 1_000_000
        same(entry['reserved_usd'], reserve, 'experiment reservation mismatch')
        require(cost <= reserve + 1e-12, 'experiment charge exceeded reservation')
        issued.add(call_id)
    require(set(variant_entries) == issued, 'unreported or resampled experiment ledger call')
    return total, variant


def validate(report, split, protocol, base_protocol, preregistration, archives, baseline_rows, v1_rows,
             history_rows, references, ledger, work_name, official_sha256, rules):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('complete') is True and report.get('dev_only') is True
            and report.get('heldout_run') is False and report.get('reader_systems') == [SYSTEM],
            'expected a complete dev-only full-history report')
    dev = dev_membership(split)
    index_rows(report['rows'], dev, 'full-history experiment')
    baseline = index_rows(baseline_rows, dev, 'baseline')
    v1 = index_rows(v1_rows, dev, 'v1 snapshots')
    history = index_rows(history_rows, dev, 'historical identities')
    require(set(archives) == set(references) == dev, 'dev source or reference membership mismatch')
    check_full_protocol(protocol, base_protocol, preregistration, rules)
    prepared = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                 'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    same(protocol['prepared_inputs'], prepared, 'frozen prepared input manifest mismatch')
    tokenizer = tiktoken.get_encoding(protocol['systems'][SYSTEM]['tokenizer'])
    call_protocol = {**protocol, 'cost': base_protocol['cost']}
    scores, calls = [], []
    for row in report['rows']:
        qid = row['qid']
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM},
                'unexpected full-history system or arm')
        for field in ('qtype', 'question', 'question_date'):
            same(row[field], baseline[qid][field], f'baseline {field} mismatch')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        require(arm['reader']['status'] != 'pending' and arm['judge']['status'] != 'pending', 'pending stage in completed report')
        candidates = list(dict.fromkeys(tid for session in archives[qid] for tid, _ in session['turns']))
        retained = check_sources(qid, arm, archives[qid], [v1[qid], baseline[qid]], candidates,
                                 max_candidates=len(candidates), require_full=True)
        same(row['candidate_groups'], len(candidates), 'full-history candidate count mismatch')
        serialized = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': arm['source_groups']}))
        count = len(tokenizer.encode(serialized))
        same(arm['reader_input'], serialized, 'reader input differs from full retained history')
        same(arm['reader_input_sha256'], digest(serialized), 'full-history input fingerprint mismatch')
        same(arm['input_tokens'], count, 'full-history input token count mismatch')
        cap = arm['reader_cap']
        for field, value in (('cap', 131072), ('local_cap', 130560), ('input_tokens', count),
                             ('uncapped_input_tokens', count), ('dropped_group_ids', []),
                             ('source_policy', 'full_user_history_with_saved_assistants')):
            same(cap[field], value, f'full-history reader cap {field} mismatch')
        failed = count > protocol['local_token_cap']
        same(cap['selection_failure'], failed, 'full-history selection failure mismatch')
        same(cap['failure_reason'], 'complete_history_input_exceeds_cap' if failed else None, 'full-history cap failure reason mismatch')
        require((arm['reader']['status'] == 'selection_failure') == failed, 'full-history reader selection status mismatch')
        actual = recompute_scores(arm)
        same(arm['scores'], actual, 'answer or citation score mismatch')
        scores.append(actual)
        for stage in ('reader', 'judge'):
            result = arm[stage]
            require(not result.get('prior_attempts'), 'full-history resampling or retries recorded')
            if stage == 'reader':
                prompt = serialized
                if failed:
                    require(result.get('call') is None, 'selection failure issued a model call')
            else:
                if arm['reader']['status'] == 'ok':
                    prompt = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], references[qid])
                else:
                    require(result['status'] == 'reader_failure' and result.get('call') is None, 'failed reader issued judge call')
                    prompt = None
            call = result.get('call')
            cost = check_call(call, result['raw_text'], result['status'], call_protocol, prompt,
                              protocol['systems'][SYSTEM][stage + '_model'], stage)
            if call is not None:
                if stage == 'reader' and call['usage'] is not None:
                    require(call['usage']['input_tokens'] <= 131072, 'actual full-history input token cap exceeded')
                calls.append((qid, stage, call, cost))
        require(type(history[qid]['scored']) is bool, 'invalid historical scoring label')
        gold = {digest(tid) for tid in history[qid]['ids'] if 'answer' in tid}
        same(row['scored_evidence'], history[qid]['scored'], 'historical scoring label mismatch')
        same(row['actual_complete'], not failed and gold <= retained, 'full-history labeled coverage mismatch')
    summary = independent_summary(report['rows'], scores, True)
    same(report['summary'], summary, 'full-history summary mismatch')
    total, variant = check_spend(report, ledger, calls, protocol, base_protocol, preregistration, work_name, official_sha256, tokenizer)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'corpus_read': False,
            'new_model_or_judge_calls': 0, 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'paid_calls_verified': len(calls), 'global_recorded_spend_usd': total,
            'incremental_spend_usd': variant, 'summary': summary}


def check_work(work):
    work = Path(work).expanduser().resolve()
    report = json.loads((work / 'answers.json').read_text())
    require(report.get('complete') is True, 'full-history experiment is not complete')
    public, results = ROOT / 'benchmarks/public', ROOT / 'benchmarks/results'
    paths = {'split': public / 'longmemeval_split.json', 'baseline': results / 'longmemeval_dev_answers_v3.json',
             'v1': results / 'longmemeval_dev_answers_v1.json', 'history': results / 'evidence_completeness_dev.json',
             'official': public / 'reader_protocol_v1.json', 'preregistration': results / 'preregistration_2026-10.json',
             'manifest': ROOT / 'target/conversation-geometry-validation/models/manifest.json',
             'export': artifact_path('~/.cache/warm-bench/holo_dev/export_dev.json')}
    required = list(paths.values()) + [public / name for name in
        ('longmemeval_full_history.py', 'longmemeval_user_first.py', 'longmemeval_reader_api.py',
         'longmemeval_answer_evidence.py', 'longmemeval_dev_sources.py', 'check_longmemeval_answers.py')]
    recorded = verify_artifacts(report, work, required)
    data = {name: json.loads(path.read_text()) for name, path in paths.items() if name != 'manifest'}
    judge_source = artifact_path(data['official']['judge_source']['path'])
    require(judge_source.resolve() in recorded, 'official judge source fingerprint missing')
    require(file_digest(paths['manifest']) == data['history']['source_manifest_sha256']
            and file_digest(paths['split']) == data['history']['split_sha256'], 'dev archive or split fingerprint mismatch')
    dev = dev_membership(data['split'])
    baseline = index_rows(data['baseline']['rows'], dev, 'baseline')
    references = check_references(data['export'], baseline, dev, data['official'])
    historical = check_historical_policy(data['official'], data['preregistration'], data['baseline'], references)
    ledger_path = work / 'spend.json'
    require(file_digest(ledger_path) == report['spend_ledger_sha256'], 'spend ledger fingerprint mismatch')
    protocol = json.loads((work / 'protocol.json').read_text())
    result = validate(report, data['split'], protocol, data['official'], data['preregistration'],
                      load_dev_sources(paths['manifest'].parent, paths['split']), data['baseline']['rows'],
                      data['v1']['rows'], data['history']['rows'], references, json.loads(ledger_path.read_text()),
                      work.name, file_digest(paths['official']), reasoning_rules(public / 'longmemeval_full_history.py'))
    same(result['summary']['user_occurrences'], json.loads(paths['manifest'].read_text())['n_turns'], 'full-history user total mismatch')
    result['historical_policy'] = historical
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'), 'spend_sha256': file_digest(ledger_path),
                        'checker_sha256': file_digest(__file__), 'source_checker_sha256': file_digest(Path(__file__).with_name('check_longmemeval_user_first.py')),
                        'protocol_sha256': report['protocol_sha256']}
    output = work / 'validation.json'
    temporary = output.with_suffix('.json.writing')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    print(json.dumps(check_work(args.work), allow_nan=False))


if __name__ == '__main__':
    main()
