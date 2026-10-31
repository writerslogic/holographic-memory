# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Independently validate a completed dev-only user-first reader experiment offline."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_call, check_protocol, digest, file_digest,
    recompute_scores, require, same, sorted_sources, wilson,
)
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_sources import load_dev_sources

SYSTEM = 'matched_gpt54'
ARM = 'user_first'
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_user_first_v1'


def dev_membership(split):
    for name, count in (('dev', 100), ('heldout', 400)):
        ids = split.get(name)
        require(isinstance(ids, list) and len(ids) == count
                and all(isinstance(qid, str) and qid for qid in ids)
                and len(set(ids)) == count, f'invalid frozen {name} membership')
    dev = set(split['dev'])
    require(not dev.intersection(split['heldout']), 'overlapping dev and heldout membership')
    return dev


def index_rows(rows, dev, label):
    require(isinstance(rows, list) and len(rows) == 100
            and all(isinstance(row, dict) and isinstance(row.get('qid'), str) for row in rows),
            f'{label} requires 100 dev rows')
    indexed = {row['qid']: row for row in rows}
    require(len(indexed) == 100 and set(indexed) == dev, f'{label} dev membership mismatch')
    return indexed


def verify_artifacts(report, work, required=()):
    protocol_path = artifact_path(report['protocol_path'])
    require(protocol_path.resolve() == (work / 'protocol.json').resolve(), 'experiment protocol path mismatch')
    require(file_digest(protocol_path) == report['protocol_sha256'], 'experiment protocol fingerprint mismatch')
    paths = set()
    for artifact in report['artifacts']:
        path = artifact_path(artifact['path'])
        require(path.resolve() not in paths, 'duplicate frozen artifact')
        paths.add(path.resolve())
        require(file_digest(path) == artifact['sha256'], f'artifact fingerprint mismatch: {path.name}')
    require(all(Path(path).resolve() in paths for path in required), 'missing required frozen artifact')
    return paths


def judge_prompt(protocol, row, answer, reference):
    official = protocol['judge_source']
    template = official['abstention_template'] if '_abs' in row['qid'] else official['templates'][row['qtype']]
    return template.format(question=row['question'], answer=reference, response=answer)


def check_references(exported, baseline, dev, protocol):
    questions = index_rows(exported['questions'], dev, 'dev reference export')
    references = {}
    for qid, row in baseline.items():
        reference = questions[qid]
        require(reference['question'] == row['question'] and reference['date'] == row['question_date'],
                'reference question mismatch')
        verified = 0
        for arm in row['systems'][SYSTEM]['arms'].values():
            if arm['judge']['status'] == 'ok':
                expected = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], reference['answer'])
                same(arm['judge']['call']['request']['input'], expected, 'reference differs from frozen judge input')
                verified += 1
        require(verified > 0, 'reference lacks a frozen successful judge input')
        references[qid] = reference['answer']
    return references


def check_historical_policy(protocol, preregistration, baseline, references):
    same(baseline['protocol_sha256'], preregistration['reader_protocol_sha256'],
         'historical baseline protocol differs from preregistration')
    settings = protocol['systems'][SYSTEM]
    counts = {'reader_inputs_verified': 0, 'reader_requests_verified': 0, 'judge_requests_verified': 0}
    for row in baseline['rows']:
        for arm in row['systems'][SYSTEM]['arms'].values():
            reader = arm['reader']
            if reader['status'] != 'selection_failure':
                require(arm['reader_input'].startswith(protocol['reader_prompt']),
                        'historical reader prompt differs from current policy')
                counts['reader_inputs_verified'] += 1
            for stage in ('reader', 'judge'):
                output = arm[stage]
                attempts = output.get('prior_attempts', []) + ([output['call']] if output.get('call') is not None else [])
                if not attempts:
                    continue
                if stage == 'reader':
                    expected_input = arm['reader_input']
                else:
                    require(reader['status'] == 'ok', 'historical judge call for failed reader')
                    expected_input = judge_prompt(protocol, row, reader['parsed']['answer'], references[row['qid']])
                expected = {'model': settings[stage + '_model'], 'input': expected_input,
                            'reasoning': {'effort': settings['reasoning_effort']},
                            'max_output_tokens': settings[stage + '_max_output_tokens'],
                            'store': False, 'truncation': 'disabled'}
                for call in attempts:
                    same(call['request'], expected, f'historical {stage} request differs from current policy')
                    same(call['request_sha256'], digest(canonical(expected)),
                         f'historical {stage} request fingerprint mismatch')
                    counts[stage + '_requests_verified'] += 1
    return {'preregistered_protocol_sha256': baseline['protocol_sha256'], **counts}


def check_sources(qid, arm, sessions, snapshots, candidates, max_candidates=64, require_full=False):
    known = {}
    for snapshot in snapshots:
        for old in snapshot['systems'][SYSTEM]['arms'].values():
            for group in old['source_groups']:
                key = group['frozen_selected_id_sha256']
                if key in known:
                    same(group['sources'], known[key], 'saved source disagreement')
                known[key] = group['sources']
    observations = {}
    for archive_index, session in enumerate(sessions):
        for tid, text in session['turns']:
            turn_index = int(tid.rsplit('_', 1)[1]) - 1
            require(turn_index >= 0, 'invalid archive turn ordinal')
            observations.setdefault(digest(tid), []).append({
                'archive_index': archive_index, 'turn_index': turn_index,
                'observation_time': session['date'], 'role': 'user', 'text': text})
    session_map = {}
    fields = ('text', 'observation_time', 'turn_index', 'role')
    for key, saved in known.items():
        require(key in observations, 'saved user handle absent from archive')
        saved_users = [source for source in saved if source['role'] == 'user']
        aliases = observations[key]
        require(len(saved_users) == len(aliases), 'saved user alias count mismatch')
        for alias, source in zip(aliases, saved_users):
            same({field: alias[field] for field in fields}, {field: source[field] for field in fields},
                 'saved user differs from dev archive')
            identity = (source['session_index'], source['session_id'])
            previous = session_map.setdefault(alias['archive_index'], identity)
            require(previous == identity, 'saved session mapping conflict')
    expected = {}
    for key, aliases in observations.items():
        users = []
        for alias in aliases:
            archive_index = alias['archive_index']
            original = session_map.get(archive_index)
            index = original[0] if original else archive_index
            prefix = 'lme' if original else 'lme-dev-archive'
            users.append({
                'source_id': digest(canonical([prefix + '-source-v1', qid, index, alias['turn_index']])),
                'session_id': original[1] if original else digest(canonical([prefix + '-session-v1', qid, index])),
                'session_index': index, **{field: alias[field] for field in fields}})
        expected[key] = users
    require(isinstance(candidates, list) and 1 <= len(candidates) <= max_candidates
            and all(isinstance(tid, str) for tid in candidates) and len(set(candidates)) == len(candidates),
            'invalid frozen candidate pool')
    ranks = {digest(tid): rank for rank, tid in enumerate(candidates)}
    require(set(ranks) <= set(expected), 'candidate source absent from archive')
    if require_full:
        same(list(ranks), list(expected), 'full history candidate order or coverage mismatch')
    groups = arm['source_groups']
    require(isinstance(groups, list) and 1 <= len(groups) <= len(candidates), 'invalid retained groups')
    seen = set()
    for group in groups:
        key = group['frozen_selected_id_sha256']
        require(key in ranks and key not in seen, 'unknown or duplicate retained group')
        seen.add(key)
        same(group['selection_rank'], ranks[key], 'retained selection rank mismatch')
        sources = group['sources']
        users = [source for source in sources if source['role'] == 'user']
        same(users, expected[key], 'retained user aliases differ from dev archive')
        same(group['group_id'], digest(canonical(['lme-reader-group-v1', qid,
                                                  [source['source_id'] for source in users]])),
             'retained group identity mismatch')
        allowed = known.get(key, expected[key])
        allowed_by_id = {source['source_id']: source for source in allowed}
        for source in sources:
            require(source['role'] in ('user', 'assistant') and source['source_id'] in allowed_by_id,
                    'source missing from saved group')
            same(source, allowed_by_id[source['source_id']], 'retained source bytes or metadata changed')
        if require_full:
            same(sources, allowed, 'full history source or saved assistant missing')
        else:
            retained_ids = {source['source_id'] for source in sources}
            same(sources, [source for source in allowed if source['source_id'] in retained_ids],
                 'retained source order mismatch')
    require(digest(candidates[0]) in seen, 'frozen anchor missing')
    require(not require_full or seen == set(ranks), 'full history user group missing')
    require(len({group['group_id'] for group in groups}) == len(groups), 'duplicate retained group identity')
    same(arm['retained_sources'], sorted_sources(groups), 'retained sources differ from source groups')
    return set(seen)


def independent_summary(rows, scores, complete):
    require(len(rows) == len(scores) == 100, 'summary requires 100 dev questions')
    correct = sum(score['correct'] for score in scores)
    arms = [row['systems'][SYSTEM]['arms'][ARM] for row in rows]
    by_type = {}
    for row, score in zip(rows, scores):
        counts = by_type.setdefault(row['qtype'], {'n': 0, 'correct': 0})
        counts['n'] += 1
        counts['correct'] += score['correct']
    return {
        'n': 100, 'correct': correct, 'answer_accuracy': correct / 100,
        'answer_accuracy_ci95': wilson(correct, 100), 'acceptance_correct': 100,
        'acceptance_met': correct == 100 and complete, 'by_type': by_type,
        'reader_status': dict(Counter(arm['reader']['status'] for arm in arms)),
        'judge_status': dict(Counter(arm['judge']['status'] for arm in arms)),
        'coverage': {'scored': sum(row['scored_evidence'] for row in rows),
                     'baseline_actual_complete': sum(row['scored_evidence'] and row['baseline_actual_complete'] for row in rows),
                     'actual_complete': sum(row['scored_evidence'] and row['actual_complete'] for row in rows)},
        'abstentions': sum(score['abstain'] for score in scores),
        'exact_quote_claims': sum(score['citation_valid_claims'] for score in scores),
        'total_claims': sum(score['citation_total_claims'] for score in scores),
    }


def check_ledger(report, ledger, calls, protocol, preregistration, official_sha256, work_name, tokenizer):
    require(type(ledger.get('schema_version')) is int and ledger['schema_version'] == 1,
            'unsupported spend ledger schema')
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'spend caps differ from preregistration')
    totals = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    entries = {}
    for entry in ledger['entries']:
        key, cost = entry['cache_key'], entry['cost_usd']
        require(isinstance(key, str) and key and key not in entries, 'duplicate or invalid spend cache identity')
        require(entry['category'] in totals and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                'invalid spend amount or category')
        require(entry['status'] != 'reserved', 'unsettled spend reservation')
        entries[key] = entry
        totals[entry['category']] += cost
    same(ledger['by_category_usd'], totals, 'spend category totals mismatch')
    total = sum(totals.values())
    same(ledger['total_usd'], total, 'spend total mismatch')
    require(total <= 250 and totals['reader_judge_dev'] <= 120
            and all(totals[category] <= ledger['caps_usd'][category] for category in totals), 'spend cap exceeded')
    same(report['incremental_spend_usd'], total - report['starting_spend_usd'], 'incremental spend mismatch')
    seen = set()
    for qid, stage, call, cost in calls:
        call_id = digest(f'{work_name}:{stage}:{qid}')
        key = digest(canonical({'request': call['request'], 'call_id': call_id, 'protocol_sha256': official_sha256}))
        require(call['cache_key'] == key and key not in seen, 'experiment paid call cache identity mismatch')
        seen.add(key)
        if call['status'] == 'budget_exhausted':
            require(key not in entries and cost == 0, 'unissued budget-exhausted call charged')
            continue
        require(key in entries, 'experiment call missing from spend ledger')
        entry = entries[key]
        require(entry['call_id'] == call_id and entry['status'] == call['status']
                and entry['category'] == 'reader_judge_dev', 'spend call identity or status mismatch')
        same(entry['cost_usd'], cost, 'spend call charge mismatch')
        same(entry['usage'], call['usage'], 'spend call usage mismatch')
        reserve = ((len(tokenizer.encode(call['request']['input'])) + 512) * protocol['cost']['input_usd_per_million']
                   + call['request']['max_output_tokens'] * protocol['cost']['output_usd_per_million']) / 1_000_000
        same(entry['reserved_usd'], reserve, 'spend reservation mismatch')
        require(cost <= reserve + 1e-12, 'API charge exceeded reserved spend')
    return total


def validate(report, split, frozen, protocol, preregistration, archives, baseline_rows, v1_rows,
             history_rows, evidence_rows, references, ledger, work_name, official_sha256):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('complete') is True and report.get('dev_only') is True
            and report.get('heldout_run') is False and report.get('reader_systems') == [SYSTEM],
            'expected a complete dev-only user-first report')
    dev = dev_membership(split)
    index_rows(report['rows'], dev, 'experiment')
    baseline = index_rows(baseline_rows, dev, 'baseline')
    v1 = index_rows(v1_rows, dev, 'v1 snapshots')
    history = index_rows(history_rows, dev, 'historical evidence')
    evidence = index_rows(evidence_rows, dev, 'frozen candidates')
    require(set(archives) == set(references) == dev, 'dev source or reference membership mismatch')
    check_protocol(protocol, preregistration)
    same(frozen['reader'], protocol['systems'][SYSTEM], 'experiment reader settings mismatch')
    require(frozen['dev_only'] is True and frozen['heldout_run'] is False
            and frozen['acceptance_correct'] == 100 and frozen['token_cap'] == protocol['token_cap'] == 8192
            and frozen['local_token_cap'] == 7680 and frozen['automatic_transport_retries'] == 0
            and frozen['reader_judge_concurrency'] == 2, 'experiment protocol invariant mismatch')
    same(frozen['reader_prompt'], protocol['reader_prompt'], 'experiment reader prompt mismatch')
    require(report['candidate_mode'] == frozen['candidate_mode'] and report['candidate_mode'] in ('pool', 'selected'),
            'experiment candidate mode mismatch')
    tokenizer = tiktoken.get_encoding(protocol['systems'][SYSTEM]['tokenizer'])
    scores, calls = [], []
    for row in report['rows']:
        qid = row['qid']
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM},
                'unexpected experiment reader or arm')
        for field in ('qtype', 'question', 'question_date'):
            same(row[field], baseline[qid][field], f'baseline {field} mismatch')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        require(arm['reader']['status'] != 'pending' and arm['judge']['status'] != 'pending', 'pending stage in completed report')
        candidates = evidence[qid]['candidates'] if report['candidate_mode'] == 'pool' else evidence[qid]['arms']['operand']['selected']
        retained = check_sources(qid, arm, archives[qid], [v1[qid], baseline[qid]], candidates)
        serialized = protocol['reader_prompt'] + canonical(reader_payload(row, {'source_groups': arm['source_groups']}))
        count = len(tokenizer.encode(serialized))
        cap = arm['reader_cap']
        require(type(cap['selection_failure']) is bool and cap['cap'] == frozen['local_token_cap'], 'invalid reader cap')
        same(cap['input_tokens'], count, 'reader cap token count mismatch')
        if cap['selection_failure']:
            require(len(arm['source_groups']) == 1 and count > frozen['local_token_cap']
                    and arm['reader']['status'] == 'selection_failure', 'invalid selection failure')
            expected_input = ''
        else:
            require(count <= frozen['local_token_cap'] and arm['reader']['status'] != 'selection_failure', 'local reader input cap exceeded')
            expected_input = serialized
        same(arm['reader_input'], expected_input, 'reader input differs from retained sources')
        same(arm['reader_input_sha256'], digest(expected_input), 'reader input fingerprint mismatch')
        same(arm['input_tokens'], len(tokenizer.encode(expected_input)), 'primary input token count mismatch')
        actual = recompute_scores(arm)
        same(arm['scores'], actual, 'answer or citation score mismatch')
        scores.append(actual)
        for stage in ('reader', 'judge'):
            result = arm[stage]
            if stage == 'reader':
                expected_prompt = expected_input
            else:
                expected_prompt = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], references[qid]) if arm['reader']['status'] == 'ok' else None
                require(arm['reader']['status'] == 'ok' or result['status'] == 'reader_failure', 'failed reader has invalid judge status')
            call = result.get('call')
            if stage == 'judge' and arm['reader']['status'] != 'ok':
                require(call is None, 'failed reader issued a judge call')
            cost = check_call(call, result['raw_text'], result['status'], protocol, expected_prompt,
                              protocol['systems'][SYSTEM][stage + '_model'], stage)
            if call is not None:
                if stage == 'reader' and call['usage'] is not None:
                    require(call['usage']['input_tokens'] <= 8192, 'actual reader input token cap exceeded')
                calls.append((qid, stage, call, cost))
        require(type(history[qid]['scored']) is bool, 'invalid historical scoring label')
        gold = {digest(tid) for tid in history[qid]['ids'] if 'answer' in tid}
        previous = {group['frozen_selected_id_sha256'] for group in baseline[qid]['systems'][SYSTEM]['arms']['operand']['source_groups']}
        same(row['scored_evidence'], history[qid]['scored'], 'scored evidence label mismatch')
        same(row['actual_complete'], gold <= retained, 'actual labeled support mismatch')
        same(row['baseline_actual_complete'], gold <= previous, 'baseline labeled support mismatch')
    summary = independent_summary(report['rows'], scores, True)
    same(report['summary'], summary, 'experiment summary mismatch')
    total = check_ledger(report, ledger, calls, protocol, preregistration, official_sha256, work_name, tokenizer)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'corpus_read': False,
            'new_model_or_judge_calls': 0, 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'paid_calls_verified': len(calls), 'session_spend_usd': total, 'summary': summary}


def check_work(work):
    work = Path(work).expanduser().resolve()
    report = json.loads((work / 'answers.json').read_text())
    require(report.get('complete') is True, 'experiment is not complete')
    public, results = ROOT / 'benchmarks/public', ROOT / 'benchmarks/results'
    paths = {'split': public / 'longmemeval_split.json', 'baseline': results / 'longmemeval_dev_answers_v3.json',
             'v1': results / 'longmemeval_dev_answers_v1.json', 'history': results / 'evidence_completeness_dev.json',
             'official': public / 'reader_protocol_v1.json', 'preregistration': results / 'preregistration_2026-10.json',
             'manifest': ROOT / 'target/conversation-geometry-validation/models/manifest.json',
             'export': artifact_path('~/.cache/warm-bench/holo_dev/export_dev.json')}
    required = list(paths.values()) + [public / name for name in
                                      ('longmemeval_user_first.py', 'longmemeval_reader_budget.py', 'longmemeval_dev_sources.py')]
    recorded = verify_artifacts(report, work, required)
    data = {name: json.loads(path.read_text()) for name, path in paths.items() if name != 'manifest'}
    evidence_path = artifact_path(data['baseline']['evidence_report_path'])
    require(evidence_path.resolve() in recorded and file_digest(evidence_path) == data['baseline']['evidence_report_sha256'],
            'frozen candidate evidence fingerprint mismatch')
    require(file_digest(paths['manifest']) == data['history']['source_manifest_sha256']
            and file_digest(paths['split']) == data['history']['split_sha256'], 'archive manifest or split fingerprint mismatch')
    official_sha256 = file_digest(paths['official'])
    dev = dev_membership(data['split'])
    baseline = index_rows(data['baseline']['rows'], dev, 'baseline')
    references = check_references(data['export'], baseline, dev, data['official'])
    historical_policy = check_historical_policy(data['official'], data['preregistration'], data['baseline'], references)
    ledger_path = work / 'spend.json'
    require(file_digest(ledger_path) == report['spend_ledger_sha256'], 'spend ledger fingerprint mismatch')
    result = validate(report, data['split'], json.loads((work / 'protocol.json').read_text()),
                      data['official'], data['preregistration'], load_dev_sources(paths['manifest'].parent, paths['split']),
                      data['baseline']['rows'], data['v1']['rows'], data['history']['rows'],
                      json.loads(evidence_path.read_text())['rows'], references, json.loads(ledger_path.read_text()),
                      work.name, official_sha256)
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'), 'spend_sha256': file_digest(ledger_path),
                        'checker_sha256': file_digest(__file__), 'protocol_sha256': report['protocol_sha256']}
    result['historical_policy'] = historical_policy
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
