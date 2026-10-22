# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Independently replay and validate a completed dev role-retrieval experiment."""

import argparse
import copy
import json
from pathlib import Path

import tiktoken

from check_longmemeval_answers import (
    ROOT, artifact_path, canonical, check_call, digest, file_digest, recompute_scores, require, same, sorted_sources,
)
from check_longmemeval_full_history import check_full_protocol, check_spend, independent_summary, reasoning_rules
from check_longmemeval_user_first import (
    check_historical_policy, check_references, dev_membership, index_rows, judge_prompt, verify_artifacts,
)
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_conversations import _validate_dev_record, load_dev_conversations
from proxies.beir_proxies import BM25, STOP, tok_hms

SYSTEM, ARM = 'matched_gpt54', 'full_history'
DEFAULT_WORK = ROOT / 'benchmarks/results/longmemeval_role_retrieval_v1'
SOURCE = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
ROLE_POLICY = 'cached user hybrid ranking fused with assistant-only lexical ranking; complete original exchanges'
FULL_POLICY = 'all complete dev user observations and all available saved v1/v3 assistant turns'
RETRIEVAL_POLICY = {'rrf_k0': 60, 'user_prior_weight': 1, 'assistant_weight': 1,
                    'reserved_user_prior': 5, 'candidate_limit': 64,
                    'budget_policy': 'drop lowest ranked whole exchanges; preserve the first anchor'}
SOURCE_GATE = {'metadata_ids_verified': 500, 'dev_records_decoded': 100, 'heldout_records_decoded': 0}


def original_groups(raw, historical_ids):
    qid = raw['question_id']
    _validate_dev_record(raw, qid)
    exchanges, orphans = {}, []
    for session_index, turns in enumerate(raw['haystack_sessions']):
        records = [{
            'source_id': digest(canonical(['lme-source-v1', qid, session_index, turn_index])),
            'session_id': digest(canonical(['lme-session-v1', qid, session_index])),
            'session_index': session_index, 'turn_index': turn_index,
            'observation_time': raw['haystack_dates'][session_index], 'role': turn['role'], 'text': turn['content'],
        } for turn_index, turn in enumerate(turns)]
        paired = set()
        for turn_index, turn in enumerate(turns):
            if turn['role'] != 'user':
                continue
            legacy = f"{raw['haystack_session_ids'][session_index]}_{turn_index + 1}"
            matches = {legacy, legacy.replace('answer', 'noans')} & historical_ids
            require(len(matches) == 1, 'ambiguous or missing historical user alias')
            end = turn_index + 1
            while end < len(turns) and turns[end]['role'] == 'assistant':
                end += 1
            exchanges.setdefault(next(iter(matches)), []).extend(records[turn_index:end])
            paired.update(range(turn_index, end))
        unpaired = [source for index, source in enumerate(records) if index not in paired]
        require(all(source['role'] == 'assistant' for source in unpaired), 'unpaired user occurrence')
        if unpaired:
            orphans.append(unpaired)
    require(set(exchanges) == historical_ids, 'historical user alias membership mismatch')
    groups, by_alias = [], {}
    for alias, sources in exchanges.items():
        by_alias[alias] = len(groups)
        occurrences = [[source['session_index'], source['turn_index']] for source in sources if source['role'] == 'user']
        groups.append({'group_id': digest(canonical(['lme-group-v1', qid, occurrences])),
                       'frozen_selected_id_sha256': digest(alias), 'sources': sources})
    for sources in orphans:
        key = digest(canonical(['lme-assistant-group-v1', qid, [source['source_id'] for source in sources]]))
        groups.append({'group_id': key, 'frozen_selected_id_sha256': key, 'sources': sources})
    flattened = [source for group in groups for source in group['sources']]
    require(len(flattened) == sum(map(len, raw['haystack_sessions']))
            and len({source['source_id'] for source in flattened}) == len(flattened), 'original occurrence coverage mismatch')
    require(len({group['group_id'] for group in groups}) == len(groups), 'duplicate original exchange')
    return groups, by_alias, len(orphans)


def replay_retrieval(question, groups, prior):
    require(prior and len(set(prior)) == len(prior) and all(type(index) is int and 0 <= index < len(groups) for index in prior),
            'invalid historical prior')
    tokens = lambda text: [token for token in tok_hms(text) if token not in STOP]
    documents = ['\n'.join(source['text'] for source in group['sources'] if source['role'] == 'assistant') for group in groups]
    scores = [float(value) for value in BM25(documents, tokens).score(question)] if any(tokens(document) for document in documents) else [0.0] * len(groups)
    position = {index: rank for rank, index in enumerate(prior)}
    tie = lambda index: position.get(index, len(prior) + index)
    assistant = sorted((index for index, score in enumerate(scores) if score > 0), key=lambda index: (-scores[index], tie(index)))
    fused = {index: 0.0 for index in range(len(groups))}
    for ranking in (prior, assistant):
        for rank, index in enumerate(ranking):
            fused[index] += 1 / (61 + rank)
    ranking = sorted(fused, key=lambda index: (-fused[index], tie(index)))
    reserved = prior[:5]
    selected = (reserved + [index for index in ranking if index not in reserved])[:64]
    return {'selected': selected, 'assistant_ranking': assistant, 'fused_ranking': ranking,
            'reserved_user_prior': reserved, 'rrf_k0': 60}


def cap_groups(question, selected, protocol, count):
    ranked = [dict(copy.deepcopy(group), selection_rank=rank) for rank, group in enumerate(selected)]
    require(1 <= len(ranked) <= 64, 'invalid selected exchange count')

    def serialized(groups):
        return protocol['reader_prompt'] + canonical(reader_payload(question, {'source_groups': groups}))

    uncapped = count(serialized(ranked))
    retained = list(ranked)
    dropped = []
    while count(serialized(retained)) > protocol['local_token_cap'] and len(retained) > 1:
        dropped.append(retained.pop()['group_id'])
    prompt = serialized(retained)
    current = count(prompt)
    failed = current > protocol['local_token_cap']
    cap = {'cap': protocol['local_token_cap'], 'input_tokens': current, 'uncapped_input_tokens': uncapped,
           'dropped_group_ids': dropped, 'selection_failure': failed,
           'failure_reason': 'anchor_complete_input_exceeds_cap' if failed else None}
    return retained, prompt, cap


def check_role_protocol(protocol, base, preregistration, rules):
    same(protocol['source_policy'], ROLE_POLICY, 'role retrieval source policy changed')
    same(protocol['retrieval'], RETRIEVAL_POLICY, 'role retrieval ranking policy changed')
    same(protocol['source_gate'], SOURCE_GATE, 'dev-only source gate changed')
    check_full_protocol(dict(protocol, source_policy=FULL_POLICY), base, preregistration, rules)


def validate(report, split, protocol, base_protocol, preregistration, raw_questions, baseline_rows, history_rows,
             references, ledger, work_name, official_sha256, rules):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('complete') is True and report.get('dev_only') is True
            and report.get('heldout_run') is False and report.get('reader_systems') == [SYSTEM],
            'expected a completed dev-only role-retrieval report')
    dev = dev_membership(split)
    index_rows(report['rows'], dev, 'role-retrieval experiment')
    baseline = index_rows(baseline_rows, dev, 'baseline')
    history = index_rows(history_rows, dev, 'historical ranking')
    require(set(raw_questions) == set(references) == dev, 'decoded dev source or reference membership mismatch')
    check_role_protocol(protocol, base_protocol, preregistration, rules)
    prepared = [{'qid': row['qid'], 'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                 'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    same(protocol['prepared_inputs'], prepared, 'frozen prepared input manifest mismatch')
    tokenizer = tiktoken.get_encoding(protocol['systems'][SYSTEM]['tokenizer'])
    count = lambda text: len(tokenizer.encode(text))
    call_protocol = {**protocol, 'cost': base_protocol['cost']}
    scores, calls = [], []
    for row in report['rows']:
        qid, raw = row['qid'], raw_questions[row['qid']]
        require(raw['question_id'] == qid, 'original dev question identity mismatch')
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM}, 'unexpected role reader or arm')
        for field, raw_field in (('question', 'question'), ('question_date', 'question_date'), ('qtype', 'question_type')):
            same(row[field], baseline[qid][field], f'baseline {field} mismatch')
            same(row[field], raw[raw_field], f'original dev {field} mismatch')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        require(arm['reader']['status'] != 'pending' and arm['judge']['status'] != 'pending', 'pending stage in completed role report')
        pool, by_alias, orphans = original_groups(raw, set(history[qid]['ids']))
        aliases = list(dict.fromkeys(history[qid]['baseline_ranking']))
        require(set(aliases) == set(by_alias), 'historical prior membership mismatch')
        result = replay_retrieval(row['question'], pool, [by_alias[alias] for alias in aliases])
        same(row['retrieval'], result, 'role retrieval replay mismatch')
        same(row['candidate_groups'], len(result['selected']), 'retrieved candidate count mismatch')
        same(row['conversation_groups'], len(pool), 'original conversation group count mismatch')
        same(row['orphan_assistant_groups'], orphans, 'orphan assistant group count mismatch')
        retained, prompt, cap = cap_groups(row, [pool[index] for index in result['selected']], protocol, count)
        same(arm['source_groups'], retained, 'retained exchanges differ from original conversation replay')
        same(arm['retained_sources'], sorted_sources(retained), 'retained source bytes or original metadata changed')
        same(arm['reader_cap'], cap, 'whole-exchange cap replay mismatch')
        same(arm['reader_input'], prompt, 'role reader prompt differs from retained original exchanges')
        same(arm['reader_input_sha256'], digest(prompt), 'role reader input fingerprint mismatch')
        same(arm['input_tokens'], count(prompt), 'role reader input token count mismatch')
        failed = cap['selection_failure']
        require((arm['reader']['status'] == 'selection_failure') == failed, 'role reader selection status mismatch')
        actual = recompute_scores(arm)
        same(arm['scores'], actual, 'answer or citation score mismatch')
        scores.append(actual)
        for stage in ('reader', 'judge'):
            output = arm[stage]
            require(not output.get('prior_attempts'), 'role reader resampling or retries recorded')
            if stage == 'reader':
                expected_input = prompt
                if failed:
                    require(output.get('call') is None, 'selection failure issued reader call')
            elif arm['reader']['status'] == 'ok':
                expected_input = judge_prompt(protocol, row, arm['reader']['parsed']['answer'], references[qid])
            else:
                require(output['status'] == 'reader_failure' and output.get('call') is None, 'failed reader issued judge call')
                expected_input = None
            call = output.get('call')
            cost = check_call(call, output['raw_text'], output['status'], call_protocol, expected_input,
                              protocol['systems'][SYSTEM][stage + '_model'], stage)
            if call is not None:
                if stage == 'reader' and call['usage'] is not None:
                    require(call['usage']['input_tokens'] <= 131072, 'actual role reader context cap exceeded')
                calls.append((qid, stage, call, cost))
        require(type(history[qid]['scored']) is bool, 'invalid historical scoring label')
        gold = {digest(alias) for alias in history[qid]['ids'] if 'answer' in alias}
        kept = {group['frozen_selected_id_sha256'] for group in retained}
        same(row['scored_evidence'], history[qid]['scored'], 'historical scoring label mismatch')
        same(row['actual_complete'], not failed and gold <= kept, 'role retrieval labeled coverage mismatch')
    summary = independent_summary(report['rows'], scores, True)
    same(report['summary'], summary, 'role reader summary mismatch')
    total, variant = check_spend(report, ledger, calls, protocol, base_protocol, preregistration, work_name, official_sha256, tokenizer)
    return {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'new_model_or_judge_calls': 0,
            'source_gate': SOURCE_GATE.copy(), 'dev_questions_verified': 100, 'heldout_questions_untouched': 400,
            'retrieval_replayed': True, 'paid_calls_verified': len(calls), 'global_recorded_spend_usd': total,
            'incremental_spend_usd': variant, 'summary': summary}


def check_work(work):
    work = Path(work).expanduser().resolve()
    report = json.loads((work / 'answers.json').read_text())
    require(report.get('complete') is True, 'role-retrieval experiment is not complete')
    public, results = ROOT / 'benchmarks/public', ROOT / 'benchmarks/results'
    paths = {'split': public / 'longmemeval_split.json', 'baseline': results / 'longmemeval_dev_answers_v3.json',
             'v1': results / 'longmemeval_dev_answers_v1.json', 'history': results / 'evidence_completeness_dev.json',
             'official': public / 'reader_protocol_v1.json', 'preregistration': results / 'preregistration_2026-10.json',
             'export': artifact_path('~/.cache/hms-bench/holo_dev/export_dev.json')}
    required = list(paths.values()) + [SOURCE] + [public / name for name in (
        'longmemeval_role_reader.py', 'longmemeval_full_history.py', 'longmemeval_user_first.py',
        'longmemeval_reader_api.py', 'longmemeval_role_retrieval.py', 'longmemeval_dev_conversations.py',
        'longmemeval_answer_evidence.py', 'check_longmemeval_answers.py', 'proxies/beir_proxies.py')]
    recorded = verify_artifacts(report, work, required)
    data = {name: json.loads(path.read_text()) for name, path in paths.items() if name != 'v1'}
    require(artifact_path(data['official']['judge_source']['path']).resolve() in recorded, 'official judge source fingerprint missing')
    same(file_digest(paths['split']), data['history']['split_sha256'], 'historical split fingerprint mismatch')
    dev = dev_membership(data['split'])
    baseline = index_rows(data['baseline']['rows'], dev, 'baseline')
    references = check_references(data['export'], baseline, dev, data['official'])
    historical = check_historical_policy(data['official'], data['preregistration'], data['baseline'], references)
    ledger_path = work / 'spend.json'
    same(file_digest(ledger_path), report['spend_ledger_sha256'], 'spend ledger fingerprint mismatch')
    result = validate(report, data['split'], json.loads((work / 'protocol.json').read_text()),
                      data['official'], data['preregistration'], load_dev_conversations(SOURCE, paths['split']),
                      data['baseline']['rows'], data['history']['rows'], references, json.loads(ledger_path.read_text()),
                      work.name, file_digest(paths['official']), reasoning_rules(public / 'longmemeval_full_history.py'))
    result['historical_policy'] = historical
    result['inputs'] = {'answers_sha256': file_digest(work / 'answers.json'), 'spend_sha256': file_digest(ledger_path),
                        'checker_sha256': file_digest(__file__),
                        'summary_spend_checker_sha256': file_digest(public / 'check_longmemeval_full_history.py'),
                        'reference_checker_sha256': file_digest(public / 'check_longmemeval_user_first.py'),
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
