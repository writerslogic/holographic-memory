# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3"]
# ///
"""Independently verify dev answers, provenance, denominators and frozen inputs."""
import argparse
import ast
import copy
import hashlib
import json
import math
import random
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path


ARMS = ('first_five', 'knapsack', 'operand')
SYSTEMS = ('matched_gpt54', 'weak_qwen4b')
SOURCE_SHA256 = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def artifact_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def same(actual, expected, message):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), message)
        for key in expected:
            same(actual[key], expected[key], f'{message}: {key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), message)
        for index, item in enumerate(expected):
            same(actual[index], item, f'{message}: {index}')
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)
    else:
        require(type(actual) is type(expected) and actual == expected, message)


def wilson(correct, total):
    require(type(correct) is int and type(total) is int and 0 <= correct <= total and total > 0,
            'invalid accuracy denominator')
    z = 1.959963984540054
    probability = correct / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def percentile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def paired_interval(deltas):
    require(deltas and all(type(value) is int and value in (-1, 0, 1) for value in deltas),
            'invalid paired predictions')
    generator = random.Random(20261009)
    n = len(deltas)
    samples = [sum(deltas[generator.randrange(n)] for _ in range(n)) / n for _ in range(2000)]
    return {'mean': sum(deltas) / n, 'ci95': [percentile(samples, 0.025), percentile(samples, 0.975)]}


def parse_answer(raw_text):
    require(isinstance(raw_text, str), 'reader text must be a string')
    content = raw_text.strip()
    parsed = json.loads(content)
    require(isinstance(parsed, dict) and set(parsed) == {'answer', 'abstain', 'claims'}
            and isinstance(parsed.get('answer'), str)
            and type(parsed.get('abstain')) is bool and isinstance(parsed.get('claims'), list),
            'invalid reader response structure')
    for claim in parsed['claims']:
        require(isinstance(claim, dict) and set(claim) == {'text', 'citations'}
                and isinstance(claim.get('text'), str)
                and isinstance(claim.get('citations'), list), 'invalid atomic claim')
        for citation in claim['citations']:
            require(isinstance(citation, dict) and set(citation) == {'source_id', 'quote'}
                    and isinstance(citation.get('source_id'), str)
                    and isinstance(citation.get('quote'), str), 'invalid citation structure')
    require(not parsed['abstain'] or not parsed['claims'], 'abstention contains asserted claims')
    return parsed


def judge_score(raw_text):
    require(isinstance(raw_text, str) and raw_text.strip().lower() in ('yes', 'no'),
            'judge response is not official yes/no')
    return raw_text.strip().lower() == 'yes'


def recompute_scores(arm, allow_incomplete=False):
    reader = arm['reader']
    judge = arm['judge']
    require(reader['status'] in ('ok', 'error', 'selection_failure') or (allow_incomplete and reader['status'] == 'pending'),
            'invalid reader status')
    require(judge['status'] in ('ok', 'error', 'selection_failure', 'reader_failure')
            or (allow_incomplete and judge['status'] == 'pending'), 'invalid judge status')
    if reader['status'] == 'ok':
        parsed = parse_answer(reader['raw_text'])
        same(reader['parsed'], parsed, 'reader parsed answer mismatch')
    else:
        parsed = {'answer': '', 'abstain': False, 'claims': []}
        require(reader.get('parsed') in (None, parsed), 'failed reader has scored claims')
        if (reader.get('call') or {}).get('status') in ('ok', 'completed'):
            try:
                parse_answer(reader['raw_text'])
            except (ValueError, TypeError):
                pass
            else:
                raise ValueError('valid completed reader response recorded as failure')
    if judge['status'] == 'ok':
        require(reader['status'] == 'ok', 'successful judge for failed reader')
        score = judge_score(judge['raw_text'])
        same(judge['score'], score, 'judge prediction mismatch')
    else:
        require(judge['score'] is False, 'failed judge score must be false')
        if (judge.get('call') or {}).get('status') == 'ok':
            require(judge['raw_text'].strip().lower() not in ('yes', 'no'),
                    'valid completed judge response recorded as failure')
        score = False
    sources = {source['source_id']: source['text'] for source in arm['retained_sources']}
    valid = 0
    for claim in parsed['claims']:
        citations = claim['citations']
        valid += bool(claim['text'].strip()) and bool(citations) and all(citation['quote'] and citation['source_id'] in sources
                                       and citation['quote'] in sources[citation['source_id']]
                                       for citation in citations)
    total = len(parsed['claims'])
    return {'correct': reader['status'] == 'ok' and score, 'citation_valid_claims': valid,
            'citation_total_claims': total,
            'all_claims_supported': reader['status'] == 'ok' and not parsed['abstain'] and total > 0 and valid == total,
            'abstain': parsed['abstain']}


def recompute_summary(rows):
    result = {}
    for system in SYSTEMS:
        family = {}
        for name in ARMS:
            scores = [row['systems'][system]['arms'][name]['scores'] for row in rows]
            correct = sum(score['correct'] for score in scores)
            valid = sum(score['citation_valid_claims'] for score in scores)
            total = sum(score['citation_total_claims'] for score in scores)
            family[name] = {'n': len(rows), 'correct_count': correct, 'answer_accuracy': correct / len(rows),
                            'answer_accuracy_ci': wilson(correct, len(rows)),
                            'citation_valid_claims': valid, 'citation_total_claims': total,
                            'citation_support': valid / total if total else 0.0,
                            'all_claims_supported_count': sum(score['all_claims_supported'] for score in scores),
                            'paired_deltas': {}}
            for baseline in ('first_five', 'knapsack'):
                deltas = [int(row['systems'][system]['arms'][name]['scores']['correct'])
                          - int(row['systems'][system]['arms'][baseline]['scores']['correct']) for row in rows]
                family[name]['paired_deltas'][baseline] = paired_interval(deltas)
        result[system] = family
    return result


def check_fingerprints(report):
    require(report['schema_version'] == 1, 'unsupported report schema')
    required = ('protocol', 'preregistration', 'evidence_report')
    for kind in required:
        path = artifact_path(report[kind + '_path'])
        require(file_digest(path) == report[kind + '_sha256'], f'{kind} fingerprint mismatch')
    seen = set()
    for artifact in report['artifacts']:
        path = artifact_path(artifact['path'])
        require(path.resolve() not in seen, 'duplicate fingerprinted artifact')
        seen.add(path.resolve())
        require(file_digest(path) == artifact['sha256'], f'artifact fingerprint mismatch: {path.name}')
    source = artifact_path(report['source_path'])
    split = artifact_path(report['split_path'])
    require(source.resolve() in seen and split.resolve() in seen, 'missing input fingerprint')
    required_paths = [artifact_path(report[kind + '_path']) for kind in required] + [
        ROOT / 'benchmarks/public/longmemeval_reader_api.py',
        ROOT / 'benchmarks/public/longmemeval_answer_evidence.py',
        ROOT / 'src/core/operand_retriever.rs', ROOT / 'src/bin/operand-retriever.rs',
    ]
    require(all(path.resolve() in seen for path in required_paths), 'missing frozen code or protocol fingerprint')
    require(file_digest(source) == SOURCE_SHA256, 'pinned LongMemEval source fingerprint mismatch')
    frozen = json.loads(split.read_text())
    require(len(frozen['dev']) == 100 and len(set(frozen['dev'])) == 100
            and len(frozen['heldout']) == 400 and not set(frozen['dev']) & set(frozen['heldout']),
            'invalid frozen dev split')
    require(len(report['rows']) == 100 and {row['qid'] for row in report['rows']} == set(frozen['dev']),
            'report must contain all 100 dev questions exactly once')
    raw = {question['question_id']: question for question in json.loads(source.read_text())
           if question['question_id'] in set(frozen['dev'])}
    require(len(raw) == 100, 'source is missing dev questions')
    protocol_path = artifact_path(report['protocol_path'])
    protocol = json.loads(protocol_path.read_text())
    preregistration = json.loads(artifact_path(report['preregistration_path']).read_text())
    return protocol, preregistration, raw


def observation_key(value):
    match = re.fullmatch(r'(\d{4})/(\d{2})/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})', value)
    require(match is not None, 'invalid observation time')
    return datetime(*(int(part) for part in match.groups()))


def reconstruct_groups(question, frozen, historical_ids):
    require(len(question['haystack_sessions']) == len(question['haystack_dates'])
            == len(question['haystack_session_ids']), 'source session lengths differ')
    groups = {}
    legacy_groups = {}
    for session_index, turns in enumerate(question['haystack_sessions']):
        date = question['haystack_dates'][session_index]
        observation_key(date)
        raw_session_id = question['haystack_session_ids'][session_index]
        user_ids = []
        for turn_index, turn in enumerate(turns):
            require(turn['role'] in ('user', 'assistant') and isinstance(turn['content'], str),
                    'invalid pinned source turn')
            if turn['role'] != 'user':
                continue
            original_id = f'{raw_session_id}_{turn_index + 1}'
            matches = historical_ids & {original_id, original_id.replace('answer', 'noans')}
            require(len(matches) == 1, 'ambiguous historical source occurrence')
            user_ids.append((turn_index, next(iter(matches))))
        observed_session_id = raw_session_id if any(
            alias == f'{raw_session_id}_{index + 1}' for index, alias in user_ids
        ) else raw_session_id.replace('answer', 'noans')
        for turn_index, alias in user_ids:
            legacy_groups.setdefault(alias, []).append([
                digest(alias), digest(observed_session_id), date, 'user', turns[turn_index]['content'],
            ])
            next_user = turn_index + 1
            while next_user < len(turns) and turns[next_user]['role'] == 'assistant':
                next_user += 1
            for index in range(turn_index, next_user):
                groups.setdefault(alias, []).append({
                    'source_id': digest(canonical(['lme-source-v1', question['question_id'], session_index, index])),
                    'session_id': digest(canonical(['lme-session-v1', question['question_id'], session_index])),
                    'session_index': session_index, 'turn_index': index, 'observation_time': date,
                    'role': turns[index]['role'], 'text': turns[index]['content'],
                })
    require(set(legacy_groups) == historical_ids, 'historical source membership differs')
    result = {}
    for name in ARMS:
        selection = frozen['arms'][name]
        selected = selection['selected']
        require(selected and len(set(selected)) == len(selected) and selected[0] == frozen['candidates'][0]
                and set(selected) <= set(frozen['candidates']), 'invalid frozen source selection')
        packet = {'v': 2, 'q': [legacy_groups[alias] for alias in selected], 'i': []} if name == 'operand' else {
            'v': 1, 'evidence': [legacy_groups[alias] for alias in selected],
        }
        blob = json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        require(digest(blob) == selection['packet_sha256'] and len(blob.encode()) == selection['packet_bytes']
                <= frozen['budget'], 'frozen source packet fingerprint mismatch')
        if name == 'operand':
            require(blob == frozen['operand']['packet'], 'frozen operand packet differs')
        selected_groups = []
        for alias in selected:
            sources = groups[alias]
            indices = [[source['session_index'], source['turn_index']] for source in sources if source['role'] == 'user']
            selected_groups.append({
                'group_id': digest(canonical(['lme-group-v1', question['question_id'], indices])),
                'selection_rank': frozen['candidates'].index(alias),
                'frozen_selected_id_sha256': digest(alias), 'sources': sources,
            })
        result[name] = selected_groups
    return result


def sorted_sources(groups):
    result = [source for group in groups for source in group['sources']]
    require(len({source['source_id'] for source in result}) == len(result), 'duplicate retained source occurrence')
    return sorted(result, key=lambda source: (observation_key(source['observation_time']),
                                             source['session_index'], source['turn_index']))


def check_source_groups(arm, groups):
    available = {group['group_id']: group for group in groups}
    retained = arm['source_groups']
    ids = [group['group_id'] for group in retained]
    require(ids and len(set(ids)) == len(ids) and ids[0] == groups[0]['group_id']
            and set(ids) <= set(available), 'invalid retained group membership')
    require(ids == [group['group_id'] for group in groups if group['group_id'] in ids],
            'retained group order differs')
    for group in retained:
        same(group, available[group['group_id']], 'retained source changed or incomplete')
    same(arm['retained_sources'], sorted_sources(retained), 'retained source chronology mismatch')


def check_instability(report, protocol):
    instability = report['instability']
    entries = instability['entries']
    require(len(entries) == 20 and len({entry['qid'] for entry in entries}) == 20,
            'judge instability must use 20 distinct dev questions')
    require([entry['qid'] for entry in entries] == protocol['instability']['sample_qids'],
            'judge instability sample fingerprint mismatch')
    rows = {row['qid']: row for row in report['rows']}
    disagreements = any_disagreement = failures = 0
    for index, entry in enumerate(entries):
        require(entry['qid'] in rows and entry['system'] == 'matched_gpt54' and entry['arm'] in ARMS,
                'invalid instability sample member')
        require(entry['arm'] == ARMS[index % len(ARMS)], 'judge instability arm assignment differs')
        outcomes = entry['outcomes']
        require(len(outcomes) == 20, 'instability question must have 20 additional judge calls')
        require([judge['repeat_index'] for judge in outcomes] == list(range(20)),
                'instability repeat indices are incomplete or duplicated')
        primary = rows[entry['qid']]['systems'][entry['system']]['arms'][entry['arm']]['scores']['correct']
        changed = 0
        for judge in outcomes:
            require(judge['status'] in ('ok', 'error'), 'invalid instability judge status')
            expected = judge_score(judge['raw_text']) if judge['status'] == 'ok' else False
            same(judge['score'], expected, 'instability prediction mismatch')
            changed += expected != primary
            failures += judge['status'] != 'ok'
        disagreements += changed
        any_disagreement += changed > 0
    expected = {'n_questions': 20, 'n_calls': 400, 'disagreements': disagreements,
                'disagreement_rate': disagreements / 400, 'questions_with_disagreement': any_disagreement,
                'question_instability_rate': any_disagreement / 20, 'failures': failures}
    same(instability['summary'], expected, 'judge instability summary mismatch')
    return expected


def check_protocol(protocol, preregistration):
    require(protocol['schema_version'] == 1 and protocol['created_before_first_reader_call'] is True
            and protocol['complete_input_counting'] is True and protocol['token_cap'] == 8192,
            'reader protocol invariant mismatch')
    require(tuple(protocol['arms']) == ARMS and set(protocol['systems']) == set(SYSTEMS),
            'frozen reader systems or arms differ')
    require(protocol['cost']['subcap_usd'] == 120 and protocol['cost']['automatic_retries'] == 0,
            'reader spend cap or retry policy differs')
    require(preregistration['session'] == 1 and preregistration['heldout_authorized'] is False
            and preregistration['spend_caps_usd']['reader_judge_dev'] == 120
            and preregistration['spend_caps_usd']['total'] == 250
            and preregistration['spend_caps_usd']['heldout400'] == 0,
            'preregistered authorization or spend cap differs')
    require(protocol['systems']['matched_gpt54']['reader_model'] == 'gpt-5.4-2026-03-05'
            and all(protocol['systems'][system]['judge_model'] == 'gpt-5.4-2026-03-05' for system in SYSTEMS),
            'reader or judge model is not the pinned matched model')
    judge_source = protocol['judge_source']
    path = artifact_path(judge_source['path'])
    require(file_digest(path) == judge_source['sha256'], 'official judge source fingerprint mismatch')
    parsed = ast.parse(path.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'get_anscheck_prompt')
    literals = {node.value for node in ast.walk(function) if isinstance(node, ast.Constant)
                and isinstance(node.value, str) and node.value.startswith('I will give you')}
    templates = list(judge_source['templates'].values()) + [judge_source['abstention_template']]
    translated = {template.replace('{question}', '{}').replace('{answer}', '{}').replace('{response}', '{}')
                  for template in templates}
    require(translated == literals, 'protocol judge templates differ from official source')


def token_counters(protocol):
    import tiktoken
    from tokenizers import Tokenizer

    matched = tiktoken.get_encoding(protocol['systems']['matched_gpt54']['tokenizer'])
    weak = protocol['systems']['weak_qwen4b']
    require(file_digest(weak['tokenizer_path']) == weak['tokenizer_sha256'], 'weak tokenizer fingerprint mismatch')
    require(file_digest(weak['binary_path']) == weak['binary_sha256'], 'weak reader binary fingerprint mismatch')
    tokenizer = Tokenizer.from_file(weak['tokenizer_path'])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return {
        'matched_gpt54': lru_cache(maxsize=None)(lambda text: len(matched.encode(text, disallowed_special=()))),
        'weak_qwen4b': lru_cache(maxsize=None)(lambda text: len(tokenizer.encode(weak['chat_template'].format(reader_input=text), add_special_tokens=False).ids)),
    }


def reader_input(question, groups, protocol):
    evidence = [{key: source[key] for key in ('source_id', 'session_id', 'observation_time', 'role', 'text')}
                for source in sorted_sources(groups)]
    payload = {'question': question['question'], 'question_date': question['question_date'], 'evidence': evidence}
    return protocol['reader_prompt'] + canonical(payload)


def check_input(arm, question, groups, protocol, counters, system):
    check_source_groups(arm, groups)
    text = reader_input(question, arm['source_groups'], protocol)
    require(arm['reader_input'] == text and arm['reader_input_sha256'] == digest(text),
            'complete reader input fingerprint mismatch')
    tokens = counters[system](text)
    uncapped = counters[system](reader_input(question, groups, protocol))
    require(type(arm['input_tokens']) is int and arm['input_tokens'] == tokens, 'reader input token count mismatch')
    require(type(arm['uncapped_input_tokens']) is int and arm['uncapped_input_tokens'] == uncapped,
            'uncapped reader input token count mismatch')
    cap = arm['reader_cap']
    require(cap['cap'] == protocol['token_cap'], 'reader token cap mismatch')
    retained_ids = {group['group_id'] for group in arm['source_groups']}
    dropped = [group['group_id'] for group in sorted(groups[1:], key=lambda group: group['selection_rank'], reverse=True)
               if group['group_id'] not in retained_ids]
    same(cap['dropped_group_ids'], dropped, 'whole-source cap removal order mismatch')
    suffix = [group['group_id'] for group in sorted(groups[1:], key=lambda group: group['selection_rank'], reverse=True)]
    require(dropped == suffix[:len(dropped)], 'cap skipped a higher-rank group')
    common_tokens = max(counter(text) for counter in counters.values())
    common_uncapped = max(counter(reader_input(question, groups, protocol)) for counter in counters.values())
    require(cap['input_tokens'] == common_tokens and cap['uncapped_input_tokens'] == common_uncapped,
            'common reader cap tokenizer mismatch')
    initial = list(groups)
    while max(counter(reader_input(question, initial, protocol)) for counter in counters.values()) > cap['cap'] and len(initial) > 1:
        initial.remove(max(initial[1:], key=lambda group: group['selection_rank']))
    checks = arm['framing_checks']
    framed_groups = list(initial)
    framed_count = None
    for index, check in enumerate(checks):
        ids = [group['group_id'] for group in framed_groups]
        same(check['group_ids'], ids, 'API framing group transition mismatch')
        expected_request = {'model': protocol['systems']['matched_gpt54']['reader_model'],
                            'input': reader_input(question, framed_groups, protocol)}
        same(check['request'], expected_request, 'API framing request differs')
        require(check['request_sha256'] == digest(canonical(check['request'])), 'API framing request fingerprint mismatch')
        framed_count = check['response']['input_tokens']
        require(type(framed_count) is int and framed_count > 0, 'API framing count invalid')
        if index < len(checks) - 1:
            require(framed_count > cap['cap'] and len(framed_groups) > 1, 'unnecessary API framing source removal')
            framed_groups.remove(max(framed_groups[1:], key=lambda group: group['selection_rank']))
    local_overflow = max(counter(reader_input(question, initial, protocol)) for counter in counters.values()) > cap['cap']
    require(checks or local_overflow, 'complete API input framing count missing')
    same(arm['source_groups'], framed_groups, 'API framing retained groups differ')
    framed_overflow = framed_count is not None and framed_count > cap['cap']
    require(type(cap['selection_failure']) is bool and cap['selection_failure'] == (common_tokens > cap['cap'] or framed_overflow),
            'reader cap failure mismatch')
    if cap['selection_failure']:
        require(len(arm['source_groups']) == 1 and arm['reader']['status'] == 'selection_failure'
                and cap['failure_reason'] in ('anchor_complete_input_exceeds_cap', 'anchor exceeds framed reader token cap'),
                'invalid anchor cap failure')
    else:
        require(tokens <= cap['cap'] and cap['failure_reason'] is None, 'reader token cap exceeded')
    return tokens


def response_text(response):
    require(isinstance(response, dict), 'API response must be an object')
    if 'output' in response:
        parts = []
        for item in response['output']:
            if item.get('type') == 'message':
                parts.extend(content['text'] for content in item.get('content', [])
                             if content.get('type') == 'output_text')
        return ''.join(parts)
    if 'choices' in response:
        return response['choices'][0]['message']['content'] or ''
    return response.get('text', '')


def check_call(call, raw_text, status, protocol, expected_input=None, model=None, stage='judge'):
    if call is None:
        require(status != 'ok', 'successful stage has no recorded call')
        return 0.0
    request = call['request']
    require(call['request_sha256'] == digest(canonical(request)), 'API request fingerprint mismatch')
    require(isinstance(call['cache_key'], str) and call['cache_key'], 'missing API cache identity')
    if model is not None:
        require(request['model'] == model, 'API model differs from frozen protocol')
    if expected_input is not None:
        require(request['input'] == expected_input, 'API input differs from frozen serialized prompt')
    if stage != 'weak_reader':
        require(request['store'] is False and request['truncation'] == 'disabled'
                and request['reasoning']['effort'] == 'medium', 'API request policy differs')
        maximum = protocol['systems']['matched_gpt54'][stage + '_max_output_tokens']
        require(request['max_output_tokens'] == maximum, 'API output limit differs')
    response = call['response']
    if 'raw_text' in call:
        require(call['raw_text'] == raw_text, 'cached API text differs from reader or judge text')
    if response is not None:
        require(response_text(response) == raw_text, 'API response text fingerprint mismatch')
    if status == 'ok':
        if stage != 'weak_reader':
            require(response.get('status') == 'completed', 'incomplete API output scored as successful')
    usage = call['usage']
    if stage == 'weak_reader':
        weak = protocol['systems']['weak_qwen4b']
        expected = {'model': weak['reader_model'], 'revision': weak['revision'],
                    'binary_sha256': weak['binary_sha256'], 'tokenizer_sha256': weak['tokenizer_sha256'],
                    'stage': 'chat', 'batch': weak['batch'], 'max_new_tokens': weak['max_new_tokens'],
                    'max_context': weak['max_context'], 'temperature': weak['temperature'], 'input': expected_input}
        same(request, expected, 'local reader generation settings differ')
        require(call['cost_usd'] == 0 and call['usage'] is None, 'local reader has paid cost or API usage')
        require(call['cache_key'] == call['request_sha256'] and call['status'] in ('completed', 'reader_failure'),
                'local reader cache identity or status differs')
        return 0.0
    require(usage is None or isinstance(usage, dict), 'API usage has invalid type')
    require(usage == (response.get('usage') if isinstance(response, dict) else None),
            'API usage differs from raw response')
    if usage:
        input_tokens = usage['input_tokens']
        output_tokens = usage['output_tokens']
        cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
        require(all(type(value) is int and value >= 0 for value in (input_tokens, output_tokens, cached))
                and cached <= input_tokens, 'invalid API token usage')
        rates = protocol['cost']
        expected_cost = ((input_tokens - cached) * rates['input_usd_per_million']
                         + cached * rates['cached_input_usd_per_million']
                         + output_tokens * rates['output_usd_per_million']) / 1_000_000
    else:
        require(status != 'ok', 'successful paid call has no usage')
        if call.get('cost_is_conservative_reservation') is True:
            import tiktoken

            count = len(tiktoken.get_encoding('o200k_base').encode(request['input'])) + 512
            expected_cost = (count * protocol['cost']['input_usd_per_million']
                             + request['max_output_tokens'] * protocol['cost']['output_usd_per_million']) / 1_000_000
        else:
            require(call['status'] == 'budget_exhausted', 'missing API usage lacks conservative reservation')
            expected_cost = 0.0
    same(call['cost_usd'], expected_cost, 'API spend mismatch')
    return expected_cost


def judge_input(question, answer, protocol):
    source = protocol['judge_source']
    template = source['abstention_template'] if '_abs' in question['question_id'] else source['templates'][question['question_type']]
    return template.format(question=question['question'], answer=question['answer'], response=answer)


def check_spend(report, calls, protocol, preregistration):
    path = artifact_path(report['spend_ledger_path'])
    require(file_digest(path) == report['spend_ledger_sha256'], 'frozen spend ledger fingerprint mismatch')
    ledger = json.loads(path.read_text())
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'spend ledger caps differ')
    require(ledger['schema_version'] == 1, 'unsupported spend ledger schema')
    categories = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    entries = {}
    for entry in ledger['entries']:
        require(entry['cache_key'] not in entries and isinstance(entry['call_id'], str) and entry['call_id'],
                'duplicate or invalid charged call identity')
        require(entry['category'] in categories and type(entry['cost_usd']) in (int, float)
                and math.isfinite(entry['cost_usd']) and entry['cost_usd'] >= 0,
                'invalid spend ledger category or amount')
        require(type(entry['unix_time']) in (int, float) and math.isfinite(entry['unix_time']) and entry['unix_time'] > 0,
                'invalid spend ledger timestamp')
        categories[entry['category']] += entry['cost_usd']
        entries[entry['cache_key']] = entry
    same(ledger['by_category_usd'], categories, 'spend category totals mismatch')
    same(ledger['total_usd'], sum(categories.values()), 'spend total mismatch')
    same(report['spend_usd'], ledger['total_usd'], 'report spend differs from ledger')
    require(ledger['total_usd'] <= ledger['caps_usd']['total']
            and all(categories[category] <= ledger['caps_usd'][category] for category in categories),
            'session spend cap exceeded')
    seen = set()
    import tiktoken

    tokenizer = tiktoken.get_encoding('o200k_base')
    for call, cost, call_id in calls:
        expected_key = digest(canonical({'request': call['request'], 'call_id': call_id,
                                         'protocol_sha256': report['protocol_sha256']}))
        require(call['cache_key'] == expected_key and expected_key not in seen, 'paid call cache identity mismatch')
        seen.add(expected_key)
        if call['status'] == 'budget_exhausted':
            require(expected_key not in entries and cost == 0, 'unissued budget-exhausted call charged')
            continue
        require(expected_key in entries, 'API call missing from spend ledger')
        entry = entries[expected_key]
        require(entry['call_id'] == call_id and entry['category'] == 'reader_judge_dev'
                and entry['status'] == call['status'], 'spend ledger call identity or status mismatch')
        same(entry['cost_usd'], cost, 'spend ledger call charge mismatch')
        same(entry['usage'], call['usage'], 'spend ledger usage mismatch')
        input_count = len(tokenizer.encode(call['request']['input'])) + 512
        expected_reserve = (input_count * protocol['cost']['input_usd_per_million']
                            + call['request']['max_output_tokens'] * protocol['cost']['output_usd_per_million']) / 1_000_000
        same(entry['reserved_usd'], expected_reserve, 'spend reservation mismatch')
        require(cost <= entry['reserved_usd'] + 1e-12, 'API charge exceeded reserved spend')
    require({key for key, entry in entries.items() if entry['category'] == 'reader_judge_dev'} <= seen,
            'reader spend ledger has unreported calls')


def check_report(report, allow_incomplete=False):
    protocol, preregistration, raw = check_fingerprints(report)
    require(report['dev_only'] is True and report['heldout_run'] is False, 'held-out run is not authorized')
    require(report['complete'] is True or allow_incomplete, 'report is incomplete')
    check_protocol(protocol, preregistration)
    require(preregistration['reader_protocol_path'] == report['protocol_path']
            and preregistration['reader_protocol_sha256'] == report['protocol_sha256'],
            'preregistered protocol fingerprint mismatch')
    counters = token_counters(protocol)
    evidence_path = artifact_path(report['evidence_report_path'])
    evidence = json.loads(evidence_path.read_text())
    require(file_digest(artifact_path(report['split_path'])) == evidence['split_sha256']
            and file_digest(artifact_path(report['source_path'])) == evidence['source_sha256'],
            'frozen evidence input fingerprints differ')
    validated_path = evidence_path.with_name('evidence_budget_summary.json')
    validated = json.loads(validated_path.read_text())
    require(validated['report_sha256'] == report['evidence_report_sha256'], 'frozen evidence validation mismatch')
    historical_path = evidence_path.with_name('evidence_completeness_dev.json')
    require(file_digest(historical_path) == evidence['input_sha256'], 'historical source identity fingerprint mismatch')
    historical = {row['qid']: set(row['ids']) for row in json.loads(historical_path.read_text())['rows']}
    frozen = {row['qid']: row for row in evidence['rows']}
    require(set(historical) == set(frozen) == set(raw), 'frozen evidence dev membership differs')
    paid_calls = []
    blind_ids = set()
    input_tokens = {system: 0 for system in SYSTEMS}
    for row in report['rows']:
        question = raw[row['qid']]
        same({key: row[key] for key in ('qtype', 'question', 'question_date')},
             {'qtype': question['question_type'], 'question': question['question'], 'question_date': question['question_date']},
             'dev question metadata mismatch')
        require(set(row['systems']) == set(SYSTEMS), 'reader system missing')
        expected_groups = reconstruct_groups(question, frozen[row['qid']], historical[row['qid']])
        for system in SYSTEMS:
            require(set(row['systems'][system]['arms']) == set(ARMS), 'reader arm missing')
            for name in ARMS:
                arm = row['systems'][system]['arms'][name]
                input_tokens[system] += check_input(arm, question, expected_groups[name], protocol, counters, system)
                same(arm['scores'], recompute_scores(arm, allow_incomplete), 'answer or citation score mismatch')
                reader = arm['reader']
                config = protocol['systems'][system]
                stage = 'reader' if system == 'matched_gpt54' else 'weak_reader'
                cost = check_call(reader.get('call'), reader['raw_text'], reader['status'], protocol,
                                  arm['reader_input'], config['reader_model'], stage)
                if stage == 'reader' and reader.get('call') is not None:
                    require(reader['call'].get('usage', {}).get('input_tokens', 0) <= protocol['token_cap']
                            if reader['call'].get('usage') else True, 'paid reader actual input exceeds cap')
                    paid_calls.append((reader['call'], cost, f"reader:{row['qid']}:{name}"))
                judge = arm['judge']
                expected_blind = digest(f"judge-blind-v1:{row['qid']}:{system}:{name}")
                if judge['status'] != 'pending':
                    require(judge['blind_id'] == expected_blind and judge['blind_id'] not in blind_ids,
                            'invalid or duplicate blinded judge identity')
                    blind_ids.add(judge['blind_id'])
                answer = reader['parsed']['answer'] if reader['status'] == 'ok' else ''
                prompt = judge_input(question, answer, protocol)
                cost = check_call(judge.get('call'), judge['raw_text'], judge['status'], protocol,
                                  prompt, config['judge_model'])
                if judge.get('call') is not None:
                    paid_calls.append((judge['call'], cost, expected_blind))
        for name in ARMS:
            first = row['systems'][SYSTEMS[0]]['arms'][name]
            second = row['systems'][SYSTEMS[1]]['arms'][name]
            same(first['source_groups'], second['source_groups'], 'reader families retain different source groups')
            same(first['reader_input'], second['reader_input'], 'reader families receive different evidence')
    expected_summary = recompute_summary(report['rows'])
    same(report['summary'], expected_summary, 'answer summary or interval mismatch')
    if allow_incomplete and not report['complete']:
        return {'complete': False, 'preliminary': True, 'dev_questions_verified': 100,
                'reader_arms_verified': 600, 'completed_readers': sum(
                    row['systems'][system]['arms'][name]['reader']['status'] != 'pending'
                    for row in report['rows'] for system in SYSTEMS for name in ARMS),
                'completed_judges': len(blind_ids), 'input_tokens_verified': input_tokens,
                'recorded_paid_calls_verified': len(paid_calls)}
    instability = check_instability(report, protocol)
    questions = {row['qid']: row for row in report['rows']}
    for entry in report['instability']['entries']:
        reader = questions[entry['qid']]['systems'][entry['system']]['arms'][entry['arm']]['reader']
        prompt = judge_input(raw[entry['qid']], reader['parsed']['answer'] if reader['status'] == 'ok'
                             else 'Reader failure: no valid answer.', protocol)
        for judge in entry['outcomes']:
            cost = check_call(judge['call'], judge['raw_text'], judge['status'], protocol, prompt,
                              protocol['systems']['matched_gpt54']['judge_model'])
            require(judge['call'] is not None, 'instability has no actual repeated call')
            blind_id = digest(f"judge-blind-v1:{entry['qid']}:{entry['system']}:{entry['arm']}")
            paid_calls.append((judge['call'], cost, f"{blind_id}:repeat:{judge['repeat_index']}"))
    cost_total = sum(cost for _, cost, _ in paid_calls)
    require(cost_total <= protocol['cost']['subcap_usd'], 'reader and judge spend cap exceeded')
    check_spend(report, paid_calls, protocol, preregistration)
    return {'schema_version': 1, 'dev_questions_verified': 100, 'reader_arms_verified': 600,
            'heldout_questions_untouched': 400,
            'validation_gates_passed': ['frozen protocol and input fingerprints', 'complete chronological raw-source occurrences',
                                       'reader token caps and shared source selections', 'raw reader and official judge scores',
                                       'failure-inclusive summaries and intervals', 'fixed repeated judge instability',
                                       'API usage and spend'],
            'input_tokens_verified': input_tokens, 'paid_calls_verified': len(paid_calls),
            'reader_judge_spend_usd': cost_total, 'instability': instability, 'summary': expected_summary}


def check_tampering(report):
    cases = {}
    valid = next((row, system, name, index, citation_index)
                 for row in report['rows'] for system in SYSTEMS for name in ARMS
                 for index, claim in enumerate((row['systems'][system]['arms'][name]['reader'].get('parsed') or {}).get('claims', []))
                 for citation_index, citation in enumerate(claim['citations'])
                 if claim['text'].strip() and claim['citations'] and all(
                     item['quote'] and any(source['source_id'] == item['source_id'] and item['quote'] in source['text']
                                           for source in row['systems'][system]['arms'][name]['retained_sources'])
                     for item in claim['citations']))
    qid, system, name, claim_index, citation_index = valid[0]['qid'], *valid[1:]
    for corruption in ('citation', 'score', 'interval', 'fingerprint', 'prediction', 'source_bytes'):
        changed = copy.deepcopy(report)
        row = next(row for row in changed['rows'] if row['qid'] == qid)
        arm = row['systems'][system]['arms'][name]
        if corruption == 'citation':
            arm['reader']['parsed']['claims'][claim_index]['citations'][citation_index]['quote'] += '\nCORRUPTED CITATION'
            text = canonical(arm['reader']['parsed'])
            arm['reader']['raw_text'] = text
            call = arm['reader']['call']
            call['raw_text'] = text
            if 'output' in call['response']:
                parts = [content for item in call['response']['output'] if item.get('type') == 'message'
                         for content in item.get('content', []) if content.get('type') == 'output_text']
                require(parts, 'tamper test needs recorded reader output text')
                parts[0]['text'] = text
                for part in parts[1:]:
                    part['text'] = ''
            else:
                call['response']['text'] = text
        elif corruption == 'score':
            arm['scores']['correct'] = not arm['scores']['correct']
        elif corruption == 'interval':
            changed['summary'][system][name]['answer_accuracy_ci'][0] += 0.01
        elif corruption == 'fingerprint':
            changed['protocol_sha256'] = '0' * 64
        elif corruption == 'prediction':
            arm['judge']['score'] = not arm['judge']['score']
        else:
            arm['source_groups'][0]['sources'][0]['text'] += '\nCORRUPTED SOURCE'
        try:
            check_report(changed)
        except ValueError as error:
            reasons = {'citation': ('citation_valid_claims',), 'score': ('correct',),
                       'interval': ('answer_accuracy_ci',), 'fingerprint': ('protocol fingerprint mismatch',),
                       'prediction': ('judge prediction mismatch', 'failed judge score must be false'),
                       'source_bytes': ('retained source changed or incomplete',)}
            require(any(reason in str(error) for reason in reasons[corruption]),
                    f'{corruption} was rejected for an unrelated reason: {error}')
            cases[corruption] = {'rejected': True, 'reason': str(error)}
        else:
            raise ValueError(f'{corruption} tampering accepted')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/results/longmemeval_dev_answers_v1_summary.json')
    parser.add_argument('--tamper-tests', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    require(not args.tamper_tests or report['complete'], 'tamper tests require a complete baseline')
    summary = check_report(report, args.allow_incomplete)
    if args.tamper_tests:
        summary['tamper_tests'] = check_tampering(report)
    summary['report_sha256'] = file_digest(args.report)
    summary['checker_sha256'] = file_digest(__file__)
    if summary.get('complete') is not False:
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    main()
