# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Settle one completed dev experiment's stale reservations from exact cached API records."""

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path

import tiktoken

from check_longmemeval_answers import artifact_path, canonical, check_call, check_protocol, digest, file_digest, require, same
from check_longmemeval_user_first import dev_membership, index_rows

SYSTEM, ARM = 'matched_gpt54', 'full_history'


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n').encode()


def byte_digest(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def frozen_json(report, name):
    matches = [item for item in report['artifacts'] if artifact_path(item['path']).name == name]
    require(len(matches) == 1, f'missing or duplicate frozen {name}')
    path = artifact_path(matches[0]['path'])
    same(file_digest(path), matches[0]['sha256'], f'frozen {name} fingerprint mismatch')
    return json.loads(path.read_text()), path


def profile(protocol, base):
    settings = protocol['systems'][SYSTEM]
    for stage in ('reader', 'judge'):
        same(settings[stage + '_model'], base['systems'][SYSTEM][stage + '_model'], 'frozen API model mismatch')
    if 'reader_reasoning_effort' in settings:
        require('reasoning_effort' not in settings and settings['reader_reasoning_effort'] == 'high'
                and settings['judge_reasoning_effort'] == 'medium' and settings['reader_max_output_tokens'] == 8192
                and settings['judge_max_output_tokens'] == 2048 and protocol['token_cap'] == 196608
                and protocol['new_spend_cap_usd'] == 50.0, 'unsupported high-reader settlement profile')
        return 50.0
    require(settings['reasoning_effort'] == 'medium' and settings['reader_max_output_tokens'] == 4096
            and settings['judge_max_output_tokens'] == 2048 and protocol['token_cap'] == 131072
            and protocol['new_spend_cap_usd'] == 40.0, 'unsupported medium-reader settlement profile')
    return 40.0


def totals(ledger):
    categories = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    for entry in ledger['entries']:
        cost = entry['cost_usd']
        require(entry['category'] in categories and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                'invalid ledger category or amount')
        categories[entry['category']] += cost
    return {'total_usd': sum(categories.values()), 'by_category_usd': categories}


def build_plan(work):
    work = Path(work).expanduser().resolve()
    report_path, ledger_path = work / 'answers.json', work / 'spend.json'
    report_bytes, ledger_bytes = report_path.read_bytes(), ledger_path.read_bytes()
    report, ledger = json.loads(report_bytes), json.loads(ledger_bytes)
    require(report.get('complete') is True and report.get('dev_only') is True and report.get('heldout_run') is False
            and report.get('reader_systems') == [SYSTEM], 'settlement requires a completed dev-only experiment')
    protocol_path = artifact_path(report['protocol_path'])
    require(protocol_path.resolve() == (work / 'protocol.json').resolve(), 'experiment protocol path mismatch')
    same(file_digest(protocol_path), report['protocol_sha256'], 'experiment protocol fingerprint mismatch')
    protocol = json.loads(protocol_path.read_text())
    base, base_path = frozen_json(report, 'reader_protocol_v1.json')
    preregistration, preregistration_path = frozen_json(report, 'preregistration_2026-10.json')
    split, split_path = frozen_json(report, 'longmemeval_split.json')
    check_protocol(base, preregistration)
    same({key: base['cost'][key] for key in ('input_usd_per_million', 'cached_input_usd_per_million', 'output_usd_per_million')},
         {'input_usd_per_million': 2.5, 'cached_input_usd_per_million': 0.25, 'output_usd_per_million': 15}, 'frozen API rates differ')
    cap = profile(protocol, base)
    index_rows(report['rows'], dev_membership(split), 'completed experiment')
    require(type(ledger.get('schema_version')) is int and ledger['schema_version'] == 1, 'invalid spend ledger schema')
    same(ledger['caps_usd'], preregistration['spend_caps_usd'], 'global spend caps differ')
    same(report['spend_ledger_sha256'], byte_digest(ledger_bytes), 'report and spend ledger are not a consistent snapshot')
    before_totals = totals(ledger)
    for field, value in before_totals.items():
        same(ledger[field], value, 'original ledger totals mismatch')
    after = copy.deepcopy(ledger)
    entries = {}
    own_ids = {digest(f"full-history:{work.name}:{stage}:{row['qid']}") for row in report['rows'] for stage in ('reader', 'judge')}
    own_entries = {}
    for entry in after['entries']:
        key = entry['cache_key']
        require(isinstance(key, str) and key and key not in entries, 'duplicate or invalid ledger cache identity')
        entries[key] = entry
        if entry.get('call_id') in own_ids:
            require(entry['call_id'] not in own_entries, 'duplicate or resampled experiment call')
            own_entries[entry['call_id']] = entry
    tokenizer = tiktoken.get_encoding(protocol['systems'][SYSTEM]['tokenizer'])
    official_sha256 = file_digest(base_path)
    inputs = [{'path': str(path), 'sha256': file_digest(path)} for path in
              (protocol_path, base_path, preregistration_path, split_path, artifact_path(base['judge_source']['path']))]
    updates, issued, seen = [], set(), set()
    for row in report['rows']:
        require(set(row['systems']) == {SYSTEM} and set(row['systems'][SYSTEM]['arms']) == {ARM}, 'unexpected experiment system or arm')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        for stage in ('reader', 'judge'):
            output = arm[stage]
            require(output['status'] != 'pending' and not output.get('prior_attempts'), 'unfinished or resampled experiment call')
            call = output.get('call')
            if call is None:
                require(output['status'] != 'ok', 'successful stage lacks a call record')
                continue
            call_id = digest(f"full-history:{work.name}:{stage}:{row['qid']}")
            key = digest(canonical({'request': call['request'], 'call_id': call_id, 'protocol_sha256': official_sha256}))
            require(call['cache_key'] == key and key not in seen, 'API cache identity or duplicate call mismatch')
            seen.add(key)
            same(call['request_sha256'], digest(canonical(call['request'])), 'API request fingerprint mismatch')
            prompt = arm['reader_input'] if stage == 'reader' else call['request']['input']
            if 'reader_reasoning_effort' in protocol['systems'][SYSTEM]:
                from check_longmemeval_complete_history import check_stage_call

                check_stage_call(call, output, protocol, base, prompt, stage, tokenizer)
            else:
                check_call(call, output['raw_text'], output['status'], {**protocol, 'cost': base['cost']},
                           prompt, protocol['systems'][SYSTEM][stage + '_model'], stage)
            cost = call['cost_usd']
            if stage == 'reader' and call['usage'] is not None:
                require(call['usage']['input_tokens'] <= protocol['token_cap'], 'actual reader input exceeded frozen context cap')
            if call['status'] == 'budget_exhausted':
                require(key not in entries and cost == 0, 'unissued budget-rejected call charged')
                continue
            cache_path = work / 'cache' / (key + '.json')
            require(cache_path.is_file(), 'cached API result missing; conservative charge preserved')
            cache_bytes = cache_path.read_bytes()
            require(canonical(json.loads(cache_bytes)) == canonical(call), 'cached API record differs from report')
            inputs.append({'path': str(cache_path), 'sha256': byte_digest(cache_bytes)})
            require(key in entries, 'charged API call missing from ledger')
            entry = entries[key]
            require(entry.get('call_id') == call_id and entry['category'] == 'reader_judge_dev', 'ledger call identity mismatch')
            reserve = ((len(tokenizer.encode(call['request']['input'])) + 512) * base['cost']['input_usd_per_million']
                       + call['request']['max_output_tokens'] * base['cost']['output_usd_per_million']) / 1_000_000
            same(entry['reserved_usd'], reserve, 'ledger reservation mismatch')
            require(cost <= reserve + 1e-12, 'API charge exceeded verified reservation')
            settled = dict(entry, status=call['status'], cost_usd=cost, usage=call['usage'])
            if canonical(entry) != canonical(settled):
                require(entry['status'] == 'reserved' and entry['usage'] is None, 'non-reservation ledger result conflicts with cached API result')
                same(entry['cost_usd'], reserve, 'stale reservation cost differs from verified reservation')
                updates.append({'cache_key': key, 'before': copy.deepcopy(entry), 'after': copy.deepcopy(settled)})
                entry.update(settled)
            issued.add(call_id)
    require(set(own_entries) == issued, 'unreported or missing experiment API result; conservative charge preserved')
    after_totals = totals(after)
    after.update(after_totals)
    require(after['total_usd'] <= 250 and after['by_category_usd']['reader_judge_dev'] <= 120
            and all(value <= after['caps_usd'][category] for category, value in after['by_category_usd'].items()), 'global spend cap exceeded')
    incremental = sum(entry['cost_usd'] for entry in after['entries'] if entry.get('call_id') in own_ids)
    require(incremental <= cap, 'experiment spend cap exceeded')
    ledger_after = encoded(after)
    report_after = copy.deepcopy(report)
    fields = {'spend_ledger_sha256': byte_digest(ledger_after), 'incremental_spend_usd': incremental,
              'global_recorded_spend_usd': after['total_usd']}
    report_after.update(fields)
    changed = bool(updates) or any(report.get(field) != value for field, value in fields.items())
    if not changed:
        return {'changed': False, 'repairs': 0, 'released_usd': 0.0}
    report_after_bytes = encoded(report_after)
    journal = {'schema_version': 1, 'status': 'prepared',
               'before': {'spend_sha256': byte_digest(ledger_bytes), 'answers_sha256': byte_digest(report_bytes)},
               'after': {'spend_sha256': byte_digest(ledger_after), 'answers_sha256': byte_digest(report_after_bytes)},
               'inputs': inputs, 'entry_updates': updates,
               'ledger_totals': {'before': before_totals, 'after': after_totals},
               'report_fields': {'before': {field: report[field] for field in fields}, 'after': fields},
               'summary': {'changed': True, 'repairs': len(updates),
                           'released_usd': ledger['total_usd'] - after['total_usd'],
                           'incremental_spend_usd': incremental, 'global_recorded_spend_usd': after['total_usd']}}
    journal['settler_sha256'] = file_digest(__file__)
    return journal


def finish(work, journal):
    for item in journal['inputs']:
        same(file_digest(Path(item['path'])), item['sha256'], 'settlement input changed during recovery')
    for filename, key in (('spend.json', 'spend_sha256'), ('answers.json', 'answers_sha256')):
        current = file_digest(work / filename)
        require(current in (journal['before'][key], journal['after'][key]), 'settlement target changed concurrently')
    for filename, key in (('spend.json', 'spend_sha256'), ('answers.json', 'answers_sha256')):
        path = work / filename
        if file_digest(path) == journal['after'][key]:
            continue
        value = json.loads(path.read_bytes())
        if filename == 'spend.json':
            entries = {entry['cache_key']: index for index, entry in enumerate(value['entries'])}
            for update in journal['entry_updates']:
                index = entries[update['cache_key']]
                require(canonical(value['entries'][index]) == canonical(update['before']), 'settlement entry changed concurrently')
                value['entries'][index] = update['after']
            value.update(journal['ledger_totals']['after'])
        else:
            value.update(journal['report_fields']['after'])
        content = encoded(value)
        same(byte_digest(content), journal['after'][key], 'recovered settlement output fingerprint mismatch')
        write_atomic(path, content)


def apply_plan(work, journal):
    work = Path(work).expanduser().resolve()
    if journal.get('changed') is False:
        return journal
    directory = work / 'ledger_settlements'
    directory.mkdir(exist_ok=True)
    sync_directory(work)
    identity = digest(canonical({'before': journal['before'], 'after': journal['after']}))
    path = directory / (identity + '.json')
    if path.exists():
        existing = json.loads(path.read_bytes())
        require(existing['before'] == journal['before'] and existing['after'] == journal['after'], 'settlement audit conflict')
        journal = existing
    else:
        with path.open('xb') as stream:
            stream.write(encoded(journal))
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(directory)
    finish(work, journal)
    journal['status'] = 'committed'
    write_atomic(path, encoded(journal))
    return {**journal['summary'], 'audit_path': str(path)}


def settle(work, apply=False):
    work = Path(work).expanduser().resolve()
    pending = [path for path in sorted((work / 'ledger_settlements').glob('*.json'))
               if json.loads(path.read_bytes()).get('status') == 'prepared']
    require(len(pending) <= 1, 'multiple pending settlement journals require inspection')
    if pending:
        require(apply, 'pending settlement requires --apply to recover the recorded transaction')
        journal = json.loads(pending[0].read_bytes())
        result = apply_plan(work, journal)
        return {**result, 'recovered': True}
    plan = build_plan(work)
    if apply:
        return apply_plan(work, plan)
    return plan.get('summary', plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='apply the verified settlement; otherwise preview only')
    args = parser.parse_args()
    print(json.dumps(settle(args.work, args.apply), allow_nan=False))


if __name__ == '__main__':
    main()
