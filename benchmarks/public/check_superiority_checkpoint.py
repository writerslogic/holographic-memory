# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Validate startup, preregistration, conservative spend and the Rust checkpoint gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    ['cargo', 'fmt', '--', '--check'],
    ['cargo', 'clippy', '--all-targets', '--', '-D', 'warnings'],
    ['cargo', 'clippy', '--all-targets', '--all-features', '--', '-D', 'warnings'],
    ['cargo', 'test', '--locked'],
    ['cargo', 'test', '--locked', '--all-features'],
]
CAPS = {'total': 250, 'reader_judge_dev': 120, 'qwen8b': 60, 'x86': 60, 'reserve': 10, 'heldout400': 0}
INPUTS = (
    'downloads/nytimes-256-angular.hdf5', 'downloads/glove-100-angular.hdf5',
    'scifact/meta.json', 'scifact/corpus.f32', 'nfcorpus/meta.json', 'nfcorpus/corpus.f32',
    'downloads/longmemeval_s_cleaned.json', 'holo_dev/export_dev.json', 'holo_dev/turn_emb.f16',
    'longmemeval_s/runs/s-dev-small-all2/scores.json',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def load(path):
    return json.loads(path.read_text())


def check(startup_path, preregistration_path, spend_path, gates_path):
    startup, registration, spend, gates = map(load, (startup_path, preregistration_path, spend_path, gates_path))
    require(startup['schema_version'] == 1 and startup['session'] == 1, 'invalid startup schema or session')
    require(startup['branch'] == 'research/2026-10-superiority', 'startup branch differs')
    require(startup['rust'] == '1.96.0' and startup['python'] == '3.13.9' and startup['numpy'] == '2.2.6'
            and isinstance(startup['uv'], str) and startup['uv'], 'startup toolchain versions differ')
    require(isinstance(startup['base_commit'], str) and len(startup['base_commit']) == 40
            and all(character in '0123456789abcdef' for character in startup['base_commit']), 'invalid base commit')
    require(len(startup['load_average']) == 3 and all(finite_nonnegative(value) for value in startup['load_average']),
            'invalid recorded machine load')
    require(type(startup['free_disk_bytes']) is int and startup['free_disk_bytes'] >= 0, 'invalid recorded free disk')
    require(type(startup['timing_lock_held_at_probe']) is bool
            and ((isinstance(startup['timing_lock_owner'], str) and startup['timing_lock_owner'])
                 if startup['timing_lock_held_at_probe'] else startup['timing_lock_owner'] is None),
            'timing lock ownership record inconsistent')
    require(startup['paid_spend_usd'] == 0 and startup['existing_uncommitted_work_preserved'] is True,
            'startup spend or work preservation record differs')
    cache = Path.home() / '.cache/hms-bench'
    expected = {(cache / item).resolve() for item in INPUTS}
    records = startup['inputs']
    require(len(records) == len(INPUTS) and {Path(record['path']).resolve() for record in records} == expected,
            'startup required input list differs')
    fingerprints, missing = [], []
    for record in records:
        path = Path(record['path'])
        require(type(record['present']) is bool and type(record['bytes']) is int and record['bytes'] >= 0,
                'invalid input availability record')
        require(record['present'] == path.is_file(), f'input availability changed: {path.name}')
        if record['present']:
            require(record['bytes'] == path.stat().st_size and record['bytes'] > 0, f'input size mismatch: {path.name}')
            fingerprints.append({'path': str(path), 'bytes': record['bytes'], 'sha256': sha(path)})
        else:
            require(record['bytes'] == 0, 'absent input has nonzero recorded bytes')
            missing.append(str(path))
    require(registration['schema_version'] == 1 and registration['session'] == 1
            and registration['branch'] == startup['branch'], 'preregistration session differs')
    require(len(registration['items']) == 5 and {item['item'] for item in registration['items']} == set(range(1, 6)),
            'preregistration must contain all five mandatory items')
    for item in registration['items']:
        require(all(isinstance(item[field], str) and item[field].strip() for field in
                    ('lever', 'kill_test', 'metric', 'condition', 'expected_failure_mode'))
                and finite_nonnegative(item['budget_hours']) and item['budget_hours'] > 0,
                'incomplete mandatory item preregistration')
    protocol_path = Path(registration['reader_protocol_path'])
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    require(sha(protocol_path) == registration['reader_protocol_sha256'], 'frozen reader protocol fingerprint mismatch')
    protocol = load(protocol_path)
    require(protocol['created_before_first_reader_call'] is True and protocol['token_cap'] == 8192
            and protocol['cost']['subcap_usd'] == 120, 'reader protocol freeze or cap differs')
    require(registration['heldout_authorized'] is False and registration['spend_caps_usd'] == CAPS,
            'preregistered spend or held-out authorization differs')
    require(registration['ann_recall_targets'] == [0.9, 0.95] and registration['ann_paired_rounds'] == 11
            and registration['load_threshold'] == 3, 'ANN preregistration condition differs')
    require(spend['schema_version'] == 1 and spend['caps_usd'] == CAPS, 'spend caps differ')
    totals = {category: 0.0 for category in CAPS if category != 'total'}
    ids = set()
    reservation_only = usage_supported = 0.0
    for entry in spend['entries']:
        require(isinstance(entry['cache_key'], str) and len(entry['cache_key']) == 64
                and all(character in '0123456789abcdef' for character in entry['cache_key'])
                and entry['cache_key'] not in ids and isinstance(entry['call_id'], str) and entry['call_id'],
                'invalid or duplicate spend call identity')
        ids.add(entry['cache_key'])
        require(entry['category'] in totals and finite_nonnegative(entry['reserved_usd'])
                and finite_nonnegative(entry['cost_usd']) and finite_nonnegative(entry['unix_time'])
                and entry['unix_time'] > 0, 'invalid spend entry')
        require(entry['status'] in ('ok', 'incomplete', 'error', 'reserved'), 'invalid spend status')
        if entry['usage'] is None:
            require(entry['status'] in ('error', 'reserved')
                    and math.isclose(entry['cost_usd'], entry['reserved_usd'], rel_tol=1e-10, abs_tol=1e-12),
                    'usage-free call is not charged as a conservative reservation')
            reservation_only += entry['cost_usd']
        else:
            usage = entry['usage']
            require(isinstance(usage, dict) and type(usage['input_tokens']) is int and usage['input_tokens'] >= 0
                    and type(usage['output_tokens']) is int and usage['output_tokens'] >= 0, 'invalid observed API usage')
            cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
            require(type(cached) is int and 0 <= cached <= usage['input_tokens'], 'invalid cached token count')
            expected_cost = ((usage['input_tokens'] - cached) * protocol['cost']['input_usd_per_million']
                             + cached * protocol['cost']['cached_input_usd_per_million']
                             + usage['output_tokens'] * protocol['cost']['output_usd_per_million']) / 1_000_000
            require(math.isclose(entry['cost_usd'], expected_cost, rel_tol=1e-10, abs_tol=1e-12), 'usage-supported charge mismatch')
            usage_supported += entry['cost_usd']
        totals[entry['category']] += entry['cost_usd']
    require(spend['by_category_usd'].keys() == totals.keys()
            and all(math.isclose(spend['by_category_usd'][category], value, rel_tol=1e-10, abs_tol=1e-12)
                    for category, value in totals.items()), 'spend category totals mismatch')
    require(math.isclose(spend['total_usd'], sum(totals.values()), rel_tol=1e-10, abs_tol=1e-12)
            and spend['total_usd'] <= CAPS['total'] and all(totals[category] <= CAPS[category] for category in totals),
            'spend totals or caps mismatch')
    require(isinstance(gates, list) and len(gates) == len(COMMANDS)
            and [gate['command'] for gate in gates] == COMMANDS, 'Rust gate command list differs')
    require(all(type(gate['exit_code']) is int and gate['exit_code'] == 0
                and finite_nonnegative(gate['elapsed_seconds']) for gate in gates), 'Rust checkpoint gate failed')
    inputs = (startup_path, preregistration_path, spend_path, gates_path, protocol_path)
    return {'schema_version': 1, 'checkpoint_validated': True, 'answer_results_validated': False,
            'D1_established': False, 'D2_established': False, 'D3_established': False,
            'rust_gate_commands_verified': 5, 'mandatory_items_preregistered': 5,
            'startup_inputs_verified': len(fingerprints), 'missing_inputs': missing,
            'startup_load_gate_met': startup['load_average'][0] < registration['load_threshold'],
            'spend_calls_logged': len(ids), 'conservative_reservation_usd': reservation_only,
            'usage_supported_cost_usd': usage_supported, 'observed_billing_established': False,
            'spend_accounting_total_usd': spend['total_usd'],
            'artifacts': [{'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                           'sha256': sha(path)} for path in inputs],
            'cached_input_fingerprints': fingerprints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup', type=Path, default=ROOT / 'benchmarks/results/startup_2026-10.json')
    parser.add_argument('--preregistration', type=Path, default=ROOT / 'benchmarks/results/preregistration_2026-10.json')
    parser.add_argument('--spend', type=Path, default=ROOT / 'benchmarks/results/spend_2026-10.json')
    parser.add_argument('--gates', type=Path, default=ROOT / 'benchmarks/results/gates_2026-10.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/results/checkpoint_2026-10_check.json')
    args = parser.parse_args()
    result = check(args.startup, args.preregistration, args.spend, args.gates)
    result['checker_sha256'] = sha(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
