# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded exact selection over caller-supplied original-ID witnesses.

This module does not determine semantic support, witness minimality,
source authenticity, temporal validity or whether a source scope is complete.
"""

from __future__ import annotations


MAX_CANDIDATES = 2048
MAX_CLAIMS = 8
MAX_WITNESSES = 64
MAX_STATES = 32768
MAX_TRANSITIONS = 1_000_000
MAX_BUDGET = 10


def _ids(values, maximum, *, unique=True):
    if (not isinstance(values, (list, tuple)) or len(values) > maximum
            or any(not isinstance(value, str) or not value.strip() or '\x00' in value
                   or len(value) > 256 or len(value.encode('utf-8')) > 256 for value in values)):
        raise ValueError('Expected bounded original source IDs')
    if unique and len(set(values)) != len(values):
        raise ValueError('Original source IDs must be distinct')
    return list(values)


def _positions(mask):
    result = []
    while mask:
        bit = mask & -mask
        result.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(result)


def _original_ids(values, maximum, *, unique=True):
    try:
        return _ids(values, maximum, unique=unique)
    except UnicodeEncodeError as exc:
        raise ValueError('Original source IDs must be valid UTF-8') from exc


def assemble(baseline_ids, candidate_ids, alternatives, protected_claims=(), budget=10):
    """Select whole witnesses while preserving already-supported protected claims.

    Admissible outputs are unions of at most one witness per claim, filled with
    baseline IDs in candidate order. The objectives are completed claims, then
    baseline retention, then candidate order; spare capacity never adds arbitrary
    new IDs. Original-ID authenticity and semantic witness validity are upstream
    responsibilities.
    """
    if type(budget) is not int or not 0 <= budget <= MAX_BUDGET:
        raise ValueError('Evidence output budget must be an integer from0through10')
    baseline = _original_ids(baseline_ids, MAX_BUDGET)
    candidates = _original_ids(candidate_ids, MAX_CANDIDATES)
    rank = {owner: index for index, owner in enumerate(candidates)}
    if not set(baseline) <= rank.keys():
        raise ValueError('Every baseline ID must remain eligible')
    if not isinstance(alternatives, (list, tuple)) or len(alternatives) > MAX_CLAIMS:
        raise ValueError('Expected at most8 required atomic claims')
    if (not isinstance(protected_claims, (list, tuple)) or len(protected_claims) > MAX_CLAIMS
            or any(type(index) is not int or not 0 <= index < len(alternatives)
                   for index in protected_claims)
            or len(set(protected_claims)) != len(protected_claims)):
        raise ValueError('Protected claims must be distinct eligible integer indices')
    protected = set(protected_claims)
    baseline_mask = sum(1 << rank[owner] for owner in baseline)
    baseline_order = sorted(baseline, key=rank.__getitem__)
    claims, total_witnesses = [], 0
    for index, choices in enumerate(alternatives):
        if not isinstance(choices, (list, tuple)):
            raise ValueError('A claim requires a list of alternative witnesses')
        total_witnesses += len(choices)
        if total_witnesses > MAX_WITNESSES:
            raise ValueError('Evidence witness count exceeds64')
        witnesses = set()
        baseline_supported = False
        for witness in choices:
            owners = _original_ids(witness, MAX_CANDIDATES, unique=False)
            if not owners:
                raise ValueError('An evidence witness requires at least one source ID')
            if not set(owners) <= rank.keys():
                raise ValueError('A witness contains an ineligible source ID')
            mask = sum(1 << rank[owner] for owner in set(owners))
            baseline_supported |= mask & baseline_mask == mask
            if mask.bit_count() <= budget:
                witnesses.add(mask)
        if index in protected and not baseline_supported:
            raise ValueError('Every protected claim must already have a whole baseline witness')
        claims.append(tuple(sorted(witnesses, key=_positions)))

    states, transitions = {0}, 0
    for index, witnesses in enumerate(claims):
        expanded = set(states)
        for selected in states:
            for witness in witnesses:
                transitions += 1
                if transitions > MAX_TRANSITIONS:
                    raise RuntimeError('Exact evidence search exceeded its transition bound')
                union = selected | witness
                if union.bit_count() <= budget:
                    expanded.add(union)
                    if len(expanded) > MAX_STATES:
                        raise RuntimeError('Exact evidence search exceeded its state bound')
        if index in protected:
            expanded = {mask for mask in expanded
                        if any(witness & mask == witness for witness in witnesses)}
        if not expanded:
            raise ValueError('Output budget cannot satisfy every protected claim')
        states = expanded

    best, best_mask = None, None
    for selected in states:
        completed = selected
        for owner in baseline_order:
            if completed.bit_count() >= budget:
                break
            completed |= 1 << rank[owner]
        covered = {index for index, witnesses in enumerate(claims)
                   if any(witness & completed == witness for witness in witnesses)}
        if not protected <= covered or completed.bit_count() > budget:
            raise RuntimeError('Witness exchange violated a protected support or output invariant')
        objective = (-len(covered), -(completed & baseline_mask).bit_count(), _positions(completed))
        if best is None or objective < best:
            best, best_mask = objective, completed
    if best_mask is None:
        raise ValueError('No feasible protected-claim evidence selection exists')
    chosen = {}
    for index, witnesses in enumerate(claims):
        supported = [witness for witness in witnesses if witness & best_mask == witness]
        if supported:
            witness = min(supported, key=_positions)
            chosen[index] = [candidates[position] for position in _positions(witness)]
    return {'ids': [candidates[position] for position in _positions(best_mask)],
            'covered_claims': list(chosen), 'chosen_witnesses': chosen}
