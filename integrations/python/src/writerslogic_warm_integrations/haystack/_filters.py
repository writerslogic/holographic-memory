# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Haystack filter handling.

Haystack's filter language (nested AND/OR/NOT, comparison operators, ISO-date aware ordering) is
evaluated by `haystack.utils.filters.document_matches_filter`, so semantics are Haystack's own.
The service only understands equality, so the top-level AND of `meta.<key> == <string>` terms is
pushed down to narrow the candidate set and everything is re-checked with Haystack's evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from haystack.errors import FilterError
from haystack.utils.filters import _looks_like_iso_date

_COMPARISON = {"==", "!=", ">", ">=", "<", "<=", "in", "not in"}
_LOGIC = {"AND", "OR", "NOT"}


def validate(filters: Optional[Dict[str, Any]]) -> None:
    """Raise `FilterError` for a malformed filter, even when no stored document would reach the bad term."""
    if not filters:
        return
    if not isinstance(filters, dict):
        raise FilterError("Filters must be a dictionary")
    if "field" in filters:
        for key in ("operator", "value"):
            if key not in filters:
                raise FilterError(f"Comparison filter is missing the {key!r} key: {filters}")
        op = filters["operator"]
        if op not in _COMPARISON:
            raise FilterError(f"Unknown comparison operator {op!r}")
        if op in ("in", "not in") and not isinstance(filters["value"], list):
            raise FilterError(f"'{op}' requires a list value, got {type(filters['value']).__name__}")
        return
    if "operator" not in filters:
        raise FilterError(f"Filter has neither 'field' nor 'operator': {filters}")
    if "conditions" not in filters:
        raise FilterError(f"Logic filter is missing the 'conditions' key: {filters}")
    if filters["operator"] not in _LOGIC:
        raise FilterError(f"Unknown logic operator {filters['operator']!r}")
    conditions = filters["conditions"]
    if not isinstance(conditions, list):
        raise FilterError("'conditions' must be a list")
    for cond in conditions:
        if not isinstance(cond, dict):
            raise FilterError("Every condition must be a dictionary")
        if "field" in cond and "operator" not in cond:
            raise FilterError(f"Comparison filter is missing the 'operator' key: {cond}")
        if "field" not in cond and "operator" in cond and cond["operator"] not in _LOGIC:
            raise FilterError(f"Comparison filter is missing the 'field' key: {cond}")
        if "field" not in cond and "operator" not in cond:
            raise FilterError(f"Condition has neither 'field' nor 'operator': {cond}")
        validate(cond)


def _terms(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "field" in filters:
        return [filters]
    if filters.get("operator") == "AND":
        out: List[Dict[str, Any]] = []
        for cond in filters["conditions"]:
            out.extend(_terms(cond))
        return out
    return [filters]


def pushdown(filters: Optional[Dict[str, Any]], reserved: Tuple[str, ...]) -> Tuple[Dict[str, Any], bool]:
    """Return `(equalities, complete)`: the service-side filter and whether it is the whole filter.

    Only string equality on a top-level metadata key is pushed. Numbers and booleans are not:
    Haystack treats 1, 1.0 and True as equal while the service compares JSON values exactly.
    ISO-date-looking strings are not pushed either, since Haystack compares them as instants.
    """
    if not filters:
        return {}, True
    pushed: Dict[str, Any] = {}
    complete = True
    for term in _terms(filters):
        field, value = term.get("field"), term.get("value")
        if (
            term.get("operator") == "=="
            and isinstance(field, str)
            and field.startswith("meta.")
            and "." not in field[5:]
            and field[5:] not in reserved
            and isinstance(value, str)
            and not _looks_like_iso_date(value)
            and field[5:] not in pushed
        ):
            pushed[field[5:]] = value
        else:
            complete = False
    return pushed, complete
