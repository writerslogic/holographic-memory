# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex `MetadataFilters` to the core filter AST.

Semantics follow `llama_index.core.vector_stores.utils.build_metadata_filter_fn`, with one
extension: nested `MetadataFilters` are supported. `FilterCondition.NOT` means none of its
filters match. A missing key matches only NE, NIN and IS_EMPTY.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from ..core.filters import And, Cmp, Eq, Filter, Not, Or


def _values(value) -> list:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _leaf(f: MetadataFilter) -> Filter:
    op, key, value = f.operator, f.key, f.value
    if op == FilterOperator.EQ:
        return Eq(key, value)
    if op == FilterOperator.NE:
        return Cmp(key, "ne", value)
    if op == FilterOperator.GT:
        return Cmp(key, "gt", value)
    if op == FilterOperator.GTE:
        return Cmp(key, "gte", value)
    if op == FilterOperator.LT:
        return Cmp(key, "lt", value)
    if op == FilterOperator.LTE:
        return Cmp(key, "lte", value)
    if op == FilterOperator.IN:
        return Cmp(key, "in", _values(value))
    if op == FilterOperator.NIN:
        return Cmp(key, "nin", _values(value))
    if op == FilterOperator.CONTAINS:
        return Cmp(key, "contains", value)
    if op == FilterOperator.ANY:
        return Or(tuple(Cmp(key, "contains", v) for v in _values(value)))
    if op == FilterOperator.ALL:
        return And(tuple(Cmp(key, "contains", v) for v in _values(value)))
    if op == FilterOperator.TEXT_MATCH:
        return Cmp(key, "contains", value)
    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        return Cmp(key, "icontains", value)
    if op == FilterOperator.IS_EMPTY:
        return Cmp(key, "empty", True)
    raise ValueError(f"unsupported filter operator {op!r}")


def to_filter(filters: Optional[MetadataFilters]) -> Optional[Filter]:
    """Convert; returns None for no filters."""
    if filters is None or not filters.filters:
        return None
    terms: List[Filter] = []
    for item in filters.filters:
        if isinstance(item, MetadataFilters):
            nested = to_filter(item)
            if nested is not None:
                terms.append(nested)
        else:  # MetadataFilter or the legacy ExactMatchFilter (key, value)
            terms.append(_leaf(item) if isinstance(item, MetadataFilter) else Eq(item.key, item.value))
    if not terms:
        return None
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.AND:
        return terms[0] if len(terms) == 1 else And(tuple(terms))
    if condition == FilterCondition.OR:
        return terms[0] if len(terms) == 1 else Or(tuple(terms))
    if condition == FilterCondition.NOT:
        return Not(Or(tuple(terms)))
    raise ValueError(f"unsupported filter condition {condition!r}")
