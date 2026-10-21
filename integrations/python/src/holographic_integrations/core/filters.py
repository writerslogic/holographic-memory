# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Metadata filter AST.

The service filters by equality on top-level metadata keys only. Equality terms that are
ANDed at the top level are pushed to the service; every other term (comparison, `in`, `or`,
`not`, nested keys, existence) is evaluated here on an over-fetched candidate list, so a
filter with such terms can return fewer than `k` results (see `HmsConfig.overfetch`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

_MISSING = object()


def _lookup(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    cur: Any = metadata
    for part in key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _ordered(a: Any, b: Any, op: str) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    numeric = (int, float)
    if (isinstance(a, numeric) and isinstance(b, numeric)) or (isinstance(a, str) and isinstance(b, str)):
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
    return False


class Filter(ABC):
    @abstractmethod
    def matches(self, metadata: Mapping[str, Any]) -> bool: ...


@dataclass(frozen=True)
class Eq(Filter):
    key: str
    value: Any

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        found = _lookup(metadata, self.key)
        return found is not _MISSING and _equal(found, self.value)


@dataclass(frozen=True)
class Cmp(Filter):
    """`op` is one of ne, gt, gte, lt, lte, in, nin, exists, empty, contains, icontains.

    `ne` and `nin` match documents that lack the key (MongoDB semantics). `contains` is
    substring for strings and membership for lists.
    """

    key: str
    op: str
    value: Any

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        found = _lookup(metadata, self.key)
        present = found is not _MISSING
        op = self.op
        if op == "exists":
            return present == bool(self.value)
        if op == "empty":
            is_empty = not present or found is None or found == "" or found == []
            return is_empty == bool(self.value)
        if op == "ne":
            return not present or not _equal(found, self.value)
        if op == "nin":
            return not present or not any(_equal(found, v) for v in self.value)
        if not present:
            return False
        if op == "in":
            return any(_equal(found, v) for v in self.value)
        if op in ("gt", "gte", "lt", "lte"):
            return _ordered(found, self.value, op)
        if op == "icontains":
            return isinstance(found, str) and isinstance(self.value, str) and self.value.lower() in found.lower()
        if op == "contains":
            if isinstance(found, str) and isinstance(self.value, str):
                return self.value in found
            if isinstance(found, list):
                return any(_equal(item, self.value) for item in found)
            return False
        raise ValueError(f"unknown filter operator {op!r}")


@dataclass(frozen=True)
class And(Filter):
    terms: Tuple[Filter, ...]

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        return all(t.matches(metadata) for t in self.terms)


@dataclass(frozen=True)
class Or(Filter):
    terms: Tuple[Filter, ...]

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        return any(t.matches(metadata) for t in self.terms)


@dataclass(frozen=True)
class Not(Filter):
    term: Filter

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        return not self.term.matches(metadata)


FilterLike = Union[Filter, Mapping[str, Any], None]

_OPS = {
    "$eq": "eq", "$ne": "ne", "$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte",
    "$in": "in", "$nin": "nin", "$exists": "exists", "$contains": "contains",
}


def all_of(terms: Sequence[Filter]) -> Optional[Filter]:
    flat: List[Filter] = []
    for t in terms:
        flat.extend(t.terms if isinstance(t, And) else [t])
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else And(tuple(flat))


def parse(spec: FilterLike) -> Optional[Filter]:
    """Build a filter from a MongoDB-style mapping.

    `{"a": 1}` is equality; `{"a": {"$gt": 1}}` a comparison; `$and` and `$or` take lists and
    `$not` a mapping. Several keys in one mapping are ANDed. A `Filter` is returned unchanged.
    """
    if spec is None or isinstance(spec, Filter):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(f"filter must be a mapping or Filter, not {type(spec).__name__}")
    terms: List[Filter] = []
    for key, value in spec.items():
        if key in ("$and", "$or"):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} takes a list of filters")
            subs = tuple(f for f in (parse(v) for v in value) if f is not None)
            if key == "$and":
                if subs:
                    terms.append(And(subs))
            else:
                terms.append(Or(subs))
        elif key == "$not":
            sub = parse(value)
            if sub is not None:
                terms.append(Not(sub))
        elif key.startswith("$"):
            raise ValueError(f"unsupported top-level operator {key}")
        elif isinstance(value, Mapping) and value and all(k.startswith("$") for k in value):
            for op_key, operand in value.items():
                op = _OPS.get(op_key)
                if op is None:
                    raise ValueError(f"unsupported operator {op_key}")
                terms.append(Eq(key, operand) if op == "eq" else Cmp(key, op, operand))
        else:
            terms.append(Eq(key, value))
    return all_of(terms)


def split(flt: Optional[Filter]) -> Tuple[Dict[str, Any], Optional[Filter]]:
    """Separate the part the service evaluates (top-level equalities) from the residual."""
    if flt is None:
        return {}, None
    terms = flt.terms if isinstance(flt, And) else (flt,)
    pushed: Dict[str, Any] = {}
    rest: List[Filter] = []
    for t in terms:
        # Floats stay client-side: the service compares JSON numbers, where 1 and 1.0 differ.
        if (
            isinstance(t, Eq)
            and (t.value is None or isinstance(t.value, (str, bool, int)))
            and "." not in t.key
            and t.key not in pushed
        ):
            pushed[t.key] = t.value
        else:
            rest.append(t)
    return pushed, all_of(rest)
