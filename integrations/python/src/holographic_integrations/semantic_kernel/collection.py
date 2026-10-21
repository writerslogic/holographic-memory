# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Semantic Kernel vector store (`VectorStore` / `VectorStoreCollection` / `VectorSearch`)."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Dict, Generic, List, Optional, TypeVar

from pydantic import ConfigDict, PrivateAttr
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.data.vector import (
    DistanceFunction,
    FieldTypes,
    GetFilteredRecordOptions,
    KernelSearchResults,
    SearchType,
    TModel,
    VectorSearch,
    VectorSearchOptions,
    VectorSearchResult,
    VectorStore,
    VectorStoreCollection,
    VectorStoreCollectionDefinition,
)
from semantic_kernel.exceptions import VectorStoreModelValidationError
from semantic_kernel.exceptions.vector_store_exceptions import (
    VectorSearchOptionsException,
    VectorStoreOperationException,
    VectorStoreOperationNotSupportedException,
)
from semantic_kernel.utils.list_handler import empty_generator

from holographic_integrations.core import AsyncHmsBackend, HmsConfig, Record
from holographic_integrations.core.filters import And, Cmp, Eq, Filter, Not, Or, all_of
from holographic_integrations.core.models import RESERVED_PREFIX

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover
    from typing_extensions import override

TKey = TypeVar("TKey", bound=str)

_COSINE_SIMILARITY = {DistanceFunction.COSINE_SIMILARITY}
_COSINE_DISTANCE = {DistanceFunction.COSINE_DISTANCE, DistanceFunction.DEFAULT}
_SCORE = "__hms_score"


async def _async_iter(items: Sequence[Any]):
    for item in items:
        yield item


class HolographicCollection(
    VectorStoreCollection[TKey, TModel],
    VectorSearch[TKey, TModel],
    Generic[TKey, TModel],
):
    """A Semantic Kernel collection stored in Holographic Memory.

    Mapping. The key field is the document id (strings only). One data field supplies the
    document text: `text_field=` if given, else the first `is_full_text_indexed` data field,
    else the first data field. Every other data field becomes metadata under its storage name
    (JSON types only). The model may have at most one vector field. Its value is used when it
    is a list of floats; a string or `None` value, or no vector field at all, makes the
    backend embed the text with `config.embedder` (or the service, with `text_queries`).
    Vectors are never returned (`include_vectors` is ignored: the service does not store
    retrievable vectors).

    Search: `search(values=...)` embeds with the field's or collection's embedding generator
    when there is one, otherwise with `config.embedder`. `top + skip` must not exceed
    `config.max_fetch`. Scores follow the field's distance function: `COSINE_SIMILARITY` is
    the cosine (higher is better), `COSINE_DISTANCE` and `DEFAULT` are `1 - cosine` (lower is
    better); other distance functions are rejected at construction. Hybrid search is available
    only with `config.text_queries` (the service's BM25 + cosine fusion).

    Filters: lambdas or strings such as `lambda x: x.team == "a" and x.rank > 2` using
    `== != < <= > >= in not in` and `.contains()`, combined with `and`, `or`, `not`.
    Top-level equalities are applied by the service before ranking; every other term is
    applied here on an over-fetch (`config.overfetch * top`), so the result can be shorter
    than `top`. The text field cannot be filtered on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    supported_key_types: ClassVar[set[str] | None] = {"str"}
    supported_search_types: ClassVar[set[SearchType]] = {SearchType.VECTOR, SearchType.KEYWORD_HYBRID}
    config: HmsConfig
    text_field: Optional[str] = None

    _backend: AsyncHmsBackend = PrivateAttr()
    _text_name: str = PrivateAttr()

    def __init__(
        self,
        record_type: type[TModel],
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        embedding_generator: EmbeddingGeneratorBase | None = None,
        *,
        config: Optional[HmsConfig] = None,
        text_field: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            record_type=record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator,
            config=config or HmsConfig(),
            text_field=text_field,
            **kwargs,
        )

    def model_post_init(self, __context: object | None = None) -> None:
        super().model_post_init(__context)
        cfg = self.config
        if cfg.collection is None or cfg.collection != self.collection_name:
            self.config = cfg = _with_collection(cfg, self.collection_name)
        self._backend = AsyncHmsBackend(cfg)

    # -- model validation -------------------------------------------------------------------
    @override
    def _validate_data_model(self) -> None:
        super()._validate_data_model()
        d = self.definition
        if len(d.vector_fields) > 1:
            raise VectorStoreModelValidationError("A document has one vector; use at most one vector field.")
        for f in d.vector_fields:
            if f.distance_function not in _COSINE_SIMILARITY | _COSINE_DISTANCE:
                raise VectorStoreModelValidationError(
                    f"Distance function {f.distance_function} is not supported; the service ranks by cosine."
                )
        data = d.data_fields
        names = [f.storage_name or f.name for f in data]
        bad = [n for n in names if n.startswith(RESERVED_PREFIX)]
        if bad:
            raise VectorStoreModelValidationError(f"Field names may not start with {RESERVED_PREFIX!r}: {bad}")
        text = self.text_field
        if text is not None:
            match = [f for f in data if text in (f.name, f.storage_name)]
            if not match:
                raise VectorStoreModelValidationError(f"text_field {text!r} is not a data field.")
            self._text_name = match[0].storage_name or match[0].name
        else:
            indexed = [f for f in data if f.is_full_text_indexed]
            chosen = (indexed or data or [None])[0]
            if chosen is None:
                raise VectorStoreModelValidationError("The model needs a data field to hold the document text.")
            self._text_name = chosen.storage_name or chosen.name

    # -- store-model conversion (store model == dict keyed by storage name) -------------------
    def _deserialize_store_models_to_dicts(self, records: Sequence[Any], **kwargs: Any) -> Sequence[dict[str, Any]]:
        return records

    def _serialize_dicts_to_store_models(self, records: Sequence[dict[str, Any]], **kwargs: Any) -> Sequence[Any]:
        return records

    def _to_record(self, rec: Dict[str, Any]) -> Record:
        d = self.definition
        key = rec[self._key_field_storage_name]
        if not isinstance(key, str) or not key:
            raise ValueError("record keys must be non-empty strings")
        text = rec.get(self._text_name)
        meta: Dict[str, Any] = {}
        for f in d.data_fields:
            name = f.storage_name or f.name
            if name != self._text_name and name in rec:
                meta[name] = rec[name]
        vector = None
        for f in d.vector_fields:
            value = rec.get(f.storage_name or f.name)
            if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
                vector = [float(x) for x in (value.tolist() if hasattr(value, "tolist") else value)]
        return Record(key, "" if text is None else str(text), meta, vector)

    def _from_record(self, rec: Record) -> Dict[str, Any]:
        out = {self._key_field_storage_name: rec.id, self._text_name: rec.text, **rec.metadata}
        return out

    # -- collection lifecycle ---------------------------------------------------------------
    @override
    async def ensure_collection_exists(self, **kwargs: Any) -> None:
        return None  # collections are implicit: records carry their collection

    @override
    async def collection_exists(self, **kwargs: Any) -> bool:
        return True

    @override
    async def ensure_collection_deleted(self, **kwargs: Any) -> None:
        await self._backend.forget_where(None)

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._backend.aclose()

    # -- CRUD -------------------------------------------------------------------------------
    @override
    async def _inner_upsert(self, records: Sequence[Any], **kwargs: Any) -> Sequence[TKey]:
        return await self._backend.remember([self._to_record(r) for r in records])  # type: ignore[return-value]

    @override
    async def _inner_get(
        self, keys: Sequence[TKey] | None = None, options: GetFilteredRecordOptions | None = None, **kwargs: Any
    ) -> Any:
        if keys:
            found = await self._backend.get(list(keys))
            return [self._from_record(r) for r in found if r is not None]
        if options is None:
            return None
        page = await self._backend.list()
        rows = [self._from_record(r) for r in page.records]
        for name, ascending in reversed(list((options.order_by or {}).items())):
            storage = next((f.storage_name or f.name for f in self.definition.fields if name in (f.name, f.storage_name)), name)
            rows.sort(key=lambda r: (r.get(storage) is None, r.get(storage)), reverse=not ascending)
        return rows[options.skip : options.skip + options.top]

    @override
    async def _inner_delete(self, keys: Sequence[TKey], **kwargs: Any) -> None:
        await self._backend.forget(list(keys))

    # -- search -----------------------------------------------------------------------------
    @override
    async def _inner_search(
        self,
        search_type: SearchType,
        options: VectorSearchOptions,
        values: Any | None = None,
        vector: Sequence[float | int] | None = None,
        **kwargs: Any,
    ) -> KernelSearchResults[VectorSearchResult[TModel]]:
        hybrid = search_type == SearchType.KEYWORD_HYBRID
        if hybrid and not self.config.text_queries:
            raise VectorStoreOperationNotSupportedException(
                "Keyword hybrid search needs HmsConfig(text_queries=True) and a service with on-device model stages."
            )
        field = self.definition.try_get_vector_field(options.vector_property_name)
        text = None if values is None else (values if isinstance(values, str) else str(values))
        if vector is None and text is not None and not hybrid and (
            (field is not None and field.embedding_generator) or self.embedding_generator
        ):
            vector = await self._generate_vector_from_values(values, options)
        if vector is None and text is None:
            raise VectorSearchOptionsException("search needs values or a vector")
        flt = self._build_filter(options.filter)
        top, skip = options.top, options.skip
        hits = await self._backend.recall(
            None if (vector is not None and not hybrid) else text,
            vector=None if vector is None else [float(x) for x in vector],
            k=top + skip,
            filter=_combine(flt),
        )
        distance = field is None or field.distance_function in _COSINE_DISTANCE
        rows: List[Dict[str, Any]] = []
        for h in hits[skip : skip + top]:
            row = self._from_record(Record(h.id, h.text, h.metadata))
            row[_SCORE] = (1.0 - h.score) if distance and not hybrid else h.score
            rows.append(row)
        total = len(hits) if options.include_total_count else None
        if not rows:
            return KernelSearchResults(results=empty_generator(), total_count=total)
        return KernelSearchResults(
            results=self._get_vector_search_results_from_results(_async_iter(rows), options), total_count=total
        )

    def _get_record_from_result(self, result: Any) -> Any:
        return {k: v for k, v in result.items() if k != _SCORE}

    def _get_score_from_result(self, result: Any) -> float | None:
        return result.get(_SCORE)

    # -- filters ----------------------------------------------------------------------------
    def _field_name(self, node: ast.AST) -> str:
        """`x.field` or `x["field"]` on the lambda parameter, mapped to the storage name."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.attr
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant):
            name = str(node.slice.value)
        else:
            raise VectorStoreOperationException(f"Unsupported filter operand: {ast.dump(node)}")
        for f in self.definition.fields:
            if name in (f.name, f.storage_name):
                storage = f.storage_name or f.name
                if f.field_type != FieldTypes.DATA:
                    raise VectorStoreOperationException(f"Only data fields can be filtered, not {name!r}.")
                if storage == self._text_name:
                    raise VectorStoreOperationException("The text field cannot be used in a filter.")
                return storage
        raise VectorStoreOperationException(f"Unknown field {name!r} in filter.")

    @staticmethod
    def _constant(node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except ValueError as exc:
            raise VectorStoreOperationException(f"Filter values must be literals: {ast.dump(node)}") from exc

    @override
    def _lambda_parser(self, node: ast.AST) -> Filter:
        if isinstance(node, ast.BoolOp):
            terms = tuple(self._lambda_parser(v) for v in node.values)
            return And(terms) if isinstance(node.op, ast.And) else Or(terms)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return Not(self._lambda_parser(node.operand))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise VectorStoreOperationException("Chained comparisons are not supported in filters.")
            op, left, right = node.ops[0], node.left, node.comparators[0]
            swap = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}
            if isinstance(left, ast.Constant) and type(op) in swap:
                left, right, op = right, left, swap[type(op)]()
            key = self._field_name(left)
            value = self._constant(right)
            table = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "lte", ast.Gt: "gt", ast.GtE: "gte",
                     ast.In: "in", ast.NotIn: "nin"}
            kind = table.get(type(op))
            if kind is None:
                raise VectorStoreOperationException(f"Comparison {type(op).__name__} is not supported in filters.")
            return Eq(key, value) if kind == "eq" else Cmp(key, kind, value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "contains" and len(node.args) == 1:
            return Cmp(self._field_name(node.func.value), "contains", self._constant(node.args[0]))
        raise VectorStoreOperationException(f"Unsupported filter expression: {ast.dump(node)}")


def _combine(parsed: Any) -> Optional[Filter]:
    if parsed is None:
        return None
    return all_of(parsed if isinstance(parsed, list) else [parsed])


def _with_collection(cfg: HmsConfig, collection: str) -> HmsConfig:
    import dataclasses

    return dataclasses.replace(cfg, collection=collection)


class HolographicStore(VectorStore):
    """`VectorStore` whose collections live in one Holographic Memory tenant.

    `list_collection_names` can only report collections opened through this store instance:
    the service has no collection listing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: HmsConfig
    _opened: List[str] = PrivateAttr(default_factory=list)

    def __init__(
        self, config: Optional[HmsConfig] = None, embedding_generator: EmbeddingGeneratorBase | None = None, **kwargs: Any
    ):
        super().__init__(config=config or HmsConfig(), embedding_generator=embedding_generator, **kwargs)

    @override
    def get_collection(
        self,
        record_type: type[TModel],
        *,
        definition: VectorStoreCollectionDefinition | None = None,
        collection_name: str | None = None,
        embedding_generator: EmbeddingGeneratorBase | None = None,
        text_field: Optional[str] = None,
        **kwargs: Any,
    ) -> HolographicCollection:
        col = HolographicCollection(
            record_type=record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator or self.embedding_generator,
            config=self.config,
            text_field=text_field,
            **kwargs,
        )
        if col.collection_name not in self._opened:
            self._opened.append(col.collection_name)
        return col

    @override
    async def list_collection_names(self, **kwargs: Any) -> Sequence[str]:
        return list(self._opened)

    @override
    async def ensure_collection_deleted(self, collection_name: str) -> None:
        async with AsyncHmsBackend(_with_collection(self.config, collection_name)) as backend:
            await backend.forget_where(None)
        if collection_name in self._opened:
            self._opened.remove(collection_name)
