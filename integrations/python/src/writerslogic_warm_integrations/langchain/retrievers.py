# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""A LangChain retriever that exposes the WARM search knobs directly."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .vectorstores import WarmVectorStore


class WarmRetriever(BaseRetriever):
    """Retrieve from a `WarmVectorStore`.

    `search_type` is `similarity` (default), `similarity_score_threshold` (drops hits whose
    relevance, see `WarmVectorStore`, is below `score_threshold`) or `mmr`
    (`fetch_k` candidates, clamped to the store's `max_fetch`). With `include_score` the raw
    score is added to each document as `metadata["warm_score"]`. `k` and `filter` can be
    overridden per call: `retriever.invoke(q, k=2, filter={"lang": "en"})`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: WarmVectorStore
    k: int = 4
    filter: Optional[Dict[str, Any]] = None
    search_type: Literal["similarity", "similarity_score_threshold", "mmr"] = "similarity"
    score_threshold: Optional[float] = None
    fetch_k: int = 20
    lambda_mult: float = 0.5
    include_score: bool = False

    def _options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        k = kwargs.get("k", self.k)
        flt = kwargs.get("filter", self.filter)
        if self.search_type == "similarity_score_threshold" and self.score_threshold is None and "score_threshold" not in kwargs:
            raise ValueError("score_threshold is required for similarity_score_threshold")
        return {"k": k, "filter": flt, "score_threshold": kwargs.get("score_threshold", self.score_threshold)}

    def _finish(self, pairs, threshold) -> List[Document]:
        fn = self.vectorstore._select_relevance_score_fn()
        docs = []
        for doc, score in pairs:
            if self.search_type == "similarity_score_threshold" and fn(score) < threshold:
                continue
            if self.include_score:
                doc = Document(id=doc.id, page_content=doc.page_content, metadata={**doc.metadata, "warm_score": score})
            docs.append(doc)
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        o = self._options(kwargs)
        if self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(query, o["k"], self.fetch_k, self.lambda_mult, o["filter"])
        return self._finish(self.vectorstore.similarity_search_with_score(query, o["k"], o["filter"]), o["score_threshold"])

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        o = self._options(kwargs)
        if self.search_type == "mmr":
            return await self.vectorstore.amax_marginal_relevance_search(query, o["k"], self.fetch_k, self.lambda_mult, o["filter"])
        return self._finish(await self.vectorstore.asimilarity_search_with_score(query, o["k"], o["filter"]), o["score_threshold"])
