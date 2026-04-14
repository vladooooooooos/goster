from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.qdrant_retriever import QdrantRetriever
from app.services.reranker_service import LocalRerankerService
from app.services.retrieval_types import RetrievedBlock, RerankedBlock, make_reranked_block
from app.services.retriever import EmptyQueryError, Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalPipelineResult:
    query: str
    candidates: list[RetrievedBlock]
    results: list[RerankedBlock]
    info: dict[str, Any]


class RetrievalPipeline:
    """Runtime retrieval flow: vector candidates, reranking, final evidence selection."""

    def __init__(
        self,
        retriever: Retriever,
        settings: Any,
        qdrant_retriever: QdrantRetriever | None = None,
        reranker: LocalRerankerService | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings
        self._qdrant_retriever = qdrant_retriever
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int) -> RetrievalPipelineResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise EmptyQueryError("Search query cannot be empty.")

        final_top_n = self._final_top_n(top_k)
        retrieval_top_k = self._retrieval_top_k(top_k)
        candidates, retrieval_info = self._retrieve_candidates(normalized_query, top_k=retrieval_top_k)
        info = {
            **retrieval_info,
            "requested_top_k": top_k,
            "retrieval_top_k": retrieval_top_k,
            "final_top_n": final_top_n,
            "retrieved_candidates_count": len(candidates),
            "reranker_enabled": self._reranker.enabled if self._reranker else False,
        }

        results = self._rerank_candidates(normalized_query, candidates, top_n=final_top_n)
        info["reranked_results_count"] = len(results)

        return RetrievalPipelineResult(
            query=normalized_query,
            candidates=candidates,
            results=results,
            info=info,
        )

    def _retrieve_candidates(self, query: str, top_k: int) -> tuple[list[RetrievedBlock], dict[str, Any]]:
        backend = self._settings.retrieval_backend.strip().lower()
        if backend in {"auto", "qdrant"} and self._qdrant_retriever is not None:
            try:
                return self._qdrant_retriever.search(query, top_k=top_k)
            except Exception as exc:
                logger.warning("Qdrant retrieval failed; falling back to JSON retrieval. Error: %s", exc)

        results, index_summary = self._retriever.retrieve_blocks(query, top_k=top_k)
        logger.info("JSON retrieval returned %s candidate block(s).", len(results))
        return results, {
            "backend": "json",
            "top_k": top_k,
            "index_summary": index_summary,
        }

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[RetrievedBlock],
        top_n: int,
    ) -> list[RerankedBlock]:
        if not candidates:
            return []
        if self._reranker is None or not self._reranker.enabled:
            logger.info("Reranker disabled; using retrieval-only order for %s candidate block(s).", len(candidates))
            return [make_reranked_block(candidate, None) for candidate in candidates[:top_n]]

        try:
            return self._reranker.rerank(query, candidates, top_n=top_n)
        except Exception as exc:
            logger.warning("Reranker failed; using retrieval-only order. Error: %s", exc)
            return [make_reranked_block(candidate, None) for candidate in candidates[:top_n]]

    def _retrieval_top_k(self, requested_top_k: int) -> int:
        if self._reranker and self._reranker.enabled:
            return max(requested_top_k, self._settings.reranker_top_k)
        return requested_top_k

    def _final_top_n(self, requested_top_k: int) -> int:
        if self._reranker and self._reranker.enabled:
            return max(1, min(requested_top_k, self._settings.reranker_top_n))
        return requested_top_k
