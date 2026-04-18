from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.retrieval_types import RetrievedBlock, RerankedBlock, make_reranked_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankerSettings:
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    batch_size: int = 2
    max_length: int = 512
    use_fp16_if_available: bool = True


class LocalRerankerService:
    def __init__(self, settings: RerankerSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._model: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model %s on %s.", self.settings.model_name, self.device)
            self._model = CrossEncoder(
                self.settings.model_name,
                device=self.device,
                max_length=max(1, self.settings.max_length),
            )
            if self._should_use_fp16():
                self._model.model.half()
                logger.info("Using fp16 reranker weights on %s.", self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedBlock],
        top_n: int,
    ) -> list[RerankedBlock]:
        if not self.enabled or not candidates:
            return [make_reranked_block(candidate, None) for candidate in candidates[:top_n]]

        pairs = [(query, candidate.evidence_text) for candidate in candidates]
        try:
            import torch
        except ImportError:
            scores = self._predict_scores(pairs)
        else:
            with torch.inference_mode():
                scores = self._predict_scores(pairs)

        reranked = [
            make_reranked_block(candidate, float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(
            key=lambda item: (
                float("inf") if item.rerank_score is None else -item.rerank_score,
                -item.retrieval_score,
                item.source_file,
                item.block_id,
            )
        )
        selected = reranked[:top_n]
        top_scores = [round(item.rerank_score, 4) for item in selected if item.rerank_score is not None]
        top_block_types = [item.block_type or "unknown" for item in selected]
        logger.info(
            "Reranked %s candidate block(s); selected %s; top scores=%s; top block types=%s.",
            len(candidates),
            len(selected),
            top_scores,
            top_block_types,
        )
        return selected

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> Any:
        return self.model.predict(
            pairs,
            batch_size=max(1, self.settings.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _should_use_fp16(self) -> bool:
        return (
            self.settings.use_fp16_if_available
            and self.device.startswith("cuda")
            and self._model is not None
            and hasattr(self._model, "model")
        )


def resolve_device(configured_device: str) -> str:
    normalized = configured_device.strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized
