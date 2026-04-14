from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalEmbeddingSettings:
    model_name: str
    device: str = "auto"
    batch_size: int = 4
    normalize_embeddings: bool = True


class LocalEmbeddingService:
    def __init__(self, settings: LocalEmbeddingSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s.", self.settings.model_name, self.device)
            self._model = SentenceTransformer(self.settings.model_name, device=self.device)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [query],
            batch_size=max(1, self.settings.batch_size),
            normalize_embeddings=self.settings.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return to_float_list(vector)


def resolve_device(configured_device: str) -> str:
    normalized = configured_device.strip().lower()
    if normalized in {"", "auto"}:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]
