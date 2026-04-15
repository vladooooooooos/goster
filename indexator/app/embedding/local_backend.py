"""Local Hugging Face embedding backend based on BAAI/bge-m3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.embedding.base import TextEmbedding


@dataclass(frozen=True)
class LocalEmbeddingSettings:
    """Runtime settings for the local BGE-M3 embedder."""

    model_name: str
    device: str = "auto"
    batch_size: int = 8
    normalize_embeddings: bool = True


class LocalBgeM3Embedder:
    """Generate dense embeddings locally with BAAI/bge-m3."""

    def __init__(self, settings: LocalEmbeddingSettings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> object:
        """Load the model lazily on the configured device."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[TextEmbedding]:
        """Return embeddings for non-empty input texts."""
        clean_texts = [text.strip() for text in texts if text.strip()]
        if not clean_texts:
            return []

        vectors = self.model.encode(
            clean_texts,
            batch_size=max(1, self.settings.batch_size),
            normalize_embeddings=self.settings.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return [
            TextEmbedding(text=text, vector=to_float_list(vector))
            for text, vector in zip(clean_texts, vectors, strict=True)
        ]

    def get_embedding_dimension(self) -> int:
        """Return the dense embedding dimension for the loaded model."""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            sample_embeddings = self.embed_texts(["dimension probe"])
            return len(sample_embeddings[0].vector) if sample_embeddings else 0
        return int(dimension)

    def describe_device_runtime(self) -> str:
        """Return a compact runtime device diagnostic for UI logs."""
        return describe_device_runtime(self.settings.device, self.device)


def resolve_device(configured_device: str) -> str:
    """Resolve auto device selection to a SentenceTransformer device string."""
    normalized = configured_device.strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if is_cuda_available() else "cpu"
    if normalized.startswith("cuda") and not is_cuda_available():
        return "cpu"
    return normalized


def is_cuda_available() -> bool:
    """Return whether the current Torch runtime can actually use CUDA."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def describe_device_runtime(configured_device: str, resolved_device: str) -> str:
    """Build a human-readable Torch device diagnostic."""
    try:
        import torch
    except ImportError:
        return (
            f"requested={configured_device}, resolved={resolved_device}, "
            "torch=not installed, cuda_available=false"
        )

    cuda_available = torch.cuda.is_available()
    cuda_version = getattr(torch.version, "cuda", None) or "none"
    device_name = "none"
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
    return (
        f"requested={configured_device}, resolved={resolved_device}, "
        f"torch={torch.__version__}, cuda_available={str(cuda_available).lower()}, "
        f"cuda_version={cuda_version}, gpu={device_name}"
    )


def to_float_list(vector: Any) -> list[float]:
    """Convert a numpy or tensor vector to a plain Python float list."""
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]
