"""Shared Qdrant vector store configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QdrantVectorStoreConfig:
    """Connection and collection settings for a local Qdrant store."""

    local_path: Path
    collection_name: str
    distance_metric: str = "Cosine"


def resolve_local_path(local_path: str | Path, app_root: Path | None = None) -> Path:
    """Resolve a local Qdrant path relative to an optional application root."""
    resolved_path = Path(local_path)
    if not resolved_path.is_absolute() and app_root is not None:
        resolved_path = app_root / resolved_path
    return resolved_path.resolve()
