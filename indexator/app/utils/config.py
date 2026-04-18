"""Configuration loading for Indexator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UiConfig:
    """UI configuration values."""

    window_width: int
    window_height: int


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local embedding configuration values."""

    model_name: str
    device: str
    batch_size: int
    normalize_embeddings: bool


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration values."""

    provider: str
    collection_name: str
    distance_metric: str
    url: str
    host: str
    port: int
    https: bool
    timeout_seconds: float
    api_key: str | None
    shared_data_path: str
    quantization_enabled: bool
    quantization_mode: str
    vectors_on_disk: bool
    quantized_vectors_always_ram: bool
    upsert_batch_size: int


@dataclass(frozen=True)
class IndexingConfig:
    """Index block selection and compaction configuration values."""

    mode: str
    min_indexable_chars: int
    target_chunk_chars: int
    max_chunk_chars: int
    store_visual_metadata: bool


@dataclass(frozen=True)
class ApplicationInfo:
    """Application metadata."""

    name: str
    version: str


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    app: ApplicationInfo
    ui: UiConfig
    embedding: EmbeddingConfig
    indexing: IndexingConfig
    storage: StorageConfig


def load_config(config_path: Path) -> AppConfig:
    """Load Indexator configuration from a JSON file."""
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    return build_config(raw_config)


def build_config(raw_config: dict[str, Any]) -> AppConfig:
    """Build a typed application config from raw JSON data."""
    return AppConfig(
        app=ApplicationInfo(
            name=str(raw_config["app"]["name"]),
            version=str(raw_config["app"]["version"]),
        ),
        ui=UiConfig(
            window_width=int(raw_config["ui"]["window_width"]),
            window_height=int(raw_config["ui"]["window_height"]),
        ),
        embedding=EmbeddingConfig(
            model_name=str(raw_config["embedding"]["model_name"]),
            device=str(raw_config["embedding"].get("device", "auto")),
            batch_size=int(raw_config["embedding"].get("batch_size", 8)),
            normalize_embeddings=bool(raw_config["embedding"].get("normalize_embeddings", True)),
        ),
        indexing=IndexingConfig(
            mode=str(raw_config.get("indexing", {}).get("mode", "compact")),
            min_indexable_chars=int(raw_config.get("indexing", {}).get("min_indexable_chars", 24)),
            target_chunk_chars=int(raw_config.get("indexing", {}).get("target_chunk_chars", 900)),
            max_chunk_chars=int(raw_config.get("indexing", {}).get("max_chunk_chars", 1600)),
            store_visual_metadata=bool(raw_config.get("indexing", {}).get("store_visual_metadata", True)),
        ),
        storage=StorageConfig(
            provider=str(raw_config["storage"]["provider"]),
            collection_name=str(raw_config["storage"]["collection_name"]),
            distance_metric=str(raw_config["storage"].get("distance_metric", "Cosine")),
            url=str(raw_config["storage"].get("url", "http://127.0.0.1:6333")),
            host=str(raw_config["storage"].get("host", "127.0.0.1")),
            port=int(raw_config["storage"].get("port", 6333)),
            https=bool(raw_config["storage"].get("https", False)),
            timeout_seconds=float(raw_config["storage"].get("timeout_seconds", 5.0)),
            api_key=optional_string(raw_config["storage"].get("api_key")),
            shared_data_path=str(raw_config["storage"].get("shared_data_path", "../shared/data")),
            quantization_enabled=bool(raw_config["storage"].get("qdrant_quantization_enabled", True)),
            quantization_mode=str(raw_config["storage"].get("qdrant_quantization_mode", "scalar")),
            vectors_on_disk=bool(raw_config["storage"].get("qdrant_vectors_on_disk", True)),
            quantized_vectors_always_ram=bool(
                raw_config["storage"].get("qdrant_quantized_vectors_always_ram", True)
            ),
            upsert_batch_size=int(raw_config["storage"].get("qdrant_upsert_batch_size", 64)),
        ),
    )


def optional_string(value: Any) -> str | None:
    """Return a non-empty stripped string or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def resolve_shared_data_path(shared_data_path: str | Path, app_root: Path | None = None) -> Path:
    """Resolve shared non-Qdrant index metadata path relative to the app root."""
    resolved_path = Path(shared_data_path)
    if not resolved_path.is_absolute() and app_root is not None:
        resolved_path = app_root / resolved_path
    return resolved_path.resolve()
