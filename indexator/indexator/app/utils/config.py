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
    local_path: str
    distance_metric: str


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
        storage=StorageConfig(
            provider=str(raw_config["storage"]["provider"]),
            collection_name=str(raw_config["storage"]["collection_name"]),
            local_path=str(raw_config["storage"]["local_path"]),
            distance_metric=str(raw_config["storage"].get("distance_metric", "Cosine")),
        ),
    )
