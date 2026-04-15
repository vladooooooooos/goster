"""Common embedding result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextEmbedding:
    """Vector embedding for one text input."""

    text: str
    vector: list[float]
