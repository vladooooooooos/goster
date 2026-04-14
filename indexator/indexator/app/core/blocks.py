"""Structured block models for future indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


BlockType = Literal[
    "heading",
    "paragraph",
    "list_item",
    "appendix_section",
    "table_of_contents",
    "table",
    "figure",
    "formula_with_context",
]


@dataclass(frozen=True)
class StructuredBlock:
    """First-pass logical block prepared for later indexing."""

    id: str
    doc_id: str
    file_name: str
    file_path: Path
    page_number: int
    block_type: BlockType
    text: str
    bbox: tuple[float, float, float, float] | None
    reading_order: int
    section_path: list[str]
    label: str | None = None
    context_text: str | None = None
