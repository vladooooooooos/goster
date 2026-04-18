"""Compact index block preparation for retrieval storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from app.core.blocks import StructuredBlock


IndexingMode = Literal["compact", "rich"]

TEXT_BLOCK_TYPES = {"paragraph", "list_item"}
BOUNDARY_BLOCK_TYPES = {"heading", "appendix_section", "table", "figure", "formula_with_context"}
VISUAL_BLOCK_TYPES = {"table", "figure", "formula_with_context"}


@dataclass(frozen=True)
class IndexCompactionSettings:
    """Settings for preparing blocks that should be embedded and stored."""

    mode: IndexingMode = "compact"
    min_indexable_chars: int = 24
    target_chunk_chars: int = 900
    max_chunk_chars: int = 1600
    store_visual_metadata: bool = True


def compact_index_blocks(
    blocks: list[StructuredBlock],
    settings: IndexCompactionSettings | None = None,
) -> list[StructuredBlock]:
    """Return retrieval-ready blocks according to compact or rich indexing mode."""
    resolved_settings = settings or IndexCompactionSettings()
    if resolved_settings.mode == "rich":
        return list(blocks)

    compacted: list[StructuredBlock] = []
    pending_text_blocks: list[StructuredBlock] = []

    for block in blocks:
        if not is_indexable_block(block, resolved_settings):
            continue

        if block.block_type in TEXT_BLOCK_TYPES:
            if can_append_to_pending(pending_text_blocks, block, resolved_settings):
                pending_text_blocks.append(block)
            else:
                flush_pending_text(compacted, pending_text_blocks)
                pending_text_blocks = [block]
            continue

        flush_pending_text(compacted, pending_text_blocks)
        pending_text_blocks = []
        compacted.append(prepare_boundary_block(block, resolved_settings))

    flush_pending_text(compacted, pending_text_blocks)
    return compacted


def is_indexable_block(block: StructuredBlock, settings: IndexCompactionSettings) -> bool:
    """Return whether a structured block should become a retrieval point."""
    if block.block_type == "table_of_contents":
        return False
    if block.block_type in BOUNDARY_BLOCK_TYPES:
        return bool(block.text.strip())
    return len(normalize_text_for_length(block.text)) >= settings.min_indexable_chars


def can_append_to_pending(
    pending_blocks: list[StructuredBlock],
    block: StructuredBlock,
    settings: IndexCompactionSettings,
) -> bool:
    """Return whether a text block can join the current compact chunk."""
    if not pending_blocks:
        return True
    first_block = pending_blocks[0]
    if first_block.doc_id != block.doc_id:
        return False
    if first_block.section_path != block.section_path:
        return False
    if pending_blocks[-1].page_number != block.page_number:
        return False

    combined_text = join_block_texts([*pending_blocks, block])
    return len(combined_text) <= settings.max_chunk_chars


def flush_pending_text(compacted: list[StructuredBlock], pending_blocks: list[StructuredBlock]) -> None:
    """Append pending text blocks as one compact retrieval block."""
    if not pending_blocks:
        return
    compacted.append(merge_text_blocks(pending_blocks))


def merge_text_blocks(blocks: list[StructuredBlock]) -> StructuredBlock:
    """Merge adjacent paragraph/list blocks while preserving local metadata."""
    first_block = blocks[0]
    if len(blocks) == 1:
        return first_block

    return replace(
        first_block,
        id=first_block.id,
        block_type="paragraph",
        text=join_block_texts(blocks),
        bbox=merge_bboxes([block.bbox for block in blocks if block.bbox is not None]),
        reading_order=first_block.reading_order,
        label=None,
        context_text=None,
    )


def prepare_boundary_block(block: StructuredBlock, settings: IndexCompactionSettings) -> StructuredBlock:
    """Prepare a non-merged block for retrieval storage."""
    if settings.store_visual_metadata or block.block_type not in VISUAL_BLOCK_TYPES:
        return block
    return replace(block, bbox=None)


def join_block_texts(blocks: list[StructuredBlock]) -> str:
    """Join block texts with stable paragraph boundaries."""
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())


def normalize_text_for_length(text: str) -> str:
    """Normalize text for lightweight usefulness checks."""
    return " ".join(text.split())


def merge_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Merge bounding boxes into one page-level text chunk bbox."""
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )
