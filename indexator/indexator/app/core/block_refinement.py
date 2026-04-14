"""Conservative raw text block refinement before structured classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from app.parsing.pdf_parser import ParsedTextBlock
from app.utils.text_cleanup import clean_text


_APPENDIX_HEADING_RE = re.compile(r"^(?:Приложение|Appendix)\s+(?:[\u0410-\u042fA-Z]|\d+)(?:\s|$)", re.IGNORECASE)
_APPENDIX_NUMBERED_RE = re.compile(r"^[\u0410-\u042fA-Z]\.\s*\d+(?:\.\d+)*(?:\s+|\n+)\S", re.IGNORECASE)
_NUMBERED_SEGMENT_RE = re.compile(r"^\d+(?:\.\d+){1,5}(?:\s+|(?=[\u0410-\u042fA-Z]))\S")
_FORMULA_LABEL_RE = re.compile(r"\((?:[\u0410-\u042fA-Z]\s*\.\s*)?\d+(?:\s*\.\s*\d+)?\)", re.IGNORECASE)
_WHERE_RE = re.compile(r"^(?:где|where)\b", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(
    r"^(?:Рисунок|Рис\.?|Чертеж|Черт\.?|Figure|Fig\.?)\s+\S",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^(?:Таблица|Table)\s+\S", re.IGNORECASE)
_CAPTION_RE = re.compile(
    r"^(?:Рисунок|Рис\.?|Чертеж|Черт\.?|Figure|Fig\.?|Таблица|Table)\s+\S",
    re.IGNORECASE,
)
_MATH_RELATION_RE = re.compile(r"[=<>≤≥≈≠]|(?:\bexp\b)", re.IGNORECASE)


@dataclass(frozen=True)
class BlockRefinementResult:
    """Result of one-page raw text block refinement."""

    blocks: list[ParsedTextBlock]
    split_reasons: dict[str, int]


def refine_text_blocks(text_blocks: list[ParsedTextBlock]) -> BlockRefinementResult:
    """Split only obvious mixed raw text blocks before final classification."""
    refined_blocks: list[ParsedTextBlock] = []
    split_reasons: Counter[str] = Counter()

    for block in text_blocks:
        split = split_text_block(block)
        refined_blocks.extend(split.blocks)
        split_reasons.update(split.split_reasons)

    return BlockRefinementResult(blocks=refined_blocks, split_reasons=dict(split_reasons))


def split_text_block(block: ParsedTextBlock) -> BlockRefinementResult:
    """Split a single PyMuPDF text block in clear high-value cases."""
    text = clean_text(block.text)
    if not text:
        return BlockRefinementResult(blocks=[block], split_reasons={})

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return BlockRefinementResult(blocks=[block], split_reasons={})

    split_lines = split_formula_explanation(lines)
    if split_lines:
        return make_split_result(block, split_lines, "formula_explanation")

    split_lines = split_caption_from_body(lines)
    if split_lines:
        return make_split_result(block, split_lines, "caption_body")

    split_lines = split_appendix_heading_from_body(lines)
    if split_lines:
        return make_split_result(block, split_lines, "appendix_heading_body")

    split_lines = split_large_numbered_segments(lines)
    if split_lines:
        return make_split_result(block, split_lines, "large_numbered_segments")

    return BlockRefinementResult(blocks=[block], split_reasons={})


def split_formula_explanation(lines: list[str]) -> list[list[str]] | None:
    """Split formula text from a following explicit variable explanation."""
    for index in range(1, min(len(lines), 8)):
        if not _WHERE_RE.match(lines[index]):
            continue

        formula_text = "\n".join(lines[:index])
        explanation_text = "\n".join(lines[index:])
        if not looks_formula_like(formula_text):
            continue
        if len(" ".join(explanation_text.split())) < 12:
            continue
        return [lines[:index], lines[index:]]

    return None


def split_caption_from_body(lines: list[str]) -> list[list[str]] | None:
    """Split a caption from a clearly unrelated following body segment."""
    if not _CAPTION_RE.match(lines[0]):
        return None

    for index in range(1, min(len(lines), 6)):
        if starts_new_logical_segment(lines[index]):
            return [lines[:index], lines[index:]]

    if not (_FIGURE_CAPTION_RE.match(lines[0]) or _TABLE_CAPTION_RE.match(lines[0])):
        return None

    first_tail_line = lines[1]
    if _TABLE_CAPTION_RE.match(lines[0]) and looks_like_table_payload(lines[1:]):
        return None
    if looks_like_caption_continuation(first_tail_line):
        return None
    if looks_like_standalone_body_sentence(first_tail_line):
        return [lines[:1], lines[1:]]

    return None


def split_appendix_heading_from_body(lines: list[str]) -> list[list[str]] | None:
    """Split appendix title lines fused with following body text."""
    if not _APPENDIX_HEADING_RE.match(lines[0]):
        return None
    if len(lines) < 3:
        return None
    if len(" ".join(lines[1:]).split()) < 12:
        return None
    return [lines[:1], lines[1:]]


def split_large_numbered_segments(lines: list[str]) -> list[list[str]] | None:
    """Split large paragraph-like blocks at repeated safe numbered cues."""
    text = "\n".join(lines)
    if len(text) < 900 and len(lines) < 8:
        return None

    split_indices = [
        index
        for index, line in enumerate(lines)
        if _NUMBERED_SEGMENT_RE.match(line) or _APPENDIX_NUMBERED_RE.match(line)
    ]
    if len(split_indices) < 3:
        return None
    if split_indices[0] not in (0, 1):
        return None

    chunks: list[list[str]] = []
    for position, start_index in enumerate(split_indices):
        end_index = split_indices[position + 1] if position + 1 < len(split_indices) else len(lines)
        chunk = lines[start_index:end_index]
        if chunk:
            chunks.append(chunk)

    if len(chunks) < 2:
        return None
    if any(len(" ".join(chunk).split()) < 5 for chunk in chunks):
        return None

    return chunks


def starts_new_logical_segment(line: str) -> bool:
    """Return whether a line is a safe cue for a new logical segment."""
    return bool(
        _NUMBERED_SEGMENT_RE.match(line)
        or _APPENDIX_NUMBERED_RE.match(line)
        or _APPENDIX_HEADING_RE.match(line)
        or _CAPTION_RE.match(line)
    )


def looks_formula_like(text: str) -> bool:
    """Return whether a short text fragment has clear formula markers."""
    normalized = " ".join(text.split())
    if len(normalized) > 360:
        return False
    return bool(_MATH_RELATION_RE.search(normalized) or _FORMULA_LABEL_RE.search(normalized))


def looks_like_caption_continuation(line: str) -> bool:
    """Return whether a line could simply be a wrapped caption title."""
    normalized = " ".join(line.split())
    if not normalized:
        return False
    if starts_new_logical_segment(normalized):
        return False
    if len(normalized) <= 90 and not normalized.endswith("."):
        return True
    return normalized[:1].islower()


def looks_like_standalone_body_sentence(line: str) -> bool:
    """Return whether a line looks like body prose rather than caption/table payload."""
    normalized = " ".join(line.split())
    if len(normalized) < 45:
        return False
    if looks_like_table_payload([normalized]):
        return False
    return normalized.endswith((".", ";", ":")) or normalized[:1].isupper()


def looks_like_table_payload(lines: list[str]) -> bool:
    """Return whether following caption lines look like table headers or rows."""
    if not lines:
        return False
    compact_lines = [" ".join(line.split()) for line in lines[:5] if line.strip()]
    if not compact_lines:
        return False

    numeric_lines = sum(1 for line in compact_lines if sum(char.isdigit() for char in line) >= 2)
    short_lines = sum(1 for line in compact_lines if len(line) <= 80)
    unit_markers = ("мм", "кН", "В", "Гц", "mm", "kg", "kN")
    has_unit_marker = any(marker in line for line in compact_lines for marker in unit_markers)
    return numeric_lines >= 2 or (short_lines >= 3 and has_unit_marker)


def make_split_result(
    block: ParsedTextBlock,
    split_lines: list[list[str]],
    reason: str,
) -> BlockRefinementResult:
    """Build inherited-bbox split blocks from text line groups."""
    refined_blocks = [
        ParsedTextBlock(
            page_number=block.page_number,
            text="\n".join(lines).strip(),
            bbox=block.bbox,
            order_index=block.order_index * 100 + index,
        )
        for index, lines in enumerate(split_lines)
        if "\n".join(lines).strip()
    ]

    if len(refined_blocks) <= 1:
        return BlockRefinementResult(blocks=[block], split_reasons={})

    return BlockRefinementResult(blocks=refined_blocks, split_reasons={reason: len(refined_blocks) - 1})
