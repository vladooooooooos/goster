"""Simple first-pass structured block builder."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from app.core.block_refinement import refine_text_blocks
from app.core.blocks import BlockType, StructuredBlock
from app.core.figure_detection import (
    FigureCandidate,
    build_figure_text,
    find_figure_candidate,
    is_figure_caption,
)
from app.core.formula_detection import (
    find_page_formula_candidates,
)
from app.parsing.pdf_parser import ParsedPage, ParsedTextBlock, ParsedDocument
from app.utils.text_cleanup import clean_text


_APPENDIX_RE = re.compile(r"^(?:Приложение|Appendix)\s+(?:[А-ЯA-Z]|\d+)(?:\s|$)", re.IGNORECASE)
_TABLE_NUMBER_RE = r"([А-ЯA-Z]?\.[\dЗ]+|\d+|[IVXLCDM]+)"
_TABLE_CAPTION_RE = re.compile(rf"^(Таблица|Table)\s+{_TABLE_NUMBER_RE}(?:\s|$|[—-])", re.IGNORECASE)
_SPACED_TABLE_CAPTION_RE = re.compile(rf"^Т\s*а\s*б\s*л\s*и\s*ц\s*а\s+{_TABLE_NUMBER_RE}(?:\s|$|[—-])", re.IGNORECASE)
_TABLE_CONTINUATION_RE = re.compile(
    rf"^\S{{0,2}}\s*(?:Продолжение|Продоло\S+|Окончание)\s+табл(?:\.|ицы)?\s+{_TABLE_NUMBER_RE}(?:\s|$)",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^([\-*•–—]|\d+[\).]|\d+\s+[А-ЯЁA-Z]{2,}|[A-Za-zА-Яа-я][\).])\s+")
_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+){0,5})(?:\s+|\n+)(\S[\s\S]*)$")
_APPENDIX_SECTION_NUMBER_RE = re.compile(r"^[А-ЯA-Z]\.\s*\d+(?:\.\d+)*(?:\s+|\n+)\S+", re.IGNORECASE)
_TOC_HEADING_RE = re.compile(r"^(Содержание|Оглавление|Table of contents)$", re.IGNORECASE)
_TOC_ENTRY_RE = re.compile(r"(?m)^\s*\d+(?:\.\d+)*\s+\S.*\.{3,}\s*\d+\s*$")
_RUNNING_HEADER_RE = re.compile(r"^ГОСТ(?:\s+Р)?\s+[\w\d]+[—-]\d{4}$", re.IGNORECASE)


class StructuredBlockBuilder:
    """Build simple logical blocks from raw parsed PDF data."""

    def __init__(self) -> None:
        self.last_refinement_counts: dict[str, int] = {}

    def build(self, parsed_document: ParsedDocument) -> list[StructuredBlock]:
        """Convert parsed PDF pages into first-pass structured text blocks."""
        doc_id = make_doc_id(parsed_document.file_path)
        toc_pages = find_table_of_contents_pages(parsed_document.pages)
        section_path: list[str] = []
        blocks: list[StructuredBlock] = []
        reading_order = 0
        refinement_counts: Counter[str] = Counter()

        for page in parsed_document.pages:
            page_is_table_of_contents = page.page_number in toc_pages
            refinement = refine_text_blocks(page.text_blocks)
            refinement_counts.update(refinement.split_reasons)
            parsed_blocks = refinement.blocks
            refined_page = replace(page, text_blocks=parsed_blocks)
            cleaned_texts = [clean_text(parsed_block.text) for parsed_block in parsed_blocks]
            figure_candidates = find_page_figure_candidates(refined_page, cleaned_texts, page_is_table_of_contents)
            consumed_figure_text_indices = {
                figure_index
                for candidate in figure_candidates.values()
                for figure_index in candidate.text_block_indices
                if figure_index != candidate.caption_index
            }
            formula_candidates = (
                {}
                if page_is_table_of_contents
                else find_page_formula_candidates(
                    refined_page,
                    cleaned_texts,
                    excluded_indices=consumed_figure_text_indices,
                )
            )
            consumed_formula_text_indices = {
                formula_index
                for candidate in formula_candidates.values()
                for formula_index in candidate.text_block_indices
                if formula_index != candidate.anchor_index
            }
            index = 0

            while index < len(parsed_blocks):
                parsed_block = parsed_blocks[index]
                text = cleaned_texts[index]
                if not text:
                    index += 1
                    continue
                if index in consumed_figure_text_indices:
                    index += 1
                    continue
                if index in consumed_formula_text_indices:
                    index += 1
                    continue

                if not page_is_table_of_contents and is_figure_caption(text):
                    figure_candidate = figure_candidates[index]
                    blocks.append(
                        StructuredBlock(
                            id=f"{doc_id}:{reading_order}",
                            doc_id=doc_id,
                            file_name=parsed_document.file_name,
                            file_path=parsed_document.file_path,
                            page_number=parsed_block.page_number,
                            block_type="figure",
                            text=build_figure_text(figure_candidate, cleaned_texts),
                            bbox=figure_candidate.bbox,
                            reading_order=reading_order,
                            section_path=list(section_path),
                            label=figure_candidate.label,
                            context_text=figure_candidate.context_text,
                        )
                    )
                    reading_order += 1
                    index += 1
                    continue

                if not page_is_table_of_contents and index in formula_candidates:
                    formula_candidate = formula_candidates[index]
                    blocks.append(
                        StructuredBlock(
                            id=f"{doc_id}:{reading_order}",
                            doc_id=doc_id,
                            file_name=parsed_document.file_name,
                            file_path=parsed_document.file_path,
                            page_number=parsed_block.page_number,
                            block_type="formula_with_context",
                            text=formula_candidate.formula_text,
                            bbox=formula_candidate.bbox,
                            reading_order=reading_order,
                            section_path=list(section_path),
                            label=formula_candidate.label,
                            context_text=formula_candidate.context_text,
                        )
                    )
                    reading_order += 1
                    index += 1
                    continue

                if not page_is_table_of_contents and is_table_caption(text):
                    table_span = collect_table_span(parsed_blocks, cleaned_texts, index)
                    table_blocks = [parsed_blocks[span_index] for span_index in table_span]
                    table_texts = [cleaned_texts[span_index] for span_index in table_span if cleaned_texts[span_index]]
                    context_text = find_table_context(cleaned_texts, index)
                    blocks.append(
                        StructuredBlock(
                            id=f"{doc_id}:{reading_order}",
                            doc_id=doc_id,
                            file_name=parsed_document.file_name,
                            file_path=parsed_document.file_path,
                            page_number=parsed_block.page_number,
                            block_type="table",
                            text="\n".join(table_texts),
                            bbox=merge_bboxes([table_block.bbox for table_block in table_blocks]),
                            reading_order=reading_order,
                            section_path=list(section_path),
                            label=extract_table_label(text),
                            context_text=context_text,
                        )
                    )
                    reading_order += 1
                    index = table_span[-1] + 1
                    continue

                block_type = detect_block_type(text, is_table_of_contents_page=page_is_table_of_contents)
                if block_type in {"heading", "appendix_section"}:
                    section_path = update_section_path(section_path, text, block_type)

                blocks.append(
                    StructuredBlock(
                        id=f"{doc_id}:{reading_order}",
                        doc_id=doc_id,
                        file_name=parsed_document.file_name,
                        file_path=parsed_document.file_path,
                        page_number=parsed_block.page_number,
                        block_type=block_type,
                        text=text,
                        bbox=parsed_block.bbox,
                        reading_order=reading_order,
                        section_path=list(section_path),
                    )
                )
                reading_order += 1
                index += 1

        self.last_refinement_counts = dict(refinement_counts)
        return blocks


def find_page_figure_candidates(
    page: ParsedPage,
    cleaned_texts: list[str],
    is_table_of_contents_page: bool,
) -> dict[int, FigureCandidate]:
    """Find figure candidates for a page before normal block emission."""
    if is_table_of_contents_page:
        return {}
    return {
        index: find_figure_candidate(page, cleaned_texts, index)
        for index, text in enumerate(cleaned_texts)
        if text and is_figure_caption(text)
    }


def detect_block_type(text: str, is_table_of_contents_page: bool = False) -> BlockType:
    """Detect a simple MVP block type from cleaned text."""
    first_line = text.splitlines()[0].strip()

    if is_table_of_contents_page or is_table_of_contents_heading(first_line) or is_table_of_contents_entry(text):
        return "table_of_contents"
    if is_appendix_heading(first_line):
        return "appendix_section"
    if is_list_item(first_line):
        return "list_item"
    if is_heading_candidate(text):
        return "heading"
    return "paragraph"


def is_appendix_heading(text: str) -> bool:
    """Return whether text starts with a standard appendix marker."""
    return bool(_APPENDIX_RE.match(text))


def is_table_caption(text: str) -> bool:
    """Return whether text starts with a table caption marker."""
    return bool(
        _TABLE_CAPTION_RE.match(text)
        or _SPACED_TABLE_CAPTION_RE.match(text)
        or _TABLE_CONTINUATION_RE.match(text)
    )


def is_table_of_contents_heading(text: str) -> bool:
    """Return whether text is a table of contents heading."""
    return bool(_TOC_HEADING_RE.match(text))


def is_table_of_contents_entry(text: str) -> bool:
    """Return whether text looks like a table of contents entry."""
    return bool(_TOC_ENTRY_RE.search(text))


def is_list_item(text: str) -> bool:
    """Return whether text looks like a simple list item."""
    return bool(_LIST_ITEM_RE.match(text))


def is_heading_candidate(text: str) -> bool:
    """Return whether text looks like a simple heading candidate."""
    if _RUNNING_HEADER_RE.match(text):
        return False
    if len(text) > 180 or len(text.splitlines()) > 2:
        return False

    first_line = text.splitlines()[0].strip()
    section_match = _SECTION_NUMBER_RE.match(text)
    if section_match:
        return is_numbered_heading_candidate(section_match.group(1), section_match.group(2))

    if first_line.startswith("©") or first_line[:1].islower():
        return False

    has_terminal_sentence_punctuation = first_line.endswith((".", ";", ","))
    has_enough_letters = sum(1 for char in first_line if char.isalpha()) >= 3
    return has_enough_letters and not has_terminal_sentence_punctuation and len(first_line) <= 80


def is_numbered_heading_candidate(section_number: str, title: str) -> bool:
    """Return whether a numbered text block is likely to be a section heading."""
    normalized_title = " ".join(title.split())
    if not normalized_title or len(normalized_title) > 140:
        return False
    if normalized_title.endswith((".", ";", ",")):
        return False
    if ":" in normalized_title and section_number.count(".") >= 1:
        return False
    if re.search(r"\s+\d+(?:\.\d+){1,5}\s+\S+", normalized_title):
        return False
    return True


def collect_table_span(
    parsed_blocks: list[ParsedTextBlock],
    cleaned_texts: list[str],
    caption_index: int,
) -> list[int]:
    """Collect a conservative same-page table span after a table caption."""
    span = [caption_index]
    caption_bottom = parsed_blocks[caption_index].bbox[3]
    last_bottom = caption_bottom

    for index in range(caption_index + 1, len(parsed_blocks)):
        text = cleaned_texts[index]
        if not text:
            continue
        if should_stop_table_span(text, index, caption_index, parsed_blocks, last_bottom):
            break

        span.append(index)
        last_bottom = parsed_blocks[index].bbox[3]

        if len(span) >= 18:
            break

    return span


def should_stop_table_span(
    text: str,
    index: int,
    caption_index: int,
    parsed_blocks: list[ParsedTextBlock],
    last_bottom: float,
) -> bool:
    """Return whether a following block should end a simple table span."""
    if is_table_caption(text):
        return True
    if is_figure_caption(text):
        return True
    if _RUNNING_HEADER_RE.match(text) or is_page_number(text):
        return True
    if index > caption_index + 1 and starts_numbered_body_section(text):
        return True

    vertical_gap = parsed_blocks[index].bbox[1] - last_bottom
    if vertical_gap > 70 and not looks_table_like_text(text):
        return True

    if len(text) > 260 and not looks_table_like_text(text):
        return True

    return False


def looks_table_like_text(text: str) -> bool:
    """Return whether text resembles compact table row or header content."""
    normalized = " ".join(text.split())
    digit_count = sum(1 for char in normalized if char.isdigit())
    has_many_spaces = normalized.count(" ") >= 3
    has_range_marker = any(marker in normalized for marker in ("До ", "Св. ", "Менее ", "От "))
    has_table_header_word = any(
        word in normalized.lower()
        for word in ("мм", "кабел", "сечение", "расстояние", "размер", "параметр", "площадь", "диаметр")
    )
    return digit_count >= 2 or has_range_marker or (has_many_spaces and has_table_header_word)


def starts_numbered_body_section(text: str) -> bool:
    """Return whether text starts a likely numbered body section after a table."""
    match = _SECTION_NUMBER_RE.match(text)
    if not match:
        return bool(_APPENDIX_SECTION_NUMBER_RE.match(text))

    section_number = match.group(1)
    normalized_title = " ".join(match.group(2).split())
    if section_number.count(".") >= 2:
        return True
    if "." in section_number and len(normalized_title) > 40:
        return True
    return len(normalized_title) > 120 or normalized_title.endswith((".", ";", ":"))


def extract_table_label(text: str) -> str | None:
    """Extract a stable table label from a caption."""
    match = _TABLE_CAPTION_RE.match(text)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    spaced_match = _SPACED_TABLE_CAPTION_RE.match(text)
    if spaced_match:
        return f"Таблица {spaced_match.group(1)}"

    continuation_match = _TABLE_CONTINUATION_RE.match(text)
    if continuation_match:
        return f"Таблица {continuation_match.group(1)}"

    return None


def find_table_context(cleaned_texts: list[str], caption_index: int) -> str | None:
    """Return nearby previous text if it explicitly references the table."""
    label = extract_table_label(cleaned_texts[caption_index])
    label_number = label.split(" ", 1)[1] if label and " " in label else None

    for index in range(caption_index - 1, max(caption_index - 4, -1), -1):
        text = cleaned_texts[index]
        if not text:
            continue
        lower_text = text.lower()
        if "таблиц" in lower_text and (not label_number or label_number in text):
            return text
    return None


def merge_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    """Merge multiple bounding boxes into one page-level table bbox."""
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def is_page_number(text: str) -> bool:
    """Return whether text looks like a standalone page number."""
    return bool(re.fullmatch(r"[IVXLCDM]+|\d{1,3}", text.strip(), re.IGNORECASE))


def update_section_path(current_path: list[str], heading_text: str, block_type: BlockType) -> list[str]:
    """Update the shallow section path using only safe heading information."""
    if block_type == "appendix_section":
        return [heading_text]

    match = _SECTION_NUMBER_RE.match(heading_text)
    if not match:
        return [heading_text]

    depth = match.group(1).count(".") + 1
    return [*current_path[: max(depth - 1, 0)], heading_text]


def find_table_of_contents_pages(pages: list[ParsedPage]) -> set[int]:
    """Find pages that belong to a simple table of contents section."""
    toc_pages: set[int] = set()
    in_table_of_contents = False
    saw_toc_content = False

    for page in pages:
        cleaned_blocks = [clean_text(block.text) for block in page.text_blocks]
        page_has_toc_heading = any(
            is_table_of_contents_heading(text.splitlines()[0].strip())
            for text in cleaned_blocks
            if text
        )
        page_has_toc_entry = any(is_table_of_contents_entry(text) for text in cleaned_blocks if text)

        if page_has_toc_heading:
            in_table_of_contents = True

        if in_table_of_contents and (page_has_toc_heading or page_has_toc_entry):
            toc_pages.add(page.page_number)
            saw_toc_content = True
            continue

        if in_table_of_contents and saw_toc_content:
            in_table_of_contents = False

    return toc_pages


def make_doc_id(file_path: Path) -> str:
    """Create a stable local document id from a resolved file path."""
    return hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()


def make_document_id(file_path: Path) -> str:
    """Create the stable shared index document id for a source PDF."""
    return make_doc_id(file_path)
