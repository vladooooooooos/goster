"""Heuristic MVP figure detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.parsing.pdf_parser import ParsedImageBlock, ParsedPage, ParsedTextBlock


_FIGURE_NUMBER_RE = r"([А-ЯA-Z]?(?:\.\s*)?\d+(?:[.\-]\d+)*(?:\s*[а-яa-z])?)"
_FIGURE_CAPTION_RE = re.compile(
    rf"^(?P<prefix>Рисунок|Рис\.?|Чертеж|Черт\.?|Figure|Fig\.?)\s*(?P<number>{_FIGURE_NUMBER_RE})(?=$|\s|[—-])",
    re.IGNORECASE,
)
_RUNNING_HEADER_RE = re.compile(r"^ГОСТ(?:\s+Р)?\s+.*\d{2,4}(?:\s+С\.\s*\d+)?$", re.IGNORECASE)
_NUMBERED_BODY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.\s+\S|\d+(?:\.\d+)+(?:\s|\n)+\S)")
_APPENDIX_BODY_RE = re.compile(r"^[А-ЯA-Z]\.\s*\d+(?:\.\d+)*\s+\S", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^(Таблица|Table)\s+", re.IGNORECASE)
_FIGURE_REFERENCE_WORDS_RE = re.compile(r"\b(?:рисунк\w*|рис\.|чертеж\w*|черт\.)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FigureCandidate:
    """Conservative figure block candidate anchored by a caption."""

    caption_index: int
    text_block_indices: list[int]
    image_bboxes: list[tuple[float, float, float, float]]
    label: str
    caption_text: str
    bbox: tuple[float, float, float, float] | None
    context_text: str | None


def find_figure_candidate(
    page: ParsedPage,
    cleaned_texts: list[str],
    caption_index: int,
) -> FigureCandidate:
    """Build a conservative figure candidate around an explicit figure caption."""
    caption_text = cleaned_texts[caption_index]
    text_block_indices = collect_nearby_figure_text_blocks(page.text_blocks, cleaned_texts, caption_index)
    image_bboxes = collect_nearby_image_bboxes(page, page.text_blocks[caption_index].bbox)
    bboxes = [page.text_blocks[index].bbox for index in text_block_indices] + image_bboxes

    return FigureCandidate(
        caption_index=caption_index,
        text_block_indices=text_block_indices,
        image_bboxes=image_bboxes,
        label=extract_figure_label(caption_text) or caption_text.splitlines()[0],
        caption_text=caption_text,
        bbox=merge_bboxes(bboxes),
        context_text=find_figure_context(cleaned_texts, caption_index),
    )


def is_figure_caption(text: str) -> bool:
    """Return whether text starts with a supported figure caption marker."""
    return bool(_FIGURE_CAPTION_RE.match(text))


def extract_figure_label(text: str) -> str | None:
    """Extract a stable figure label from a caption."""
    match = _FIGURE_CAPTION_RE.match(text)
    if not match:
        return None

    prefix = match.group("prefix")
    number = " ".join(match.group("number").split())
    return f"{prefix} {number}"


def build_figure_text(candidate: FigureCandidate, cleaned_texts: list[str]) -> str:
    """Build a useful text representation with the caption first."""
    nearby_texts = [
        cleaned_texts[index]
        for index in candidate.text_block_indices
        if index != candidate.caption_index and cleaned_texts[index]
    ]
    return "\n".join([candidate.caption_text, *nearby_texts])


def collect_nearby_figure_text_blocks(
    text_blocks: list[ParsedTextBlock],
    cleaned_texts: list[str],
    caption_index: int,
) -> list[int]:
    """Collect caption-adjacent text blocks that look like figure labels or legends."""
    caption_block = text_blocks[caption_index]
    caption_top = caption_block.bbox[1]
    caption_bottom = caption_block.bbox[3]
    min_y = max(0.0, caption_top - 260.0)
    max_y = caption_bottom + 8.0
    previous_caption_bottom = find_previous_caption_bottom(text_blocks, cleaned_texts, caption_index)
    if previous_caption_bottom is not None:
        min_y = max(min_y, previous_caption_bottom)
    indices = [caption_index]

    for index, block in enumerate(text_blocks):
        if index == caption_index:
            continue

        text = cleaned_texts[index]
        if not is_likely_figure_visual_text(text):
            continue
        if block.bbox[1] < min_y or block.bbox[3] > max_y:
            continue
        if not has_horizontal_relation(block.bbox, caption_block.bbox):
            continue

        indices.append(index)

    return sorted(indices, key=lambda index: text_blocks[index].bbox[1])


def collect_nearby_image_bboxes(
    page: ParsedPage,
    caption_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """Collect non-page-sized image blocks near the caption."""
    image_bboxes: list[tuple[float, float, float, float]] = []
    min_y = max(0.0, caption_bbox[1] - 320.0)
    max_y = caption_bbox[3] + 80.0

    for image_block in page.image_blocks:
        if is_page_sized_image_block(image_block, page):
            continue
        if image_block.bbox[3] < min_y or image_block.bbox[1] > max_y:
            continue
        if has_horizontal_relation(image_block.bbox, caption_bbox):
            image_bboxes.append(image_block.bbox)

    return image_bboxes


def find_previous_caption_bottom(
    text_blocks: list[ParsedTextBlock],
    cleaned_texts: list[str],
    caption_index: int,
) -> float | None:
    """Return the nearest previous figure caption bottom coordinate on the page."""
    previous_caption_bboxes = [
        text_blocks[index].bbox
        for index in range(caption_index)
        if is_figure_caption(cleaned_texts[index])
    ]
    if not previous_caption_bboxes:
        return None
    return max(bbox[3] for bbox in previous_caption_bboxes)


def is_likely_figure_visual_text(text: str) -> bool:
    """Return whether text is a short graphic label, dimension, or legend."""
    if not text:
        return False
    if is_figure_caption(text) or is_page_number(text) or _RUNNING_HEADER_RE.match(text):
        return False
    if _TABLE_CAPTION_RE.match(text) or _NUMBERED_BODY_RE.match(text) or _APPENDIX_BODY_RE.match(text):
        return False

    normalized = " ".join(text.split())
    if len(normalized) > 180:
        return False
    if len(normalized) <= 3:
        return True
    if re.match(r"^[А-ЯЁ][а-яё]+", normalized):
        return False
    if re.fullmatch(r"[А-ЯЁA-Z0-9\s\.\-—]+", normalized) and len(normalized) > 12:
        return False

    has_graphic_markers = any(char.isdigit() for char in normalized) or any(
        marker in normalized for marker in ("—", "-", "/", "\\", "(", ")", "<", ">", "°")
    )
    has_many_line_breaks = text.count("\n") >= 1
    return has_graphic_markers or has_many_line_breaks


def has_horizontal_relation(
    bbox: tuple[float, float, float, float],
    caption_bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether a nearby block plausibly belongs to the same figure area."""
    bbox_center = (bbox[0] + bbox[2]) / 2
    caption_center = (caption_bbox[0] + caption_bbox[2]) / 2
    overlap = min(bbox[2], caption_bbox[2]) - max(bbox[0], caption_bbox[0])
    return overlap > 0 or abs(bbox_center - caption_center) <= 170.0


def find_figure_context(cleaned_texts: list[str], caption_index: int) -> str | None:
    """Return nearby previous text if it explicitly references the figure label."""
    label = extract_figure_label(cleaned_texts[caption_index])
    label_number = label.split(" ", 1)[1] if label and " " in label else None

    for index in range(caption_index - 1, max(caption_index - 7, -1), -1):
        text = cleaned_texts[index]
        if not text or is_figure_caption(text) or is_page_number(text):
            continue
        if _FIGURE_REFERENCE_WORDS_RE.search(text) and (not label_number or label_number in text):
            return text
    return None


def is_page_number(text: str) -> bool:
    """Return whether text looks like a standalone page number."""
    return bool(re.fullmatch(r"[IVXLCDM]+|\d{1,3}", text.strip(), re.IGNORECASE))


def is_page_sized_image_block(image_block: ParsedImageBlock, page: ParsedPage) -> bool:
    """Return whether an image block is likely a full-page scanned background."""
    x0, y0, x1, y1 = image_block.bbox
    image_area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    page_area = page.width * page.height
    if page_area <= 0:
        return False
    return image_area / page_area >= 0.75


def merge_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    """Merge multiple bounding boxes into one page-level figure bbox."""
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )
