"""Heuristic MVP formula detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.parsing.pdf_parser import ParsedPage, ParsedTextBlock


_FORMULA_LABEL_RE = re.compile(
    r"\((?P<label>(?:[A-ZА-Я]\s*\.\s*)?[0-9OОЗ]+(?:\s*\.\s*[0-9OОЗ]+)?)\)",
    re.IGNORECASE,
)
_LABEL_ONLY_RE = re.compile(
    r"^\((?:[A-ZА-Я]\s*\.\s*)?[0-9OОЗ]+(?:\s*\.\s*[0-9OОЗ]+)?\)\*?$",
    re.IGNORECASE,
)
_LABEL_SEQUENCE_RE = re.compile(
    r"^(?:\((?:[A-ZА-Я]\s*\.\s*)?[0-9OОЗ]+(?:\s*\.\s*[0-9OОЗ]+)?\)\*?\s*)+$",
    re.IGNORECASE,
)
_VARIABLE_RELATION_RE = re.compile(
    r"(?:^|[\s\[\(])(?:[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_\+\-/\^\.,]{0,12}|[A-ZА-Я]\s*[a-zа-я0-9]{0,4})\s*(?:=|<|>|≤|≥)"
)
_RELATION_SYMBOL_RE = re.compile(r"[=≈≠]|(?:\bexp\b|\bехр\b)", re.IGNORECASE)
_RUNNING_HEADER_RE = re.compile(r"^ГОСТ(?:\s+Р)?\s+.*\d{2,4}(?:\s+С\.\s*\d+)?$", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^(Таблица|Table)\s+", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^(Рисунок|Рис\.?|Чертеж|Черт\.?|Figure|Fig\.?)\s+", re.IGNORECASE)
_FORMULA_CONTEXT_RE = re.compile(
    r"\b(?:где|where|формул[аеуы]|вычисля|расчет|определя|принима|коэффициент|условие)\b",
    re.IGNORECASE,
)
_NUMBERED_BODY_RE = re.compile(
    r"^(?:\d+(?:\.\d+){1,5}|[\u0410-\u042fA-Z]\.\s*\d+(?:\.\d+)*)(?:\s+|(?=[\u0410-\u042fA-Z]))\S",
    re.IGNORECASE,
)
_VARIABLE_EXPLANATION_RE = re.compile(
    r"^\s*(?:где|where)\b|^\s*[A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_\+\-/\^\.,]{0,12}\s*[—-]\s+\S",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FormulaCandidate:
    """Conservative formula block candidate with nearby explanatory context."""

    anchor_index: int
    text_block_indices: list[int]
    label: str | None
    formula_text: str
    bbox: tuple[float, float, float, float] | None
    context_text: str | None


def find_page_formula_candidates(
    page: ParsedPage,
    cleaned_texts: list[str],
    excluded_indices: set[int] | None = None,
) -> dict[int, FormulaCandidate]:
    """Find conservative formula candidates on a page."""
    excluded = excluded_indices or set()
    candidates: dict[int, FormulaCandidate] = {}
    consumed: set[int] = set()

    for index, text in enumerate(cleaned_texts):
        if index in excluded or index in consumed or not is_formula_anchor(text):
            continue

        span = collect_formula_span(page.text_blocks, cleaned_texts, index, excluded)
        if not span:
            continue

        candidate = build_formula_candidate(page.text_blocks, cleaned_texts, index, span)
        candidates[index] = candidate
        consumed.update(span)

    return candidates


def build_formula_candidate(
    text_blocks: list[ParsedTextBlock],
    cleaned_texts: list[str],
    anchor_index: int,
    span: list[int],
) -> FormulaCandidate:
    """Build a formula candidate from collected formula-like text blocks."""
    formula_texts = [cleaned_texts[index] for index in span if not is_formula_label_only(cleaned_texts[index])]
    label = extract_formula_label("\n".join(cleaned_texts[index] for index in span))
    bboxes = [text_blocks[index].bbox for index in span]
    return FormulaCandidate(
        anchor_index=anchor_index,
        text_block_indices=span,
        label=label,
        formula_text="\n".join(formula_texts),
        bbox=merge_bboxes(bboxes),
        context_text=find_formula_context(cleaned_texts, anchor_index),
    )


def collect_formula_span(
    text_blocks: list[ParsedTextBlock],
    cleaned_texts: list[str],
    anchor_index: int,
    excluded_indices: set[int],
) -> list[int]:
    """Collect close formula fragments and nearby standalone formula labels."""
    span = {anchor_index}
    anchor_block = text_blocks[anchor_index]

    for index, text in enumerate(cleaned_texts):
        if index == anchor_index or index in excluded_indices or not text:
            continue
        if is_formula_label_only(text) and is_same_formula_row(text_blocks[index].bbox, anchor_block.bbox):
            span.add(index)

    for direction in (-1, 1):
        index = anchor_index + direction
        last_bbox = anchor_block.bbox
        while 0 <= index < len(cleaned_texts):
            text = cleaned_texts[index]
            if index in excluded_indices or not text:
                index += direction
                continue
            if not is_formula_continuation(text):
                break
            if abs(text_blocks[index].bbox[1] - last_bbox[3]) > 34 and not is_same_formula_row(
                text_blocks[index].bbox,
                anchor_block.bbox,
            ):
                break
            span.add(index)
            last_bbox = text_blocks[index].bbox
            if len(span) >= 6:
                break
            index += direction

    return sorted(span, key=lambda item: (text_blocks[item].bbox[1], text_blocks[item].bbox[0]))


def is_formula_anchor(text: str) -> bool:
    """Return whether text is strong enough to anchor a formula block."""
    if not text or is_formula_label_only(text) or is_only_formula_labels(text) or is_non_formula_boilerplate(text):
        return False

    normalized = " ".join(text.split())
    if len(normalized) > 900:
        return False
    if looks_like_vertical_ocr_noise(text):
        return False

    label = extract_formula_label(text)
    relation = has_math_relation(text)
    if not label and not relation:
        return False
    if not label and "=" not in normalized and len(normalized) <= 24:
        return False
    if not label and is_numbered_body_text(text) and len(normalized) > 120:
        return False
    if looks_like_plain_dimension_note(text):
        return False

    score = 0
    if label and relation:
        score += 3
    if _VARIABLE_RELATION_RE.search(text):
        score += 3
    if has_math_relation(text) and len(normalized) <= 320:
        score += 2
    if math_symbol_density(text) >= 0.18 and any(char.isdigit() for char in text) and len(normalized) <= 240:
        score += 3
    elif math_symbol_density(text) >= 0.12 and any(char.isdigit() for char in text) and len(normalized) <= 180:
        score += 2
    if len(text.splitlines()) <= 7 and has_math_relation(text):
        score += 1
    if len(normalized) > 500:
        score -= 2
    if looks_like_plain_reference_text(text):
        score -= 3

    return score >= 3


def is_formula_continuation(text: str) -> bool:
    """Return whether a nearby text block can be part of the same formula object."""
    if is_formula_label_only(text):
        return True
    if is_non_formula_boilerplate(text):
        return False
    normalized = " ".join(text.split())
    if len(normalized) > 360:
        return False
    return has_math_relation(text) or math_symbol_density(text) >= 0.18


def extract_formula_label(text: str) -> str | None:
    """Extract a formula label such as (1) or (A.1)."""
    matches = list(_FORMULA_LABEL_RE.finditer(text))
    if not matches:
        return None
    label = matches[-1].group(0)
    return " ".join(label.split())


def is_formula_label_only(text: str) -> bool:
    """Return whether text is only a standalone formula label."""
    return bool(_LABEL_ONLY_RE.fullmatch(" ".join(text.split())))


def is_only_formula_labels(text: str) -> bool:
    """Return whether text contains only one or more standalone labels."""
    return bool(_LABEL_SEQUENCE_RE.fullmatch(" ".join(text.split())))


def has_math_relation(text: str) -> bool:
    """Return whether text contains a mathematical relation or operator."""
    return bool(_RELATION_SYMBOL_RE.search(text) or _VARIABLE_RELATION_RE.search(text))


def math_symbol_density(text: str) -> float:
    """Return a rough density of math-like characters."""
    normalized = "".join(text.split())
    if not normalized:
        return 0.0
    math_chars = sum(1 for char in normalized if char in "=<>≤≥±∑Σ√∞≈≠·•∙■*/^°+-()[]")
    return math_chars / len(normalized)


def is_same_formula_row(
    bbox: tuple[float, float, float, float],
    anchor_bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether two blocks sit on the same visual formula row."""
    vertical_overlap = min(bbox[3], anchor_bbox[3]) - max(bbox[1], anchor_bbox[1])
    min_height = min(bbox[3] - bbox[1], anchor_bbox[3] - anchor_bbox[1])
    return vertical_overlap >= min_height * 0.45


def find_formula_context(cleaned_texts: list[str], anchor_index: int) -> str | None:
    """Return nearby explanatory text for a formula candidate."""
    context_parts: list[str] = []

    for index in range(anchor_index - 1, max(anchor_index - 4, -1), -1):
        text = cleaned_texts[index]
        if is_useful_formula_context(text):
            context_parts.insert(0, text)
            break

    for index in range(anchor_index + 1, min(anchor_index + 5, len(cleaned_texts))):
        text = cleaned_texts[index]
        if not text or is_formula_label_only(text):
            continue
        if is_formula_anchor(text):
            break
        if context_parts and is_numbered_body_text(text):
            break
        if _VARIABLE_EXPLANATION_RE.search(text) or (
            context_parts and len(" ".join(text.split())) <= 260 and not is_non_formula_boilerplate(text)
        ):
            context_parts.append(text)
            continue
        break

    if not context_parts:
        return None
    return "\n".join(context_parts)


def is_useful_formula_context(text: str) -> bool:
    """Return whether text is useful explanatory context near a formula."""
    if not text or is_non_formula_boilerplate(text) or is_formula_label_only(text):
        return False
    normalized = " ".join(text.split())
    if len(normalized) > 700:
        return False
    return bool(_FORMULA_CONTEXT_RE.search(text) or normalized.endswith(":"))


def looks_like_plain_reference_text(text: str) -> bool:
    """Return whether parenthesized numbers are probably references, not formulas."""
    normalized = " ".join(text.split())
    if "=" in normalized or any(symbol in normalized for symbol in ("<", ">", "≤", "≥", "±")):
        return False
    if len(normalized) <= 60:
        return False
    return len(_FORMULA_LABEL_RE.findall(normalized)) >= 1


def is_numbered_body_text(text: str) -> bool:
    """Return whether text starts a new numbered body or appendix point."""
    return bool(_NUMBERED_BODY_RE.match(text))


def looks_like_vertical_ocr_noise(text: str) -> bool:
    """Return whether a compact multiline block looks like vertical OCR noise."""
    normalized = " ".join(text.split())
    return len(normalized) <= 120 and text.count("\n") >= 4 and not extract_formula_label(text)


def looks_like_plain_dimension_note(text: str) -> bool:
    """Return whether a short prose note with one assignment is likely not a formula."""
    normalized = " ".join(text.split())
    if extract_formula_label(text) or _FORMULA_CONTEXT_RE.search(text):
        return False
    if normalized.count("=") != 1 or len(normalized) > 150:
        return False
    long_words = re.findall(r"\b[А-Яа-яA-Za-z]{4,}\b", normalized)
    return len(long_words) >= 3


def is_non_formula_boilerplate(text: str) -> bool:
    """Return whether text is a page artifact or another supported object caption."""
    normalized = " ".join(text.split())
    if not normalized:
        return True
    if _RUNNING_HEADER_RE.match(normalized):
        return True
    if _TABLE_CAPTION_RE.match(normalized) or _FIGURE_CAPTION_RE.match(normalized):
        return True
    return bool(re.fullmatch(r"[IVXLCDM]+|\d{1,3}", normalized, re.IGNORECASE))


def merge_bboxes(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    """Merge multiple bounding boxes into one page-level formula bbox."""
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )
