from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.services.retrieval_types import RerankedBlock

VisualDecisionMode = Literal["text_only", "show_visual", "inspect_visual_and_show"]


@dataclass(frozen=True)
class VisualEvidenceRef:
    block_id: str
    document_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    block_type: str | None
    label: str | None
    source_file: str
    text_preview: str
    crop_path: str | None = None
    crop_status: str | None = None
    crop_width: int | None = None
    crop_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class ContextVisualHints:
    selected: list[VisualEvidenceRef]
    candidates: list[VisualEvidenceRef]

    @property
    def total_count(self) -> int:
        return len({ref.block_id for ref in [*self.selected, *self.candidates]})

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [ref.to_dict() for ref in self.selected],
            "candidates": [ref.to_dict() for ref in self.candidates],
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class VisualEvidenceDecision:
    mode: VisualDecisionMode
    target_block_ids: list[str]
    show_in_sources: bool
    show_in_answer: bool
    needs_multimodal_followup: bool
    reason: str

    @classmethod
    def text_only(cls, reason: str) -> VisualEvidenceDecision:
        return cls(
            mode="text_only",
            target_block_ids=[],
            show_in_sources=False,
            show_in_answer=False,
            needs_multimodal_followup=False,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visual_ref_from_block(block: RerankedBlock, text_preview: str) -> VisualEvidenceRef | None:
    if block.payload.get("has_visual_evidence") is not True:
        return None
    bbox = _bbox(block.payload.get("bbox"))
    page_number = _int_value(block.payload.get("page_number")) or block.page_start or block.page
    if not block.block_id or not block.document_id or not page_number or bbox is None:
        return None
    return VisualEvidenceRef(
        block_id=block.block_id,
        document_id=block.document_id,
        page_number=page_number,
        bbox=bbox,
        block_type=block.block_type,
        label=block.label,
        source_file=block.source_file,
        text_preview=text_preview,
        crop_path=_str_or_none(block.payload.get("crop_path")),
        crop_status=_str_or_none(block.payload.get("crop_status")),
        crop_width=_int_value(block.payload.get("crop_width")),
        crop_height=_int_value(block.payload.get("crop_height")),
    )


def parse_visual_decision(raw: str) -> VisualEvidenceDecision:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return VisualEvidenceDecision.text_only("Invalid visual decision JSON.")
    if not isinstance(payload, dict):
        return VisualEvidenceDecision.text_only("Visual decision was not an object.")
    mode = payload.get("mode")
    if mode not in {"text_only", "show_visual", "inspect_visual_and_show"}:
        return VisualEvidenceDecision.text_only("Visual decision mode was invalid.")
    target_block_ids = [
        item.strip()
        for item in payload.get("target_block_ids", [])
        if isinstance(item, str) and item.strip()
    ]
    return VisualEvidenceDecision(
        mode=mode,
        target_block_ids=target_block_ids,
        show_in_sources=bool(payload.get("show_in_sources")),
        show_in_answer=bool(payload.get("show_in_answer")),
        needs_multimodal_followup=bool(payload.get("needs_multimodal_followup")),
        reason=_str_or_none(payload.get("reason")) or "",
    )


def guard_visual_decision(
    decision: VisualEvidenceDecision,
    available_refs: list[VisualEvidenceRef | None],
    max_targets: int,
) -> VisualEvidenceDecision:
    if decision.mode == "text_only":
        return decision
    available = {ref.block_id: ref for ref in available_refs if ref}
    target_block_ids = [block_id for block_id in decision.target_block_ids if block_id in available]
    target_block_ids = target_block_ids[: max(0, max_targets)]
    if not target_block_ids:
        return VisualEvidenceDecision.text_only("No valid visual target blocks were requested.")
    return VisualEvidenceDecision(
        mode=decision.mode,
        target_block_ids=target_block_ids,
        show_in_sources=decision.show_in_sources,
        show_in_answer=decision.show_in_answer,
        needs_multimodal_followup=decision.needs_multimodal_followup,
        reason=decision.reason,
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _str_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
