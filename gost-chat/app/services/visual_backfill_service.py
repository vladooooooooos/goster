from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from app.services.context_builder import BuiltContext, ContextEvidence
from app.services.retrieval_types import RerankedBlock
from app.services.visual_evidence import VisualEvidenceRef, visual_ref_from_block
from app.services.visual_reference_extractor import VisualReferenceExtractor, VisualReferenceMention

logger = logging.getLogger(__name__)


class VisualBlockLookup(Protocol):
    def find_visual_blocks(self, document_id: str, limit: int) -> list[RerankedBlock]: ...


@dataclass(frozen=True)
class VisualBackfillResult:
    refs: list[VisualEvidenceRef]
    reference_mentions: list[str]
    attempted: bool
    backfilled_block_ids: list[str]
    missing_references: list[str]
    fallback_reason: str | None

    @classmethod
    def empty(cls, fallback_reason: str | None = None) -> VisualBackfillResult:
        return cls(
            refs=[],
            reference_mentions=[],
            attempted=False,
            backfilled_block_ids=[],
            missing_references=[],
            fallback_reason=fallback_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_mentions": self.reference_mentions,
            "backfill_attempted": self.attempted,
            "backfilled_count": len(self.refs),
            "backfilled_block_ids": self.backfilled_block_ids,
            "missing_references": self.missing_references,
            "backfill_fallback_reason": self.fallback_reason,
        }


class VisualBackfillService:
    def __init__(
        self,
        lookup: VisualBlockLookup | None,
        candidate_limit: int = 8,
        page_window: int = 2,
        extractor: VisualReferenceExtractor | None = None,
    ) -> None:
        self._lookup = lookup
        self._candidate_limit = max(1, candidate_limit)
        self._page_window = max(0, page_window)
        self._extractor = extractor or VisualReferenceExtractor()

    def backfill(self, built_context: BuiltContext) -> VisualBackfillResult:
        return self._backfill_from_evidence(built_context, answer_text=None)

    def backfill_from_answer(self, built_context: BuiltContext, answer_text: str) -> VisualBackfillResult:
        return self._backfill_from_evidence(built_context, answer_text=answer_text)

    def _backfill_from_evidence(self, built_context: BuiltContext, answer_text: str | None) -> VisualBackfillResult:
        if self._lookup is None:
            return VisualBackfillResult.empty("visual_lookup_unavailable")

        anchors = _collect_anchors(built_context, self._extractor, answer_text)
        mentions = [mention for _, mention in anchors]
        mention_labels = _unique([mention.normalized_label for mention in mentions])
        if not mentions:
            return VisualBackfillResult.empty("no_graphic_references")

        logger.info(
            "Visual references extracted: %s from block ids=%s.",
            mention_labels,
            _unique([evidence.block.block_id for evidence, _ in anchors]),
        )

        refs: list[VisualEvidenceRef] = []
        missing: list[str] = []
        seen_ref_ids: set[str] = set()
        candidates_by_doc: dict[str, list[RerankedBlock]] = {}

        for evidence, mention in anchors:
            document_id = evidence.block.document_id
            if not document_id:
                missing.append(mention.normalized_label)
                continue
            if document_id not in candidates_by_doc:
                candidates_by_doc[document_id] = self._lookup.find_visual_blocks(
                    document_id,
                    limit=self._candidate_limit * 4,
                )
            matched = self._best_match(evidence, mention, candidates_by_doc[document_id])
            if matched is None:
                missing.append(mention.normalized_label)
                continue
            ref = visual_ref_from_block(matched, matched.evidence_text[:320])
            if ref is None:
                missing.append(mention.normalized_label)
                continue
            if ref.block_id in seen_ref_ids:
                continue
            seen_ref_ids.add(ref.block_id)
            refs.append(
                replace(
                    ref,
                    selection_reason="Matched graphic reference in selected evidence.",
                    source_block_id=evidence.block.block_id,
                    crop_kind="backfilled_bbox",
                )
            )
            if len(refs) >= self._candidate_limit:
                break

        fallback_reason = None if refs else "no_matching_visual_blocks"
        logger.info(
            "Visual backfill found %s candidate(s): %s.",
            len(refs),
            [ref.block_id for ref in refs],
        )
        return VisualBackfillResult(
            refs=refs,
            reference_mentions=mention_labels,
            attempted=True,
            backfilled_block_ids=[ref.block_id for ref in refs],
            missing_references=_unique(missing),
            fallback_reason=fallback_reason,
        )

    def _best_match(
        self,
        evidence: ContextEvidence,
        mention: VisualReferenceMention,
        candidates: list[RerankedBlock],
    ) -> RerankedBlock | None:
        scored: list[tuple[int, int, RerankedBlock]] = []
        source_page = evidence.block.page_start or evidence.block.page
        for candidate in candidates:
            ref = visual_ref_from_block(candidate, candidate.evidence_text[:320])
            if ref is None:
                continue
            match_score = _reference_match_score(mention, candidate)
            page_distance = _page_distance(source_page, candidate.page_start or candidate.page)
            if match_score <= 0 and (source_page is None or page_distance > self._page_window):
                continue
            scored.append((match_score, -page_distance, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]


def _collect_anchors(
    built_context: BuiltContext,
    extractor: VisualReferenceExtractor,
    answer_text: str | None,
) -> list[tuple[ContextEvidence, VisualReferenceMention]]:
    anchors: list[tuple[ContextEvidence, VisualReferenceMention]] = []
    for evidence in built_context.selected:
        text = answer_text if answer_text is not None else evidence.prompt_text
        for mention in extractor.extract(text):
            anchors.append((evidence, mention))
    return anchors


def _reference_match_score(mention: VisualReferenceMention, candidate: RerankedBlock) -> int:
    haystack = " ".join(
        value
        for value in [
            candidate.label or "",
            candidate.text,
            candidate.retrieval_text,
        ]
        if value
    )
    normalized = _normalize_text(haystack)
    normalized_label = _normalize_text(mention.normalized_label)
    normalized_value = _normalize_text(mention.value)
    if normalized_label in normalized:
        return 100
    if normalized_value in normalized and _candidate_kind_matches(mention, candidate):
        return 80
    if normalized_value in normalized:
        return 60
    return 0


def _candidate_kind_matches(mention: VisualReferenceMention, candidate: RerankedBlock) -> bool:
    block_type = (candidate.block_type or "").casefold()
    if mention.kind in {"figure", "drawing", "appendix"}:
        return block_type == "figure"
    if mention.kind == "table":
        return block_type == "table"
    if mention.kind == "formula":
        return block_type == "formula_with_context"
    return False


def _page_distance(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 999999
    return abs(left - right)


def _normalize_text(value: str) -> str:
    normalized = value.casefold().replace("\u2014", " ").replace("-", " ")
    normalized = normalized.translate(str.maketrans({
        "\u0430": "a",
        "\u0432": "b",
        "\u0435": "e",
        "\u043a": "k",
        "\u043c": "m",
        "\u043d": "h",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0442": "t",
        "\u0445": "x",
    }))
    return " ".join(normalized.split())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
