from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from app.services.query_planner import QueryPlan
from app.services.retrieval_types import RerankedBlock
from app.services.visual_evidence import ContextVisualHints, VisualEvidenceRef, visual_ref_from_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBuilderSettings:
    min_blocks: int = 2
    soft_target_blocks: int = 5
    max_blocks: int = 12
    max_context_chars: int = 18000
    max_chars_per_block: int = 4000
    evidence_preview_chars: int = 320
    enable_near_duplicate_filter: bool = False
    adaptive_score_threshold: float = 0.12


@dataclass(frozen=True)
class ContextEvidence:
    index: int
    block: RerankedBlock
    prompt_text: str
    evidence_preview: str
    source_label: str
    page_label: str


@dataclass(frozen=True)
class ContextBuildStats:
    input_count: int
    selected_count: int
    dropped_duplicate_count: int
    dropped_budget_count: int
    truncated_count: int
    total_chars_included: int
    max_context_chars: int
    min_blocks: int
    soft_target_blocks: int
    max_blocks: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuiltContext:
    query: str
    selected: list[ContextEvidence]
    formatted_context: str
    stats: ContextBuildStats
    visual_hints: ContextVisualHints
    coverage: dict[str, list[str]]
    query_plan: QueryPlan | None = None


class ContextBuilder:
    def __init__(self, settings: ContextBuilderSettings | None = None) -> None:
        self.settings = settings or ContextBuilderSettings()

    def build(
        self,
        query: str,
        ranked_blocks: list[RerankedBlock],
        query_plan: QueryPlan | None = None,
    ) -> BuiltContext:
        selected: list[ContextEvidence] = []
        seen_block_ids: set[str] = set()
        seen_texts: set[str] = set()
        dropped_duplicate_count = 0
        dropped_budget_count = 0
        truncated_count = 0
        total_chars_included = 0
        best_score: float | None = None
        stop_reason = "exhausted"

        effective_soft_target_blocks = _effective_soft_target_blocks(self.settings, query_plan)

        for block in ranked_blocks:
            if len(selected) >= max(1, self.settings.max_blocks):
                dropped_budget_count += 1
                stop_reason = "max_blocks"
                continue

            score = block.rerank_score if block.rerank_score is not None else block.retrieval_score
            if best_score is None:
                best_score = score
            if (
                selected
                and len(selected) >= self.settings.min_blocks
                and len(selected) >= effective_soft_target_blocks
                and best_score > 0
                and score < best_score * (1 - self.settings.adaptive_score_threshold)
            ):
                stop_reason = "score_drop"
                break

            block_id = block.block_id.strip()
            if block_id and block_id in seen_block_ids:
                dropped_duplicate_count += 1
                continue

            normalized_text = normalize_text(block.evidence_text)
            if not normalized_text:
                dropped_duplicate_count += 1
                continue

            text_key = normalized_text.casefold()
            if text_key in seen_texts:
                dropped_duplicate_count += 1
                continue

            prompt_text = truncate_text(normalized_text, self.settings.max_chars_per_block)
            if len(prompt_text) < len(normalized_text):
                truncated_count += 1

            context_piece = self._format_context_piece(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
            )
            piece_chars = len(context_piece)
            separator_chars = 7 if selected else 0
            projected_chars = total_chars_included + separator_chars + piece_chars
            if selected and projected_chars > self.settings.max_context_chars:
                dropped_budget_count += 1
                stop_reason = "budget"
                continue

            evidence = ContextEvidence(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
                evidence_preview=truncate_text(normalized_text, self.settings.evidence_preview_chars),
                source_label=block.source_file,
                page_label=format_page_range(block.page_start, block.page_end),
            )
            selected.append(evidence)
            if block_id:
                seen_block_ids.add(block_id)
            seen_texts.add(text_key)
            total_chars_included = projected_chars

        visual_added = self._add_best_visual_evidence_if_needed(
            selected=selected,
            ranked_blocks=ranked_blocks,
            seen_block_ids=seen_block_ids,
            seen_texts=seen_texts,
            total_chars_included=total_chars_included,
        )
        if visual_added is not None:
            selected.append(visual_added)

        selected = self._add_visual_coverage_if_needed(
            selected=selected,
            ranked_blocks=ranked_blocks,
            seen_block_ids=seen_block_ids,
            seen_texts=seen_texts,
            total_chars_included=len("\n\n---\n\n".join(
                self._format_context_piece(
                    index=evidence.index,
                    block=evidence.block,
                    prompt_text=evidence.prompt_text,
                )
                for evidence in selected
            )),
            query_plan=query_plan,
        )
        selected = _reindex_evidence(selected)

        formatted_context = "\n\n---\n\n".join(
            self._format_context_piece(
                index=evidence.index,
                block=evidence.block,
                prompt_text=evidence.prompt_text,
            )
            for evidence in selected
        )
        visual_hints = self._build_visual_hints(selected, ranked_blocks)
        stats = ContextBuildStats(
            input_count=len(ranked_blocks),
            selected_count=len(selected),
            dropped_duplicate_count=dropped_duplicate_count,
            dropped_budget_count=dropped_budget_count,
            truncated_count=truncated_count,
            total_chars_included=len(formatted_context),
            max_context_chars=self.settings.max_context_chars,
            min_blocks=self.settings.min_blocks,
            soft_target_blocks=self.settings.soft_target_blocks,
            max_blocks=self.settings.max_blocks,
            stop_reason=stop_reason,
        )
        built_context = BuiltContext(
            query=query,
            selected=selected,
            formatted_context=formatted_context,
            stats=stats,
            visual_hints=visual_hints,
            coverage=_build_coverage(selected, query_plan),
            query_plan=query_plan,
        )
        logger.info(
            "ContextBuilder input block types=%s; selected evidence block types=%s; "
            "selected visual refs=%s; stop reason=%s.",
            _block_types(ranked_blocks),
            _block_types([evidence.block for evidence in selected]),
            len(visual_hints.selected),
            stop_reason,
        )
        return built_context

    def _format_context_piece(self, index: int, block: RerankedBlock, prompt_text: str) -> str:
        rerank_score = "none" if block.rerank_score is None else f"{block.rerank_score:.6f}"
        return "\n".join(
            [
                (
                    f"[{index}] {block.source_file} | "
                    f"document_id={block.document_id or 'unknown'} | "
                    f"block_id={block.block_id} | "
                    f"{format_page_range(block.page_start, block.page_end)} | "
                    f"retrieval_score={block.retrieval_score:.6f} | "
                    f"rerank_score={rerank_score}"
                ),
                prompt_text,
            ]
        )

    def _add_best_visual_evidence_if_needed(
        self,
        selected: list[ContextEvidence],
        ranked_blocks: list[RerankedBlock],
        seen_block_ids: set[str],
        seen_texts: set[str],
        total_chars_included: int,
    ) -> ContextEvidence | None:
        if len(selected) >= max(1, self.settings.max_blocks):
            return None
        if any(visual_ref_from_block(evidence.block, evidence.evidence_preview) for evidence in selected):
            return None

        for block in ranked_blocks:
            if block.block_id in seen_block_ids:
                continue
            normalized_text = normalize_text(block.evidence_text)
            if not normalized_text:
                continue
            ref = visual_ref_from_block(block, truncate_text(normalized_text, self.settings.evidence_preview_chars))
            if ref is None:
                continue

            prompt_text = truncate_text(normalized_text, self.settings.max_chars_per_block)
            context_piece = self._format_context_piece(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
            )
            separator_chars = 7 if selected else 0
            projected_chars = total_chars_included + separator_chars + len(context_piece)
            if selected and projected_chars > self.settings.max_context_chars:
                return None

            text_key = normalized_text.casefold()
            if text_key in seen_texts:
                continue
            return ContextEvidence(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
                evidence_preview=truncate_text(normalized_text, self.settings.evidence_preview_chars),
                source_label=block.source_file,
                page_label=format_page_range(block.page_start, block.page_end),
            )
        return None

    def _build_visual_hints(
        self,
        selected: list[ContextEvidence],
        ranked_blocks: list[RerankedBlock],
    ) -> ContextVisualHints:
        selected_refs = [
            ref
            for evidence in selected
            if (ref := visual_ref_from_block(evidence.block, evidence.evidence_preview))
        ]
        selected_ids = {ref.block_id for ref in selected_refs}
        candidate_refs: list[VisualEvidenceRef] = []
        for block in ranked_blocks:
            if block.block_id in selected_ids:
                continue
            preview = truncate_text(block.evidence_text, self.settings.evidence_preview_chars)
            ref = visual_ref_from_block(block, preview)
            if ref:
                candidate_refs.append(ref)
        return ContextVisualHints(selected=selected_refs, candidates=candidate_refs)

    def _add_visual_coverage_if_needed(
        self,
        selected: list[ContextEvidence],
        ranked_blocks: list[RerankedBlock],
        seen_block_ids: set[str],
        seen_texts: set[str],
        total_chars_included: int,
        query_plan: QueryPlan | None,
    ) -> list[ContextEvidence]:
        if query_plan is None or not query_plan.needs_visual:
            return selected

        next_selected = list(selected)
        min_visual_count = min(
            max(2, sum(1 for task in query_plan.tasks if task.needs_visual)),
            max(1, self.settings.max_blocks),
        )
        current_visual_count = sum(
            1 for evidence in next_selected if visual_ref_from_block(evidence.block, evidence.evidence_preview)
        )
        if current_visual_count >= min_visual_count:
            return next_selected

        for block in ranked_blocks:
            if len(next_selected) >= max(1, self.settings.max_blocks):
                break
            if block.block_id in seen_block_ids:
                continue
            normalized_text = normalize_text(block.evidence_text)
            if not normalized_text:
                continue
            text_key = normalized_text.casefold()
            if text_key in seen_texts:
                continue
            ref = visual_ref_from_block(block, truncate_text(normalized_text, self.settings.evidence_preview_chars))
            if ref is None:
                continue

            evidence = self._make_evidence(
                block=block,
                normalized_text=normalized_text,
                index=len(next_selected) + 1,
                total_chars_included=total_chars_included,
            )
            if evidence is None:
                continue
            next_selected.append(evidence)
            seen_block_ids.add(block.block_id)
            seen_texts.add(text_key)
            current_visual_count += 1
            if current_visual_count >= min_visual_count:
                break
        return next_selected

    def _make_evidence(
        self,
        block: RerankedBlock,
        normalized_text: str,
        index: int,
        total_chars_included: int,
    ) -> ContextEvidence | None:
        prompt_text = truncate_text(normalized_text, self.settings.max_chars_per_block)
        context_piece = self._format_context_piece(index=index, block=block, prompt_text=prompt_text)
        separator_chars = 7 if total_chars_included else 0
        if total_chars_included and total_chars_included + separator_chars + len(context_piece) > self.settings.max_context_chars:
            return None
        return ContextEvidence(
            index=index,
            block=block,
            prompt_text=prompt_text,
            evidence_preview=truncate_text(normalized_text, self.settings.evidence_preview_chars),
            source_label=block.source_file,
            page_label=format_page_range(block.page_start, block.page_end),
        )


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def truncate_text(text: str, max_chars: int) -> str:
    normalized = normalize_text(text)
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return "." * max_chars
    return f"{normalized[: max_chars - 3].rstrip()}..."


def format_page_range(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "page unknown"
    if page_end is not None and page_end != page_start:
        return f"pages {page_start}-{page_end}"
    return f"page {page_start}"


def _block_types(blocks: list[RerankedBlock]) -> list[str]:
    return [block.block_type or "unknown" for block in blocks]


def _effective_soft_target_blocks(settings: ContextBuilderSettings, query_plan: QueryPlan | None) -> int:
    if query_plan is None or query_plan.complexity == "simple" and not query_plan.needs_visual:
        return settings.soft_target_blocks
    return settings.max_blocks


def _reindex_evidence(selected: list[ContextEvidence]) -> list[ContextEvidence]:
    return [
        ContextEvidence(
            index=index,
            block=evidence.block,
            prompt_text=evidence.prompt_text,
            evidence_preview=evidence.evidence_preview,
            source_label=evidence.source_label,
            page_label=evidence.page_label,
        )
        for index, evidence in enumerate(selected, start=1)
    ]


def _build_coverage(selected: list[ContextEvidence], query_plan: QueryPlan | None) -> dict[str, list[str]]:
    if query_plan is None:
        return {}
    selected_ids = [evidence.block.block_id for evidence in selected]
    return {task.id: selected_ids for task in query_plan.tasks}
