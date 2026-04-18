from __future__ import annotations

from dataclasses import asdict, dataclass

from app.services.retrieval_types import RerankedBlock


@dataclass(frozen=True)
class ContextBuilderSettings:
    min_blocks: int = 2
    soft_target_blocks: int = 5
    max_blocks: int = 10
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

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BuiltContext:
    query: str
    selected: list[ContextEvidence]
    formatted_context: str
    stats: ContextBuildStats


class ContextBuilder:
    def __init__(self, settings: ContextBuilderSettings | None = None) -> None:
        self.settings = settings or ContextBuilderSettings()

    def build(self, query: str, ranked_blocks: list[RerankedBlock]) -> BuiltContext:
        selected: list[ContextEvidence] = []
        seen_block_ids: set[str] = set()
        seen_texts: set[str] = set()
        dropped_duplicate_count = 0
        dropped_budget_count = 0
        truncated_count = 0
        total_chars_included = 0

        for block in ranked_blocks:
            if len(selected) >= max(1, self.settings.max_blocks):
                dropped_budget_count += 1
                continue

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

        formatted_context = "\n\n---\n\n".join(
            self._format_context_piece(
                index=evidence.index,
                block=evidence.block,
                prompt_text=evidence.prompt_text,
            )
            for evidence in selected
        )
        stats = ContextBuildStats(
            input_count=len(ranked_blocks),
            selected_count=len(selected),
            dropped_duplicate_count=dropped_duplicate_count,
            dropped_budget_count=dropped_budget_count,
            truncated_count=truncated_count,
            total_chars_included=len(formatted_context),
            max_context_chars=self.settings.max_context_chars,
        )
        return BuiltContext(
            query=query,
            selected=selected,
            formatted_context=formatted_context,
            stats=stats,
        )

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
