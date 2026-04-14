from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedBlock:
    block_id: str
    text: str
    retrieval_text: str
    source_file: str
    page: int | None
    section_path: list[str]
    retrieval_score: float
    payload: dict[str, Any]
    document_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    block_type: str | None = None
    label: str | None = None

    @property
    def evidence_text(self) -> str:
        if self.retrieval_text.strip():
            return self.retrieval_text
        if self.text.strip():
            return self.text

        parts: list[str] = []
        if self.section_path:
            parts.append("Section: " + " > ".join(self.section_path))
        if self.label:
            parts.append(f"Label: {self.label}")

        context_text = self.payload.get("context_text")
        if isinstance(context_text, str) and context_text.strip():
            parts.append(f"Context: {context_text.strip()}")
        return "\n".join(parts).strip()


@dataclass(frozen=True)
class RerankedBlock:
    block_id: str
    text: str
    retrieval_text: str
    source_file: str
    page: int | None
    section_path: list[str]
    retrieval_score: float
    rerank_score: float | None
    payload: dict[str, Any]
    document_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    block_type: str | None = None
    label: str | None = None

    @property
    def evidence_text(self) -> str:
        if self.retrieval_text.strip():
            return self.retrieval_text
        if self.text.strip():
            return self.text

        parts: list[str] = []
        if self.section_path:
            parts.append("Section: " + " > ".join(self.section_path))
        if self.label:
            parts.append(f"Label: {self.label}")

        context_text = self.payload.get("context_text")
        if isinstance(context_text, str) and context_text.strip():
            parts.append(f"Context: {context_text.strip()}")
        return "\n".join(parts).strip()


def make_reranked_block(candidate: RetrievedBlock, rerank_score: float | None) -> RerankedBlock:
    return RerankedBlock(
        block_id=candidate.block_id,
        text=candidate.text,
        retrieval_text=candidate.retrieval_text,
        source_file=candidate.source_file,
        page=candidate.page,
        section_path=candidate.section_path,
        retrieval_score=candidate.retrieval_score,
        rerank_score=rerank_score,
        payload=candidate.payload,
        document_id=candidate.document_id,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        block_type=candidate.block_type,
        label=candidate.label,
    )
