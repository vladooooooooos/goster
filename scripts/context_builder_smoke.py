"""Smoke check for chat-side context building and RagService integration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gost-chat"))

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.rag_service import RagService
from app.services.retrieval_pipeline import RetrievalPipelineResult
from app.services.retrieval_types import RetrievedBlock, RerankedBlock, make_reranked_block


def main() -> int:
    ranked_blocks = [
        make_block(
            block_id="block-a",
            text="Primary evidence about bolts and tolerances.",
            document_id="doc-a",
            source_file="standard-a.pdf",
            page_start=1,
            page_end=1,
            retrieval_score=0.9,
            rerank_score=0.99,
            section_path=["General"],
            block_type="paragraph",
            label="A",
        ),
        make_block(
            block_id="block-a",
            text="Primary evidence about bolts and tolerances.",
            document_id="doc-a",
            source_file="standard-a.pdf",
            page_start=1,
            page_end=1,
            retrieval_score=0.8,
            rerank_score=0.9,
            section_path=["General"],
            block_type="paragraph",
            label="A duplicate id",
        ),
        make_block(
            block_id="block-b",
            text="Primary   evidence about bolts and tolerances.",
            document_id="doc-b",
            source_file="standard-b.pdf",
            page_start=2,
            page_end=2,
            retrieval_score=0.7,
            rerank_score=0.8,
            section_path=["Duplicate text"],
            block_type="paragraph",
            label="B duplicate text",
        ),
        make_block(
            block_id="block-c",
            text="Secondary evidence that should not fit the tight context budget.",
            document_id="doc-c",
            source_file="standard-c.pdf",
            page_start=3,
            page_end=4,
            retrieval_score=0.6,
            rerank_score=0.7,
            section_path=["Budget"],
            block_type="table",
            label="C",
        ),
    ]

    builder = ContextBuilder(
        ContextBuilderSettings(
            max_blocks=5,
            max_context_chars=180,
            max_chars_per_block=200,
            evidence_preview_chars=24,
        )
    )
    built_context = builder.build("bolt question", ranked_blocks)

    assert [evidence.block.block_id for evidence in built_context.selected] == ["block-a"]
    assert built_context.stats.input_count == 4
    assert built_context.stats.selected_count == 1
    assert built_context.stats.dropped_duplicate_count == 2
    assert built_context.stats.dropped_budget_count == 1
    assert built_context.stats.total_chars_included == len(built_context.formatted_context)
    assert len(built_context.selected[0].evidence_preview) <= 24
    assert built_context.selected[0].evidence_preview.endswith("...")
    assert built_context.selected[0].page_label == "page 1"

    rag_service = RagService(
        ollama_client=FakeOllamaClient(),
        retrieval_pipeline=FakeRetrievalPipeline(ranked_blocks),
        context_builder=builder,
    )
    answer = asyncio.run(rag_service.answer_question(" bolt question ", top_k=4))

    assert answer.answer == "grounded answer"
    assert answer.retrieved_results_count == 1
    assert len(answer.citations) == 1
    assert answer.citations[0].block_id == "block-a"
    assert answer.citations[0].document_id == "doc-a"
    assert answer.citations[0].source_file == "standard-a.pdf"
    assert answer.citations[0].section_path == ["General"]
    assert answer.citations[0].retrieval_score == 0.9
    assert answer.citations[0].rerank_score == 0.99
    assert answer.retrieved_chunks[0].text == "Primary evidence about bolts and tolerances."
    assert answer.retrieval_info is not None
    assert answer.retrieval_info["reranked_results_count"] == 4
    assert answer.retrieval_info["context"]["selected_count"] == 1
    assert answer.retrieval_info["context"]["dropped_duplicate_count"] == 2
    assert answer.retrieval_info["context"]["dropped_budget_count"] == 1

    print("Context builder smoke passed: selection, deduplication, budget, and metadata verified.")
    return 0


class FakeRetrievalPipeline:
    def __init__(self, results: list[RerankedBlock]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> RetrievalPipelineResult:
        normalized_query = query.strip()
        self.calls.append((normalized_query, top_k))
        candidates = [
            RetrievedBlock(
                block_id=result.block_id,
                text=result.text,
                retrieval_text=result.retrieval_text,
                source_file=result.source_file,
                page=result.page,
                section_path=result.section_path,
                retrieval_score=result.retrieval_score,
                payload=result.payload,
                document_id=result.document_id,
                page_start=result.page_start,
                page_end=result.page_end,
                block_type=result.block_type,
                label=result.label,
            )
            for result in self.results
        ]
        return RetrievalPipelineResult(
            query=normalized_query,
            candidates=candidates,
            results=self.results,
            info={
                "backend": "fake",
                "requested_top_k": top_k,
                "reranked_results_count": len(self.results),
            },
        )


class FakeOllamaClient:
    async def chat(self, messages: list[dict[str, str]]) -> str:
        assert messages
        prompt = messages[-1]["content"]
        assert "block-a" in prompt
        assert "block-b" not in prompt
        assert "block-c" not in prompt
        return "grounded answer"


def make_block(
    block_id: str,
    text: str,
    document_id: str,
    source_file: str,
    page_start: int,
    page_end: int,
    retrieval_score: float,
    rerank_score: float,
    section_path: list[str],
    block_type: str,
    label: str,
) -> RerankedBlock:
    candidate = RetrievedBlock(
        block_id=block_id,
        text=text,
        retrieval_text=text,
        source_file=source_file,
        page=page_start,
        section_path=section_path,
        retrieval_score=retrieval_score,
        payload={
            "doc_id": document_id,
            "file_name": source_file,
            "file_path": str(REPO_ROOT / "docs" / source_file),
            "block_type": block_type,
            "label": label,
        },
        document_id=document_id,
        page_start=page_start,
        page_end=page_end,
        block_type=block_type,
        label=label,
    )
    return make_reranked_block(candidate, rerank_score)


if __name__ == "__main__":
    raise SystemExit(main())
