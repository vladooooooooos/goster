"""Smoke check for chat-side retrieval orchestration and reranking."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gost-chat"))

from app.services.retrieval_pipeline import RetrievalPipeline
from app.services.retrieval_types import RetrievedBlock, RerankedBlock, make_reranked_block
from app.services.rag_service import RagService
from app.services.context_builder import ContextBuilder


def main() -> int:
    qdrant_retriever = FakeQdrantRetriever()
    json_retriever = FakeJsonRetriever()
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        retriever=json_retriever,
        settings=FakeSettings(),
        qdrant_retriever=qdrant_retriever,
        reranker=reranker,
    )

    result = pipeline.retrieve(" target query ", top_k=2)

    assert result.query == "target query"
    assert qdrant_retriever.calls == [("target query", 4)]
    assert json_retriever.calls == []
    assert reranker.calls == [("target query", ["candidate-a", "candidate-b"], 2)]
    assert result.info["backend"] == "qdrant"
    assert result.info["requested_top_k"] == 2
    assert result.info["retrieval_top_k"] == 4
    assert result.info["final_top_n"] == 2
    assert result.info["retrieved_candidates_count"] == 2
    assert result.info["reranked_results_count"] == 2
    assert result.info["reranker_enabled"] is True

    assert [block.block_id for block in result.results] == ["candidate-b", "candidate-a"]
    top = result.results[0]
    assert top.rerank_score == 0.95
    assert top.retrieval_score == 0.5
    assert top.document_id == "doc-b"
    assert top.source_file == "standard-b.pdf"
    assert top.page_start == 5
    assert top.page_end == 6
    assert top.section_path == ["Section B"]
    assert top.block_type == "table"
    assert top.label == "Table B"
    assert top.payload["file_path"] == str(REPO_ROOT / "docs" / "standard-b.pdf")

    rag_service = RagService(
        llm_service=FakeLlmService(),
        retrieval_pipeline=pipeline,
        context_builder=ContextBuilder(),
    )
    answer = asyncio.run(rag_service.answer_question("target query", top_k=2))
    assert answer.answer == "grounded answer"
    assert answer.retrieved_results_count == 2
    assert answer.citations[0].block_id == "candidate-b"
    assert answer.citations[0].retrieval_score == 0.5
    assert answer.citations[0].rerank_score == 0.95
    assert answer.retrieved_chunks[0].section_path == ["Section B"]
    assert answer.retrieval_info is not None
    assert answer.retrieval_info["reranked_results_count"] == 2
    assert answer.retrieval_info["context"]["selected_count"] == 2

    print("Retrieval pipeline smoke passed: vector retrieval, reranking, and metadata preservation verified.")
    return 0


@dataclass(frozen=True)
class FakeSettings:
    retrieval_backend: str = "auto"
    reranker_top_k: int = 4
    reranker_top_n: int = 2


class FakeQdrantRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> tuple[list[RetrievedBlock], dict[str, object]]:
        self.calls.append((query, top_k))
        return [
            make_candidate(
                block_id="candidate-a",
                document_id="doc-a",
                source_file="standard-a.pdf",
                retrieval_score=0.9,
                page_start=3,
                page_end=3,
                section_path=["Section A"],
                block_type="paragraph",
                label="A",
            ),
            make_candidate(
                block_id="candidate-b",
                document_id="doc-b",
                source_file="standard-b.pdf",
                retrieval_score=0.5,
                page_start=5,
                page_end=6,
                section_path=["Section B"],
                block_type="table",
                label="Table B",
            ),
        ], {
            "backend": "qdrant",
            "top_k": top_k,
            "collection_name": "smoke",
        }


class FakeJsonRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve_blocks(self, query: str, top_k: int) -> tuple[list[RetrievedBlock], dict[str, object]]:
        self.calls.append((query, top_k))
        return [], {"backend": "json"}


class FakeReranker:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, candidates: list[RetrievedBlock], top_n: int) -> list[RerankedBlock]:
        self.calls.append((query, [candidate.block_id for candidate in candidates], top_n))
        scores = {
            "candidate-a": 0.1,
            "candidate-b": 0.95,
        }
        reranked = [make_reranked_block(candidate, scores[candidate.block_id]) for candidate in candidates]
        reranked.sort(key=lambda block: -(block.rerank_score or 0.0))
        return reranked[:top_n]


class FakeLlmService:
    async def chat(self, messages: list[dict[str, str]]) -> str:
        assert messages
        assert "candidate-b" in messages[-1]["content"]
        return "grounded answer"


def make_candidate(
    block_id: str,
    document_id: str,
    source_file: str,
    retrieval_score: float,
    page_start: int,
    page_end: int,
    section_path: list[str],
    block_type: str,
    label: str,
) -> RetrievedBlock:
    payload = {
        "doc_id": document_id,
        "file_name": source_file,
        "file_path": str(REPO_ROOT / "docs" / source_file),
        "block_type": block_type,
        "label": label,
    }
    return RetrievedBlock(
        block_id=block_id,
        text=f"Text for {block_id}",
        retrieval_text=f"Evidence for {block_id}",
        source_file=source_file,
        page=page_start,
        section_path=section_path,
        retrieval_score=retrieval_score,
        payload=payload,
        document_id=document_id,
        page_start=page_start,
        page_end=page_end,
        block_type=block_type,
        label=label,
    )


if __name__ == "__main__":
    raise SystemExit(main())
