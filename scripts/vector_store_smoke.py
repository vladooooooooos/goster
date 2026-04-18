"""End-to-end smoke check for the shared Qdrant vector store integration."""

from __future__ import annotations

import argparse
from contextlib import suppress
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "goster_vector_store_smoke"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
VECTOR_SIZE = 4
TARGET_VECTOR = [1.0, 0.0, 0.0, 0.0]


def main() -> int:
    args = parse_args()
    if args.phase == "writer":
        run_writer(args.qdrant_url)
        return 0
    if args.phase == "reader":
        run_reader(args.qdrant_url)
        return 0

    run_wrapper(args.qdrant_url)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["wrapper", "writer", "reader"], default="wrapper")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    return parser.parse_args()


def run_wrapper(qdrant_url: str) -> None:
    run_phase("writer", qdrant_url)
    run_phase("reader", qdrant_url)
    print(
        "Shared vector store smoke passed: "
        f"collection={COLLECTION_NAME}, endpoint={qdrant_url}"
    )


def run_phase(phase: str, qdrant_url: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--qdrant-url",
        qdrant_url,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_writer(qdrant_url: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "indexator"))

    from app.core.blocks import StructuredBlock
    from app.services.embedding_service import BlockEmbedding, BlockEmbeddingRun
    from app.storage.qdrant_store import QdrantStore
    from shared.vector_store import QdrantVectorStoreConfig

    blocks = [
        StructuredBlock(
            id="smoke-doc:0001",
            doc_id="smoke-doc",
            file_name="smoke-standard.pdf",
            file_path=REPO_ROOT / "smoke-standard.pdf",
            page_number=7,
            block_type="paragraph",
            text="Smoke target text for shared vector store retrieval.",
            bbox=(1.0, 2.0, 3.0, 4.0),
            reading_order=1,
            section_path=["Smoke section", "Target"],
            label="target-label",
            context_text="Target context text.",
        ),
        StructuredBlock(
            id="smoke-doc:0002",
            doc_id="smoke-doc",
            file_name="smoke-standard.pdf",
            file_path=REPO_ROOT / "smoke-standard.pdf",
            page_number=8,
            block_type="table",
            text="Smoke distractor text for shared vector store retrieval.",
            bbox=None,
            reading_order=2,
            section_path=["Smoke section", "Distractor"],
            label="distractor-label",
            context_text="Distractor context text.",
        ),
    ]
    embedding_run = BlockEmbeddingRun(
        embeddings=[
            BlockEmbedding(
                block_id="smoke-doc:0001",
                block_type="paragraph",
                page_number=7,
                reading_order=1,
                text="Section: Smoke section > Target\nLabel: target-label\nText: Smoke target text",
                vector=TARGET_VECTOR,
            ),
            BlockEmbedding(
                block_id="smoke-doc:0002",
                block_type="table",
                page_number=8,
                reading_order=2,
                text="Section: Smoke section > Distractor\nLabel: distractor-label\nText: Smoke distractor text",
                vector=[0.0, 1.0, 0.0, 0.0],
            ),
        ],
        model_name="smoke-embedding-model",
        device="cpu",
        embedding_dimension=VECTOR_SIZE,
        elapsed_seconds=0.0,
    )

    store = QdrantStore(
        QdrantVectorStoreConfig(
            collection_name=COLLECTION_NAME,
            url=qdrant_url,
            distance_metric="Cosine",
        )
    )
    try:
        with suppress(Exception):
            store.clear_all()
        storage_run = store.upsert_block_embeddings(blocks, embedding_run)
        assert storage_run.collection_name == COLLECTION_NAME
        assert storage_run.endpoint == qdrant_url
        assert storage_run.stored_blocks == 2
        assert storage_run.embedding_dimension == VECTOR_SIZE
    finally:
        store.close()


def run_reader(qdrant_url: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "gost-chat"))

    from app.services.qdrant_retriever import QdrantRetriever

    retriever = QdrantRetriever(
        qdrant_url=qdrant_url,
        qdrant_host="127.0.0.1",
        qdrant_port=6333,
        qdrant_https=False,
        qdrant_api_key=None,
        qdrant_timeout_seconds=5.0,
        collection_name=COLLECTION_NAME,
        embedding_service=FakeEmbeddingService(),
    )
    try:
        results, info = retriever.search("find target", top_k=2)
        assert info["backend"] == "qdrant"
        assert info["collection_name"] == COLLECTION_NAME
        assert info["endpoint"] == qdrant_url
        assert info["embedding_model"] == "smoke-embedding-model"
        assert info["embedding_device"] == "cpu"
        assert len(results) >= 1

        target = results[0]
        assert target.block_id == "smoke-doc:0001"
        assert target.text == "Smoke target text for shared vector store retrieval."
        assert target.retrieval_text == "Section: Smoke section > Target\nLabel: target-label\nText: Smoke target text"
        assert target.source_file == "smoke-standard.pdf"
        assert target.page == 7
        assert target.page_start == 7
        assert target.page_end == 7
        assert target.section_path == ["Smoke section", "Target"]
        assert target.document_id == "smoke-doc"
        assert target.block_type == "paragraph"
        assert target.label == "target-label"
        assert target.retrieval_score > 0.0

        payload = target.payload
        expected_payload = {
            "block_id": "smoke-doc:0001",
            "doc_id": "smoke-doc",
            "document_id": "smoke-doc",
            "doc_title": "smoke-standard.pdf",
            "file_name": "smoke-standard.pdf",
            "file_path": str(REPO_ROOT / "smoke-standard.pdf"),
            "source_path": str(REPO_ROOT / "smoke-standard.pdf"),
            "indexed_at": payload.get("indexed_at"),
            "block_type": "paragraph",
            "page_start": 7,
            "page_end": 7,
            "section_path": ["Smoke section", "Target"],
            "label": "target-label",
            "text": "Smoke target text for shared vector store retrieval.",
            "embedding_text": "Section: Smoke section > Target\nLabel: target-label\nText: Smoke target text",
            "context_text": "Target context text.",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "reading_order": 1,
            "tokens_estimate": 18,
        }
        for key, expected_value in expected_payload.items():
            assert payload.get(key) == expected_value, f"{key}: {payload.get(key)!r} != {expected_value!r}"
    finally:
        retriever.close()


@dataclass(frozen=True)
class FakeEmbeddingSettings:
    model_name: str = "smoke-embedding-model"


class FakeEmbeddingService:
    settings = FakeEmbeddingSettings()
    device = "cpu"

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Smoke query must not be empty.")
        return TARGET_VECTOR


if __name__ == "__main__":
    raise SystemExit(main())
