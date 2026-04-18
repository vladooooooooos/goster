from __future__ import annotations

import atexit
import shutil
import unittest
import uuid
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TEMP_ROOT = APP_ROOT.parent / ".test_tmp"
atexit.register(shutil.rmtree, TEST_TEMP_ROOT, ignore_errors=True)

from app.core.blocks import StructuredBlock
from app.services.embedding_service import BlockEmbedding, BlockEmbeddingRun
from app.storage.qdrant_store import QdrantStore
from shared.vector_store import QdrantVectorStoreConfig


class QdrantDeletionTest(unittest.TestCase):
    def test_delete_documents_handles_missing_collection(self) -> None:
        temp_dir = make_temp_dir()
        try:
            store = make_store(FakeQdrantClient())
            try:
                run = store.delete_documents(["doc-1"])
                self.assertEqual(run.requested_documents, 1)
                self.assertEqual(run.removed_points, 0)
                self.assertEqual(run.skipped_documents, 1)
            finally:
                store.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_delete_documents_removes_matching_document_only(self) -> None:
        temp_dir = make_temp_dir()
        try:
            fake_client = FakeQdrantClient()
            store = make_store(fake_client)
            try:
                blocks = [
                    make_block("doc-1", 0),
                    make_block("doc-1", 1),
                    make_block("doc-2", 0),
                ]
                embeddings = BlockEmbeddingRun(
                    embeddings=[
                        make_embedding("doc-1:0", [1.0, 0.0, 0.0, 0.0]),
                        make_embedding("doc-1:1", [0.8, 0.1, 0.0, 0.0]),
                        make_embedding("doc-2:0", [0.0, 1.0, 0.0, 0.0]),
                    ],
                    model_name="test",
                    device="cpu",
                    embedding_dimension=4,
                    elapsed_seconds=0.0,
                )
                store.upsert_block_embeddings(blocks, embeddings, indexed_at="2026-04-14T00:00:00+00:00")

                run = store.delete_documents(["doc-1", "unknown"])

                self.assertEqual(run.removed_points, 2)
                self.assertEqual(store.count_document_points("doc-1"), 0)
                self.assertEqual(store.count_document_points("doc-2"), 1)
                self.assertEqual(store.count_document_points("unknown"), 0)
            finally:
                store.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_clear_all_removes_collection_without_removing_root(self) -> None:
        temp_dir = make_temp_dir()
        try:
            fake_client = FakeQdrantClient()
            store = make_store(fake_client)
            try:
                blocks = [make_block("doc-1", 0)]
                embeddings = BlockEmbeddingRun(
                    embeddings=[make_embedding("doc-1:0", [1.0, 0.0, 0.0, 0.0])],
                    model_name="test",
                    device="cpu",
                    embedding_dimension=4,
                    elapsed_seconds=0.0,
                )
                store.upsert_block_embeddings(blocks, embeddings, indexed_at="2026-04-14T00:00:00+00:00")

                run = store.clear_all()

                self.assertTrue(run.collection_existed)
                self.assertEqual(run.removed_points, 1)
                self.assertFalse(store.client.collection_exists("gost_blocks"))
            finally:
                store.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def make_block(document_id: str, reading_order: int) -> StructuredBlock:
    return StructuredBlock(
        id=f"{document_id}:{reading_order}",
        doc_id=document_id,
        file_name=f"{document_id}.pdf",
        file_path=Path(f"C:/{document_id}.pdf"),
        page_number=1,
        block_type="paragraph",
        text=f"Text for {document_id}",
        bbox=None,
        reading_order=reading_order,
        section_path=[],
    )


def make_embedding(block_id: str, vector: list[float]) -> BlockEmbedding:
    return BlockEmbedding(
        block_id=block_id,
        block_type="paragraph",
        page_number=1,
        reading_order=0,
        text="Text",
        vector=vector,
    )


def make_temp_dir() -> Path:
    TEST_TEMP_ROOT.mkdir(exist_ok=True)
    temp_dir = TEST_TEMP_ROOT / f"run_{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def make_store(fake_client: "FakeQdrantClient") -> QdrantStore:
    return QdrantStore(
        QdrantVectorStoreConfig(
            collection_name="gost_blocks",
            url="http://127.0.0.1:6333",
        ),
        client_factory=lambda **kwargs: fake_client,
    )


class FakeCountResult:
    def __init__(self, count: int) -> None:
        self.count = count


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections.setdefault(collection_name, {})

    def upsert(self, collection_name: str, points: list[object], wait: bool) -> None:
        collection = self.collections.setdefault(collection_name, {})
        for point in points:
            collection[str(getattr(point, "id"))] = point

    def count(self, collection_name: str, count_filter: object | None = None, exact: bool = True) -> FakeCountResult:
        collection = self.collections.get(collection_name, {})
        if count_filter is None:
            return FakeCountResult(len(collection))
        document_ids = filter_document_ids(count_filter)
        return FakeCountResult(
            sum(
                1
                for point in collection.values()
                if getattr(point, "payload", {}).get("document_id") in document_ids
                or getattr(point, "payload", {}).get("doc_id") in document_ids
            )
        )

    def delete(self, collection_name: str, points_selector: object, wait: bool) -> None:
        collection = self.collections.get(collection_name, {})
        document_ids = filter_document_ids(getattr(points_selector, "filter"))
        for point_id, point in list(collection.items()):
            payload = getattr(point, "payload", {})
            if payload.get("document_id") in document_ids or payload.get("doc_id") in document_ids:
                del collection[point_id]

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def close(self) -> None:
        pass


def filter_document_ids(qdrant_filter: object) -> set[str]:
    document_ids: set[str] = set()
    for condition in getattr(qdrant_filter, "should", []) or []:
        match = getattr(condition, "match", None)
        values = getattr(match, "any", []) or []
        document_ids.update(str(value) for value in values)
    return document_ids


if __name__ == "__main__":
    unittest.main()
