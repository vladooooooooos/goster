from __future__ import annotations

import unittest

from shared.vector_store import (
    QdrantServerConnectionError,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorPoint,
)


class SharedQdrantServerConfigTest(unittest.TestCase):
    def test_endpoint_prefers_url(self) -> None:
        config = QdrantVectorStoreConfig(
            collection_name="gost_blocks",
            url="http://127.0.0.1:6333",
            host="ignored-host",
            port=6334,
            https=True,
        )

        self.assertEqual(config.endpoint, "http://127.0.0.1:6333")

    def test_endpoint_uses_host_port_when_url_is_empty(self) -> None:
        config = QdrantVectorStoreConfig(
            collection_name="gost_blocks",
            url="",
            host="qdrant.local",
            port=6333,
            https=False,
        )

        self.assertEqual(config.endpoint, "http://qdrant.local:6333")

    def test_vector_store_constructs_url_client(self) -> None:
        calls: list[dict[str, object]] = []

        def make_client(**kwargs: object) -> FakeQdrantClient:
            calls.append(kwargs)
            return FakeQdrantClient()

        store = QdrantVectorStore(
            QdrantVectorStoreConfig(
                collection_name="gost_blocks",
                url="http://127.0.0.1:6333",
                api_key="secret",
                timeout_seconds=2.5,
            ),
            client_factory=make_client,
        )

        self.assertEqual(store.endpoint, "http://127.0.0.1:6333")
        self.assertEqual(
            calls,
            [{"url": "http://127.0.0.1:6333", "api_key": "secret", "timeout": 2.5}],
        )

    def test_vector_store_wraps_reachability_errors(self) -> None:
        def make_client(**kwargs: object) -> FailingQdrantClient:
            return FailingQdrantClient()

        store = QdrantVectorStore(
            QdrantVectorStoreConfig(
                collection_name="gost_blocks",
                url="http://127.0.0.1:6333",
            ),
            client_factory=make_client,
        )

        with self.assertRaisesRegex(
            QdrantServerConnectionError,
            "Qdrant server is not reachable at http://127.0.0.1:6333",
        ):
            store.search(vector=[1.0, 0.0], top_k=3)

    def test_vector_store_upserts_points_in_configured_batches(self) -> None:
        fake_client = UpsertRecordingQdrantClient()
        store = QdrantVectorStore(
            QdrantVectorStoreConfig(
                collection_name="gost_blocks",
                url="http://127.0.0.1:6333",
                upsert_batch_size=2,
            ),
            client_factory=lambda **kwargs: fake_client,
        )
        points = [
            VectorPoint(id=f"point-{index}", vector=[float(index), 0.0], payload={"document_id": "doc"})
            for index in range(5)
        ]

        run = store.upsert_points(points, embedding_dimension=2)

        self.assertEqual(run.stored_points, 5)
        self.assertEqual([len(batch) for batch in fake_client.upserted_batches], [2, 2, 1])


class FakeQdrantClient:
    def collection_exists(self, collection_name: str) -> bool:
        return False

    def close(self) -> None:
        pass


class UpsertRecordingQdrantClient:
    def __init__(self) -> None:
        self.collection_created = False
        self.upserted_batches: list[list[object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_created

    def create_collection(self, **kwargs: object) -> None:
        self.collection_created = True

    def upsert(self, collection_name: str, points: list[object], wait: bool) -> None:
        self.upserted_batches.append(points)

    def close(self) -> None:
        pass


class FailingQdrantClient:
    def collection_exists(self, collection_name: str) -> bool:
        raise RuntimeError("connection refused")

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
