from __future__ import annotations

import unittest
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qdrant_client import models

from shared.vector_store import QdrantVectorStore, QdrantVectorStoreConfig


class QdrantCollectionConfigTest(unittest.TestCase):
    def test_collection_creation_uses_on_disk_vectors_and_scalar_quantization(self) -> None:
        fake_client = FakeQdrantClient()
        store = QdrantVectorStore(
            QdrantVectorStoreConfig(
                collection_name="gost_blocks",
                url="http://127.0.0.1:6333",
                vectors_on_disk=True,
                quantization_enabled=True,
                quantization_mode="scalar",
                quantized_vectors_always_ram=True,
            ),
            client_factory=lambda **kwargs: fake_client,
        )

        store.ensure_collection(1024)

        self.assertEqual(fake_client.created_kwargs["collection_name"], "gost_blocks")
        vector_params = fake_client.created_kwargs["vectors_config"]
        self.assertTrue(vector_params.on_disk)
        self.assertEqual(vector_params.size, 1024)
        self.assertEqual(vector_params.distance, models.Distance.COSINE)

        quantization_config = fake_client.created_kwargs["quantization_config"]
        self.assertIsInstance(quantization_config, models.ScalarQuantization)
        self.assertEqual(quantization_config.scalar.type, models.ScalarType.INT8)
        self.assertTrue(quantization_config.scalar.always_ram)

    def test_collection_creation_can_disable_quantization_and_on_disk_vectors(self) -> None:
        fake_client = FakeQdrantClient()
        store = QdrantVectorStore(
            QdrantVectorStoreConfig(
                collection_name="gost_blocks",
                url="http://127.0.0.1:6333",
                vectors_on_disk=False,
                quantization_enabled=False,
            ),
            client_factory=lambda **kwargs: fake_client,
        )

        store.ensure_collection(8)

        self.assertFalse(fake_client.created_kwargs["vectors_config"].on_disk)
        self.assertNotIn("quantization_config", fake_client.created_kwargs)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created_kwargs: dict[str, object] = {}
        self.exists = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: object) -> None:
        self.created_kwargs = kwargs
        self.exists = True

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
