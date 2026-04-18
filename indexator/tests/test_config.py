from __future__ import annotations

import unittest
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.utils.config import build_config


class IndexatorConfigTest(unittest.TestCase):
    def test_storage_config_reads_qdrant_server_settings(self) -> None:
        config = build_config(
            {
                "app": {"name": "Indexator", "version": "0.1.0"},
                "ui": {"window_width": 100, "window_height": 100},
                "embedding": {"model_name": "BAAI/bge-m3"},
                "storage": {
                    "provider": "qdrant",
                    "collection_name": "gost_blocks",
                    "url": "http://127.0.0.1:6333",
                    "host": "127.0.0.1",
                    "port": 6333,
                    "https": False,
                    "timeout_seconds": 3.0,
                    "api_key": "",
                    "distance_metric": "Cosine",
                    "shared_data_path": "../shared/data",
                },
            }
        )

        self.assertEqual(config.storage.url, "http://127.0.0.1:6333")
        self.assertEqual(config.storage.host, "127.0.0.1")
        self.assertEqual(config.storage.port, 6333)
        self.assertFalse(config.storage.https)
        self.assertEqual(config.storage.timeout_seconds, 3.0)
        self.assertIsNone(config.storage.api_key)
        self.assertEqual(config.storage.shared_data_path, "../shared/data")

    def test_storage_config_reads_compact_indexing_and_quantization_settings(self) -> None:
        config = build_config(
            {
                "app": {"name": "Indexator", "version": "0.1.0"},
                "ui": {"window_width": 100, "window_height": 100},
                "embedding": {"model_name": "BAAI/bge-m3"},
                "indexing": {
                    "mode": "compact",
                    "min_indexable_chars": 32,
                    "target_chunk_chars": 900,
                    "max_chunk_chars": 1600,
                    "store_visual_metadata": True,
                },
                "storage": {
                    "provider": "qdrant",
                    "collection_name": "gost_blocks",
                    "url": "http://127.0.0.1:6333",
                    "qdrant_quantization_enabled": True,
                    "qdrant_quantization_mode": "scalar",
                    "qdrant_vectors_on_disk": True,
                    "qdrant_quantized_vectors_always_ram": True,
                    "qdrant_upsert_batch_size": 32,
                },
            }
        )

        self.assertEqual(config.indexing.mode, "compact")
        self.assertEqual(config.indexing.min_indexable_chars, 32)
        self.assertEqual(config.indexing.target_chunk_chars, 900)
        self.assertEqual(config.indexing.max_chunk_chars, 1600)
        self.assertTrue(config.indexing.store_visual_metadata)
        self.assertTrue(config.storage.quantization_enabled)
        self.assertEqual(config.storage.quantization_mode, "scalar")
        self.assertTrue(config.storage.vectors_on_disk)
        self.assertTrue(config.storage.quantized_vectors_always_ram)
        self.assertEqual(config.storage.upsert_batch_size, 32)


if __name__ == "__main__":
    unittest.main()
