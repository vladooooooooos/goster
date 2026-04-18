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


if __name__ == "__main__":
    unittest.main()
