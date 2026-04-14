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

from app.storage.document_registry import DocumentRegistry, IndexedDocumentRecord


class DocumentRegistryTest(unittest.TestCase):
    def test_register_remove_and_clear_documents(self) -> None:
        temp_dir = make_temp_dir()
        try:
            registry = DocumentRegistry(temp_dir)
            record = IndexedDocumentRecord(
                document_id="doc-1",
                doc_id="doc-1",
                source_path="C:/docs/source.pdf",
                file_name="source.pdf",
                indexed_at="2026-04-14T00:00:00+00:00",
                stored_points=3,
                file_size=123,
                modified_at="2026-04-14T00:00:00+00:00",
                source_fingerprint="abc123",
                status="indexed",
            )

            registry.register_document(record)
            self.assertEqual(registry.get_document("doc-1"), record)

            removed = registry.remove_documents(["doc-1", "missing"])
            self.assertEqual(removed, [record])
            self.assertIsNone(registry.get_document("doc-1"))

            registry.register_document(record)
            self.assertEqual(registry.clear(), 1)
            self.assertEqual(registry.load_documents(), {})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def make_temp_dir() -> Path:
    TEST_TEMP_ROOT.mkdir(exist_ok=True)
    temp_dir = TEST_TEMP_ROOT / f"run_{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


if __name__ == "__main__":
    unittest.main()
