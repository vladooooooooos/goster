from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TEMP_ROOT = APP_ROOT.parent / ".test_tmp"

from app.core.block_builder import make_document_id
from app.services.file_fingerprint import FileFingerprintService
from app.services.indexed_state import INDEXED, INDEXED_STALE, MISSING_SOURCE, READY, IndexedStateResolver
from app.services.pdf_scanner import PdfScanResult
from app.storage.document_registry import IndexedDocumentRecord


class IndexedStateResolverTest(unittest.TestCase):
    def test_ready_indexed_and_stale_states(self) -> None:
        temp_dir = make_temp_dir()
        try:
            pdf_path = temp_dir / "source.pdf"
            pdf_path.write_bytes(b"first")
            fingerprint_service = FileFingerprintService()
            fingerprint = fingerprint_service.get_content_fingerprint(pdf_path)
            document_id = make_document_id(pdf_path)
            resolver = IndexedStateResolver(fingerprint_service)
            scan_result = make_scan_result(pdf_path)

            ready_state = resolver.resolve_scan_results([scan_result], {}, temp_dir)[0]
            self.assertEqual(ready_state.status, READY)

            record = IndexedDocumentRecord(
                document_id=document_id,
                doc_id=document_id,
                source_path=str(pdf_path.resolve()),
                file_name=pdf_path.name,
                indexed_at="2026-04-14T00:00:00+00:00",
                stored_points=1,
                file_size=fingerprint.file_size,
                modified_at=fingerprint.modified_at,
                source_fingerprint=fingerprint.source_fingerprint or "",
            )
            indexed_state = resolver.resolve_scan_results([scan_result], {document_id: record}, temp_dir)[0]
            self.assertEqual(indexed_state.status, INDEXED)

            stale_record = IndexedDocumentRecord(
                document_id=document_id,
                doc_id=document_id,
                source_path=str(pdf_path.resolve()),
                file_name=pdf_path.name,
                indexed_at="2026-04-14T00:00:00+00:00",
                stored_points=1,
                file_size=fingerprint.file_size + 1,
                modified_at=fingerprint.modified_at,
                source_fingerprint=fingerprint.source_fingerprint or "",
            )
            stale_state = resolver.resolve_scan_results([scan_result], {document_id: stale_record}, temp_dir)[0]
            self.assertEqual(stale_state.status, INDEXED_STALE)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_missing_source_record_is_included_for_scanned_folder(self) -> None:
        temp_dir = make_temp_dir()
        try:
            missing_path = temp_dir / "missing.pdf"
            document_id = make_document_id(missing_path)
            record = IndexedDocumentRecord(
                document_id=document_id,
                doc_id=document_id,
                source_path=str(missing_path.resolve()),
                file_name=missing_path.name,
                indexed_at="2026-04-14T00:00:00+00:00",
                stored_points=2,
                file_size=10,
                modified_at="2026-04-14T00:00:00+00:00",
                source_fingerprint="hash",
            )

            states = IndexedStateResolver(FileFingerprintService()).resolve_scan_results([], {document_id: record}, temp_dir)

            self.assertEqual(len(states), 1)
            self.assertEqual(states[0].status, MISSING_SOURCE)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def make_scan_result(pdf_path: Path) -> PdfScanResult:
    return PdfScanResult(
        file_name=pdf_path.name,
        file_path=pdf_path.resolve(),
        file_size_bytes=pdf_path.stat().st_size,
        page_count=1,
        status=READY,
    )


def make_temp_dir() -> Path:
    TEST_TEMP_ROOT.mkdir(exist_ok=True)
    temp_dir = TEST_TEMP_ROOT / f"state_{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


if __name__ == "__main__":
    unittest.main()
