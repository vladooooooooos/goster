import json
import tempfile
import unittest
import sys
from pathlib import Path

import pymupdf

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.visual_crop_service import VisualCropService, VisualCropSettings
from app.services.visual_evidence import VisualEvidenceRef


class VisualCropServiceTest(unittest.TestCase):
    def test_generates_deterministic_crop_from_registered_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "source.pdf"
            self._write_pdf(pdf_path)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "documents.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "source_path": str(pdf_path),
                                "file_name": "source.pdf",
                                "indexed_at": "now",
                                "stored_points": 1,
                                "file_size": pdf_path.stat().st_size,
                                "modified_at": "now",
                                "source_fingerprint": "abc",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service = VisualCropService(
                VisualCropSettings(indexer_output_dir=root, crops_dir=root / "crops", dpi=120)
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(10.0, 10.0, 120.0, 80.0),
                block_type="figure",
                label="Figure 1",
                source_file="source.pdf",
                text_preview="Figure context",
            )

            crop = service.get_or_create_crop(ref)

            self.assertTrue(Path(crop.file_path).exists())
            self.assertEqual(crop.block_id, "block-1")
            self.assertEqual(crop.document_id, "doc-1")
            self.assertEqual(crop.format, "png")
            self.assertEqual(crop.dpi, 120)
            self.assertGreater(crop.width, 0)
            self.assertGreater(crop.height, 0)
            self.assertIn("/crops/doc-1/", crop.url_path.replace("\\", "/"))

    def test_invalid_bbox_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = VisualCropService(
                VisualCropSettings(indexer_output_dir=root, crops_dir=root / "crops", dpi=120)
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(20.0, 20.0, 10.0, 10.0),
                block_type="figure",
                label=None,
                source_file="source.pdf",
                text_preview="Figure context",
            )

            self.assertIsNone(service.get_or_create_crop(ref))

    def test_finds_source_pdf_by_fingerprint_when_registered_path_is_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "docs"
            source_root.mkdir()
            pdf_path = source_root / "source.pdf"
            self._write_pdf(pdf_path)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "documents.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "source_path": str(root / "missing" / "source.pdf"),
                                "file_name": "source.pdf",
                                "indexed_at": "now",
                                "stored_points": 1,
                                "file_size": pdf_path.stat().st_size,
                                "modified_at": "now",
                                "source_fingerprint": self._sha256(pdf_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service = VisualCropService(
                VisualCropSettings(
                    indexer_output_dir=root,
                    crops_dir=root / "crops",
                    dpi=120,
                    source_roots=(source_root,),
                )
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(10.0, 10.0, 120.0, 80.0),
                block_type="figure",
                label="Figure 1",
                source_file="source.pdf",
                text_preview="Figure context",
            )

            crop = service.get_or_create_crop(ref)

            self.assertIsNotNone(crop)
            self.assertTrue(Path(crop.file_path).exists())

    def test_expands_small_figure_bbox_on_scanned_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "source.pdf"
            self._write_scanned_like_pdf(pdf_path)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "documents.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "source_path": str(pdf_path),
                                "file_name": "source.pdf",
                                "indexed_at": "now",
                                "stored_points": 1,
                                "file_size": pdf_path.stat().st_size,
                                "modified_at": "now",
                                "source_fingerprint": self._sha256(pdf_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service = VisualCropService(
                VisualCropSettings(indexer_output_dir=root, crops_dir=root / "crops", dpi=72)
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(210.0, 260.0, 478.0, 330.0),
                block_type="figure",
                label="Figure 1",
                source_file="source.pdf",
                text_preview="Figure context",
            )

            crop = service.get_or_create_crop(ref)

            self.assertIsNotNone(crop)
            self.assertGreater(crop.width, 450)
            self.assertGreater(crop.height, 200)

    def _write_pdf(self, path: Path) -> None:
        document = pymupdf.open()
        page = document.new_page(width=200, height=120)
        page.insert_text((20, 40), "Visual crop test")
        document.save(path)
        document.close()

    def _write_scanned_like_pdf(self, path: Path) -> None:
        document = pymupdf.open()
        page = document.new_page(width=687, height=971)
        pixmap = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 687, 971), 0)
        pixmap.clear_with(255)
        page.insert_image(page.rect, pixmap=pixmap)
        page.insert_text((230, 320), "Figure 1 - Sink layout")
        document.save(path)
        document.close()

    def _sha256(self, path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
