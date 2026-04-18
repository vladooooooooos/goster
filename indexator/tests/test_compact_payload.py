from __future__ import annotations

import unittest
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
for path in (APP_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.vector_store import GostBlockVector, make_gost_block_payload, parse_gost_payload


class CompactPayloadTest(unittest.TestCase):
    def test_payload_keeps_retrieval_and_visual_metadata_without_debug_duplicates(self) -> None:
        payload = make_gost_block_payload(
            GostBlockVector(
                block_id="doc:1",
                doc_id="doc",
                document_id="doc",
                file_name="doc.pdf",
                file_path=Path("C:/docs/doc.pdf"),
                block_type="figure",
                page_number=3,
                text="Figure 1 - Cable route",
                embedding_text="Section: 1 Scope\nText: Figure 1 - Cable route",
                vector=[0.1, 0.2],
                section_path=["1 Scope"],
                reading_order=7,
                indexed_at="2026-04-18T00:00:00+00:00",
                label="Figure 1",
                context_text="The cable route is shown in the figure.",
                bbox=(1.0, 2.0, 30.0, 40.0),
            )
        )

        self.assertEqual(payload["block_id"], "doc:1")
        self.assertEqual(payload["document_id"], "doc")
        self.assertEqual(payload["file_name"], "doc.pdf")
        self.assertEqual(payload["page_start"], 3)
        self.assertEqual(payload["page_end"], 3)
        self.assertEqual(payload["block_type"], "figure")
        self.assertEqual(payload["label"], "Figure 1")
        self.assertEqual(payload["section_path"], ["1 Scope"])
        self.assertEqual(payload["text"], "Figure 1 - Cable route")
        self.assertEqual(payload["reading_order"], 7)
        self.assertTrue(payload["has_visual_evidence"])
        self.assertEqual(payload["bbox"], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(payload["page_number"], 3)
        self.assertEqual(payload["crop_status"], "available")

        for removed_key in (
            "doc_id",
            "doc_title",
            "file_path",
            "source_path",
            "indexed_at",
            "embedding_text",
            "context_text",
            "tokens_estimate",
        ):
            self.assertNotIn(removed_key, payload)

    def test_parse_gost_payload_still_reads_legacy_document_id_and_embedding_text(self) -> None:
        parsed = parse_gost_payload(
            {
                "block_id": "legacy:1",
                "doc_id": "legacy-doc",
                "file_path": "C:/docs/legacy.pdf",
                "text": "Stored text",
                "embedding_text": "Legacy embedding text",
                "page": 5,
            },
            fallback_id="fallback",
        )

        self.assertEqual(parsed.block_id, "legacy:1")
        self.assertEqual(parsed.document_id, "legacy-doc")
        self.assertEqual(parsed.source_file, "C:/docs/legacy.pdf")
        self.assertEqual(parsed.retrieval_text, "Legacy embedding text")
        self.assertEqual(parsed.page_start, 5)


if __name__ == "__main__":
    unittest.main()
