import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.retrieval_types import RerankedBlock
from app.services.visual_backfill_service import VisualBackfillService


class VisualBackfillServiceTest(unittest.TestCase):
    def test_backfill_prefers_exact_label_match_in_same_document(self):
        selected_blocks = [
            _block(
                "text-1",
                "Paragraph references \u0440\u0438\u0441. 3.3.4 for the equipment layout.",
                document_id="doc-1",
                page=61,
                block_type="paragraph",
            )
        ]
        built_context = ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=12)).build(
            "equipment layout",
            selected_blocks,
        )
        lookup = FakeVisualLookup(
            [
                _visual_block("nearby", "Figure 2", page=61),
                _visual_block("target", "Figure 3.3.4", page=63),
            ]
        )

        result = VisualBackfillService(lookup, candidate_limit=8).backfill(built_context)

        self.assertTrue(result.attempted)
        self.assertEqual([ref.block_id for ref in result.refs], ["target"])
        self.assertEqual(result.refs[0].source_block_id, "text-1")
        self.assertEqual(result.refs[0].selection_reason, "Matched graphic reference in selected evidence.")

    def test_backfill_ignores_blocks_without_usable_visual_metadata(self):
        selected_blocks = [
            _block(
                "text-1",
                "Paragraph references Figure 1.",
                document_id="doc-1",
                page=5,
                block_type="paragraph",
            )
        ]
        built_context = ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=12)).build(
            "figure",
            selected_blocks,
        )
        lookup = FakeVisualLookup([_visual_block("bad", "Figure 1", page=5, bbox=None)])

        result = VisualBackfillService(lookup, candidate_limit=8).backfill(built_context)

        self.assertTrue(result.attempted)
        self.assertEqual(result.refs, [])
        self.assertEqual(result.missing_references, ["figure 1"])


class FakeVisualLookup:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []

    def find_visual_blocks(self, document_id, limit):
        self.calls.append((document_id, limit))
        return self.blocks


def _block(
    block_id,
    text,
    document_id="doc-1",
    page=1,
    block_type="paragraph",
    label=None,
    payload=None,
):
    return RerankedBlock(
        block_id=block_id,
        text=text,
        retrieval_text=text,
        source_file="source.pdf",
        page=page,
        section_path=[],
        retrieval_score=0.9,
        rerank_score=0.9,
        payload=payload or {},
        document_id=document_id,
        page_start=page,
        page_end=page,
        block_type=block_type,
        label=label,
    )


def _visual_block(block_id, label, page, bbox=(10.0, 20.0, 110.0, 120.0)):
    payload = {
        "has_visual_evidence": True,
        "page_number": page,
    }
    if bbox is not None:
        payload["bbox"] = list(bbox)
    return _block(
        block_id,
        label,
        page=page,
        block_type="figure",
        label=label,
        payload=payload,
    )


if __name__ == "__main__":
    unittest.main()
