import unittest

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.retrieval_types import RerankedBlock
from app.services.visual_evidence import (
    VisualEvidenceDecision,
    guard_visual_decision,
    parse_visual_decision,
    visual_ref_from_block,
)


def visual_block(block_id: str, score: float = 0.9) -> RerankedBlock:
    return RerankedBlock(
        block_id=block_id,
        text="Formula context",
        retrieval_text="Formula context",
        source_file="source.pdf",
        page=2,
        section_path=[],
        retrieval_score=score,
        rerank_score=score,
        payload={
            "has_visual_evidence": True,
            "bbox": [10.0, 20.0, 110.0, 120.0],
            "page_number": 2,
        },
        document_id="doc-1",
        page_start=2,
        page_end=2,
        block_type="formula_with_context",
        label="Formula 1",
    )


class VisualEvidenceTest(unittest.TestCase):
    def test_visual_ref_from_block_requires_metadata(self):
        ref = visual_ref_from_block(visual_block("v1"), text_preview="Formula context")

        self.assertIsNotNone(ref)
        self.assertEqual(ref.block_id, "v1")
        self.assertEqual(ref.document_id, "doc-1")
        self.assertEqual(ref.page_number, 2)
        self.assertEqual(ref.bbox, (10.0, 20.0, 110.0, 120.0))
        self.assertEqual(ref.block_type, "formula_with_context")
        self.assertEqual(ref.label, "Formula 1")

    def test_context_builder_returns_visual_hints(self):
        builder = ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2))

        built = builder.build("query", [visual_block("v1")])

        self.assertEqual(len(built.visual_hints.selected), 1)
        self.assertEqual(built.visual_hints.selected[0].block_id, "v1")
        self.assertEqual(built.visual_hints.total_count, 1)

    def test_parse_and_guard_decision_keeps_known_targets(self):
        decision = parse_visual_decision(
            '{"mode":"inspect_visual_and_show","target_block_ids":["v1","missing"],'
            '"show_in_sources":true,"show_in_answer":true,'
            '"needs_multimodal_followup":true,"reason":"Formula may be needed."}'
        )
        ref = visual_ref_from_block(visual_block("v1"), text_preview="Formula context")

        guarded = guard_visual_decision(decision, [ref], max_targets=1)

        self.assertEqual(guarded.mode, "inspect_visual_and_show")
        self.assertEqual(guarded.target_block_ids, ["v1"])
        self.assertTrue(guarded.show_in_sources)
        self.assertTrue(guarded.show_in_answer)

    def test_malformed_decision_falls_back_to_text_only(self):
        decision = parse_visual_decision("not json")

        self.assertEqual(decision, VisualEvidenceDecision.text_only("Invalid visual decision JSON."))
