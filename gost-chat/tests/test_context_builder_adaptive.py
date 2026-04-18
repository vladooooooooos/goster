import unittest
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.retrieval_types import RerankedBlock


def block(
    block_id: str,
    text: str,
    rerank_score: float | None,
    retrieval_score: float = 0.5,
    payload: dict | None = None,
    block_type: str = "paragraph",
) -> RerankedBlock:
    return RerankedBlock(
        block_id=block_id,
        text=text,
        retrieval_text=text,
        source_file="source.pdf",
        page=1,
        section_path=[],
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        payload=payload or {},
        document_id="doc-1",
        page_start=1,
        page_end=1,
        block_type=block_type,
        label="Figure 1" if block_type == "figure" else None,
    )


class AdaptiveContextBuilderTest(unittest.TestCase):
    def test_selects_more_than_soft_target_until_score_drop(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=2,
                soft_target_blocks=3,
                max_blocks=10,
                adaptive_score_threshold=0.20,
                max_context_chars=20000,
            )
        )
        ranked = [
            block("b1", "first relevant block", 1.00),
            block("b2", "second relevant block", 0.95),
            block("b3", "third relevant block", 0.90),
            block("b4", "fourth still close block", 0.83),
            block("b5", "fifth dropped block", 0.50),
        ]

        built = builder.build("query", ranked)

        self.assertEqual([item.block.block_id for item in built.selected], ["b1", "b2", "b3", "b4"])
        self.assertEqual(built.stats.stop_reason, "score_drop")
        self.assertEqual(built.stats.input_count, 5)
        self.assertEqual(built.stats.selected_count, 4)

    def test_keeps_minimum_blocks_even_when_second_score_drops(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=2,
                soft_target_blocks=2,
                max_blocks=5,
                adaptive_score_threshold=0.10,
                max_context_chars=20000,
            )
        )
        ranked = [
            block("b1", "first relevant block", 1.00),
            block("b2", "second low but required block", 0.20),
            block("b3", "third low block", 0.19),
        ]

        built = builder.build("query", ranked)

        self.assertEqual([item.block.block_id for item in built.selected], ["b1", "b2"])
        self.assertEqual(built.stats.stop_reason, "score_drop")

    def test_tracks_duplicates_and_budget(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=1,
                soft_target_blocks=5,
                max_blocks=5,
                max_context_chars=130,
                max_chars_per_block=200,
            )
        )
        ranked = [
            block("b1", "same text", 1.0),
            block("b1", "same text again by id", 0.9),
            block("b2", "same text", 0.8),
            block("b3", "large unique text " * 20, 0.7),
        ]

        built = builder.build("query", ranked)

        self.assertEqual(len(built.selected), 1)
        self.assertGreaterEqual(built.stats.dropped_duplicate_count, 2)
        self.assertGreaterEqual(built.stats.dropped_budget_count, 1)

    def test_includes_best_visual_candidate_when_text_selection_has_no_visuals(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=2,
                soft_target_blocks=3,
                max_blocks=5,
                adaptive_score_threshold=0.20,
                max_context_chars=20000,
            )
        )
        ranked = [
            block("text-1", "first text block", 1.00),
            block("text-2", "second text block", 0.95),
            block("text-3", "third text block", 0.90),
            block(
                "visual-1",
                "figure surrogate text",
                0.50,
                payload={
                    "has_visual_evidence": True,
                    "bbox": [10.0, 20.0, 110.0, 120.0],
                    "page_number": 1,
                },
                block_type="figure",
            ),
        ]

        built = builder.build("query", ranked)

        self.assertIn("visual-1", [item.block.block_id for item in built.selected])
        self.assertEqual(len(built.visual_hints.selected), 1)
        self.assertEqual(built.visual_hints.selected[0].block_id, "visual-1")

    def test_logs_input_and_selected_block_types(self):
        builder = ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2))

        with self.assertLogs("app.services.context_builder", level="INFO") as logs:
            builder.build("query", [block("text-1", "text", 1.0)])

        output = "\n".join(logs.output)
        self.assertIn("input block types", output)
        self.assertIn("selected evidence block types", output)
