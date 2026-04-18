from __future__ import annotations

import unittest
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.core.blocks import StructuredBlock
from app.core.index_compaction import IndexCompactionSettings, compact_index_blocks


class IndexCompactionTest(unittest.TestCase):
    def test_compact_mode_filters_toc_and_short_noise(self) -> None:
        blocks = [
            make_block("doc:0", "heading", "1 Scope", 0),
            make_block("doc:1", "table_of_contents", "1 Scope................1", 1),
            make_block("doc:2", "paragraph", "ok", 2),
            make_block("doc:3", "paragraph", "This paragraph has enough useful retrieval text.", 3),
        ]

        compacted = compact_index_blocks(
            blocks,
            IndexCompactionSettings(mode="compact", min_indexable_chars=12),
        )

        self.assertEqual([block.block_type for block in compacted], ["heading", "paragraph"])
        self.assertEqual(compacted[1].text, "This paragraph has enough useful retrieval text.")

    def test_compact_mode_merges_adjacent_text_blocks_in_same_section(self) -> None:
        blocks = [
            make_block("doc:0", "paragraph", "First requirement text for cable routing.", 0),
            make_block("doc:1", "list_item", "a) Additional related requirement.", 1),
            make_block("doc:2", "paragraph", "Next subsection starts here.", 2, section_path=["2 Other"]),
        ]

        compacted = compact_index_blocks(
            blocks,
            IndexCompactionSettings(
                mode="compact",
                min_indexable_chars=8,
                target_chunk_chars=120,
                max_chunk_chars=200,
            ),
        )

        self.assertEqual(len(compacted), 2)
        self.assertEqual(compacted[0].block_type, "paragraph")
        self.assertIn("First requirement text", compacted[0].text)
        self.assertIn("a) Additional related requirement.", compacted[0].text)
        self.assertEqual(compacted[0].bbox, (10.0, 10.0, 120.0, 30.0))
        self.assertEqual(compacted[0].reading_order, 0)
        self.assertEqual(compacted[1].section_path, ["2 Other"])

    def test_compact_mode_preserves_visual_block_boundaries(self) -> None:
        blocks = [
            make_block("doc:0", "paragraph", "Introductory paragraph before a figure.", 0),
            make_block(
                "doc:1",
                "figure",
                "Figure 1 - Cable route",
                1,
                bbox=(40.0, 50.0, 200.0, 180.0),
                label="Figure 1",
                context_text="The route is shown below.",
            ),
            make_block("doc:2", "paragraph", "Paragraph after the figure.", 2),
        ]

        compacted = compact_index_blocks(
            blocks,
            IndexCompactionSettings(mode="compact", min_indexable_chars=8),
        )

        self.assertEqual([block.block_type for block in compacted], ["paragraph", "figure", "paragraph"])
        self.assertEqual(compacted[1].bbox, (40.0, 50.0, 200.0, 180.0))
        self.assertEqual(compacted[1].label, "Figure 1")
        self.assertEqual(compacted[1].context_text, "The route is shown below.")

    def test_compact_mode_can_strip_visual_metadata_without_dropping_visual_blocks(self) -> None:
        blocks = [
            make_block(
                "doc:1",
                "figure",
                "Figure 1 - Cable route",
                1,
                bbox=(40.0, 50.0, 200.0, 180.0),
                label="Figure 1",
            ),
        ]

        compacted = compact_index_blocks(
            blocks,
            IndexCompactionSettings(mode="compact", store_visual_metadata=False),
        )

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].block_type, "figure")
        self.assertIsNone(compacted[0].bbox)
        self.assertEqual(compacted[0].label, "Figure 1")

    def test_rich_mode_keeps_table_of_contents(self) -> None:
        blocks = [
            make_block("doc:0", "table_of_contents", "1 Scope................1", 0),
            make_block("doc:1", "paragraph", "Body paragraph with useful content.", 1),
        ]

        compacted = compact_index_blocks(blocks, IndexCompactionSettings(mode="rich"))

        self.assertEqual([block.block_type for block in compacted], ["table_of_contents", "paragraph"])


def make_block(
    block_id: str,
    block_type: str,
    text: str,
    reading_order: int,
    section_path: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    label: str | None = None,
    context_text: str | None = None,
) -> StructuredBlock:
    return StructuredBlock(
        id=block_id,
        doc_id="doc",
        file_name="doc.pdf",
        file_path=Path("C:/docs/doc.pdf"),
        page_number=1,
        block_type=block_type,  # type: ignore[arg-type]
        text=text,
        bbox=bbox or (10.0, 10.0 + reading_order * 10, 120.0, 20.0 + reading_order * 10),
        reading_order=reading_order,
        section_path=section_path or ["1 Scope"],
        label=label,
        context_text=context_text,
    )


if __name__ == "__main__":
    unittest.main()
