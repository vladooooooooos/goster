import asyncio
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.orchestration.tool_contracts import ToolContext
from app.orchestration.tools import DocumentRagTool, VisualAssetTool, VisualCropTool


@dataclass(frozen=True)
class FakeVisual:
    block_id: str = "v1"
    document_id: str = "doc-1"
    source_file: str = "source.pdf"
    page_number: int = 7
    block_type: str | None = "figure"
    label: str | None = "Figure 1"
    crop_path: str = "data/crops/doc-1/v1.png"
    crop_url: str = "/crops/doc-1/v1.png"
    width: int = 200
    height: int = 100
    format: str = "png"
    dpi: int = 160
    selection_reason: str | None = "Relevant."
    confidence: float | None = 0.9
    source_block_id: str | None = "v1"
    crop_kind: str | None = "rag_selected"


class FakeRagAnswer:
    query = "query"
    answer = "Answer [1]."
    citations = []
    retrieved_results_count = 1
    retrieval_used = True
    retrieved_chunks = []
    retrieval_info = {"backend": "test"}
    visual_evidence = [FakeVisual()]


class FakeRagService:
    def __init__(self):
        self.calls = []

    async def answer_question(self, query, top_k=12):
        self.calls.append((query, top_k))
        return FakeRagAnswer()


class FakeCropService:
    def __init__(self):
        self.refs = []

    def get_or_create_crop(self, ref):
        self.refs.append(ref)
        return type(
            "Crop",
            (),
            {
                "block_id": ref.block_id,
                "document_id": ref.document_id,
                "file_path": "data/crops/doc-1/v1.png",
                "url_path": "/crops/doc-1/v1.png",
                "width": 200,
                "height": 100,
                "format": "png",
                "dpi": 160,
            },
        )()


class OrchestrationToolsTest(unittest.TestCase):
    def test_document_rag_tool_returns_structured_content(self):
        rag = FakeRagService()
        tool = DocumentRagTool(rag)

        result = asyncio.run(tool.run({"query": "query", "top_k": 9}, ToolContext(session_id="s1", message_id="m1")))

        self.assertTrue(result.ok)
        self.assertEqual(rag.calls, [("query", 9)])
        self.assertEqual(result.content["answer"], "Answer [1].")
        self.assertEqual(result.content["visual_evidence"][0]["crop_url"], "/crops/doc-1/v1.png")

    def test_visual_asset_tool_normalizes_visual_evidence(self):
        tool = VisualAssetTool()

        result = asyncio.run(tool.run({"visual_evidence": [FakeVisual().__dict__]}, ToolContext("s1", "m1")))

        self.assertTrue(result.ok)
        self.assertEqual(result.content["attachments"][0]["asset_id"], "doc-1:v1")
        self.assertEqual(result.content["attachments"][0]["url"], "/crops/doc-1/v1.png")

    def test_visual_crop_tool_creates_crop_from_reference_payload(self):
        crop_service = FakeCropService()
        tool = VisualCropTool(crop_service)
        payload = {
            "block_id": "v1",
            "document_id": "doc-1",
            "page_number": 7,
            "bbox": [1.0, 2.0, 50.0, 60.0],
            "block_type": "figure",
            "label": "Figure 1",
            "source_file": "source.pdf",
            "text_preview": "preview",
        }

        result = asyncio.run(tool.run(payload, ToolContext("s1", "m1")))

        self.assertTrue(result.ok)
        self.assertEqual(result.content["crop"]["url_path"], "/crops/doc-1/v1.png")
        self.assertEqual(crop_service.refs[0].bbox, (1.0, 2.0, 50.0, 60.0))


if __name__ == "__main__":
    unittest.main()
