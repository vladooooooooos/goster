import unittest
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.rag_service import RagService
from app.services.retrieval_pipeline import RetrievalPipelineResult
from app.services.retrieval_types import RetrievedBlock, RerankedBlock
from app.services.visual_crop_service import GeneratedCrop


class FakeLlmService:
    def __init__(self, visual_decision: str | None = None):
        self.calls = []
        self.visual_decision = visual_decision or (
            '{"mode":"inspect_visual_and_show","target_block_ids":["v1"],'
            '"show_in_sources":true,"show_in_answer":true,'
            '"needs_multimodal_followup":true,"reason":"Formula may be visual."}'
        )

    async def chat(self, messages):
        self.calls.append(messages)
        if "Return only JSON" in messages[-1]["content"]:
            return self.visual_decision
        return "The text supports the answer [1]. A related visual fragment is attached."


class FakeRetrievalPipeline:
    def __init__(self, visual_count: int = 1):
        self.visual_count = visual_count

    def retrieve(self, query, top_k):
        results = [
            RerankedBlock(
                block_id=f"v{index}",
                text=f"Formula context {index}",
                retrieval_text=f"Formula context {index}",
                source_file="source.pdf",
                page=2,
                section_path=[],
                retrieval_score=0.9 - index * 0.01,
                rerank_score=0.95 - index * 0.01,
                payload={
                    "has_visual_evidence": True,
                    "bbox": [10.0, 20.0, 110.0, 120.0],
                    "page_number": 2,
                },
                document_id="doc-1",
                page_start=2,
                page_end=2,
                block_type="formula_with_context",
                label=f"Formula {index}",
            )
            for index in range(1, self.visual_count + 1)
        ]
        candidates = [
            RetrievedBlock(
                block_id=result.block_id,
                text=result.text,
                retrieval_text=result.retrieval_text,
                source_file=result.source_file,
                page=result.page,
                section_path=result.section_path,
                retrieval_score=result.retrieval_score,
                payload=result.payload,
                document_id=result.document_id,
                page_start=result.page_start,
                page_end=result.page_end,
                block_type=result.block_type,
                label=result.label,
            )
            for result in results
        ]
        return RetrievalPipelineResult(
            query=query,
            candidates=candidates,
            results=results,
            info={"backend": "test"},
        )


class FakeCropService:
    def get_or_create_crop(self, ref):
        return GeneratedCrop(
            block_id=ref.block_id,
            document_id=ref.document_id,
            file_path="data/crops/doc-1/page-2-v1.png",
            url_path="/crops/doc-1/page-2-v1.png",
            width=200,
            height=100,
            format="png",
            dpi=160,
        )


class RagVisualFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_visual_decision_generates_crop_and_adds_metadata(self):
        service = RagService(
            llm_service=FakeLlmService(),
            retrieval_pipeline=FakeRetrievalPipeline(),
            context_builder=ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2)),
            visual_crop_service=FakeCropService(),
            visual_decision_enabled=True,
            visual_max_crops_per_answer=1,
        )

        with self.assertLogs("app.services.rag_service", level="INFO") as logs:
            answer = await service.answer_question("query", top_k=12)

        self.assertEqual(len(answer.visual_evidence), 1)
        self.assertEqual(answer.visual_evidence[0].block_id, "v1")
        self.assertEqual(answer.visual_evidence[0].crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertTrue(answer.citations[0].has_visual_evidence)
        self.assertEqual(answer.citations[0].visual_evidence.crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertEqual(answer.retrieval_info["visual"]["decision"]["mode"], "inspect_visual_and_show")
        self.assertIn("Returned 1 visual attachment", "\n".join(logs.output))

    async def test_explicit_visual_request_overrides_text_only_decision(self):
        service = RagService(
            llm_service=FakeLlmService(
                visual_decision=(
                    '{"mode":"text_only","target_block_ids":[],"show_in_sources":false,'
                    '"show_in_answer":false,"needs_multimodal_followup":false,'
                    '"reason":"The text is enough."}'
                )
            ),
            retrieval_pipeline=FakeRetrievalPipeline(),
            context_builder=ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2)),
            visual_crop_service=FakeCropService(),
            visual_decision_enabled=True,
            visual_max_crops_per_answer=1,
        )

        with self.assertLogs("app.services.rag_service", level="INFO") as logs:
            answer = await service.answer_question("дай фотографию расположения раковины", top_k=12)

        self.assertEqual(len(answer.visual_evidence), 1)
        self.assertEqual(answer.visual_evidence[0].block_id, "v1")
        self.assertEqual(answer.retrieval_info["visual"]["decision"]["mode"], "show_visual")
        self.assertTrue(answer.retrieval_info["visual"]["decision"]["show_in_sources"])
        self.assertIn("Explicit visual request promoted visual decision", "\n".join(logs.output))

    async def test_implicit_visual_query_returns_multiple_visual_attachments(self):
        service = RagService(
            llm_service=FakeLlmService(
                visual_decision=(
                    '{"mode":"text_only","target_block_ids":[],"show_in_sources":false,'
                    '"show_in_answer":false,"needs_multimodal_followup":false,'
                    '"reason":"The text is enough."}'
                )
            ),
            retrieval_pipeline=FakeRetrievalPipeline(visual_count=5),
            context_builder=ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=2, max_blocks=12)),
            visual_crop_service=FakeCropService(),
            visual_decision_enabled=True,
            visual_max_crops_per_answer=4,
        )

        answer = await service.answer_question("sink and mixer layout", top_k=12)

        self.assertEqual(len(answer.visual_evidence), 4)
        self.assertEqual(answer.retrieval_info["query_plan"]["tasks"][0]["needs_visual"], True)
        self.assertEqual(answer.retrieval_info["visual"]["selected_count"], 4)
        self.assertEqual(answer.retrieval_info["visual"]["candidate_count"], 5)
        answer_prompt = service._llm_service.calls[-1][-1]["content"]
        self.assertIn("Visual evidence has been attached", answer_prompt)
