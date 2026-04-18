import unittest

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.rag_service import RagService
from app.services.retrieval_pipeline import RetrievalPipelineResult
from app.services.retrieval_types import RetrievedBlock, RerankedBlock
from app.services.visual_crop_service import GeneratedCrop


class FakeLlmService:
    def __init__(self):
        self.calls = []

    async def chat(self, messages):
        self.calls.append(messages)
        if "Return only JSON" in messages[-1]["content"]:
            return (
                '{"mode":"inspect_visual_and_show","target_block_ids":["v1"],'
                '"show_in_sources":true,"show_in_answer":true,'
                '"needs_multimodal_followup":true,"reason":"Formula may be visual."}'
            )
        return "The text supports the answer [1]. A related visual fragment is attached."


class FakeRetrievalPipeline:
    def retrieve(self, query, top_k):
        visual_payload = {
            "has_visual_evidence": True,
            "bbox": [10.0, 20.0, 110.0, 120.0],
            "page_number": 2,
        }
        result = RerankedBlock(
            block_id="v1",
            text="Formula context",
            retrieval_text="Formula context",
            source_file="source.pdf",
            page=2,
            section_path=[],
            retrieval_score=0.9,
            rerank_score=0.95,
            payload=visual_payload,
            document_id="doc-1",
            page_start=2,
            page_end=2,
            block_type="formula_with_context",
            label="Formula 1",
        )
        candidate = RetrievedBlock(
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
        return RetrievalPipelineResult(
            query=query,
            candidates=[candidate],
            results=[result],
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

        answer = await service.answer_question("query", top_k=12)

        self.assertEqual(len(answer.visual_evidence), 1)
        self.assertEqual(answer.visual_evidence[0].block_id, "v1")
        self.assertEqual(answer.visual_evidence[0].crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertTrue(answer.citations[0].has_visual_evidence)
        self.assertEqual(answer.citations[0].visual_evidence.crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertEqual(answer.retrieval_info["visual"]["decision"]["mode"], "inspect_visual_and_show")
