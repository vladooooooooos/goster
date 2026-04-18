import unittest

from app.schemas.ask import AskCitation, AskResponse, AskRetrievedChunk, AskVisualEvidence


class AskSchemaVisualsTest(unittest.TestCase):
    def test_response_serializes_visual_evidence(self):
        visual = AskVisualEvidence(
            block_id="v1",
            document_id="doc-1",
            source_file="source.pdf",
            page_number=2,
            block_type="figure",
            label="Figure 1",
            crop_path="data/crops/doc-1/page-2-v1.png",
            crop_url="/crops/doc-1/page-2-v1.png",
            width=200,
            height=100,
            format="png",
            dpi=160,
        )
        response = AskResponse(
            query="query",
            answer="answer",
            citations=[
                AskCitation(
                    document_id="doc-1",
                    file_name="source.pdf",
                    chunk_id="v1",
                    page_start=2,
                    page_end=2,
                    score=0.9,
                    evidence_preview="preview",
                    has_visual_evidence=True,
                    visual_evidence=visual,
                )
            ],
            retrieved_results_count=1,
            retrieval_used=True,
            retrieved_chunks=[
                AskRetrievedChunk(
                    document_id="doc-1",
                    file_name="source.pdf",
                    chunk_id="v1",
                    page_start=2,
                    page_end=2,
                    score=0.9,
                    text="preview",
                    has_visual_evidence=True,
                    visual_evidence=visual,
                )
            ],
            visual_evidence=[visual],
        )

        payload = response.model_dump()

        self.assertEqual(payload["visual_evidence"][0]["crop_url"], "/crops/doc-1/page-2-v1.png")
        self.assertEqual(payload["citations"][0]["visual_evidence"]["block_id"], "v1")
