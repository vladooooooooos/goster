import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.reranker_service import LocalRerankerService, RerankerSettings
from app.services.retrieval_types import RetrievedBlock


def candidate(block_id: str, block_type: str) -> RetrievedBlock:
    return RetrievedBlock(
        block_id=block_id,
        text=f"{block_type} text",
        retrieval_text=f"{block_type} text",
        source_file="source.pdf",
        page=1,
        section_path=[],
        retrieval_score=0.5,
        payload={},
        document_id="doc-1",
        page_start=1,
        page_end=1,
        block_type=block_type,
        label=None,
    )


class RerankerDiagnosticsTest(unittest.TestCase):
    def test_logs_selected_block_types(self):
        service = LocalRerankerService(RerankerSettings(enabled=True, device="cpu"))
        service._predict_scores = lambda pairs: [0.9, 0.8, 0.7]

        with self.assertLogs("app.services.reranker_service", level="INFO") as logs:
            service.rerank(
                "query",
                [
                    candidate("text-1", "paragraph"),
                    candidate("visual-1", "figure"),
                    candidate("table-1", "table"),
                ],
                top_n=3,
            )

        output = "\n".join(logs.output)
        self.assertIn("top block types", output)
        self.assertIn("figure", output)
