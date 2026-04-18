import unittest
from types import SimpleNamespace

from app.services.retrieval_pipeline import RetrievalPipeline
from app.services.retrieval_types import RetrievedBlock, make_reranked_block


def candidate(index: int) -> RetrievedBlock:
    return RetrievedBlock(
        block_id=f"b{index}",
        text=f"text {index}",
        retrieval_text=f"text {index}",
        source_file="source.pdf",
        page=1,
        section_path=[],
        retrieval_score=1.0 / index,
        payload={},
        document_id="doc-1",
        page_start=1,
        page_end=1,
    )


class FakeRetriever:
    def __init__(self):
        self.top_k = None

    def retrieve_blocks(self, query, top_k):
        self.top_k = top_k
        return [candidate(index) for index in range(1, top_k + 1)], {}


class FakeReranker:
    enabled = True

    def __init__(self):
        self.top_n = None

    def rerank(self, query, candidates, top_n):
        self.top_n = top_n
        return [make_reranked_block(item, 1.0 / (index + 1)) for index, item in enumerate(candidates[:top_n])]


class RetrievalPipelinePoolTest(unittest.TestCase):
    def test_reranker_receives_broad_pool_and_returns_context_pool(self):
        retriever = FakeRetriever()
        reranker = FakeReranker()
        settings = SimpleNamespace(
            retrieval_backend="json",
            reranker_top_k=40,
            reranker_top_n=12,
        )
        pipeline = RetrievalPipeline(retriever=retriever, settings=settings, reranker=reranker)

        result = pipeline.retrieve("query", top_k=12)

        self.assertEqual(retriever.top_k, 40)
        self.assertEqual(reranker.top_n, 12)
        self.assertEqual(len(result.candidates), 40)
        self.assertEqual(len(result.results), 12)
        self.assertEqual(result.info["candidate_pool_top_n"], 12)

    def test_requested_top_k_can_raise_retrieval_depth(self):
        retriever = FakeRetriever()
        reranker = FakeReranker()
        settings = SimpleNamespace(
            retrieval_backend="json",
            reranker_top_k=40,
            reranker_top_n=12,
        )
        pipeline = RetrievalPipeline(retriever=retriever, settings=settings, reranker=reranker)

        pipeline.retrieve("query", top_k=45)

        self.assertEqual(retriever.top_k, 45)
        self.assertEqual(reranker.top_n, 12)
