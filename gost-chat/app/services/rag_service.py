from dataclasses import dataclass
from typing import Any

from app.services.context_builder import BuiltContext, ContextBuilder, ContextEvidence
from app.services.llm_service import LlmService
from app.services.retrieval_pipeline import RetrievalPipeline

NO_RELIABLE_ANSWER = "В документах не найдено достаточно надежной информации для ответа."

RAG_SYSTEM_PROMPT = (
    "Ты — инженер-консультант по нормативно-технической документации и ГОСТам.\n\n"
    "Отвечай на русском языке, понятно и профессионально, как опытный инженер-конструктор.\n"
    "Основывай ответ прежде всего на найденном документном контексте и цитируй релевантные фрагменты.\n"
    "Допускается кратко объяснять смысл требований простым инженерным языком, но не придумывай нормативные положения, которых нет в предоставленном контексте.\n"
    "Если контекст неполный, прямо укажи, что по найденным документам нельзя сделать надёжный однозначный вывод.\n"
    "Старайся давать не сухой пересказ, а практичный ответ: что именно требуется, в каких условиях это применимо, и на какие пункты документа это опирается.\n"
    "Сохраняй ожидаемое поведение цитирования: используй номера источников вида [1] или [2] для утверждений, подтвержденных контекстом.\n"
    "Не выдумывай факты, определения, требования, выводы или ссылки на документы, которых нет в контексте."
)


@dataclass(frozen=True)
class RagCitation:
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    score: float
    evidence_preview: str
    block_id: str | None = None
    source_file: str | None = None
    page: int | None = None
    section_path: list[str] | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class RagRetrievedChunk:
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    score: float
    text: str
    block_id: str | None = None
    source_file: str | None = None
    page: int | None = None
    section_path: list[str] | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    citations: list[RagCitation]
    retrieved_results_count: int
    retrieval_used: bool
    retrieved_chunks: list[RagRetrievedChunk]
    retrieval_info: dict[str, Any] | None


class RagService:
    def __init__(
        self,
        llm_service: LlmService,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
    ) -> None:
        self._llm_service = llm_service
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder

    async def answer_question(self, query: str, top_k: int = 5) -> RagAnswer:
        retrieval_result = self._retrieval_pipeline.retrieve(query, top_k=top_k)

        if not retrieval_result.candidates:
            return RagAnswer(
                query=retrieval_result.query,
                answer=NO_RELIABLE_ANSWER,
                citations=[],
                retrieved_results_count=0,
                retrieval_used=False,
                retrieved_chunks=[],
                retrieval_info=retrieval_result.info,
            )

        built_context = self._context_builder.build(retrieval_result.query, retrieval_result.results)
        retrieval_info = _with_context_info(retrieval_result.info, built_context)
        if not built_context.selected:
            return RagAnswer(
                query=retrieval_result.query,
                answer=NO_RELIABLE_ANSWER,
                citations=[],
                retrieved_results_count=0,
                retrieval_used=True,
                retrieved_chunks=[],
                retrieval_info=retrieval_info,
            )

        prompt = _build_grounded_prompt(built_context)
        answer = await self._llm_service.chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        citations = [_citation_from_evidence(evidence) for evidence in built_context.selected]
        retrieved_chunks = [_retrieved_chunk_from_evidence(evidence) for evidence in built_context.selected]

        return RagAnswer(
            query=retrieval_result.query,
            answer=answer,
            citations=citations,
            retrieved_results_count=len(built_context.selected),
            retrieval_used=True,
            retrieved_chunks=retrieved_chunks,
            retrieval_info=retrieval_info,
        )


def _build_grounded_prompt(built_context: BuiltContext) -> str:
    return "\n\n".join(
        [
            "Answer the question using only the retrieved document context below.",
            "Do not add facts from outside the retrieved context.",
            f"If the retrieved context is insufficient, answer exactly: {NO_RELIABLE_ANSWER}",
            "If the evidence is only partial, say what is supported and what is not supported.",
            "Always write the final answer in Russian.",
            "Use source numbers like [1] or [2] for statements supported by the context.",
            "",
            f"Question: {built_context.query}",
            "",
            "Retrieved context:",
            built_context.formatted_context,
        ]
    )


def _citation_from_evidence(evidence: ContextEvidence) -> RagCitation:
    result = evidence.block
    return RagCitation(
        document_id=result.document_id or "",
        file_name=result.source_file,
        chunk_id=result.block_id,
        page_start=result.page_start,
        page_end=result.page_end,
        score=result.rerank_score if result.rerank_score is not None else result.retrieval_score,
        evidence_preview=evidence.evidence_preview,
        block_id=result.block_id,
        source_file=result.source_file,
        page=result.page,
        section_path=result.section_path,
        retrieval_score=result.retrieval_score,
        rerank_score=result.rerank_score,
    )


def _retrieved_chunk_from_evidence(evidence: ContextEvidence) -> RagRetrievedChunk:
    result = evidence.block
    return RagRetrievedChunk(
        document_id=result.document_id or "",
        file_name=result.source_file,
        chunk_id=result.block_id,
        page_start=result.page_start,
        page_end=result.page_end,
        score=result.rerank_score if result.rerank_score is not None else result.retrieval_score,
        text=result.evidence_text,
        block_id=result.block_id,
        source_file=result.source_file,
        page=result.page,
        section_path=result.section_path,
        retrieval_score=result.retrieval_score,
        rerank_score=result.rerank_score,
    )


def _with_context_info(info: dict[str, Any] | None, built_context: BuiltContext) -> dict[str, Any]:
    merged = dict(info or {})
    merged["context"] = built_context.stats.to_dict()
    return merged
