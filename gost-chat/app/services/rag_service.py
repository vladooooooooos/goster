import logging
from dataclasses import dataclass
from typing import Any

from app.services.context_builder import BuiltContext, ContextBuilder, ContextEvidence
from app.services.llm_service import LlmService
from app.services.retrieval_pipeline import RetrievalPipeline
from app.services.visual_crop_service import GeneratedCrop
from app.services.visual_evidence import (
    VisualEvidenceDecision,
    VisualEvidenceRef,
    guard_visual_decision,
    parse_visual_decision,
)

logger = logging.getLogger(__name__)

_VISUAL_REQUEST_TERMS = (
    "show",
    "display",
    "give",
    "provide",
    "attach",
    "return",
    "\u043f\u043e\u043a\u0430\u0436",
    "\u0434\u0430\u0439",
    "\u0434\u0430\u0439\u0442\u0435",
    "\u0432\u044b\u0434\u0430\u0439",
    "\u0432\u044b\u0432\u0435\u0434",
    "\u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432",
    "\u043f\u0440\u0438\u0448\u043b",
)
_VISUAL_OBJECT_TERMS = (
    "image",
    "picture",
    "photo",
    "figure",
    "drawing",
    "diagram",
    "scheme",
    "table",
    "formula",
    "\u0444\u043e\u0442\u043e",
    "\u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444",
    "\u043a\u0430\u0440\u0442\u0438\u043d",
    "\u0438\u0437\u043e\u0431\u0440\u0430\u0436",
    "\u0440\u0438\u0441\u0443\u043d",
    "\u0447\u0435\u0440\u0442\u0435\u0436",
    "\u0441\u0445\u0435\u043c",
    "\u0434\u0438\u0430\u0433\u0440\u0430\u043c",
    "\u0442\u0430\u0431\u043b\u0438\u0446",
    "\u0444\u043e\u0440\u043c\u0443\u043b",
)

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
class RagVisualEvidence:
    block_id: str
    document_id: str
    source_file: str
    page_number: int
    block_type: str | None
    label: str | None
    crop_path: str
    crop_url: str
    width: int
    height: int
    format: str
    dpi: int


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
    block_type: str | None = None
    label: str | None = None
    has_visual_evidence: bool = False
    visual_evidence: RagVisualEvidence | None = None


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
    block_type: str | None = None
    label: str | None = None
    has_visual_evidence: bool = False
    visual_evidence: RagVisualEvidence | None = None


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    citations: list[RagCitation]
    retrieved_results_count: int
    retrieval_used: bool
    retrieved_chunks: list[RagRetrievedChunk]
    retrieval_info: dict[str, Any] | None
    visual_evidence: list[RagVisualEvidence]


class RagService:
    def __init__(
        self,
        llm_service: LlmService,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        visual_crop_service: Any | None = None,
        visual_decision_enabled: bool = True,
        visual_max_crops_per_answer: int = 1,
    ) -> None:
        self._llm_service = llm_service
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._visual_crop_service = visual_crop_service
        self._visual_decision_enabled = visual_decision_enabled
        self._visual_max_crops_per_answer = visual_max_crops_per_answer

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
                visual_evidence=[],
            )

        built_context = self._context_builder.build(retrieval_result.query, retrieval_result.results)
        if not built_context.selected:
            retrieval_info = _with_context_info(
                retrieval_result.info,
                built_context,
                VisualEvidenceDecision.text_only("No text context was selected."),
                [],
            )
            return RagAnswer(
                query=retrieval_result.query,
                answer=NO_RELIABLE_ANSWER,
                citations=[],
                retrieved_results_count=0,
                retrieval_used=True,
                retrieved_chunks=[],
                retrieval_info=retrieval_info,
                visual_evidence=[],
            )

        visual_decision = await self._decide_visual_evidence(built_context)
        visual_evidence = self._generate_visual_evidence(built_context, visual_decision)
        logger.info(
            "Returned %s visual attachment(s): %s.",
            len(visual_evidence),
            [visual.block_id for visual in visual_evidence],
        )
        visual_by_block_id = {visual.block_id: visual for visual in visual_evidence}
        retrieval_info = _with_context_info(retrieval_result.info, built_context, visual_decision, visual_evidence)
        prompt = _build_grounded_prompt(built_context)
        answer = await self._llm_service.chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )

        citations = [_citation_from_evidence(evidence, visual_by_block_id) for evidence in built_context.selected]
        retrieved_chunks = [
            _retrieved_chunk_from_evidence(evidence, visual_by_block_id) for evidence in built_context.selected
        ]

        return RagAnswer(
            query=retrieval_result.query,
            answer=answer,
            citations=citations,
            retrieved_results_count=len(built_context.selected),
            retrieval_used=True,
            retrieved_chunks=retrieved_chunks,
            retrieval_info=retrieval_info,
            visual_evidence=visual_evidence,
        )

    async def _decide_visual_evidence(self, built_context: BuiltContext) -> VisualEvidenceDecision:
        refs = [*built_context.visual_hints.selected, *built_context.visual_hints.candidates]
        if not self._visual_decision_enabled or not refs:
            return VisualEvidenceDecision.text_only("No visual decision was requested.")
        prompt = _build_visual_decision_prompt(built_context)
        raw = await self._llm_service.chat([{"role": "user", "content": prompt}])
        decision = parse_visual_decision(raw)
        guarded = guard_visual_decision(decision, refs, max_targets=self._visual_max_crops_per_answer)
        if guarded.mode == "text_only" and _is_explicit_visual_request(built_context.query):
            promoted = _visual_decision_for_explicit_request(refs, self._visual_max_crops_per_answer)
            if promoted.mode != "text_only":
                logger.info(
                    "Explicit visual request promoted visual decision from text_only to %s; target block ids=%s.",
                    promoted.mode,
                    promoted.target_block_ids,
                )
                return promoted
        logger.info(
            "Visual decision mode=%s; target block ids=%s; reason=%s.",
            guarded.mode,
            guarded.target_block_ids,
            guarded.reason,
        )
        return guarded

    def _generate_visual_evidence(
        self,
        built_context: BuiltContext,
        decision: VisualEvidenceDecision,
    ) -> list[RagVisualEvidence]:
        if decision.mode == "text_only" or self._visual_crop_service is None:
            return []
        refs_by_id = {
            ref.block_id: ref for ref in [*built_context.visual_hints.selected, *built_context.visual_hints.candidates]
        }
        visual_evidence: list[RagVisualEvidence] = []
        for block_id in decision.target_block_ids:
            ref = refs_by_id.get(block_id)
            if ref is None:
                continue
            crop = self._visual_crop_service.get_or_create_crop(ref)
            if crop is None:
                logger.info("Visual crop was not generated for block_id=%s.", block_id)
                continue
            visual_evidence.append(_rag_visual_from_crop(ref, crop))
        return visual_evidence


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


def _build_visual_decision_prompt(built_context: BuiltContext) -> str:
    return "\n\n".join(
        [
            "Return only JSON for whether visual evidence is needed.",
            "Allowed mode values: text_only, show_visual, inspect_visual_and_show.",
            "Use only the provided retrieved visual hints.",
            f"Question: {built_context.query}",
            "Visual hints:",
            str(built_context.visual_hints.to_dict()),
        ]
    )


def _is_explicit_visual_request(query: str) -> bool:
    normalized = query.casefold()
    asks_to_show = any(term in normalized for term in _VISUAL_REQUEST_TERMS)
    names_visual_object = any(term in normalized for term in _VISUAL_OBJECT_TERMS)
    return asks_to_show and names_visual_object


def _visual_decision_for_explicit_request(
    refs: list[VisualEvidenceRef],
    max_targets: int,
) -> VisualEvidenceDecision:
    target_block_ids = [ref.block_id for ref in refs[: max(0, max_targets)]]
    if not target_block_ids:
        return VisualEvidenceDecision.text_only("No visual target blocks are available for the explicit request.")
    return VisualEvidenceDecision(
        mode="show_visual",
        target_block_ids=target_block_ids,
        show_in_sources=True,
        show_in_answer=True,
        needs_multimodal_followup=False,
        reason="The user explicitly asked to show visual evidence.",
    )


def _citation_from_evidence(
    evidence: ContextEvidence,
    visual_by_block_id: dict[str, RagVisualEvidence],
) -> RagCitation:
    result = evidence.block
    visual = visual_by_block_id.get(result.block_id)
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
        block_type=result.block_type,
        label=result.label,
        has_visual_evidence=result.payload.get("has_visual_evidence") is True,
        visual_evidence=visual,
    )


def _retrieved_chunk_from_evidence(
    evidence: ContextEvidence,
    visual_by_block_id: dict[str, RagVisualEvidence],
) -> RagRetrievedChunk:
    result = evidence.block
    visual = visual_by_block_id.get(result.block_id)
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
        block_type=result.block_type,
        label=result.label,
        has_visual_evidence=result.payload.get("has_visual_evidence") is True,
        visual_evidence=visual,
    )


def _rag_visual_from_crop(ref: VisualEvidenceRef, crop: GeneratedCrop) -> RagVisualEvidence:
    return RagVisualEvidence(
        block_id=ref.block_id,
        document_id=ref.document_id,
        source_file=ref.source_file,
        page_number=ref.page_number,
        block_type=ref.block_type,
        label=ref.label,
        crop_path=crop.file_path,
        crop_url=crop.url_path,
        width=crop.width,
        height=crop.height,
        format=crop.format,
        dpi=crop.dpi,
    )


def _with_context_info(
    info: dict[str, Any] | None,
    built_context: BuiltContext,
    visual_decision: VisualEvidenceDecision,
    visual_evidence: list[RagVisualEvidence],
) -> dict[str, Any]:
    merged = dict(info or {})
    merged["context"] = built_context.stats.to_dict()
    merged["visual"] = {
        "hints": built_context.visual_hints.to_dict(),
        "decision": visual_decision.to_dict(),
        "generated_count": len(visual_evidence),
    }
    return merged
