from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from indexed documents.")
    top_k: int = Field(12, ge=1, le=50, description="Maximum number of source chunks to retrieve.")


class AskVisualEvidence(BaseModel):
    block_id: str
    document_id: str
    source_file: str
    page_number: int
    block_type: str | None = None
    label: str | None = None
    crop_path: str
    crop_url: str
    width: int
    height: int
    format: str
    dpi: int
    selection_reason: str | None = None
    confidence: float | None = None
    source_block_id: str | None = None
    crop_kind: str | None = None


class AskCitation(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
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
    visual_evidence: AskVisualEvidence | None = None


class AskRetrievedChunk(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
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
    visual_evidence: AskVisualEvidence | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[AskCitation]
    retrieved_results_count: int
    retrieval_used: bool
    retrieved_chunks: list[AskRetrievedChunk]
    retrieval_info: dict[str, object] | None = None
    visual_evidence: list[AskVisualEvidence] = Field(default_factory=list)
