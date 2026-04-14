from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Plain text query to search in indexed chunks.")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of source chunks to return.")


class SearchResult(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]
    total_results: int
    index_summary: dict[str, object] | None = None

