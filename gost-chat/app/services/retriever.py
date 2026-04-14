import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.retrieval_types import RetrievedBlock

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
DOCUMENTS_FILE = "documents.json"
CHUNKS_FILE = "chunks.jsonl"
SUMMARY_FILE = "indexing_summary.json"


class RetrievalError(RuntimeError):
    """Base error for retrieval failures."""


class IndexNotFoundError(RetrievalError):
    """Raised when the indexer output files are missing."""


class IndexLoadError(RetrievalError):
    """Raised when the indexer output cannot be parsed."""


class EmptyQueryError(ValueError):
    """Raised when a retrieval query is empty."""


@dataclass(frozen=True)
class IndexedChunk:
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    text: str
    token_counts: Counter[str]
    token_total: int


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    text: str
    score: float


@dataclass(frozen=True)
class LoadedIndex:
    chunks: list[IndexedChunk]
    documents: dict[str, dict[str, Any]]
    summary: dict[str, Any] | None
    document_frequency: Counter[str]
    fingerprint: tuple[tuple[str, float | None, int | None], ...]


class Retriever:
    def __init__(self, indexer_output_dir: Path) -> None:
        self._output_dir = indexer_output_dir
        self._loaded_index: LoadedIndex | None = None

    @property
    def chunks_path(self) -> Path:
        return self._output_dir / "index" / CHUNKS_FILE

    @property
    def documents_path(self) -> Path:
        return self._output_dir / "metadata" / DOCUMENTS_FILE

    @property
    def summary_path(self) -> Path:
        return self._output_dir / "metadata" / SUMMARY_FILE

    def search(self, query: str, top_k: int = 5) -> tuple[list[RetrievalResult], dict[str, Any] | None]:
        normalized_query = query.strip()
        if not normalized_query:
            raise EmptyQueryError("Search query cannot be empty.")

        query_terms = _tokenize(normalized_query)
        if not query_terms:
            return [], self._load_index().summary

        index = self._load_index()
        query_counts = Counter(query_terms)
        query_unique_terms = set(query_counts)
        scored_results: list[RetrievalResult] = []

        for chunk in index.chunks:
            score = self._score_chunk(
                chunk=chunk,
                query=normalized_query,
                query_counts=query_counts,
                query_unique_terms=query_unique_terms,
                chunk_count=len(index.chunks),
                document_frequency=index.document_frequency,
            )
            if score <= 0:
                continue

            scored_results.append(
                RetrievalResult(
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    chunk_id=chunk.chunk_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    score=round(score, 6),
                )
            )

        scored_results.sort(key=lambda result: (-result.score, result.file_name, result.chunk_id))
        return scored_results[:top_k], index.summary

    def retrieve_blocks(self, query: str, top_k: int = 5) -> tuple[list[RetrievedBlock], dict[str, Any] | None]:
        results, index_summary = self.search(query, top_k=top_k)
        return [retrieved_block_from_result(result) for result in results], index_summary

    def _load_index(self) -> LoadedIndex:
        fingerprint = self._fingerprint()
        if self._loaded_index and self._loaded_index.fingerprint == fingerprint:
            return self._loaded_index

        if not self.chunks_path.exists() or not self.documents_path.exists():
            raise IndexNotFoundError(
                f"Indexer output is missing. Expected {self.chunks_path} and {self.documents_path}."
            )

        documents = self._load_documents()
        summary = self._load_summary()
        chunks = self._load_chunks(documents)
        document_frequency: Counter[str] = Counter()

        for chunk in chunks:
            document_frequency.update(chunk.token_counts.keys())

        loaded_index = LoadedIndex(
            chunks=chunks,
            documents=documents,
            summary=summary,
            document_frequency=document_frequency,
            fingerprint=fingerprint,
        )
        self._loaded_index = loaded_index
        logger.info("Loaded retrieval index with %s chunks from %s.", len(chunks), self.chunks_path)
        return loaded_index

    def _load_documents(self) -> dict[str, dict[str, Any]]:
        try:
            with self.documents_path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise IndexLoadError(f"Document metadata is not valid JSON: {self.documents_path}.") from exc
        except OSError as exc:
            raise IndexLoadError(f"Could not read document metadata: {self.documents_path}.") from exc

        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise IndexLoadError("Document metadata must contain a documents list.")

        loaded: dict[str, dict[str, Any]] = {}
        for document in documents:
            if not isinstance(document, dict) or not isinstance(document.get("document_id"), str):
                raise IndexLoadError("Document metadata contains an invalid document record.")
            loaded[document["document_id"]] = document
        return loaded

    def _load_summary(self) -> dict[str, Any] | None:
        if not self.summary_path.exists():
            return None

        try:
            with self.summary_path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise IndexLoadError(f"Indexing summary is not valid JSON: {self.summary_path}.") from exc
        except OSError as exc:
            raise IndexLoadError(f"Could not read indexing summary: {self.summary_path}.") from exc

        if not isinstance(payload, dict):
            raise IndexLoadError("Indexing summary must be a JSON object.")
        return payload

    def _load_chunks(self, documents: dict[str, dict[str, Any]]) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        try:
            with self.chunks_path.open("r", encoding="utf-8-sig") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise IndexLoadError(
                            f"Chunk index contains invalid JSON at line {line_number}: {self.chunks_path}."
                        ) from exc
                    chunks.append(self._chunk_from_payload(payload, documents, line_number))
        except OSError as exc:
            raise IndexLoadError(f"Could not read chunk index: {self.chunks_path}.") from exc
        return chunks

    def _chunk_from_payload(
        self,
        payload: Any,
        documents: dict[str, dict[str, Any]],
        line_number: int,
    ) -> IndexedChunk:
        if not isinstance(payload, dict):
            raise IndexLoadError(f"Chunk record at line {line_number} must be a JSON object.")

        document_id = payload.get("document_id")
        chunk_id = payload.get("chunk_id")
        text = payload.get("text")
        if not isinstance(document_id, str) or not isinstance(chunk_id, str) or not isinstance(text, str):
            raise IndexLoadError(f"Chunk record at line {line_number} is missing required string fields.")

        document = documents.get(document_id, {})
        file_name = payload.get("file_name") or document.get("filename")
        if not isinstance(file_name, str):
            file_name = "unknown"

        tokens = _tokenize(text)
        return IndexedChunk(
            document_id=document_id,
            file_name=file_name,
            chunk_id=chunk_id,
            page_start=_optional_int(payload.get("page_start")),
            page_end=_optional_int(payload.get("page_end")),
            text=text,
            token_counts=Counter(tokens),
            token_total=len(tokens),
        )

    def _fingerprint(self) -> tuple[tuple[str, float | None, int | None], ...]:
        paths = (self.chunks_path, self.documents_path, self.summary_path)
        fingerprint = []
        for path in paths:
            try:
                stat = path.stat()
            except FileNotFoundError:
                fingerprint.append((str(path), None, None))
            else:
                fingerprint.append((str(path), stat.st_mtime, stat.st_size))
        return tuple(fingerprint)

    def _score_chunk(
        self,
        chunk: IndexedChunk,
        query: str,
        query_counts: Counter[str],
        query_unique_terms: set[str],
        chunk_count: int,
        document_frequency: Counter[str],
    ) -> float:
        if not chunk.token_counts or chunk.token_total <= 0:
            return 0.0

        score = 0.0
        matched_terms = 0
        for term, query_count in query_counts.items():
            term_frequency = chunk.token_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            matched_terms += 1
            inverse_document_frequency = math.log((chunk_count + 1) / (document_frequency[term] + 1)) + 1
            normalized_frequency = term_frequency / chunk.token_total
            score += normalized_frequency * inverse_document_frequency * (1 + math.log(query_count))

        if matched_terms == 0:
            return 0.0

        coverage = matched_terms / len(query_unique_terms)
        score *= 1 + coverage

        if query.casefold() in chunk.text.casefold():
            score += 0.25

        return score


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def retrieved_block_from_result(result: RetrievalResult) -> RetrievedBlock:
    payload = {
        "document_id": result.document_id,
        "file_name": result.file_name,
        "chunk_id": result.chunk_id,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "text": result.text,
    }
    return RetrievedBlock(
        block_id=result.chunk_id,
        text=result.text,
        retrieval_text=result.text,
        source_file=result.file_name,
        page=result.page_start,
        section_path=[],
        retrieval_score=result.score,
        payload=payload,
        document_id=result.document_id,
        page_start=result.page_start,
        page_end=result.page_end,
    )
