"""End-to-end local indexing pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.core.block_builder import StructuredBlockBuilder, make_document_id
from app.parsing.pdf_parser import PdfParser
from app.services.embedding_service import StructuredBlockEmbeddingService
from app.services.file_fingerprint import FileFingerprintService
from app.storage.document_registry import DocumentRegistry, IndexedDocumentRecord
from app.storage.qdrant_store import QdrantStore


ProgressCallback = Callable[["IndexingProgress"], None]


@dataclass(frozen=True)
class IndexedFileResult:
    """Per-file result from one indexing pipeline run."""

    file_name: str
    file_path: Path
    document_id: str | None
    indexed_at: str | None
    page_count: int | None
    structured_blocks: int
    embedded_blocks: int
    stored_points: int
    elapsed_seconds: float
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class IndexingRunSummary:
    """Summary for a multi-file indexing run."""

    selected_files: int
    successful_files: int
    failed_files: int
    total_structured_blocks: int
    total_stored_points: int
    elapsed_seconds: float
    results: list[IndexedFileResult]


@dataclass(frozen=True)
class IndexingProgress:
    """Simple progress event for UI logging and progress bars."""

    file_name: str
    current_file: int
    total_files: int
    stage: str


class IndexingPipeline:
    """Run parse, block building, embedding, and Qdrant server storage for PDFs."""

    def __init__(
        self,
        parser: PdfParser,
        block_builder: StructuredBlockBuilder,
        embedding_service: StructuredBlockEmbeddingService,
        qdrant_store: QdrantStore,
        document_registry: DocumentRegistry | None = None,
        fingerprint_service: FileFingerprintService | None = None,
    ) -> None:
        self.parser = parser
        self.block_builder = block_builder
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.document_registry = document_registry
        self.fingerprint_service = fingerprint_service or FileFingerprintService()

    def index_pdfs(
        self,
        pdf_paths: list[Path],
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingRunSummary:
        """Index selected PDFs into Qdrant server."""
        start_time = perf_counter()
        results: list[IndexedFileResult] = []
        total_files = len(pdf_paths)

        for file_index, pdf_path in enumerate(pdf_paths, start=1):
            self._report(progress_callback, pdf_path.name, file_index, total_files, "parse")
            results.append(self._index_one_pdf(pdf_path, file_index, total_files, progress_callback))

        elapsed_seconds = perf_counter() - start_time
        successful_files = sum(1 for result in results if result.success)
        failed_files = total_files - successful_files

        return IndexingRunSummary(
            selected_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_structured_blocks=sum(result.structured_blocks for result in results),
            total_stored_points=sum(result.stored_points for result in results),
            elapsed_seconds=elapsed_seconds,
            results=results,
        )

    def _index_one_pdf(
        self,
        pdf_path: Path,
        file_index: int,
        total_files: int,
        progress_callback: ProgressCallback | None,
    ) -> IndexedFileResult:
        file_start_time = perf_counter()
        page_count: int | None = None

        try:
            parsed_document = self.parser.parse(pdf_path)
            page_count = parsed_document.page_count
            document_id = make_document_id(parsed_document.file_path)
            indexed_at = utc_now_iso()
            fingerprint = self.fingerprint_service.get_content_fingerprint(parsed_document.file_path)

            self._report(progress_callback, pdf_path.name, file_index, total_files, "build_blocks")
            structured_blocks = self.block_builder.build(parsed_document)

            self._report(progress_callback, pdf_path.name, file_index, total_files, "embed")
            embedding_run = self.embedding_service.embed_blocks(structured_blocks)

            self._report(progress_callback, pdf_path.name, file_index, total_files, "store")
            storage_run = self.qdrant_store.upsert_block_embeddings(
                structured_blocks,
                embedding_run,
                indexed_at=indexed_at,
            )
            if self.document_registry is not None and document_id is not None:
                self.document_registry.register_document(
                    IndexedDocumentRecord(
                        document_id=document_id,
                        doc_id=document_id,
                        source_path=str(parsed_document.file_path),
                        file_name=parsed_document.file_name,
                        indexed_at=indexed_at,
                        stored_points=storage_run.stored_blocks,
                        file_size=fingerprint.file_size,
                        modified_at=fingerprint.modified_at,
                        source_fingerprint=fingerprint.source_fingerprint or "",
                        status="indexed",
                    )
                )

            self._report(progress_callback, pdf_path.name, file_index, total_files, "done")
            return IndexedFileResult(
                file_name=pdf_path.name,
                file_path=pdf_path,
                document_id=document_id,
                indexed_at=indexed_at,
                page_count=page_count,
                structured_blocks=len(structured_blocks),
                embedded_blocks=len(embedding_run.embeddings),
                stored_points=storage_run.stored_blocks,
                elapsed_seconds=perf_counter() - file_start_time,
                success=True,
            )
        except Exception as error:
            self._report(progress_callback, pdf_path.name, file_index, total_files, "failed")
            return IndexedFileResult(
                file_name=pdf_path.name,
                file_path=pdf_path,
                document_id=None,
                indexed_at=None,
                page_count=page_count,
                structured_blocks=0,
                embedded_blocks=0,
                stored_points=0,
                elapsed_seconds=perf_counter() - file_start_time,
                success=False,
                error_message=str(error),
            )

    def _report(
        self,
        progress_callback: ProgressCallback | None,
        file_name: str,
        current_file: int,
        total_files: int,
        stage: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            IndexingProgress(
                file_name=file_name,
                current_file=current_file,
                total_files=total_files,
                stage=stage,
            )
        )


def utc_now_iso() -> str:
    """Return the current UTC timestamp for indexing metadata."""
    return datetime.now(UTC).isoformat()
