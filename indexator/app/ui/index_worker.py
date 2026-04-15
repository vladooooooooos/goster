"""Background worker for long-running UI actions."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.core.block_builder import StructuredBlockBuilder
from app.core.pipeline import IndexingPipeline, IndexingProgress
from app.parsing.pdf_parser import PdfParser
from app.services.deletion_service import IndexDeletionService
from app.services.embedding_service import StructuredBlockEmbeddingService
from app.services.pdf_scanner import PdfScanner
from app.services.reindex_service import ReindexService
from app.storage.qdrant_store import QdrantStore
from app.utils.debug_export import (
    export_blocks_jsonl,
    export_embedding_summary,
    export_indexing_summary,
    export_qdrant_storage_summary,
)


class IndexWorker(QObject):
    """Run heavy document and storage work away from the UI thread."""

    progress_changed = Signal(object)
    log_message = Signal(str)
    file_finished = Signal(str)
    finished = Signal(str, object, object)
    error = Signal(str, str, str)

    def __init__(
        self,
        mode: str,
        pdf_paths: list[Path] | None,
        document_ids: list[str] | None,
        scan_folder: Path | None,
        pdf_scanner: PdfScanner,
        parser: PdfParser,
        block_builder: StructuredBlockBuilder,
        embedding_service: StructuredBlockEmbeddingService,
        qdrant_store: QdrantStore,
        indexing_pipeline: IndexingPipeline,
        reindex_service: ReindexService,
        deletion_service: IndexDeletionService,
        output_dir: Path,
        embedding_device: str,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.pdf_paths = pdf_paths or []
        self.document_ids = document_ids or []
        self.scan_folder = scan_folder
        self.pdf_scanner = pdf_scanner
        self.parser = parser
        self.block_builder = block_builder
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.indexing_pipeline = indexing_pipeline
        self.reindex_service = reindex_service
        self.deletion_service = deletion_service
        self.output_dir = output_dir
        self.embedding_device = embedding_device

    @Slot()
    def run(self) -> None:
        """Execute the selected heavy operation and report results with signals."""
        try:
            if self.mode == "scan":
                result = self._run_scan()
            elif self.mode == "index":
                result = self._run_index()
            elif self.mode == "reindex":
                result = self._run_reindex()
            elif self.mode == "preview":
                result = self._run_preview()
            elif self.mode == "embed_preview":
                result = self._run_embed_preview()
            elif self.mode == "store_preview":
                result = self._run_store_preview()
            elif self.mode == "clear_selected":
                result = self._run_clear_selected()
            elif self.mode == "clear_all":
                result = self._run_clear_all()
            else:
                raise ValueError(f"Unsupported indexing worker mode: {self.mode}")

            self.finished.emit(self.mode, result, None)
        except Exception as error:
            for pdf_path in self.pdf_paths:
                self._report(pdf_path.name, 1, max(1, len(self.pdf_paths)), "failed")
            self.error.emit(self.mode, str(error), traceback.format_exc())

    def _run_index(self) -> dict[str, object]:
        summary = self.indexing_pipeline.index_pdfs(
            self.pdf_paths,
            progress_callback=self._handle_progress,
        )
        summary_path = export_indexing_summary(summary, self.output_dir / "last_indexing_summary.json")
        return {"summary": summary, "summary_path": summary_path}

    def _run_scan(self) -> dict[str, object]:
        if self.scan_folder is None:
            raise ValueError("Scan mode requires a folder path.")
        self._report(str(self.scan_folder), 1, 1, "scan")
        results = self.pdf_scanner.scan(self.scan_folder)
        self._report(str(self.scan_folder), 1, 1, "done")
        return {"scan_folder": self.scan_folder, "scan_results": results}

    def _run_reindex(self) -> dict[str, object]:
        self.log_message.emit(f"Reindex embedding device: {self.embedding_device}")
        summary = self.reindex_service.reindex_pdfs(
            self.pdf_paths,
            progress_callback=self._handle_progress,
        )
        summary_path = self.reindex_service.export_summary(summary, "last_reindex_summary.json")
        return {"summary": summary, "summary_path": summary_path}

    def _run_preview(self) -> dict[str, object]:
        pdf_path = self._single_pdf_path()
        self._report(pdf_path.name, 1, 1, "parse")
        parsed_document = self.parser.parse(pdf_path)
        self._report(pdf_path.name, 1, 1, "build_blocks")
        structured_blocks = self.block_builder.build(parsed_document)
        debug_path = export_blocks_jsonl(
            structured_blocks,
            self.output_dir / f"{pdf_path.stem}_structured_blocks.jsonl",
        )
        self._report(pdf_path.name, 1, 1, "done")
        return {
            "parsed_document": parsed_document,
            "structured_blocks": structured_blocks,
            "debug_path": debug_path,
        }

    def _run_embed_preview(self) -> dict[str, object]:
        pdf_path = self._single_pdf_path()
        self._report(pdf_path.name, 1, 1, "parse")
        parsed_document = self.parser.parse(pdf_path)
        self._report(pdf_path.name, 1, 1, "build_blocks")
        structured_blocks = self.block_builder.build(parsed_document)
        self._report(pdf_path.name, 1, 1, "embed")
        embedding_run = self.embedding_service.embed_blocks(structured_blocks)
        debug_path = export_embedding_summary(
            embedding_run,
            self.output_dir / f"{pdf_path.stem}_embedding_summary.json",
        )
        self._report(pdf_path.name, 1, 1, "done")
        return {"embedding_run": embedding_run, "debug_path": debug_path}

    def _run_store_preview(self) -> dict[str, object]:
        pdf_path = self._single_pdf_path()
        self._report(pdf_path.name, 1, 1, "parse")
        parsed_document = self.parser.parse(pdf_path)
        self._report(pdf_path.name, 1, 1, "build_blocks")
        structured_blocks = self.block_builder.build(parsed_document)
        self._report(pdf_path.name, 1, 1, "embed")
        embedding_run = self.embedding_service.embed_blocks(structured_blocks)
        self._report(pdf_path.name, 1, 1, "store")
        storage_run = self.qdrant_store.upsert_block_embeddings(structured_blocks, embedding_run)
        debug_path = export_qdrant_storage_summary(
            storage_run,
            self.output_dir / f"{pdf_path.stem}_qdrant_storage_summary.json",
        )
        self._report(pdf_path.name, 1, 1, "done")
        return {"pdf_path": pdf_path, "storage_run": storage_run, "debug_path": debug_path}

    def _run_clear_selected(self) -> dict[str, object]:
        self._report("selected documents", 1, 1, "clear")
        summary = self.deletion_service.clear_selected(self.document_ids)
        summary_path = self.deletion_service.export_summary(summary, "last_clear_selected_summary.json")
        self._report("selected documents", 1, 1, "done")
        return {"summary": summary, "summary_path": summary_path}

    def _run_clear_all(self) -> dict[str, object]:
        self._report("shared index data", 1, 1, "clear")
        summary = self.deletion_service.clear_all()
        summary_path = self.deletion_service.export_summary(summary, "last_clear_all_summary.json")
        self._report("shared index data", 1, 1, "done")
        return {"summary": summary, "summary_path": summary_path}

    def _single_pdf_path(self) -> Path:
        if len(self.pdf_paths) != 1:
            raise ValueError(f"{self.mode} expects exactly one PDF path.")
        return self.pdf_paths[0]

    def _handle_progress(self, progress: IndexingProgress) -> None:
        self.progress_changed.emit(progress)
        if progress.stage in {"done", "failed"}:
            self.file_finished.emit(progress.file_name)

    def _report(self, file_name: str, current_file: int, total_files: int, stage: str) -> None:
        self._handle_progress(
            IndexingProgress(
                file_name=file_name,
                current_file=current_file,
                total_files=total_files,
                stage=stage,
            )
        )
