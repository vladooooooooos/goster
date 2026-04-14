"""Main window for the Indexator desktop shell."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.block_builder import StructuredBlockBuilder, make_document_id
from app.core.blocks import StructuredBlock
from app.core.pipeline import IndexedFileResult, IndexingPipeline, IndexingProgress, IndexingRunSummary
from app.parsing.pdf_parser import ParsedDocument, PdfParser
from app.services.deletion_service import ClearOperationSummary, IndexDeletionService
from app.services.embedding_service import StructuredBlockEmbeddingService
from app.services.file_fingerprint import FileFingerprintService
from app.services.indexed_state import INDEXED, INDEXED_STALE, INDEX_ERROR, MISSING_SOURCE, READY, IndexedStateResolver
from app.services.pdf_scanner import PdfScanner, PdfScanResult
from app.services.reindex_service import ReindexRunSummary, ReindexService
from app.storage.document_registry import DocumentRegistry
from app.storage.qdrant_store import QdrantStore
from app.utils.config import AppConfig
from app.utils.debug_export import (
    export_blocks_jsonl,
    export_embedding_summary,
    export_indexing_summary,
    export_qdrant_storage_summary,
)


class MainWindow(QMainWindow):
    """Minimal MVP shell for folder selection, scanning, logging, and indexing actions."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.pdf_scanner = PdfScanner()
        self.pdf_parser = PdfParser()
        self.block_builder = StructuredBlockBuilder()
        self.fingerprint_service = FileFingerprintService()
        self.indexed_state_resolver = IndexedStateResolver(self.fingerprint_service)
        self.embedding_service = StructuredBlockEmbeddingService.from_config(config.embedding)
        self.app_root = Path(__file__).resolve().parents[2]
        self.qdrant_store = QdrantStore.from_config(config.storage, self.app_root)
        self.document_registry = DocumentRegistry(self.qdrant_store.local_path.parent)
        self.deletion_service = IndexDeletionService(self.qdrant_store, self.document_registry)
        self.indexing_pipeline = IndexingPipeline(
            parser=self.pdf_parser,
            block_builder=self.block_builder,
            embedding_service=self.embedding_service,
            qdrant_store=self.qdrant_store,
            document_registry=self.document_registry,
            fingerprint_service=self.fingerprint_service,
        )
        self.reindex_service = ReindexService(
            self.indexing_pipeline,
            self.deletion_service,
            self.document_registry,
            self.fingerprint_service,
        )
        self.output_dir = self.app_root / "output"
        self.current_scan_folder: Path | None = None
        self.setWindowTitle(config.app.name)
        self.resize(config.ui.window_width, config.ui.window_height)

        self.folder_path_field = QLineEdit()
        self.folder_path_field.setReadOnly(True)
        self.folder_path_field.setPlaceholderText("Select a folder with PDF files")

        self.select_folder_button = QPushButton("Select Folder")
        self.scan_button = QPushButton("Scan PDFs")
        self.preview_button = QPushButton("Preview blocks")
        self.embed_preview_button = QPushButton("Embed preview")
        self.store_preview_button = QPushButton("Store preview")
        self.index_button = QPushButton("Index selected")
        self.reindex_button = QPushButton("Reindex selected")
        self.clear_selected_button = QPushButton("Clear selected")
        self.clear_all_button = QPushButton("Clear all")

        self.pdf_table = QTableWidget(0, 6)
        self.log_panel = QPlainTextEdit()
        self.progress_bar = QProgressBar()

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()
        self._append_log("Indexator shell started.")
        self._append_log(f"Python runtime: {sys.executable}")
        self._append_log(f"Embedding device: {self.embedding_service.embedder.describe_device_runtime()}")

    def _configure_widgets(self) -> None:
        self.pdf_table.setHorizontalHeaderLabels(["Selected", "File name", "Path", "Size", "Pages", "Status"])
        self.pdf_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.pdf_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.pdf_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.pdf_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.pdf_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.pdf_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.pdf_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pdf_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Logs will appear here")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.reindex_button.setEnabled(False)
        self.clear_selected_button.setEnabled(False)

    def _build_layout(self) -> None:
        root = QWidget(self)
        main_layout = QVBoxLayout(root)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("PDF folder:"))
        folder_layout.addWidget(self.folder_path_field, stretch=1)
        folder_layout.addWidget(self.select_folder_button)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.scan_button)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.embed_preview_button)
        action_layout.addWidget(self.store_preview_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.reindex_button)
        action_layout.addWidget(self.clear_selected_button)
        action_layout.addWidget(self.clear_all_button)
        action_layout.addWidget(self.index_button)

        main_layout.addLayout(folder_layout)
        main_layout.addLayout(action_layout)
        main_layout.addWidget(self.pdf_table, stretch=3)
        main_layout.addWidget(QLabel("Logs:"))
        main_layout.addWidget(self.log_panel, stretch=1)
        main_layout.addWidget(self.progress_bar)

        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.select_folder_button.clicked.connect(self._select_folder)
        self.scan_button.clicked.connect(self._scan_pdfs)
        self.preview_button.clicked.connect(self._preview_selected_pdf)
        self.embed_preview_button.clicked.connect(self._embed_selected_pdf_preview)
        self.store_preview_button.clicked.connect(self._store_selected_pdf_preview)
        self.index_button.clicked.connect(self._index_selected)
        self.reindex_button.clicked.connect(self._reindex_selected)
        self.clear_selected_button.clicked.connect(self._clear_selected)
        self.clear_all_button.clicked.connect(self._clear_all)
        self.pdf_table.itemChanged.connect(self._update_clear_selected_enabled)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Release local resources before the desktop window closes."""
        self.qdrant_store.close()
        super().closeEvent(event)

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select PDF folder")
        if not folder:
            return

        self.folder_path_field.setText(folder)
        self._append_log(f"Selected folder: {folder}")

    def _scan_pdfs(self) -> None:
        folder = self.folder_path_field.text().strip()
        if not folder:
            self._append_log("Select a folder before scanning.")
            return

        self.pdf_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.current_scan_folder = Path(folder)
        self._append_log(f"Scanning PDF files in: {folder}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            results = self.pdf_scanner.scan(Path(folder))
        except OSError as error:
            self._append_log(f"Scan failed: {error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)

        self._populate_pdf_table(results)
        self.progress_bar.setValue(100)

        unreadable_count = sum(1 for result in results if result.status != "Ready")
        self._append_log(f"Found {len(results)} PDF file(s).")
        if unreadable_count:
            self._append_log(f"{unreadable_count} PDF file(s) could not be read.")

    def _index_selected(self) -> None:
        selected_rows = self._checked_pdf_rows()
        if not selected_rows:
            self._append_log("No PDF files selected for indexing.")
            return

        pdf_paths = self._checked_pdf_paths_with_statuses({READY})
        skipped_count = len(selected_rows) - len(pdf_paths)
        if skipped_count:
            self._append_log(f"Skipped {skipped_count} selected file(s) that are not ready.")
        if not pdf_paths:
            self._append_log("No Ready PDF files selected for indexing.")
            return

        self.progress_bar.setValue(0)
        self._append_log(f"Indexing selected PDF file(s): {len(pdf_paths)}.")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            summary = self.indexing_pipeline.index_pdfs(
                pdf_paths,
                progress_callback=self._handle_indexing_progress,
            )
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)
            self.progress_bar.setValue(100)

        self._append_indexing_summary(summary)
        debug_path = export_indexing_summary(summary, self.output_dir / "last_indexing_summary.json")
        self._append_log(f"Indexing summary export: {debug_path}")
        self._refresh_indexed_statuses()

    def _reindex_selected(self) -> None:
        selected_rows = self._checked_pdf_rows()
        if not selected_rows:
            self._append_log("No PDF files selected for reindexing.")
            return

        pdf_paths = self._checked_pdf_paths_with_statuses({READY, INDEXED, INDEXED_STALE})
        skipped_count = len(selected_rows) - len(pdf_paths)
        if skipped_count:
            self._append_log(f"Skipped {skipped_count} selected file(s) that are not available for reindexing.")
        if not pdf_paths:
            self._append_log("No available checked PDF files selected for reindexing.")
            return

        answer = QMessageBox.question(
            self,
            "Reindex selected documents",
            (
                f"Force reindex {len(pdf_paths)} checked document(s)?\n\n"
                "Existing indexed data for those documents will be replaced. Source PDFs will not be deleted."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._append_log("Reindex selected cancelled.")
            return

        self.progress_bar.setValue(0)
        self._append_log(f"Reindexing selected PDF file(s): {len(pdf_paths)}.")
        self._append_log(f"Reindex embedding device: {self.embedding_service.embedder.describe_device_runtime()}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            summary = self.reindex_service.reindex_pdfs(pdf_paths)
            summary_path = self.reindex_service.export_summary(summary, "last_reindex_summary.json")
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)
            self.progress_bar.setValue(100)

        self._append_reindex_summary(summary)
        self._append_log(f"Reindex summary export: {summary_path}")
        self._refresh_indexed_statuses()

    def _preview_selected_pdf(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log("Select one ready PDF file before previewing blocks.")
            return

        self._append_log(f"Parsing preview for: {pdf_path.name}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            parsed_document = self.pdf_parser.parse(pdf_path)
            structured_blocks = self.block_builder.build(parsed_document)
        except Exception as error:
            self._append_log(f"Parse failed for {pdf_path.name}: {error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)

        self._append_parsed_preview(parsed_document)
        self._append_structured_block_preview(structured_blocks)
        debug_path = export_blocks_jsonl(
            structured_blocks,
            self.output_dir / f"{pdf_path.stem}_structured_blocks.jsonl",
        )
        self._append_log(f"Structured block debug export: {debug_path}")

    def _embed_selected_pdf_preview(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log("Select one ready PDF file before embedding blocks.")
            return

        self._append_log(f"Embedding structured block preview for: {pdf_path.name}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            parsed_document = self.pdf_parser.parse(pdf_path)
            structured_blocks = self.block_builder.build(parsed_document)
            embedding_run = self.embedding_service.embed_blocks(structured_blocks)
        except Exception as error:
            self._append_log(f"Embedding preview failed for {pdf_path.name}: {error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)

        self._append_log(
            f"Embedded {len(embedding_run.embeddings)} block(s) with {embedding_run.model_name}: "
            f"dimension={embedding_run.embedding_dimension}, device={embedding_run.device}, "
            f"elapsed={embedding_run.elapsed_seconds:.2f}s."
        )
        debug_path = export_embedding_summary(
            embedding_run,
            self.output_dir / f"{pdf_path.stem}_embedding_summary.json",
        )
        self._append_log(f"Embedding debug summary export: {debug_path}")

    def _store_selected_pdf_preview(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log("Select one ready PDF file before storing blocks.")
            return

        self._append_log(f"Storing structured block preview in local Qdrant for: {pdf_path.name}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            parsed_document = self.pdf_parser.parse(pdf_path)
            structured_blocks = self.block_builder.build(parsed_document)
            embedding_run = self.embedding_service.embed_blocks(structured_blocks)
            storage_run = self.qdrant_store.upsert_block_embeddings(structured_blocks, embedding_run)
        except Exception as error:
            self._append_log(f"Local Qdrant storage preview failed for {pdf_path.name}: {error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)

        self._append_log(
            f"Stored {storage_run.stored_blocks} block(s) from {pdf_path.name} in local Qdrant: "
            f"collection={storage_run.collection_name}, dimension={storage_run.embedding_dimension}, "
            f"path={storage_run.local_path}, elapsed={storage_run.elapsed_seconds:.2f}s."
        )
        debug_path = export_qdrant_storage_summary(
            storage_run,
            self.output_dir / f"{pdf_path.stem}_qdrant_storage_summary.json",
        )
        self._append_log(f"Local Qdrant storage summary export: {debug_path}")

    def _append_log(self, message: str) -> None:
        self.log_panel.appendPlainText(message)
        self.log_panel.verticalScrollBar().setValue(self.log_panel.verticalScrollBar().maximum())

    def _populate_pdf_table(self, results: list[PdfScanResult]) -> None:
        states = self.indexed_state_resolver.resolve_scan_results(
            scan_results=results,
            records=self.document_registry.load_documents(),
            scanned_folder=self.current_scan_folder,
        )
        self.pdf_table.blockSignals(True)
        self.pdf_table.setRowCount(len(states))

        try:
            for row, state in enumerate(states):
                selected_item = make_checkbox_item(checked=state.status in {READY, INDEXED_STALE})
                file_name_item = make_table_item(state.file_name)
                path_item = make_table_item(str(state.file_path))
                size_item = make_table_item(format_file_size(state.file_size_bytes))
                pages_item = make_table_item(str(state.page_count) if state.page_count is not None else "-")
                status_item = make_table_item(state.status)

                if state.error_message:
                    status_item.setToolTip(state.error_message)
                    self._append_log(f"{state.status} for {state.file_name}: {state.error_message}")

                self.pdf_table.setItem(row, 0, selected_item)
                self.pdf_table.setItem(row, 1, file_name_item)
                self.pdf_table.setItem(row, 2, path_item)
                self.pdf_table.setItem(row, 3, size_item)
                self.pdf_table.setItem(row, 4, pages_item)
                self.pdf_table.setItem(row, 5, status_item)
        finally:
            self.pdf_table.blockSignals(False)
        self._update_action_buttons_enabled()

    def _set_scan_controls_enabled(self, enabled: bool) -> None:
        self.select_folder_button.setEnabled(enabled)
        self.scan_button.setEnabled(enabled)
        self.preview_button.setEnabled(enabled)
        self.embed_preview_button.setEnabled(enabled)
        self.store_preview_button.setEnabled(enabled)
        self.index_button.setEnabled(enabled)
        self.reindex_button.setEnabled(enabled)
        self.clear_all_button.setEnabled(enabled)
        self._update_action_buttons_enabled()

    def _checked_pdf_rows(self) -> list[int]:
        rows: list[int] = []
        for row in range(self.pdf_table.rowCount()):
            selected_item = self.pdf_table.item(row, 0)
            if selected_item and selected_item.checkState() == Qt.CheckState.Checked:
                rows.append(row)
        return rows

    def _ready_pdf_paths_from_rows(self, rows: list[int]) -> list[Path]:
        paths: list[Path] = []
        for row in rows:
            status_item = self.pdf_table.item(row, 5)
            path_item = self.pdf_table.item(row, 2)
            if status_item and path_item and status_item.text() in {READY, INDEXED, INDEXED_STALE}:
                paths.append(Path(path_item.text()))
        return paths

    def _checked_pdf_paths_with_statuses(self, statuses: set[str]) -> list[Path]:
        paths: list[Path] = []
        for row in self._checked_pdf_rows():
            status_item = self.pdf_table.item(row, 5)
            path_item = self.pdf_table.item(row, 2)
            if status_item and path_item and status_item.text() in statuses:
                paths.append(Path(path_item.text()))
        return paths

    def _first_selected_pdf_path(self) -> Path | None:
        checked_rows = self._checked_pdf_rows()
        selected_rows = [index.row() for index in self.pdf_table.selectionModel().selectedRows()]

        for row in checked_rows or selected_rows:
            status_item = self.pdf_table.item(row, 5)
            path_item = self.pdf_table.item(row, 2)
            if status_item and path_item and status_item.text() in {READY, INDEXED, INDEXED_STALE}:
                return Path(path_item.text())

        return None

    def _append_parsed_preview(self, parsed_document: ParsedDocument) -> None:
        total_blocks = sum(len(page.text_blocks) for page in parsed_document.pages)
        landscape_pages = [page.page_number for page in parsed_document.pages if page.width > page.height]
        rotated_pages = [page.page_number for page in parsed_document.pages if page.rotation]
        self._append_log(
            f"Parsed {parsed_document.file_name}: {parsed_document.page_count} page(s), "
            f"{total_blocks} text block(s)."
        )
        if landscape_pages or rotated_pages:
            self._append_log(
                f"Layout notes: landscape_pages={landscape_pages[:20]} "
                f"rotated_pages={rotated_pages[:20]}."
            )

        for page in parsed_document.pages[:2]:
            self._append_log(
                f"Page {page.page_number}: {len(page.text)} text character(s), "
                f"{len(page.text_blocks)} text block(s), "
                f"size={page.width:.1f}x{page.height:.1f}, rotation={page.rotation}."
            )
            for block in page.text_blocks[:3]:
                preview_text = " ".join(block.text.split())[:180]
                self._append_log(
                    f"  Block {block.order_index}, bbox={format_bbox(block.bbox)}: {preview_text}"
                )

    def _append_structured_block_preview(self, blocks: list[StructuredBlock]) -> None:
        self._append_log(f"Built {len(blocks)} structured block(s).")
        counts = Counter(block.block_type for block in blocks)
        self._append_log(
            "Block counts: "
            f"headings={counts['heading']}, "
            f"paragraphs={counts['paragraph']}, "
            f"list_items={counts['list_item']}, "
            f"appendix_sections={counts['appendix_section']}, "
            f"table_of_contents={counts['table_of_contents']}, "
            f"tables={counts['table']}, "
            f"figures={counts['figure']}, "
            f"formulas={counts['formula_with_context']}."
        )
        refinement_counts = self.block_builder.last_refinement_counts
        if refinement_counts:
            formatted_counts = ", ".join(
                f"{reason}={count}" for reason, count in sorted(refinement_counts.items())
            )
            self._append_log(f"Refinement splits: {formatted_counts}.")
        table_blocks = [block for block in blocks if block.block_type == "table"]
        for table_block in table_blocks[:3]:
            preview_text = " ".join(table_block.text.split())[:220]
            label = table_block.label or "unlabeled"
            self._append_log(
                f"  table example label={label} page={table_block.page_number} "
                f"order={table_block.reading_order}: {preview_text}"
            )
        figure_blocks = [block for block in blocks if block.block_type == "figure"]
        for figure_block in figure_blocks[:5]:
            preview_text = " ".join(figure_block.text.split())[:220]
            label = figure_block.label or "unlabeled"
            self._append_log(
                f"  figure example label={label} page={figure_block.page_number} "
                f"order={figure_block.reading_order}: {preview_text}"
            )
        formula_blocks = [block for block in blocks if block.block_type == "formula_with_context"]
        for formula_block in formula_blocks[:5]:
            preview_text = " ".join(formula_block.text.split())[:180]
            context_preview = " ".join((formula_block.context_text or "").split())[:160]
            label = formula_block.label or "unlabeled"
            self._append_log(
                f"  formula example label={label} page={formula_block.page_number} "
                f"order={formula_block.reading_order}: {preview_text}"
            )
            if context_preview:
                self._append_log(f"    context: {context_preview}")
        for block in blocks[:8]:
            preview_text = " ".join(block.text.split())[:180]
            section_path = " > ".join(block.section_path) if block.section_path else "-"
            self._append_log(
                f"  {block.block_type} page={block.page_number} order={block.reading_order} "
                f"section={section_path}: {preview_text}"
            )

    def _handle_indexing_progress(self, progress: IndexingProgress) -> None:
        stage_messages = {
            "parse": "Parsing",
            "build_blocks": "Building structured blocks",
            "embed": "Embedding structured blocks",
            "store": "Storing vectors in local Qdrant",
            "done": "Finished",
            "failed": "Failed",
        }
        stage_label = stage_messages.get(progress.stage, progress.stage)
        self._append_log(
            f"[{progress.current_file}/{progress.total_files}] {stage_label}: {progress.file_name}"
        )

        stage_offsets = {
            "parse": 0.10,
            "build_blocks": 0.30,
            "embed": 0.60,
            "store": 0.85,
            "done": 1.0,
            "failed": 1.0,
        }
        file_fraction = (progress.current_file - 1 + stage_offsets.get(progress.stage, 0.0)) / progress.total_files
        self.progress_bar.setValue(max(0, min(100, int(file_fraction * 100))))
        QApplication.processEvents()

    def _append_indexing_summary(self, summary: IndexingRunSummary) -> None:
        for result in summary.results:
            self._append_indexed_file_result(result)

        self._append_log(
            "Indexing finished: "
            f"selected_files={summary.selected_files}, "
            f"successful_files={summary.successful_files}, "
            f"failed_files={summary.failed_files}, "
            f"total_structured_blocks={summary.total_structured_blocks}, "
            f"total_stored_points={summary.total_stored_points}, "
            f"elapsed={summary.elapsed_seconds:.2f}s."
        )

    def _append_indexed_file_result(self, result: IndexedFileResult) -> None:
        if not result.success:
            self._append_log(
                f"Index failed for {result.file_name}: {result.error_message or 'unknown error'} "
                f"elapsed={result.elapsed_seconds:.2f}s."
            )
            return

        page_count = result.page_count if result.page_count is not None else "-"
        self._append_log(
            f"Indexed {result.file_name}: pages={page_count}, "
            f"document_id={result.document_id or '-'}, "
            f"structured_blocks={result.structured_blocks}, "
            f"embeddings={result.embedded_blocks}, "
            f"stored_points={result.stored_points}, "
            f"elapsed={result.elapsed_seconds:.2f}s."
        )

    def _clear_selected(self) -> None:
        document_ids = self._checked_document_ids()
        if not document_ids:
            self._append_log("No checked PDF files selected for clearing.")
            return

        answer = QMessageBox.question(
            self,
            "Clear selected indexed documents",
            (
                f"Remove checked indexed data for {len(document_ids)} selected document(s)?\n\n"
                "Files that are not indexed will be skipped and reported.\n"
                "Source PDFs will not be deleted."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._append_log("Clear selected cancelled.")
            return

        self.progress_bar.setValue(0)
        self._append_log(f"Clearing selected document(s): {len(document_ids)}.")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            summary = self.deletion_service.clear_selected(document_ids)
            summary_path = self.deletion_service.export_summary(summary, "last_clear_selected_summary.json")
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)
            self.progress_bar.setValue(100)

        self._append_clear_summary(summary)
        self._append_log(f"Clear selected summary export: {summary_path}")
        self._refresh_indexed_statuses()

    def _clear_all(self) -> None:
        shared_root = self.document_registry.shared_data_root
        answer = QMessageBox.warning(
            self,
            "Clear all shared index data",
            (
                "This will wipe the whole shared local index for Indexator and chat retrieval.\n\n"
                f"Shared index root: {shared_root}\n"
                f"Qdrant collection: {self.qdrant_store.collection_name}\n\n"
                "Only index-owned data will be removed. Source PDFs will not be deleted."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._append_log("Clear all cancelled.")
            return

        self.progress_bar.setValue(0)
        self._append_log(f"Clearing all shared index data under: {shared_root}")
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            summary = self.deletion_service.clear_all()
            summary_path = self.deletion_service.export_summary(summary, "last_clear_all_summary.json")
        finally:
            QApplication.restoreOverrideCursor()
            self._set_scan_controls_enabled(True)
            self.progress_bar.setValue(100)

        self._append_clear_summary(summary)
        self._append_log(f"Clear all summary export: {summary_path}")
        self._refresh_indexed_statuses()

    def _checked_document_ids(self) -> list[str]:
        document_ids: list[str] = []
        for row in self._checked_pdf_rows():
            path_item = self.pdf_table.item(row, 2)
            if not path_item:
                continue
            document_id = make_document_id(Path(path_item.text()))
            document_ids.append(document_id)
        return document_ids

    def _refresh_indexed_statuses(self) -> None:
        if self.current_scan_folder is not None and self.current_scan_folder.exists():
            try:
                self._populate_pdf_table(self.pdf_scanner.scan(self.current_scan_folder))
                return
            except OSError as error:
                self._append_log(f"Indexed state refresh failed: {error}")
        self._update_action_buttons_enabled()

    def _update_action_buttons_enabled(self) -> None:
        if not hasattr(self, "clear_selected_button"):
            return
        has_checked_clearable = False
        has_checked_reindexable = False
        for row in self._checked_pdf_rows():
            status_item = self.pdf_table.item(row, 5)
            if not status_item:
                continue
            status = status_item.text()
            if status in {INDEXED, INDEXED_STALE, MISSING_SOURCE, INDEX_ERROR}:
                has_checked_clearable = True
            if status in {READY, INDEXED, INDEXED_STALE}:
                has_checked_reindexable = True
        controls_enabled = self.index_button.isEnabled()
        self.clear_selected_button.setEnabled(controls_enabled and has_checked_clearable)
        self.reindex_button.setEnabled(controls_enabled and has_checked_reindexable)

    def _update_clear_selected_enabled(self) -> None:
        self._update_action_buttons_enabled()

    def _append_clear_summary(self, summary: ClearOperationSummary) -> None:
        for detail in summary.details:
            if detail.status == "removed":
                label = detail.file_name or detail.document_id
                self._append_log(
                    f"Removed {label}: document_id={detail.document_id}, "
                    f"points={detail.removed_points}, cache_entries={detail.removed_cache_entries}."
                )
            elif detail.status.startswith("skipped"):
                self._append_log(f"Skipped {detail.document_id}: {detail.status}.")
            else:
                self._append_log(
                    f"Clear failed for {detail.document_id}: {detail.error_message or 'unknown error'}."
                )

        self._append_log(
            f"{summary.operation} finished: requested_documents={summary.requested_documents}, "
            f"removed_documents={summary.removed_documents}, "
            f"skipped_documents={summary.skipped_documents}, "
            f"removed_points={summary.removed_points}, "
            f"removed_cache_entries={summary.removed_cache_entries}, "
            f"elapsed={summary.elapsed_seconds:.2f}s."
        )

    def _append_reindex_summary(self, summary: ReindexRunSummary) -> None:
        for result in summary.results:
            label = result.file_name or result.document_id or result.source_path
            if result.status in {"reindexed", "newly_indexed"}:
                self._append_log(
                    f"{result.status} {label}: document_id={result.document_id}, "
                    f"removed_points={result.removed_points}, stored_points={result.stored_points}."
                )
            elif result.status.startswith("skipped"):
                self._append_log(f"Skipped {label}: {result.error_message or result.status}.")
            else:
                self._append_log(f"Reindex failed for {label}: {result.error_message or 'unknown error'}.")

        self._append_log(
            "Reindex finished: "
            f"requested_documents={summary.requested_documents}, "
            f"reindexed_documents={summary.reindexed_documents}, "
            f"newly_indexed_documents={summary.newly_indexed_documents}, "
            f"skipped_documents={summary.skipped_documents}, "
            f"failed_documents={summary.failed_documents}, "
            f"removed_points={summary.removed_points}, "
            f"stored_points={summary.stored_points}, "
            f"elapsed={summary.elapsed_seconds:.2f}s."
        )


def make_table_item(text: str) -> QTableWidgetItem:
    """Create a readonly table item with a stable alignment."""
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    return item


def make_checkbox_item(checked: bool) -> QTableWidgetItem:
    """Create a table item with a checkbox for selecting PDF files."""
    item = QTableWidgetItem()
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def format_file_size(size_bytes: int) -> str:
    """Format a file size for the PDF table."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Format a PDF block bounding box for debug preview logs."""
    return "(" + ", ".join(f"{value:.1f}" for value in bbox) + ")"
