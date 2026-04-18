"""Main window for the Indexator desktop shell."""

from __future__ import annotations

import sys
import json
from collections import Counter
from pathlib import Path

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation, QThread, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.block_builder import StructuredBlockBuilder, make_document_id
from app.core.blocks import StructuredBlock
from app.core.index_compaction import IndexCompactionSettings
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
from app.ui.index_worker import IndexWorker
from app.utils.config import AppConfig, resolve_shared_data_path


TRANSLATIONS = {
    "en": {
        "language_toggle": "RU",
        "folder_label": "PDF folder:",
        "folder_placeholder": "Select a folder with PDF files",
        "select_folder": "Select Folder",
        "refresh_scan": "Refresh scan",
        "preview_blocks": "Preview blocks",
        "embed_preview": "Embed preview",
        "store_preview": "Store preview",
        "index_selected": "Index selected",
        "reindex_selected": "Reindex selected",
        "clear_selected": "Clear selected",
        "clear_all": "Clear all",
        "logs_label": "Logs:",
        "logs_placeholder": "Logs will appear here",
        "table_selected": "",
        "table_file_name": "File name",
        "table_path": "Path",
        "table_size": "Size",
        "table_pages": "Pages",
        "table_status": "Status",
        "status_ready": "Ready",
        "status_indexed": "Indexed",
        "status_indexed_stale": "Indexed (stale)",
        "status_missing_source": "Missing source",
        "status_index_error": "Index error",
        "status_unreadable": "Unreadable",
        "app_started": "Indexator shell started.",
        "runtime": "Python runtime: {runtime}",
        "embedding_device": "Embedding device: {device}",
        "select_folder_title": "Select PDF folder",
        "selected_folder": "Selected folder: {folder}",
        "ui_state_save_failed": "Could not save UI state: {error}",
        "select_folder_before_scan": "Select a folder before scanning.",
        "background_running": "Background operation is already running.",
        "scanning_folder": "Scanning PDF files in: {folder}",
        "found_pdfs": "Found {count} PDF file(s).",
        "unreadable_pdfs": "{count} PDF file(s) could not be read.",
        "close_running_title": "Background operation is running",
        "close_running_message": "Wait until the current background operation finishes before closing Indexator.",
        "no_index_selection": "No PDF files selected for indexing.",
        "skipped_not_ready": "Skipped {count} selected file(s) that are not ready.",
        "no_ready_files": "No Ready PDF files selected for indexing.",
        "indexing_selected": "Indexing selected PDF file(s): {count}.",
        "no_reindex_selection": "No PDF files selected for reindexing.",
        "skipped_not_reindexable": "Skipped {count} selected file(s) that are not available for reindexing.",
        "no_reindexable_files": "No available checked PDF files selected for reindexing.",
        "reindex_question_title": "Reindex selected documents",
        "reindex_question_message": (
            "Force reindex {count} checked document(s)?\n\n"
            "Existing indexed data for those documents will be replaced. Source PDFs will not be deleted."
        ),
        "reindex_cancelled": "Reindex selected cancelled.",
        "reindexing_selected": "Reindexing selected PDF file(s): {count}.",
        "no_preview_selection": "Select one ready PDF file before previewing blocks.",
        "parsing_preview": "Parsing preview for: {file_name}",
        "no_embed_selection": "Select one ready PDF file before embedding blocks.",
        "embedding_preview": "Embedding structured block preview for: {file_name}",
        "no_store_selection": "Select one ready PDF file before storing blocks.",
        "storing_preview": "Storing structured block preview in Qdrant server for: {file_name}",
        "no_clear_selection": "No checked PDF files selected for clearing.",
        "clear_selected_title": "Clear selected indexed documents",
        "clear_selected_message": (
            "Remove checked indexed data for {count} selected document(s)?\n\n"
            "Files that are not indexed will be skipped and reported.\n"
            "Source PDFs will not be deleted."
        ),
        "clear_selected_cancelled": "Clear selected cancelled.",
        "clearing_selected": "Clearing selected document(s): {count}.",
        "clear_all_title": "Clear all shared index data",
        "clear_all_message": (
            "This will wipe the configured Qdrant collection for Indexator and chat retrieval.\n\n"
            "Qdrant endpoint: {endpoint}\n"
            "Qdrant collection: {collection}\n\n"
            "Index-owned metadata and cache data will also be cleared. Source PDFs will not be deleted."
        ),
        "clear_all_cancelled": "Clear all cancelled.",
        "clearing_all": "Clearing Qdrant collection {collection} at {endpoint}",
        "stage_parse": "Parsing",
        "stage_scan": "Scanning PDFs",
        "stage_build_blocks": "Building structured blocks",
        "stage_embed": "Embedding structured blocks",
        "stage_store": "Storing vectors in Qdrant server",
        "stage_clear": "Clearing index data",
        "stage_done": "Finished",
        "stage_failed": "Failed",
        "yes": "Yes",
        "no": "No",
    },
    "ru": {
        "language_toggle": "EN",
        "folder_label": "Папка PDF:",
        "folder_placeholder": "Выберите папку с PDF-файлами",
        "select_folder": "Выбрать папку",
        "refresh_scan": "Обновить",
        "preview_blocks": "Предпросмотр",
        "embed_preview": "Эмбеддинг",
        "store_preview": "Сохранить",
        "index_selected": "Индексировать",
        "reindex_selected": "Переиндекс.",
        "clear_selected": "Очистить выбран.",
        "clear_all": "Очистить всё",
        "logs_label": "Логи:",
        "logs_placeholder": "Здесь будут появляться логи",
        "table_selected": "",
        "table_file_name": "Файл",
        "table_path": "Путь",
        "table_size": "Размер",
        "table_pages": "Стр.",
        "table_status": "Статус",
        "status_ready": "Готов",
        "status_indexed": "Индексирован",
        "status_indexed_stale": "Устарел",
        "status_missing_source": "Файл отсутствует",
        "status_index_error": "Ошибка индекса",
        "status_unreadable": "Не читается",
        "app_started": "Indexator запущен.",
        "runtime": "Python: {runtime}",
        "embedding_device": "Устройство эмбеддингов: {device}",
        "select_folder_title": "Выберите папку с PDF",
        "selected_folder": "Выбрана папка: {folder}",
        "ui_state_save_failed": "Не удалось сохранить UI state: {error}",
        "select_folder_before_scan": "Сначала выберите папку для сканирования.",
        "background_running": "Фоновая операция уже выполняется.",
        "scanning_folder": "Сканирование PDF-файлов в папке: {folder}",
        "found_pdfs": "Найдено PDF-файлов: {count}.",
        "unreadable_pdfs": "Не удалось прочитать PDF-файлов: {count}.",
        "close_running_title": "Фоновая операция выполняется",
        "close_running_message": "Дождитесь завершения текущей фоновой операции перед закрытием Indexator.",
        "no_index_selection": "Не выбраны PDF-файлы для индексации.",
        "skipped_not_ready": "Пропущено выбранных файлов не в статусе готовности: {count}.",
        "no_ready_files": "Нет выбранных PDF-файлов со статусом готовности.",
        "indexing_selected": "Индексация выбранных PDF-файлов: {count}.",
        "no_reindex_selection": "Не выбраны PDF-файлы для переиндексации.",
        "skipped_not_reindexable": "Пропущено файлов, недоступных для переиндексации: {count}.",
        "no_reindexable_files": "Нет выбранных доступных PDF-файлов для переиндексации.",
        "reindex_question_title": "Переиндексировать документы",
        "reindex_question_message": (
            "Принудительно переиндексировать выбранные документы: {count}?\n\n"
            "Старые индексные данные будут заменены. Исходные PDF не будут удалены."
        ),
        "reindex_cancelled": "Переиндексация отменена.",
        "reindexing_selected": "Переиндексация выбранных PDF-файлов: {count}.",
        "no_preview_selection": "Выберите один готовый PDF-файл для предпросмотра блоков.",
        "parsing_preview": "Предпросмотр парсинга: {file_name}",
        "no_embed_selection": "Выберите один готовый PDF-файл для предпросмотра эмбеддингов.",
        "embedding_preview": "Предпросмотр эмбеддингов для: {file_name}",
        "no_store_selection": "Выберите один готовый PDF-файл перед сохранением блоков.",
        "storing_preview": "Storing preview blocks in Qdrant server for: {file_name}",
        "no_clear_selection": "Не выбраны PDF-файлы для очистки.",
        "clear_selected_title": "Очистить выбранные индексные данные",
        "clear_selected_message": (
            "Удалить индексные данные выбранных документов: {count}?\n\n"
            "Неиндексированные файлы будут пропущены и отражены в отчёте.\n"
            "Исходные PDF не будут удалены."
        ),
        "clear_selected_cancelled": "Очистка выбранного отменена.",
        "clearing_selected": "Очистка выбранных документов: {count}.",
        "clear_all_title": "Очистить общий индекс",
        "clear_all_message": (
            "This will wipe the configured Qdrant collection for Indexator and chat retrieval.\n\n"
            "Qdrant endpoint: {endpoint}\n"
            "Qdrant collection: {collection}\n\n"
            "Index-owned metadata and cache data will also be cleared. Source PDFs will not be deleted."
        ),
        "clear_all_cancelled": "Полная очистка отменена.",
        "clearing_all": "Clearing Qdrant collection {collection} at {endpoint}",
        "stage_parse": "Парсинг",
        "stage_scan": "Сканирование PDF",
        "stage_build_blocks": "Построение структурных блоков",
        "stage_embed": "Расчёт эмбеддингов",
        "stage_store": "Storing vectors in Qdrant server",
        "stage_clear": "Очистка индекса",
        "stage_done": "Готово",
        "stage_failed": "Ошибка",
        "yes": "Да",
        "no": "Нет",
    },
}

STATUS_TRANSLATION_KEYS = {
    READY: "status_ready",
    INDEXED: "status_indexed",
    INDEXED_STALE: "status_indexed_stale",
    MISSING_SOURCE: "status_missing_source",
    INDEX_ERROR: "status_index_error",
    "Unreadable": "status_unreadable",
}


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
        self.compaction_settings = IndexCompactionSettings(
            mode=config.indexing.mode,  # type: ignore[arg-type]
            min_indexable_chars=config.indexing.min_indexable_chars,
            target_chunk_chars=config.indexing.target_chunk_chars,
            max_chunk_chars=config.indexing.max_chunk_chars,
            store_visual_metadata=config.indexing.store_visual_metadata,
        )
        self.app_root = Path(__file__).resolve().parents[2]
        self.qdrant_store = QdrantStore.from_config(config.storage, self.app_root)
        self.document_registry = DocumentRegistry(resolve_shared_data_path(config.storage.shared_data_path, self.app_root))
        self.deletion_service = IndexDeletionService(self.qdrant_store, self.document_registry)
        self.indexing_pipeline = IndexingPipeline(
            parser=self.pdf_parser,
            block_builder=self.block_builder,
            embedding_service=self.embedding_service,
            qdrant_store=self.qdrant_store,
            compaction_settings=self.compaction_settings,
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
        self.ui_state_path = self.output_dir / "ui_state.json"
        self.current_scan_folder: Path | None = None
        self.active_index_thread: QThread | None = None
        self.active_index_worker: IndexWorker | None = None
        self.current_language = "ru"
        self.setWindowTitle(config.app.name)
        self.resize(config.ui.window_width, config.ui.window_height)

        self.folder_path_field = QLineEdit()
        self.folder_path_field.setReadOnly(True)
        self.folder_path_field.setPlaceholderText("")

        self.select_folder_button = QPushButton()
        self.language_button = QPushButton()
        self.scan_button = QPushButton()
        self.preview_button = QPushButton()
        self.embed_preview_button = QPushButton()
        self.store_preview_button = QPushButton()
        self.index_button = QPushButton()
        self.reindex_button = QPushButton()
        self.clear_selected_button = QPushButton()
        self.clear_all_button = QPushButton()
        self.select_folder_blink_effect = QGraphicsOpacityEffect(self.select_folder_button)
        self.select_folder_blink_animation = QPropertyAnimation(self.select_folder_blink_effect, b"opacity", self)

        self.pdf_table = QTableWidget(0, 6)
        self.log_panel = QPlainTextEdit()
        self.progress_bar = QProgressBar()
        self.folder_label = QLabel()
        self.logs_label = QLabel()
        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        self.log_container = QWidget()

        self._configure_widgets()
        self._build_layout()
        self._connect_signals()
        self._apply_language()
        self._update_select_folder_attention()
        self._append_log(self._text("app_started"))
        self._append_log(self._text("runtime", runtime=sys.executable))
        self._append_log(self._text("embedding_device", device=self.embedding_service.embedder.describe_device_runtime()))
        self._restore_startup_folder()

    def _configure_widgets(self) -> None:
        self.select_folder_button.setObjectName("selectFolderButton")
        self.language_button.setObjectName("languageButton")
        self.scan_button.setObjectName("refreshScanButton")
        self.preview_button.setObjectName("previewButton")
        self.embed_preview_button.setObjectName("previewButton")
        self.store_preview_button.setObjectName("previewButton")
        self.reindex_button.setObjectName("reindexButton")
        self.index_button.setObjectName("indexButton")
        self.clear_selected_button.setObjectName("clearButton")
        self.clear_all_button.setObjectName("clearButton")
        self.folder_label.setFixedWidth(64)
        self.select_folder_button.setFixedWidth(118)
        self.scan_button.setFixedWidth(96)
        self.preview_button.setFixedWidth(122)
        self.embed_preview_button.setFixedWidth(112)
        self.store_preview_button.setFixedWidth(104)
        for button in (
            self.reindex_button,
            self.index_button,
            self.clear_selected_button,
            self.clear_all_button,
        ):
            button.setFixedWidth(142)
        self.language_button.setFixedWidth(44)
        self._apply_button_styles()
        self.select_folder_button.setGraphicsEffect(self.select_folder_blink_effect)
        self.select_folder_blink_effect.setOpacity(1.0)
        self.select_folder_blink_animation.setKeyValueAt(0.0, 1.0)
        self.select_folder_blink_animation.setKeyValueAt(0.5, 0.42)
        self.select_folder_blink_animation.setKeyValueAt(1.0, 1.0)
        self.select_folder_blink_animation.setDuration(900)
        self.select_folder_blink_animation.setLoopCount(-1)
        self.select_folder_blink_animation.setEasingCurve(QEasingCurve.Type.InOutSine)

        self.pdf_table.setHorizontalHeaderLabels(["Selected", "File name", "Path", "Size", "Pages", "Status"])
        self._configure_pdf_table_columns()
        self.pdf_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.pdf_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Logs will appear here")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.reindex_button.setEnabled(False)
        self.clear_selected_button.setEnabled(False)

    def _configure_pdf_table_columns(self) -> None:
        header = self.pdf_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(42)
        for column in range(self.pdf_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.pdf_table.setColumnWidth(0, 38)
        self.pdf_table.setColumnWidth(1, 170)
        self.pdf_table.setColumnWidth(2, 560)
        self.pdf_table.setColumnWidth(3, 90)
        self.pdf_table.setColumnWidth(4, 60)
        self.pdf_table.setColumnWidth(5, 130)

    def _build_layout(self) -> None:
        root = QWidget(self)
        main_layout = QVBoxLayout(root)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_path_field, stretch=1)
        folder_layout.addWidget(self.language_button)
        folder_layout.addWidget(self.select_folder_button)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.scan_button)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.embed_preview_button)
        action_layout.addWidget(self.store_preview_button)
        action_layout.addStretch(1)

        right_action_layout = QVBoxLayout()

        index_layout = QHBoxLayout()
        index_layout.addWidget(self.reindex_button)
        index_layout.addWidget(self.index_button)

        clear_layout = QHBoxLayout()
        clear_layout.addWidget(self.clear_selected_button)
        clear_layout.addWidget(self.clear_all_button)

        right_action_layout.addLayout(index_layout)
        right_action_layout.addLayout(clear_layout)
        action_layout.addLayout(right_action_layout)

        main_layout.addLayout(folder_layout)
        main_layout.addLayout(action_layout)
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.logs_label)
        log_layout.addWidget(self.log_panel)

        self.content_splitter.addWidget(self.pdf_table)
        self.content_splitter.addWidget(self.log_container)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([420, 150])

        main_layout.addWidget(self.content_splitter, stretch=1)
        main_layout.addWidget(self.progress_bar)

        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.select_folder_button.clicked.connect(self._select_folder)
        self.language_button.clicked.connect(self._toggle_language)
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
        if self.active_index_thread is not None:
            QMessageBox.warning(
                self,
                self._text("close_running_title"),
                self._text("close_running_message"),
            )
            event.ignore()
            return

        self.qdrant_store.close()
        super().closeEvent(event)

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self._text("select_folder_title"))
        if not folder:
            return

        self.folder_path_field.setText(folder)
        self._save_startup_folder(Path(folder))
        self._append_log(self._text("selected_folder", folder=folder))
        self._update_select_folder_attention()
        self._scan_pdfs()

    def _restore_startup_folder(self) -> None:
        folder = self._load_startup_folder()
        if folder is None:
            self._update_select_folder_attention()
            return

        self.folder_path_field.setText(str(folder))
        self.current_scan_folder = folder
        self._update_select_folder_attention()
        self._append_log(self._text("selected_folder", folder=folder))
        self._scan_pdfs()

    def _load_startup_folder(self) -> Path | None:
        try:
            raw_state = json.loads(self.ui_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        folder_value = raw_state.get("startup_folder") if isinstance(raw_state, dict) else None
        if not isinstance(folder_value, str) or not folder_value.strip():
            return None

        folder = Path(folder_value)
        if folder.exists() and folder.is_dir():
            return folder

        return None

    def _save_startup_folder(self, folder: Path) -> None:
        try:
            self.ui_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.ui_state_path.write_text(
                json.dumps({"startup_folder": str(folder)}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            self._append_log(self._text("ui_state_save_failed", error=error))

    def _text(self, key: str, **values: object) -> str:
        template = TRANSLATIONS[self.current_language].get(key, TRANSLATIONS["en"].get(key, key))
        return template.format(**values)

    def _apply_language(self) -> None:
        self.folder_label.setText(self._text("folder_label"))
        self.folder_path_field.setPlaceholderText(self._text("folder_placeholder"))
        self.language_button.setText(self._text("language_toggle"))
        self.select_folder_button.setText(self._text("select_folder"))
        self.scan_button.setText(self._text("refresh_scan"))
        self.preview_button.setText(self._text("preview_blocks"))
        self.embed_preview_button.setText(self._text("embed_preview"))
        self.store_preview_button.setText(self._text("store_preview"))
        self.index_button.setText(self._text("index_selected"))
        self.reindex_button.setText(self._text("reindex_selected"))
        self.clear_selected_button.setText(self._text("clear_selected"))
        self.clear_all_button.setText(self._text("clear_all"))
        self.logs_label.setText(self._text("logs_label"))
        self.log_panel.setPlaceholderText(self._text("logs_placeholder"))
        self.pdf_table.setHorizontalHeaderLabels(
            [
                self._text("table_selected"),
                self._text("table_file_name"),
                self._text("table_path"),
                self._text("table_size"),
                self._text("table_pages"),
                self._text("table_status"),
            ]
        )
        self._refresh_table_language()

    def _toggle_language(self) -> None:
        self.current_language = "en" if self.current_language == "ru" else "ru"
        self._apply_language()

    def _confirm(self, title: str, message: str, icon: QMessageBox.Icon = QMessageBox.Icon.Question) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(icon)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        yes_button = dialog.addButton(self._text("yes"), QMessageBox.ButtonRole.YesRole)
        dialog.addButton(self._text("no"), QMessageBox.ButtonRole.NoRole)
        dialog.setDefaultButton(yes_button)
        dialog.exec()
        return dialog.clickedButton() == yes_button

    def _apply_button_styles(self) -> None:
        self.setStyleSheet(
            """
            QPushButton {
                border: 1px solid rgba(255, 255, 255, 38);
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background-color: #24282d;
                color: #6e7681;
                border-color: rgba(255, 255, 255, 18);
            }
            QPushButton#selectFolderButton {
                background-color: #3a4656;
                color: #f2f7ff;
                border-color: #7893b8;
            }
            QPushButton#selectFolderButton:pressed {
                background-color: #242d39;
            }
            QPushButton#languageButton {
                background-color: #303741;
                color: #e6edf5;
                border-color: #596575;
            }
            QPushButton#languageButton:pressed {
                background-color: #20262e;
            }
            QPushButton#refreshScanButton {
                background-color: #3b5f9f;
                color: #ffffff;
            }
            QPushButton#refreshScanButton:pressed {
                background-color: #253e6d;
            }
            QPushButton#previewButton {
                background-color: #6b4f9f;
                color: #ffffff;
            }
            QPushButton#previewButton:pressed {
                background-color: #46366e;
            }
            QPushButton#reindexButton {
                background-color: #c7a646;
                color: #1d1703;
            }
            QPushButton#reindexButton:pressed {
                background-color: #806824;
                color: #fff1bd;
            }
            QPushButton#indexButton {
                background-color: #7fae55;
                color: #0f1d08;
            }
            QPushButton#indexButton:pressed {
                background-color: #4f7232;
                color: #ecffd8;
            }
            QPushButton#clearButton {
                background-color: #b64c4c;
                color: #ffffff;
            }
            QPushButton#clearButton:pressed {
                background-color: #793030;
            }
            """
        )

    def _update_select_folder_attention(self) -> None:
        if self.folder_path_field.text().strip():
            self.select_folder_blink_animation.stop()
            self.select_folder_blink_effect.setOpacity(1.0)
            return

        if self.select_folder_blink_animation.state() != QAbstractAnimation.State.Running:
            self.select_folder_blink_animation.start()

    def _scan_pdfs(self) -> None:
        folder = self.folder_path_field.text().strip()
        if not folder:
            self._append_log(self._text("select_folder_before_scan"))
            return
        if self.active_index_thread is not None:
            self._append_log(self._text("background_running"))
            return

        self.pdf_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.current_scan_folder = Path(folder)
        self._append_log(self._text("scanning_folder", folder=folder))
        self._start_index_worker("scan", scan_folder=Path(folder))

    def _index_selected(self) -> None:
        selected_rows = self._checked_pdf_rows()
        if not selected_rows:
            self._append_log(self._text("no_index_selection"))
            return

        pdf_paths = self._checked_pdf_paths_with_statuses({READY})
        skipped_count = len(selected_rows) - len(pdf_paths)
        if skipped_count:
            self._append_log(self._text("skipped_not_ready", count=skipped_count))
        if not pdf_paths:
            self._append_log(self._text("no_ready_files"))
            return

        self.progress_bar.setValue(0)
        self._append_log(self._text("indexing_selected", count=len(pdf_paths)))
        self._start_index_worker("index", pdf_paths=pdf_paths)

    def _reindex_selected(self) -> None:
        selected_rows = self._checked_pdf_rows()
        if not selected_rows:
            self._append_log(self._text("no_reindex_selection"))
            return

        pdf_paths = self._checked_pdf_paths_with_statuses({READY, INDEXED, INDEXED_STALE})
        skipped_count = len(selected_rows) - len(pdf_paths)
        if skipped_count:
            self._append_log(self._text("skipped_not_reindexable", count=skipped_count))
        if not pdf_paths:
            self._append_log(self._text("no_reindexable_files"))
            return

        if not self._confirm(
            self._text("reindex_question_title"),
            self._text("reindex_question_message", count=len(pdf_paths)),
        ):
            self._append_log(self._text("reindex_cancelled"))
            return

        self.progress_bar.setValue(0)
        self._append_log(self._text("reindexing_selected", count=len(pdf_paths)))
        self._start_index_worker("reindex", pdf_paths=pdf_paths)

    def _start_index_worker(
        self,
        mode: str,
        pdf_paths: list[Path] | None = None,
        document_ids: list[str] | None = None,
        scan_folder: Path | None = None,
    ) -> None:
        if self.active_index_thread is not None:
            self._append_log(self._text("background_running"))
            return

        thread = QThread(self)
        worker = IndexWorker(
            mode=mode,
            pdf_paths=pdf_paths,
            document_ids=document_ids,
            scan_folder=scan_folder,
            pdf_scanner=self.pdf_scanner,
            parser=self.pdf_parser,
            block_builder=self.block_builder,
            embedding_service=self.embedding_service,
            qdrant_store=self.qdrant_store,
            indexing_pipeline=self.indexing_pipeline,
            reindex_service=self.reindex_service,
            deletion_service=self.deletion_service,
            output_dir=self.output_dir,
            embedding_device=self.embedding_service.embedder.describe_device_runtime(),
            compaction_settings=self.compaction_settings,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress_changed.connect(self._handle_indexing_progress)
        worker.log_message.connect(self._append_log)
        worker.file_finished.connect(self._handle_index_worker_file_finished)
        worker.finished.connect(self._handle_index_worker_finished)
        worker.error.connect(self._handle_index_worker_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_index_worker_refs)

        self.active_index_thread = thread
        self.active_index_worker = worker
        self._set_scan_controls_enabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        thread.start()

    def _handle_index_worker_finished(self, mode: str, result: object, _unused: object) -> None:
        QApplication.restoreOverrideCursor()
        self._set_scan_controls_enabled(True)
        self.progress_bar.setValue(100)

        if not isinstance(result, dict):
            self._append_log(f"Finished {mode} operation with unexpected result type.")
            if self._should_refresh_indexed_statuses(mode):
                self._refresh_indexed_statuses()
            return

        summary = result.get("summary")
        summary_path = result.get("summary_path")

        if mode == "scan":
            scan_results = result.get("scan_results")
            scan_folder = result.get("scan_folder")
            if isinstance(scan_results, list):
                if isinstance(scan_folder, Path):
                    self.current_scan_folder = scan_folder
                self._populate_pdf_table(scan_results)
                unreadable_count = sum(1 for scan_result in scan_results if scan_result.status != READY)
                self._append_log(self._text("found_pdfs", count=len(scan_results)))
                if unreadable_count:
                    self._append_log(self._text("unreadable_pdfs", count=unreadable_count))
        elif mode == "index" and isinstance(summary, IndexingRunSummary):
            self._append_indexing_summary(summary)
            self._append_log(f"Indexing summary export: {summary_path}")
        elif mode == "reindex" and isinstance(summary, ReindexRunSummary):
            self._append_reindex_summary(summary)
            self._append_log(f"Reindex summary export: {summary_path}")
        elif mode == "preview":
            parsed_document = result.get("parsed_document")
            structured_blocks = result.get("structured_blocks")
            if isinstance(parsed_document, ParsedDocument) and isinstance(structured_blocks, list):
                self._append_parsed_preview(parsed_document)
                self._append_structured_block_preview(structured_blocks)
                self._append_log(f"Structured block debug export: {result.get('debug_path')}")
        elif mode == "embed_preview":
            embedding_run = result.get("embedding_run")
            if embedding_run is not None:
                self._append_log(
                    f"Embedded {len(embedding_run.embeddings)} block(s) with {embedding_run.model_name}: "
                    f"dimension={embedding_run.embedding_dimension}, device={embedding_run.device}, "
                    f"elapsed={embedding_run.elapsed_seconds:.2f}s."
                )
                self._append_log(f"Embedding debug summary export: {result.get('debug_path')}")
        elif mode == "store_preview":
            storage_run = result.get("storage_run")
            pdf_path = result.get("pdf_path")
            if storage_run is not None and isinstance(pdf_path, Path):
                self._append_log(
                    f"Stored {storage_run.stored_blocks} block(s) from {pdf_path.name} in Qdrant server: "
                    f"collection={storage_run.collection_name}, dimension={storage_run.embedding_dimension}, "
                    f"endpoint={storage_run.endpoint}, elapsed={storage_run.elapsed_seconds:.2f}s."
                )
                self._append_log(f"Qdrant server storage summary export: {result.get('debug_path')}")
        elif mode == "clear_selected" and isinstance(summary, ClearOperationSummary):
            self._append_clear_summary(summary)
            self._append_log(f"Clear selected summary export: {summary_path}")
        elif mode == "clear_all" and isinstance(summary, ClearOperationSummary):
            self._append_clear_summary(summary)
            self._append_log(f"Clear all summary export: {summary_path}")
        else:
            self._append_log(f"Finished {mode} operation with unexpected result payload.")

        if self._should_refresh_indexed_statuses(mode):
            self._refresh_indexed_statuses()

    def _handle_index_worker_error(self, mode: str, message: str, details: str) -> None:
        QApplication.restoreOverrideCursor()
        self._set_scan_controls_enabled(True)
        self.progress_bar.setValue(100)
        self._append_log(f"{mode.capitalize()} operation failed: {message}")
        self._append_log(details.rstrip())
        if self._should_refresh_indexed_statuses(mode):
            self._refresh_indexed_statuses()

    def _handle_index_worker_file_finished(self, _file_name: str) -> None:
        return

    def _clear_index_worker_refs(self) -> None:
        self.active_index_thread = None
        self.active_index_worker = None

    def _should_refresh_indexed_statuses(self, mode: str) -> bool:
        return mode in {"index", "reindex", "clear_selected", "clear_all"}

    def _preview_selected_pdf(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log(self._text("no_preview_selection"))
            return

        self._append_log(self._text("parsing_preview", file_name=pdf_path.name))
        self.progress_bar.setValue(0)
        self._start_index_worker("preview", pdf_paths=[pdf_path])

    def _embed_selected_pdf_preview(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log(self._text("no_embed_selection"))
            return

        self._append_log(self._text("embedding_preview", file_name=pdf_path.name))
        self.progress_bar.setValue(0)
        self._start_index_worker("embed_preview", pdf_paths=[pdf_path])

    def _store_selected_pdf_preview(self) -> None:
        pdf_path = self._first_selected_pdf_path()
        if not pdf_path:
            self._append_log(self._text("no_store_selection"))
            return

        self._append_log(self._text("storing_preview", file_name=pdf_path.name))
        self.progress_bar.setValue(0)
        self._start_index_worker("store_preview", pdf_paths=[pdf_path])

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
                status_item = make_table_item(self._status_label(state.status))
                status_item.setData(Qt.ItemDataRole.UserRole, state.status)

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

    def _refresh_table_language(self) -> None:
        if not hasattr(self, "pdf_table"):
            return
        for row in range(self.pdf_table.rowCount()):
            status_item = self.pdf_table.item(row, 5)
            if status_item is None:
                continue
            status = status_item.data(Qt.ItemDataRole.UserRole) or status_item.text()
            status_item.setText(self._status_label(str(status)))

    def _status_label(self, status: str) -> str:
        translation_key = STATUS_TRANSLATION_KEYS.get(status)
        if translation_key is None:
            return status
        return self._text(translation_key)

    def _row_status(self, row: int) -> str | None:
        status_item = self.pdf_table.item(row, 5)
        if status_item is None:
            return None
        status = status_item.data(Qt.ItemDataRole.UserRole)
        return str(status) if status is not None else status_item.text()

    def _set_scan_controls_enabled(self, enabled: bool) -> None:
        self.select_folder_button.setEnabled(enabled)
        self.language_button.setEnabled(enabled)
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
            path_item = self.pdf_table.item(row, 2)
            status = self._row_status(row)
            if status in {READY, INDEXED, INDEXED_STALE} and path_item:
                paths.append(Path(path_item.text()))
        return paths

    def _checked_pdf_paths_with_statuses(self, statuses: set[str]) -> list[Path]:
        paths: list[Path] = []
        for row in self._checked_pdf_rows():
            path_item = self.pdf_table.item(row, 2)
            status = self._row_status(row)
            if status in statuses and path_item:
                paths.append(Path(path_item.text()))
        return paths

    def _first_selected_pdf_path(self) -> Path | None:
        checked_rows = self._checked_pdf_rows()
        selected_rows = [index.row() for index in self.pdf_table.selectionModel().selectedRows()]

        for row in checked_rows or selected_rows:
            path_item = self.pdf_table.item(row, 2)
            status = self._row_status(row)
            if status in {READY, INDEXED, INDEXED_STALE} and path_item:
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
            "parse": self._text("stage_parse"),
            "scan": self._text("stage_scan"),
            "build_blocks": self._text("stage_build_blocks"),
            "embed": self._text("stage_embed"),
            "store": self._text("stage_store"),
            "clear": self._text("stage_clear"),
            "done": self._text("stage_done"),
            "failed": self._text("stage_failed"),
        }
        stage_label = stage_messages.get(progress.stage, progress.stage)
        self._append_log(
            f"[{progress.current_file}/{progress.total_files}] {stage_label}: {progress.file_name}"
        )

        stage_offsets = {
            "parse": 0.10,
            "scan": 0.50,
            "build_blocks": 0.30,
            "embed": 0.60,
            "store": 0.85,
            "clear": 0.50,
            "done": 1.0,
            "failed": 1.0,
        }
        file_fraction = (progress.current_file - 1 + stage_offsets.get(progress.stage, 0.0)) / progress.total_files
        self.progress_bar.setValue(max(0, min(100, int(file_fraction * 100))))

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
            self._append_log(self._text("no_clear_selection"))
            return

        if not self._confirm(
            self._text("clear_selected_title"),
            self._text("clear_selected_message", count=len(document_ids)),
        ):
            self._append_log(self._text("clear_selected_cancelled"))
            return

        self.progress_bar.setValue(0)
        self._append_log(self._text("clearing_selected", count=len(document_ids)))
        self._start_index_worker("clear_selected", document_ids=document_ids)

    def _clear_all(self) -> None:
        if not self._confirm(
            self._text("clear_all_title"),
            self._text(
                "clear_all_message",
                endpoint=self.qdrant_store.endpoint,
                collection=self.qdrant_store.collection_name,
            ),
            QMessageBox.Icon.Warning,
        ):
            self._append_log(self._text("clear_all_cancelled"))
            return

        self.progress_bar.setValue(0)
        self._append_log(
            self._text(
                "clearing_all",
                endpoint=self.qdrant_store.endpoint,
                collection=self.qdrant_store.collection_name,
            )
        )
        self._start_index_worker("clear_all")

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
            status = self._row_status(row)
            if status is None:
                continue
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
