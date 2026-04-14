import json
from pathlib import Path
from typing import Any

from services.chunker import TextChunk


DOCUMENTS_FILE = "documents.json"
CHUNKS_FILE = "chunks.jsonl"
SUMMARY_FILE = "indexing_summary.json"


class IndexWriter:
    def __init__(self, index_dir: Path, metadata_dir: Path) -> None:
        self.index_dir = index_dir
        self.metadata_dir = metadata_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @property
    def documents_path(self) -> Path:
        return self.metadata_dir / DOCUMENTS_FILE

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / CHUNKS_FILE

    @property
    def summary_path(self) -> Path:
        return self.metadata_dir / SUMMARY_FILE

    def load_document_metadata(self) -> dict[str, dict[str, Any]]:
        if not self.documents_path.exists():
            return {}

        with self.documents_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        documents = payload.get("documents", [])
        return {document["document_id"]: document for document in documents}

    def save_document_metadata(self, documents: dict[str, dict[str, Any]]) -> None:
        payload = {
            "schema_version": 1,
            "documents": sorted(documents.values(), key=lambda item: item["filename"].lower()),
        }
        self._write_json(self.documents_path, payload)

    def save_chunks(self, chunks_by_document: dict[str, list[TextChunk]]) -> None:
        replaced_document_ids = set(chunks_by_document)
        existing_chunks = self._load_existing_chunks(exclude_document_ids=replaced_document_ids)
        new_chunks = [
            self._chunk_to_dict(chunk)
            for document_chunks in chunks_by_document.values()
            for chunk in document_chunks
        ]

        with self.chunks_path.open("w", encoding="utf-8") as file:
            for chunk in existing_chunks + new_chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def save_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.summary_path, summary)

    def clear_outputs(self) -> list[Path]:
        removed_paths: list[Path] = []
        for path in (self.chunks_path, self.documents_path, self.summary_path):
            if path.exists():
                path.unlink()
                removed_paths.append(path)
        return removed_paths

    def _load_existing_chunks(self, exclude_document_ids: set[str]) -> list[dict[str, Any]]:
        if not self.chunks_path.exists():
            return []

        chunks: list[dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("document_id") not in exclude_document_ids:
                    chunks.append(chunk)
        return chunks

    def _chunk_to_dict(self, chunk: TextChunk) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
