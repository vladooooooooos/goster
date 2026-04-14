"""Debug export helpers for structured block previews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.blocks import StructuredBlock

if TYPE_CHECKING:
    from app.core.pipeline import IndexingRunSummary
    from app.services.embedding_service import BlockEmbeddingRun
    from app.storage.qdrant_store import QdrantStorageRun


def export_blocks_jsonl(blocks: list[StructuredBlock], output_path: Path) -> Path:
    """Write structured blocks to a JSONL debug file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for block in blocks:
            file.write(json.dumps(block_to_json(block), ensure_ascii=False) + "\n")
    return output_path


def export_embedding_summary(run: "BlockEmbeddingRun", output_path: Path) -> Path:
    """Write a compact embedding debug summary without dumping full vectors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": run.model_name,
        "device": run.device,
        "embedded_blocks": len(run.embeddings),
        "embedding_dimension": run.embedding_dimension,
        "elapsed_seconds": round(run.elapsed_seconds, 3),
        "examples": [
            {
                "block_id": embedding.block_id,
                "block_type": embedding.block_type,
                "page_number": embedding.page_number,
                "reading_order": embedding.reading_order,
                "text_preview": " ".join(embedding.text.split())[:240],
                "vector_preview": embedding.vector[:5],
            }
            for embedding in run.embeddings[:5]
        ],
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def export_qdrant_storage_summary(run: "QdrantStorageRun", output_path: Path) -> Path:
    """Write a compact local Qdrant storage debug summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "collection_name": run.collection_name,
        "local_path": str(run.local_path),
        "stored_blocks": run.stored_blocks,
        "embedding_dimension": run.embedding_dimension,
        "elapsed_seconds": round(run.elapsed_seconds, 3),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def export_indexing_summary(summary: "IndexingRunSummary", output_path: Path) -> Path:
    """Write a compact full indexing run summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_data = {
        "selected_files": summary.selected_files,
        "successful_files": summary.successful_files,
        "failed_files": summary.failed_files,
        "total_structured_blocks": summary.total_structured_blocks,
        "total_stored_points": summary.total_stored_points,
        "elapsed_seconds": round(summary.elapsed_seconds, 3),
        "files": [
            {
                "file_name": result.file_name,
                "file_path": str(result.file_path),
                "document_id": result.document_id,
                "indexed_at": result.indexed_at,
                "page_count": result.page_count,
                "structured_blocks": result.structured_blocks,
                "embedded_blocks": result.embedded_blocks,
                "stored_points": result.stored_points,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "success": result.success,
                "error_message": result.error_message,
            }
            for result in summary.results
        ],
    }
    output_path.write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def block_to_json(block: StructuredBlock) -> dict[str, object]:
    """Convert a structured block to JSON-serializable debug data."""
    return {
        "id": block.id,
        "doc_id": block.doc_id,
        "file_name": block.file_name,
        "file_path": str(block.file_path),
        "page_number": block.page_number,
        "block_type": block.block_type,
        "text": block.text,
        "bbox": block.bbox,
        "reading_order": block.reading_order,
        "section_path": block.section_path,
        "label": block.label,
        "context_text": block.context_text,
    }
