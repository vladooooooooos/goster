"""Shared payload helpers for GOST Qdrant points."""

from __future__ import annotations

import uuid
from typing import Any

from .models import GostBlockVector, GostPayloadFields, VectorPoint


def make_gost_block_point(block: GostBlockVector) -> VectorPoint:
    """Convert a GOST block vector into a generic vector store point."""
    return VectorPoint(
        id=make_point_id(block.block_id),
        vector=block.vector,
        payload=make_gost_block_payload(block),
    )


def make_point_id(block_id: str) -> str:
    """Create a deterministic Qdrant-compatible UUID from the block id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, block_id))


def make_gost_block_payload(block: GostBlockVector) -> dict[str, Any]:
    """Build retrieval-friendly Qdrant payload metadata for one GOST block."""
    has_visual_evidence = block.block_type in {"table", "figure", "formula_with_context"} and block.bbox is not None
    payload = {
        "block_id": block.block_id,
        "document_id": block.document_id,
        "file_name": block.file_name,
        "page_start": block.page_number,
        "page_end": block.page_number,
        "block_type": block.block_type,
        "label": block.label,
        "section_path": block.section_path,
        "text": block.text,
        "reading_order": block.reading_order,
        "has_visual_evidence": has_visual_evidence,
    }
    if has_visual_evidence:
        payload["bbox"] = list(block.bbox) if block.bbox else None
        payload["page_number"] = block.page_number
        payload["crop_status"] = "available"
    return payload


def parse_gost_payload(payload: dict[str, Any], fallback_id: Any) -> GostPayloadFields:
    """Parse common retrieval fields from a GOST vector payload."""
    block_id = string_value(payload.get("block_id")) or string_value(payload.get("chunk_id")) or str(fallback_id)
    text = string_value(payload.get("text"))
    retrieval_text = string_value(payload.get("retrieval_text")) or string_value(payload.get("embedding_text"))
    source_file = (
        string_value(payload.get("source_file"))
        or string_value(payload.get("file_name"))
        or string_value(payload.get("doc_title"))
        or string_value(payload.get("file_path"))
        or "unknown"
    )
    page_start = optional_int(payload.get("page_start")) or optional_int(payload.get("page"))
    page_end = optional_int(payload.get("page_end")) or page_start

    return GostPayloadFields(
        block_id=block_id,
        text=text,
        retrieval_text=retrieval_text,
        source_file=source_file,
        page_start=page_start,
        page_end=page_end,
        section_path=list_of_strings(payload.get("section_path")),
        document_id=string_value(payload.get("doc_id")) or string_value(payload.get("document_id")) or None,
        block_type=string_value(payload.get("block_type")) or None,
        label=string_value(payload.get("label")) or None,
    )


def estimate_tokens(text: str) -> int:
    """Return a simple token estimate for debug payloads."""
    return max(1, len(text) // 4) if text else 0


def string_value(value: Any) -> str:
    """Return a stripped string value or an empty string."""
    return value.strip() if isinstance(value, str) else ""


def optional_int(value: Any) -> int | None:
    """Return an int payload value when present."""
    if isinstance(value, int):
        return value
    return None


def list_of_strings(value: Any) -> list[str]:
    """Normalize section path payload values to a list of strings."""
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(">") if part.strip()]
    return []
