from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.orchestration.tool_contracts import ToolContext, ToolDefinition, ToolResult
from app.services.visual_evidence import VisualEvidenceRef


class DocumentRagTool:
    def __init__(self, rag_service: Any) -> None:
        self._rag_service = rag_service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="document_rag",
            description="Answer a question using indexed GOST documents with citations and visual evidence.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            },
        )

    async def run(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k") or 12)
        answer = await self._rag_service.answer_question(query, top_k=top_k)
        return ToolResult(
            tool_name=self.definition.name,
            ok=True,
            content={
                "query": answer.query,
                "answer": answer.answer,
                "citations": [_to_dict(item) for item in answer.citations],
                "retrieved_results_count": answer.retrieved_results_count,
                "retrieval_used": answer.retrieval_used,
                "retrieved_chunks": [_to_dict(item) for item in answer.retrieved_chunks],
                "retrieval_info": answer.retrieval_info,
                "visual_evidence": [_to_dict(item) for item in answer.visual_evidence],
            },
        )


class VisualCropTool:
    def __init__(self, crop_service: Any) -> None:
        self._crop_service = crop_service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="visual_crop",
            description="Create or reuse a PDF crop from a visual evidence reference.",
            input_schema={"type": "object", "required": ["block_id", "document_id", "page_number", "bbox"]},
        )

    async def run(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        bbox = payload.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return ToolResult(tool_name=self.definition.name, ok=False, error="Invalid bbox.")
        ref = VisualEvidenceRef(
            block_id=str(payload["block_id"]),
            document_id=str(payload["document_id"]),
            page_number=int(payload["page_number"]),
            bbox=tuple(float(value) for value in bbox),
            block_type=payload.get("block_type"),
            label=payload.get("label"),
            source_file=str(payload.get("source_file") or ""),
            text_preview=str(payload.get("text_preview") or ""),
        )
        crop = self._crop_service.get_or_create_crop(ref)
        if crop is None:
            return ToolResult(tool_name=self.definition.name, ok=False, error="Crop could not be generated.")
        return ToolResult(tool_name=self.definition.name, ok=True, content={"crop": _to_dict(crop)})


class VisualAssetTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="visual_asset",
            description="Convert generated visual evidence into frontend attachment metadata.",
            input_schema={"type": "object", "properties": {"visual_evidence": {"type": "array"}}},
        )

    async def run(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        visuals = payload.get("visual_evidence")
        if not isinstance(visuals, list):
            visuals = []
        attachments = [_attachment(_to_dict(visual)) for visual in visuals if isinstance(_to_dict(visual), dict)]
        return ToolResult(tool_name=self.definition.name, ok=True, content={"attachments": attachments})


def _attachment(visual: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": f"{visual.get('document_id')}:{visual.get('block_id')}",
        "block_id": visual.get("block_id"),
        "document_id": visual.get("document_id"),
        "url": visual.get("crop_url") or visual.get("url_path"),
        "source_file": visual.get("source_file"),
        "page_number": visual.get("page_number"),
        "width": visual.get("width"),
        "height": visual.get("height"),
        "format": visual.get("format"),
        "label": visual.get("label"),
        "block_type": visual.get("block_type"),
    }


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        payload = dict(value.__dict__)
        for name in (
            "block_id",
            "document_id",
            "file_path",
            "url_path",
            "crop_path",
            "crop_url",
            "width",
            "height",
            "format",
            "dpi",
            "source_file",
            "page_number",
            "block_type",
            "label",
        ):
            if name not in payload and hasattr(value, name):
                payload[name] = getattr(value, name)
        return payload
    return {}
