from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.services.visual_evidence import VisualEvidenceRef


@dataclass(frozen=True)
class VisualCropSettings:
    indexer_output_dir: Path
    crops_dir: Path
    dpi: int = 160
    image_format: str = "png"


@dataclass(frozen=True)
class GeneratedCrop:
    block_id: str
    document_id: str
    file_path: str
    url_path: str
    width: int
    height: int
    format: str
    dpi: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisualCropService:
    def __init__(self, settings: VisualCropSettings) -> None:
        self.settings = settings

    def get_or_create_crop(self, ref: VisualEvidenceRef) -> GeneratedCrop | None:
        if not self._bbox_is_valid(ref.bbox):
            return None
        source_path = self._source_path(ref.document_id)
        if source_path is None or not source_path.exists():
            return None
        crop_path = self._crop_path(ref)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not crop_path.exists() and not self._render_crop(source_path, ref, crop_path):
            return None
        return self._crop_metadata(ref, crop_path)

    def _source_path(self, document_id: str) -> Path | None:
        documents_path = self.settings.indexer_output_dir / "metadata" / "documents.json"
        try:
            payload = json.loads(documents_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        for document in payload.get("documents", []):
            if isinstance(document, dict) and document.get("document_id") == document_id:
                source_path = document.get("source_path")
                if isinstance(source_path, str) and source_path.strip():
                    return Path(source_path)
        return None

    def _crop_path(self, ref: VisualEvidenceRef) -> Path:
        digest = hashlib.sha1(
            f"{ref.block_id}|{ref.page_number}|{ref.bbox}|{self.settings.dpi}".encode("utf-8")
        ).hexdigest()[:12]
        safe_block_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in ref.block_id)
        filename = f"page-{ref.page_number}-{safe_block_id}-{digest}.{self.settings.image_format}"
        return self.settings.crops_dir / ref.document_id / filename

    def _render_crop(self, source_path: Path, ref: VisualEvidenceRef, crop_path: Path) -> bool:
        try:
            with pymupdf.open(source_path) as document:
                page_index = ref.page_number - 1
                if page_index < 0 or page_index >= len(document):
                    return False
                page = document[page_index]
                clip = pymupdf.Rect(*ref.bbox) & page.rect
                if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                    return False
                pixmap = page.get_pixmap(dpi=self.settings.dpi, clip=clip, alpha=False)
                pixmap.save(crop_path)
                return True
        except (OSError, RuntimeError, ValueError):
            return False

    def _crop_metadata(self, ref: VisualEvidenceRef, crop_path: Path) -> GeneratedCrop | None:
        try:
            with pymupdf.open(crop_path) as image:
                page = image[0]
                width = int(page.rect.width)
                height = int(page.rect.height)
        except (OSError, RuntimeError, ValueError):
            return None
        relative = crop_path.relative_to(self.settings.crops_dir)
        url_path = "/crops/" + relative.as_posix()
        return GeneratedCrop(
            block_id=ref.block_id,
            document_id=ref.document_id,
            file_path=str(crop_path),
            url_path=url_path,
            width=width,
            height=height,
            format=self.settings.image_format,
            dpi=self.settings.dpi,
        )

    def _bbox_is_valid(self, bbox: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = bbox
        return x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0
