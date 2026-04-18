from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.services.visual_evidence import VisualEvidenceRef

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VisualCropSettings:
    indexer_output_dir: Path
    crops_dir: Path
    dpi: int = 160
    image_format: str = "png"
    source_roots: tuple[Path, ...] = ()


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
            logger.info("Skipping visual crop for block_id=%s: invalid bbox=%s.", ref.block_id, ref.bbox)
            return None
        source_path = self._source_path(ref.document_id)
        if source_path is None or not source_path.exists():
            logger.info(
                "Skipping visual crop for block_id=%s: source PDF was not found for document_id=%s.",
                ref.block_id,
                ref.document_id,
            )
            return None
        crop_path = self._crop_path(ref)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not crop_path.exists() and not self._render_crop(source_path, ref, crop_path):
            logger.info(
                "Skipping visual crop for block_id=%s: render failed for source=%s page=%s bbox=%s.",
                ref.block_id,
                source_path,
                ref.page_number,
                ref.bbox,
            )
            return None
        crop = self._crop_metadata(ref, crop_path)
        if crop is None:
            logger.info("Skipping visual crop for block_id=%s: generated crop metadata was unreadable.", ref.block_id)
        return crop

    def _source_path(self, document_id: str) -> Path | None:
        for document in self._document_records(document_id):
            source_path = document.get("source_path")
            if isinstance(source_path, str) and source_path.strip():
                path = Path(source_path)
                if path.exists():
                    return path
                logger.info("Registered source path does not exist for document_id=%s: %s.", document_id, path)

            fingerprint = document.get("source_fingerprint")
            file_size = _int_or_none(document.get("file_size"))
            if isinstance(fingerprint, str) and fingerprint.strip():
                path = self._find_source_by_fingerprint(fingerprint.strip(), file_size)
                if path is not None:
                    logger.info("Resolved source PDF by fingerprint for document_id=%s: %s.", document_id, path)
                    return path
        return None

    def _document_records(self, document_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for documents_path in self._documents_paths():
            try:
                payload = json.loads(documents_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            for document in payload.get("documents", []):
                if isinstance(document, dict) and document.get("document_id") == document_id:
                    records.append(document)
        return records

    def _documents_paths(self) -> list[Path]:
        roots = [
            self.settings.indexer_output_dir,
            PROJECT_ROOT / "shared" / "data",
            PROJECT_ROOT / "gost-chat" / "data",
        ]
        paths: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            path = (root / "metadata" / "documents.json").resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def _find_source_by_fingerprint(self, fingerprint: str, file_size: int | None) -> Path | None:
        expected = fingerprint.casefold()
        for source_root in self._source_roots():
            if not source_root.exists():
                continue
            for path in source_root.rglob("*.pdf"):
                try:
                    if file_size is not None and path.stat().st_size != file_size:
                        continue
                    if _sha256(path) == expected:
                        return path
                except OSError:
                    continue
        return None

    def _source_roots(self) -> list[Path]:
        roots = [
            *self.settings.source_roots,
            PROJECT_ROOT / "docs",
            PROJECT_ROOT / "gost-chat" / "docs",
        ]
        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                unique_roots.append(resolved)
                seen.add(resolved)
        return unique_roots

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
