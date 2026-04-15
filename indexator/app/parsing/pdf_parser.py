"""PyMuPDF-based raw PDF parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class ParsedTextBlock:
    """Raw text block extracted from a PDF page."""

    page_number: int
    text: str
    bbox: tuple[float, float, float, float]
    order_index: int


@dataclass(frozen=True)
class ParsedImageBlock:
    """Raw image block extracted from a PDF page."""

    page_number: int
    bbox: tuple[float, float, float, float]
    order_index: int


@dataclass(frozen=True)
class ParsedPage:
    """Raw page-level parsing result."""

    page_number: int
    text: str
    text_blocks: list[ParsedTextBlock]
    image_blocks: list[ParsedImageBlock]
    width: float
    height: float
    rotation: int


@dataclass(frozen=True)
class ParsedDocument:
    """Raw document-level parsing result for future block building."""

    file_name: str
    file_path: Path
    page_count: int
    pages: list[ParsedPage]


class PdfParser:
    """Parser for extracting raw page text and text block geometry from PDFs."""

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF file into raw page-level data."""
        path = Path(file_path)

        with pymupdf.open(path) as document:
            pages = [self._parse_page(page, page_index + 1) for page_index, page in enumerate(document)]

        return ParsedDocument(
            file_name=path.name,
            file_path=path.resolve(),
            page_count=len(pages),
            pages=pages,
        )

    def _parse_page(self, page: pymupdf.Page, page_number: int) -> ParsedPage:
        page_text = page.get_text("text")
        raw_blocks = page.get_text("blocks", sort=False)
        text_blocks = [
            ParsedTextBlock(
                page_number=page_number,
                text=str(block[4]).strip(),
                bbox=(float(block[0]), float(block[1]), float(block[2]), float(block[3])),
                order_index=order_index,
            )
            for order_index, block in enumerate(raw_blocks)
            if len(block) >= 5 and str(block[4]).strip()
        ]
        image_blocks = self._parse_image_blocks(page, page_number)

        return ParsedPage(
            page_number=page_number,
            text=page_text,
            text_blocks=text_blocks,
            image_blocks=image_blocks,
            width=float(page.rect.width),
            height=float(page.rect.height),
            rotation=int(page.rotation),
        )

    def _parse_image_blocks(self, page: pymupdf.Page, page_number: int) -> list[ParsedImageBlock]:
        """Parse image block geometry from the page text dictionary."""
        raw_blocks = page.get_text("dict", sort=False).get("blocks", [])
        return [
            ParsedImageBlock(
                page_number=page_number,
                bbox=(
                    float(block["bbox"][0]),
                    float(block["bbox"][1]),
                    float(block["bbox"][2]),
                    float(block["bbox"][3]),
                ),
                order_index=order_index,
            )
            for order_index, block in enumerate(raw_blocks)
            if block.get("type") == 1 and "bbox" in block
        ]
