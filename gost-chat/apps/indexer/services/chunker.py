from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.pdf_loader import ExtractedPdf


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    file_name: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document_id: str, extracted_pdf: "ExtractedPdf") -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_text = ""
        page_start: int | None = None
        page_end: int | None = None

        for page in extracted_pdf.pages:
            for segment in self._split_text(page.text):
                if not segment:
                    continue

                if not chunk_text:
                    chunk_text = segment
                    page_start = page.page_number
                    page_end = page.page_number
                    continue

                candidate = f"{chunk_text}\n\n{segment}"
                if len(candidate) <= self.chunk_size:
                    chunk_text = candidate
                    page_end = page.page_number
                    continue

                chunks.append(
                    self._build_chunk(
                        document_id=document_id,
                        file_name=extracted_pdf.file_name,
                        ordinal=len(chunks) + 1,
                        page_start=page_start or page.page_number,
                        page_end=page_end or page.page_number,
                        text=chunk_text,
                    )
                )
                chunk_text = self._with_overlap(chunk_text, segment)
                page_start = page.page_number
                page_end = page.page_number

        if chunk_text:
            chunks.append(
                self._build_chunk(
                    document_id=document_id,
                    file_name=extracted_pdf.file_name,
                    ordinal=len(chunks) + 1,
                    page_start=page_start or 1,
                    page_end=page_end or page_start or 1,
                    text=chunk_text,
                )
            )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized = "\n".join(line.strip() for line in text.splitlines())
        paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
        segments: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                segments.append(paragraph)
                continue
            segments.extend(self._split_long_text(paragraph))

        return segments

    def _split_long_text(self, text: str) -> list[str]:
        words = text.split()
        segments: list[str] = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                segments.append(current)
            current = word

        if current:
            segments.append(current)

        return segments

    def _with_overlap(self, previous_text: str, next_segment: str) -> str:
        if self.chunk_overlap == 0:
            return next_segment

        overlap = previous_text[-self.chunk_overlap :].strip()
        if not overlap:
            return next_segment
        return f"{overlap}\n\n{next_segment}"

    def _build_chunk(
        self,
        document_id: str,
        file_name: str,
        ordinal: int,
        page_start: int,
        page_end: int,
        text: str,
    ) -> TextChunk:
        return TextChunk(
            document_id=document_id,
            file_name=file_name,
            chunk_id=f"{document_id}:{ordinal:05d}",
            page_start=page_start,
            page_end=page_end,
            text=text.strip(),
        )
