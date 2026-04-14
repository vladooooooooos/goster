from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPdf:
    source_path: Path
    file_name: str
    pages: list[ExtractedPage]


class PdfLoader:
    def extract(self, source_path: Path) -> ExtractedPdf:
        try:
            import fitz
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyMuPDF is not installed. Activate the project virtual environment "
                "or run 'pip install -r requirements.txt'."
            ) from exc

        pages: list[ExtractedPage] = []

        with fitz.open(source_path) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                pages.append(ExtractedPage(page_number=index, text=text))

        return ExtractedPdf(
            source_path=source_path,
            file_name=source_path.name,
            pages=pages,
        )
