import logging
from pathlib import Path

from config import IndexerConfig, load_config
from services.chunker import TextChunker
from services.index_writer import IndexWriter
from services.metadata_service import calculate_file_hash, build_document_id, utc_now_iso
from services.pdf_loader import PdfLoader


logger = logging.getLogger("indexer")


def find_pdf_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []
    if not input_dir.is_dir():
        logger.warning("Input path is not a directory: %s", input_dir)
        return []
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def run_indexer(config: IndexerConfig) -> dict[str, object]:
    writer = IndexWriter(index_dir=config.index_dir, metadata_dir=config.metadata_dir)
    if config.clear:
        removed_paths = writer.clear_outputs()
        logger.info("Cleared indexer output files: %s", len(removed_paths))
        return {
            "schema_version": 1,
            "indexed_at": utc_now_iso(),
            "input_dir": str(config.input_dir.resolve()),
            "output_dir": str(config.output_dir.resolve()),
            "clear": True,
            "removed_files": [str(path.resolve()) for path in removed_paths],
            "outputs": {
                "chunks": str(writer.chunks_path.resolve()),
                "documents": str(writer.documents_path.resolve()),
                "summary": str(writer.summary_path.resolve()),
            },
        }

    loader = PdfLoader()
    chunker = TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    documents = writer.load_document_metadata()
    chunks_by_document = {}

    pdf_files = find_pdf_files(config.input_dir)
    logger.info("PDF files found: %s", len(pdf_files))

    indexed = 0
    skipped = 0
    failed = 0
    errors: list[dict[str, str]] = []

    for pdf_path in pdf_files:
        document_id = build_document_id(pdf_path)
        file_hash = calculate_file_hash(pdf_path)
        existing = documents.get(document_id)

        if existing and existing.get("file_hash") == file_hash and not config.reindex:
            skipped += 1
            logger.info("Skipped unchanged PDF: %s", pdf_path)
            continue

        try:
            extracted_pdf = loader.extract(pdf_path)
            chunks = chunker.chunk_document(document_id, extracted_pdf)
            now = utc_now_iso()
            documents[document_id] = {
                "document_id": document_id,
                "filename": pdf_path.name,
                "source_path": str(pdf_path.resolve()),
                "file_hash": file_hash,
                "indexed_at": now,
                "chunk_count": len(chunks),
                "status": "indexed",
            }
            chunks_by_document[document_id] = chunks
            indexed += 1
            logger.info("Indexed PDF: %s chunks=%s", pdf_path, len(chunks))
        except Exception as exc:
            failed += 1
            error_message = str(exc)
            errors.append({"source_path": str(pdf_path), "error": error_message})
            chunks_by_document[document_id] = []
            documents[document_id] = {
                "document_id": document_id,
                "filename": pdf_path.name,
                "source_path": str(pdf_path.resolve()),
                "file_hash": file_hash,
                "indexed_at": utc_now_iso(),
                "chunk_count": 0,
                "status": "failed",
                "error": error_message,
            }
            logger.exception("Failed to index PDF: %s", pdf_path)

    if chunks_by_document:
        writer.save_chunks(chunks_by_document)
    elif not writer.chunks_path.exists():
        writer.save_chunks({})

    writer.save_document_metadata(documents)

    summary = {
        "schema_version": 1,
        "indexed_at": utc_now_iso(),
        "input_dir": str(config.input_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "reindex": config.reindex,
        "clear": False,
        "files_found": len(pdf_files),
        "files_indexed": indexed,
        "files_skipped": skipped,
        "files_failed": failed,
        "errors": errors,
        "outputs": {
            "chunks": str(writer.chunks_path.resolve()),
            "documents": str(writer.documents_path.resolve()),
            "summary": str(writer.summary_path.resolve()),
        },
    }
    writer.save_summary(summary)
    return summary


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    configure_logging()
    config = load_config()
    summary = run_indexer(config)

    if summary.get("clear"):
        print()
        print("Index cleared")
        print("-------------")
        print(f"Files removed: {len(summary['removed_files'])}")
        for path in summary["removed_files"]:
            print(f"Removed:       {path}")
        print(f"Chunks file:   {summary['outputs']['chunks']}")
        print(f"Metadata file: {summary['outputs']['documents']}")
        print(f"Summary file:  {summary['outputs']['summary']}")
        return

    print()
    print("Indexing summary")
    print("----------------")
    print(f"Files found:   {summary['files_found']}")
    print(f"Files indexed: {summary['files_indexed']}")
    print(f"Files skipped: {summary['files_skipped']}")
    print(f"Files failed:  {summary['files_failed']}")
    print(f"Chunks file:   {summary['outputs']['chunks']}")
    print(f"Metadata file: {summary['outputs']['documents']}")
    print(f"Summary file:  {summary['outputs']['summary']}")


if __name__ == "__main__":
    main()
