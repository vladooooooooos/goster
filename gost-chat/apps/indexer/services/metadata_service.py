import hashlib
from datetime import UTC, datetime
from pathlib import Path


def build_document_id(source_path: Path) -> str:
    normalized_path = source_path.resolve().as_posix().lower()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:16]


def calculate_file_hash(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
