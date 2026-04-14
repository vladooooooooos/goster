import argparse
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_DIR = "docs"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 300


@dataclass(frozen=True)
class IndexerConfig:
    input_dir: Path
    output_dir: Path
    chunk_size: int
    chunk_overlap: int
    reindex: bool
    clear: bool

    @property
    def index_dir(self) -> Path:
        return self.output_dir / "index"

    @property
    def metadata_dir(self) -> Path:
        return self.output_dir / "metadata"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index local PDF files into JSON data for future RAG retrieval."
    )
    parser.add_argument(
        "--input-dir",
        default=os.getenv("GOST_INDEXER_INPUT_DIR", DEFAULT_INPUT_DIR),
        help="Directory with source PDF files. Defaults to env GOST_INDEXER_INPUT_DIR or docs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("GOST_INDEXER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Directory where index and metadata files are written. Defaults to env GOST_INDEXER_OUTPUT_DIR or data.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("GOST_INDEXER_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("GOST_INDEXER_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex all PDF files even when their file hash has not changed.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete indexer output files and exit without scanning PDF files.",
    )
    return parser


def load_config() -> IndexerConfig:
    args = build_parser().parse_args()
    config = IndexerConfig(
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reindex=args.reindex,
        clear=args.clear,
    )
    validate_config(config)
    return config


def validate_config(config: IndexerConfig) -> None:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater.")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
