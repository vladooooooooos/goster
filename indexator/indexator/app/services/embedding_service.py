"""Service helpers for embedding structured blocks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.core.blocks import StructuredBlock
from app.embedding.base import TextEmbedding
from app.embedding.local_backend import LocalBgeM3Embedder, LocalEmbeddingSettings
from app.utils.config import EmbeddingConfig


@dataclass(frozen=True)
class BlockEmbedding:
    """Embedding result linked back to a structured block."""

    block_id: str
    block_type: str
    page_number: int
    reading_order: int
    text: str
    vector: list[float]


@dataclass(frozen=True)
class BlockEmbeddingRun:
    """Debug metadata for one block embedding run."""

    embeddings: list[BlockEmbedding]
    model_name: str
    device: str
    embedding_dimension: int
    elapsed_seconds: float


class StructuredBlockEmbeddingService:
    """Prepare retrieval text and generate local embeddings for structured blocks."""

    def __init__(self, embedder: LocalBgeM3Embedder) -> None:
        self.embedder = embedder

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "StructuredBlockEmbeddingService":
        """Create the service from application embedding config."""
        settings = LocalEmbeddingSettings(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
        )
        return cls(LocalBgeM3Embedder(settings))

    def embed_blocks(self, blocks: list[StructuredBlock]) -> BlockEmbeddingRun:
        """Embed structured blocks and return debug run metadata."""
        prepared_inputs = [
            (block, text)
            for block in blocks
            if (text := build_block_embedding_text(block))
        ]
        if not prepared_inputs:
            return BlockEmbeddingRun(
                embeddings=[],
                model_name=self.embedder.settings.model_name,
                device=self.embedder.device,
                embedding_dimension=0,
                elapsed_seconds=0.0,
            )

        start_time = perf_counter()
        text_embeddings = self.embedder.embed_texts([text for _, text in prepared_inputs])
        elapsed_seconds = perf_counter() - start_time

        block_embeddings = [
            make_block_embedding(block, text_embedding)
            for (block, _), text_embedding in zip(prepared_inputs, text_embeddings, strict=True)
        ]
        embedding_dimension = len(block_embeddings[0].vector)

        return BlockEmbeddingRun(
            embeddings=block_embeddings,
            model_name=self.embedder.settings.model_name,
            device=self.embedder.device,
            embedding_dimension=embedding_dimension,
            elapsed_seconds=elapsed_seconds,
        )


def build_block_embedding_text(block: StructuredBlock) -> str:
    """Build stable retrieval text for one structured block."""
    parts: list[str] = []
    section_path = " > ".join(item.strip() for item in block.section_path if item.strip())

    if section_path:
        parts.append(f"Section: {section_path}")
    if block.label:
        parts.append(f"Label: {block.label.strip()}")

    main_text = block.text.strip()
    context_text = (block.context_text or "").strip()

    if block.block_type in {"figure", "formula_with_context", "table"} and context_text:
        parts.append(f"Context: {context_text}")
    if main_text:
        parts.append(f"Text: {main_text}")

    return "\n".join(parts).strip()


def make_block_embedding(block: StructuredBlock, text_embedding: TextEmbedding) -> BlockEmbedding:
    """Attach one text embedding to its source structured block."""
    return BlockEmbedding(
        block_id=block.id,
        block_type=block.block_type,
        page_number=block.page_number,
        reading_order=block.reading_order,
        text=text_embedding.text,
        vector=text_embedding.vector,
    )
