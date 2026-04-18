# Adaptive Grounded Visual Evidence Design

## Goal

Build one text-first grounded RAG pipeline that can adaptively select enough textual evidence, decide when indexed visual metadata matters, lazily crop the referenced PDF fragment, and expose that visual evidence in both the answer payload and the source list.

## Current Context

GOST Chat already has a layered retrieval flow:

- `gost-chat/app/services/retrieval_pipeline.py` retrieves vector or JSON candidates and optionally reranks them.
- `gost-chat/app/services/context_builder.py` converts reranked blocks into prompt context and citation previews.
- `gost-chat/app/services/rag_service.py` performs the grounded LLM answer pass.
- `shared/vector_store/payloads.py` already stores compact visual metadata for visual-capable blocks, including `has_visual_evidence`, `bbox`, `page_number`, `block_type`, and `label`.
- `indexator/app/storage/document_registry.py` persists each indexed document's `source_path`, which is enough to reopen the original PDF for lazy crop generation.
- `gost-chat/app/static/app.js` and `gost-chat/app/static/style.css` already render text citations and can be extended to render optional visual previews.

The design should preserve existing grounded behavior and response compatibility while adding optional fields for adaptive and visual-aware behavior.

## Recommended Approach

Use a single adaptive grounded pipeline:

1. Retrieve a broad candidate pool.
2. Rerank that broad pool.
3. Let `ContextBuilder` adaptively select the final text evidence.
4. Build visual hints from selected and high-ranked visual-capable blocks.
5. Ask the LLM whether text is enough or whether a retrieved visual fragment should be shown.
6. Apply backend guardrails to any visual decision.
7. Lazily crop the requested PDF region only when needed.
8. Produce the final grounded answer and attach visual artifacts to answer metadata and citations.

This keeps visual support connected to the text-grounded answer flow rather than creating a separate image display feature.

## Non-Goals

- Do not store raw image bytes in Qdrant payloads.
- Do not pre-render crops for every visual block during indexing.
- Do not replace text retrieval with image retrieval.
- Do not make every answer include visual evidence.
- Do not claim automated image interpretation when no multimodal inspection has happened.
- Do not introduce unrelated refactors or large UI redesigns.

## Architecture

### Retrieval Pipeline

`RetrievalPipeline.retrieve()` should treat `top_k` as requested retrieval depth, not as the final prompt evidence count. When the reranker is enabled, retrieval should request at least `settings.reranker_top_k`, and reranking should return a broad pool bounded by `settings.reranker_top_n`. The context builder then decides how many blocks enter the prompt.

The default `/ask` `top_k` should increase from `5` to `12`, with a wider upper bound, so the reranker and context builder are not starved.

### Adaptive Context Builder

`ContextBuilderSettings` should add:

- `min_blocks: int = 2`
- `soft_target_blocks: int = 5`
- `max_blocks: int = 10`
- `max_context_chars: int = 18000`
- `adaptive_score_threshold: float = 0.12`

Selection rules:

- Always include the first relevant non-duplicate block if it fits.
- Skip duplicate block IDs and duplicate normalized text.
- Respect `max_chars_per_block` and `max_context_chars`.
- Continue through reranked blocks until `min_blocks` is satisfied when possible.
- After `soft_target_blocks`, stop when the current rerank score has dropped significantly from the best score.
- Stop at `max_blocks`.
- Track why selection stopped or skipped blocks in debug stats.

`BuiltContext` should include both final text context and a visual hint bundle.

### Visual Hint Models

Add compact dataclasses in a focused service module, for example `gost-chat/app/services/visual_evidence.py`:

- `VisualEvidenceRef`
- `ContextVisualHints`
- `VisualEvidenceDecision`
- `GeneratedCrop`

`VisualEvidenceRef` should be derived from retrieval payload metadata. It should include:

- `block_id`
- `document_id`
- `page_number`
- `bbox`
- `block_type`
- `label`
- `source_file`
- `text_preview`
- optional existing crop metadata

Only blocks with `has_visual_evidence` and usable `document_id`, `page_number`, and `bbox` should be crop-capable.

### Visual Decision Layer

`RagService` should perform a lightweight LLM decision step after building text context and visual hints.

The decision output should be constrained to a small JSON object:

- `mode`: `text_only`, `show_visual`, or `inspect_visual_and_show`
- `target_block_ids`: list of retrieved visual block IDs
- `show_in_sources`: boolean
- `show_in_answer`: boolean
- `needs_multimodal_followup`: boolean
- `reason`: short text

Backend guardrails should validate:

- The requested block IDs exist in `ContextVisualHints`.
- Each target has `document_id`, `page_number`, and `bbox`.
- The number of crops is within a small configured limit.
- A source PDF path can be resolved.
- Crop dimensions and bbox values are sane.

If the decision is malformed or unsafe, fall back to `text_only`.

### Lazy Crop Service

Add `gost-chat/app/services/visual_crop_service.py`.

Responsibilities:

- Resolve `document_id` to the original PDF path using `data/metadata/documents.json`.
- Validate page number and bbox.
- Render the crop with PyMuPDF `Page.get_pixmap(..., clip=...)`.
- Save the crop as a deterministic sidecar file under `data/crops/<document_id>/`.
- Reuse an existing crop file when the same block, page, bbox, DPI, and format are requested.

Default settings:

- `GOST_CHAT_VISUAL_CROPS_DIR=data/crops`
- `GOST_CHAT_VISUAL_CROP_DPI=160`
- `GOST_CHAT_VISUAL_MAX_CROPS_PER_ANSWER=1`
- `GOST_CHAT_VISUAL_ENABLE_DECISION=true`

The initial format should be PNG because formulas, tables, drawings, and line art benefit from lossless output.

### RAG Answer Flow

The answer flow should be:

1. Retrieve candidates.
2. Rerank candidates.
3. Build adaptive text context and visual hints.
4. If no selected text context exists, return the existing no-reliable-answer response.
5. Ask for a visual decision when visual hints exist.
6. Generate guarded lazy crops if the decision requests visual evidence.
7. Build the final grounded prompt.
8. Return citations, retrieved chunks, retrieval info, and optional visual evidence metadata.

If a multimodal second pass is unavailable, the final answer must not claim the crop was interpreted. It may state that a related visual fragment was found and attached when the surrounding text supports that statement.

### API Schema

Extend `AskCitation` and `AskRetrievedChunk` with optional visual metadata fields:

- `block_type`
- `label`
- `has_visual_evidence`
- `visual_evidence`

Extend `AskResponse` with:

- `visual_evidence: list[AskVisualEvidence] = []`

`AskVisualEvidence` should expose safe, UI-oriented fields:

- `block_id`
- `document_id`
- `source_file`
- `page_number`
- `block_type`
- `label`
- `crop_path`
- `crop_url`
- `width`
- `height`
- `format`
- `dpi`

Existing clients should continue to work when these fields are absent or empty.

### UI Support

The chat UI should:

- Render answer-level visual previews when `response.visual_evidence` is present.
- Render citation-level visual previews when a citation references a generated crop.
- Keep text citation metadata next to the visual preview.
- Avoid showing an image without source metadata.
- Store visual evidence in existing local chat session state along with the assistant response.

The UI should not be redesigned beyond what is required to expose visual evidence clearly.

## Configuration Defaults

Add chat settings:

- `context_min_blocks = 2`
- `context_soft_target_blocks = 5`
- `context_max_blocks = 10`
- `context_max_chars = 18000`
- `context_adaptive_score_threshold = 0.12`
- `visual_enable_decision = true`
- `visual_crops_dir = Path("data/crops")`
- `visual_crop_dpi = 160`
- `visual_max_crops_per_answer = 1`

Update `.env.example` with:

- `GOST_CHAT_LLM_MAX_TOKENS=5000`
- `GOST_CHAT_RERANKER_TOP_K=40`
- `GOST_CHAT_RERANKER_TOP_N=12`
- the context settings above
- the visual settings above

## Error Handling

- Missing index data should keep the current API error behavior.
- Missing or invalid visual metadata should skip visual crop generation and keep the text answer path.
- Missing source PDF should be reported in debug `retrieval_info.visual`, not as a fatal answer failure.
- Crop generation errors should not prevent a text-grounded answer.
- Invalid LLM decision JSON should be treated as `text_only`.

## Testing Strategy

Use the existing `unittest` style.

Add focused tests for:

- adaptive context selection with min, soft target, max, score drop, duplicate, and budget behavior
- visual hint extraction from selected and candidate reranked blocks
- retrieval pipeline broad candidate/rerank pool behavior
- crop service PDF path resolution, bbox validation, deterministic file naming, and successful crop generation
- RAG service visual fallback behavior without multimodal inspection
- API schema serialization of optional visual evidence

Run:

- `python -m unittest discover -s tests`
- `python -m unittest discover -s gost-chat/tests`
- `python -m unittest discover -s indexator/tests`
- `python -m compileall app tests` from `gost-chat`

## Rollout Plan

1. Implement adaptive text grounding and context settings first.
2. Add visual hint models and extraction.
3. Add guarded visual decision parsing and fallback behavior.
4. Add lazy crop generation.
5. Wire visual evidence into `RagService` response objects and API schemas.
6. Add UI rendering for answer-level and citation-level crops.
7. Extend debug transparency in `retrieval_info`.

Each step should be covered by a failing test before production code changes.

## Open Decisions

The first implementation should not add a full multimodal image-reading pass. It should prepare the interfaces for it and honestly expose generated visual evidence as attached document fragments. A later implementation can extend `LlmService` with provider-specific multimodal messages once the target provider contract is confirmed.
