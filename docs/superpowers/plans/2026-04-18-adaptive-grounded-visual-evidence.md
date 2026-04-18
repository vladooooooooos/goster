# Adaptive Grounded Visual Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one adaptive grounded RAG pipeline that can select textual evidence dynamically, route to indexed visual evidence when needed, lazily crop PDF fragments, and expose those visual fragments in answers and sources.

**Architecture:** Retrieval returns a broad reranked candidate pool, `ContextBuilder` performs final adaptive evidence selection and visual hint extraction, `RagService` runs a guarded visual decision step, and `VisualCropService` lazily renders crop files from original PDFs. API and UI changes add optional visual evidence fields without breaking existing text-only responses.

**Tech Stack:** Python 3, FastAPI, Pydantic, PyMuPDF, unittest, browser JavaScript, CSS, existing Qdrant/vector payload metadata.

---

## File Structure

- Modify `gost-chat/app/schemas/ask.py`: raise request default `top_k`, add visual evidence response models and optional visual fields.
- Modify `gost-chat/app/config.py`: add adaptive context and visual crop settings.
- Modify `gost-chat/app/main.py`: pass configured `ContextBuilderSettings`, initialize `VisualCropService`, and mount crop static files when available.
- Modify `gost-chat/app/services/retrieval_pipeline.py`: keep broad reranked candidate pools available for context building.
- Modify `gost-chat/app/services/context_builder.py`: add adaptive evidence selection, expanded stats, and visual hint output.
- Create `gost-chat/app/services/visual_evidence.py`: visual evidence dataclasses, payload extraction, decision parsing, and guardrail helpers.
- Create `gost-chat/app/services/visual_crop_service.py`: document registry lookup, bbox validation, deterministic crop rendering, and URL-safe crop metadata.
- Modify `gost-chat/app/services/rag_service.py`: add visual decision flow, crop generation, richer citations, and honest fallback behavior without multimodal inspection.
- Modify `gost-chat/app/api/ask.py`: serialize optional visual evidence.
- Modify `gost-chat/app/static/app.js`: render answer-level and citation-level visual previews.
- Modify `gost-chat/app/static/style.css`: style visual previews inside existing chat layout.
- Modify `gost-chat/.env.example`: document new defaults.
- Add tests under `gost-chat/tests/` for context builder, retrieval pipeline, visual evidence, crop service, RAG service, and schema serialization.

---

### Task 1: Request And Configuration Defaults

**Files:**
- Modify: `gost-chat/app/schemas/ask.py`
- Modify: `gost-chat/app/config.py`
- Modify: `gost-chat/app/main.py`
- Modify: `gost-chat/.env.example`
- Test: `gost-chat/tests/test_adaptive_settings.py`

- [ ] **Step 1: Write the failing test**

Create `gost-chat/tests/test_adaptive_settings.py`:

```python
import unittest

from app.config import Settings
from app.schemas.ask import AskRequest
from app.services.context_builder import ContextBuilderSettings


class AdaptiveSettingsTest(unittest.TestCase):
    def test_ask_request_uses_wider_default_top_k(self):
        request = AskRequest(query="What is required?")

        self.assertEqual(request.top_k, 12)

    def test_settings_expose_adaptive_context_defaults(self):
        settings = Settings()

        self.assertEqual(settings.context_min_blocks, 2)
        self.assertEqual(settings.context_soft_target_blocks, 5)
        self.assertEqual(settings.context_max_blocks, 10)
        self.assertEqual(settings.context_max_chars, 18000)
        self.assertEqual(settings.context_adaptive_score_threshold, 0.12)

    def test_settings_expose_visual_defaults(self):
        settings = Settings()

        self.assertTrue(settings.visual_enable_decision)
        self.assertEqual(settings.visual_crops_dir.as_posix(), "data/crops")
        self.assertEqual(settings.visual_crop_dpi, 160)
        self.assertEqual(settings.visual_max_crops_per_answer, 1)

    def test_context_builder_settings_accept_adaptive_values(self):
        settings = ContextBuilderSettings(
            min_blocks=2,
            soft_target_blocks=5,
            max_blocks=10,
            max_context_chars=18000,
            adaptive_score_threshold=0.12,
        )

        self.assertEqual(settings.min_blocks, 2)
        self.assertEqual(settings.soft_target_blocks, 5)
        self.assertEqual(settings.max_blocks, 10)
        self.assertEqual(settings.max_context_chars, 18000)
        self.assertEqual(settings.adaptive_score_threshold, 0.12)
```

- [ ] **Step 2: Run test to verify it fails**

Run from repository root:

```powershell
python -m unittest gost-chat.tests.test_adaptive_settings -v
```

Expected: FAIL because `AskRequest.top_k` is still `5`, `Settings` has no adaptive or visual fields, and `ContextBuilderSettings` has no new adaptive fields.

- [ ] **Step 3: Write minimal implementation**

In `gost-chat/app/schemas/ask.py`, change `AskRequest`:

```python
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer from indexed documents.")
    top_k: int = Field(12, ge=1, le=50, description="Maximum number of source chunks to retrieve.")
```

In `gost-chat/app/config.py`, add fields to `Settings`:

```python
    context_min_blocks: int = 2
    context_soft_target_blocks: int = 5
    context_max_blocks: int = 10
    context_max_chars: int = 18000
    context_adaptive_score_threshold: float = 0.12
    visual_enable_decision: bool = True
    visual_crops_dir: Path = Path("data/crops")
    visual_crop_dpi: int = 160
    visual_max_crops_per_answer: int = 1
```

In `gost-chat/app/services/context_builder.py`, expand `ContextBuilderSettings`:

```python
@dataclass(frozen=True)
class ContextBuilderSettings:
    min_blocks: int = 2
    soft_target_blocks: int = 5
    max_blocks: int = 10
    max_context_chars: int = 18000
    max_chars_per_block: int = 4000
    evidence_preview_chars: int = 320
    enable_near_duplicate_filter: bool = False
    adaptive_score_threshold: float = 0.12
```

In `gost-chat/app/main.py`, import `ContextBuilderSettings` and initialize:

```python
app.state.context_builder = ContextBuilder(
    ContextBuilderSettings(
        min_blocks=settings.context_min_blocks,
        soft_target_blocks=settings.context_soft_target_blocks,
        max_blocks=settings.context_max_blocks,
        max_context_chars=settings.context_max_chars,
        adaptive_score_threshold=settings.context_adaptive_score_threshold,
    )
)
```

Update `gost-chat/.env.example`:

```env
GOST_CHAT_LLM_MAX_TOKENS=5000
GOST_CHAT_RERANKER_TOP_K=40
GOST_CHAT_RERANKER_TOP_N=12
GOST_CHAT_CONTEXT_MIN_BLOCKS=2
GOST_CHAT_CONTEXT_SOFT_TARGET_BLOCKS=5
GOST_CHAT_CONTEXT_MAX_BLOCKS=10
GOST_CHAT_CONTEXT_MAX_CHARS=18000
GOST_CHAT_CONTEXT_ADAPTIVE_SCORE_THRESHOLD=0.12
GOST_CHAT_VISUAL_ENABLE_DECISION=true
GOST_CHAT_VISUAL_CROPS_DIR=data/crops
GOST_CHAT_VISUAL_CROP_DPI=160
GOST_CHAT_VISUAL_MAX_CROPS_PER_ANSWER=1
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest gost-chat.tests.test_adaptive_settings -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gost-chat/app/schemas/ask.py gost-chat/app/config.py gost-chat/app/main.py gost-chat/app/services/context_builder.py gost-chat/.env.example gost-chat/tests/test_adaptive_settings.py
git commit -m "feat: add adaptive chat settings"
```

---

### Task 2: Adaptive Context Selection

**Files:**
- Modify: `gost-chat/app/services/context_builder.py`
- Test: `gost-chat/tests/test_context_builder_adaptive.py`

- [ ] **Step 1: Write the failing test**

Create `gost-chat/tests/test_context_builder_adaptive.py`:

```python
import unittest

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.retrieval_types import RerankedBlock


def block(
    block_id: str,
    text: str,
    rerank_score: float | None,
    retrieval_score: float = 0.5,
    payload: dict | None = None,
) -> RerankedBlock:
    return RerankedBlock(
        block_id=block_id,
        text=text,
        retrieval_text=text,
        source_file="source.pdf",
        page=1,
        section_path=[],
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        payload=payload or {},
        document_id="doc-1",
        page_start=1,
        page_end=1,
        block_type="paragraph",
        label=None,
    )


class AdaptiveContextBuilderTest(unittest.TestCase):
    def test_selects_more_than_soft_target_until_score_drop(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=2,
                soft_target_blocks=3,
                max_blocks=10,
                adaptive_score_threshold=0.20,
                max_context_chars=20000,
            )
        )
        ranked = [
            block("b1", "first relevant block", 1.00),
            block("b2", "second relevant block", 0.95),
            block("b3", "third relevant block", 0.90),
            block("b4", "fourth still close block", 0.83),
            block("b5", "fifth dropped block", 0.50),
        ]

        built = builder.build("query", ranked)

        self.assertEqual([item.block.block_id for item in built.selected], ["b1", "b2", "b3", "b4"])
        self.assertEqual(built.stats.stop_reason, "score_drop")
        self.assertEqual(built.stats.input_count, 5)
        self.assertEqual(built.stats.selected_count, 4)

    def test_keeps_minimum_blocks_even_when_second_score_drops(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=2,
                soft_target_blocks=2,
                max_blocks=5,
                adaptive_score_threshold=0.10,
                max_context_chars=20000,
            )
        )
        ranked = [
            block("b1", "first relevant block", 1.00),
            block("b2", "second low but required block", 0.20),
            block("b3", "third low block", 0.19),
        ]

        built = builder.build("query", ranked)

        self.assertEqual([item.block.block_id for item in built.selected], ["b1", "b2"])
        self.assertEqual(built.stats.stop_reason, "score_drop")

    def test_tracks_duplicates_and_budget(self):
        builder = ContextBuilder(
            ContextBuilderSettings(
                min_blocks=1,
                soft_target_blocks=5,
                max_blocks=5,
                max_context_chars=130,
                max_chars_per_block=200,
            )
        )
        ranked = [
            block("b1", "same text", 1.0),
            block("b1", "same text again by id", 0.9),
            block("b2", "same text", 0.8),
            block("b3", "large unique text " * 20, 0.7),
        ]

        built = builder.build("query", ranked)

        self.assertEqual(len(built.selected), 1)
        self.assertGreaterEqual(built.stats.dropped_duplicate_count, 2)
        self.assertGreaterEqual(built.stats.dropped_budget_count, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest gost-chat.tests.test_context_builder_adaptive -v
```

Expected: FAIL because `ContextBuildStats.stop_reason` does not exist and selection is still fixed by `max_blocks`.

- [ ] **Step 3: Write minimal implementation**

In `gost-chat/app/services/context_builder.py`, extend `ContextBuildStats`:

```python
@dataclass(frozen=True)
class ContextBuildStats:
    input_count: int
    selected_count: int
    dropped_duplicate_count: int
    dropped_budget_count: int
    truncated_count: int
    total_chars_included: int
    max_context_chars: int
    min_blocks: int
    soft_target_blocks: int
    max_blocks: int
    stop_reason: str
```

Replace the selection loop with adaptive logic:

```python
        best_score: float | None = None
        stop_reason = "exhausted"

        for block in ranked_blocks:
            if len(selected) >= max(1, self.settings.max_blocks):
                dropped_budget_count += 1
                stop_reason = "max_blocks"
                continue

            score = block.rerank_score if block.rerank_score is not None else block.retrieval_score
            if best_score is None:
                best_score = score
            if (
                selected
                and len(selected) >= self.settings.min_blocks
                and len(selected) >= self.settings.soft_target_blocks
                and best_score > 0
                and score < best_score * (1 - self.settings.adaptive_score_threshold)
            ):
                stop_reason = "score_drop"
                break

            block_id = block.block_id.strip()
            if block_id and block_id in seen_block_ids:
                dropped_duplicate_count += 1
                continue

            normalized_text = normalize_text(block.evidence_text)
            if not normalized_text:
                dropped_duplicate_count += 1
                continue

            text_key = normalized_text.casefold()
            if text_key in seen_texts:
                dropped_duplicate_count += 1
                continue

            prompt_text = truncate_text(normalized_text, self.settings.max_chars_per_block)
            if len(prompt_text) < len(normalized_text):
                truncated_count += 1

            context_piece = self._format_context_piece(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
            )
            piece_chars = len(context_piece)
            separator_chars = 7 if selected else 0
            projected_chars = total_chars_included + separator_chars + piece_chars
            if selected and projected_chars > self.settings.max_context_chars:
                dropped_budget_count += 1
                stop_reason = "budget"
                continue

            evidence = ContextEvidence(
                index=len(selected) + 1,
                block=block,
                prompt_text=prompt_text,
                evidence_preview=truncate_text(normalized_text, self.settings.evidence_preview_chars),
                source_label=block.source_file,
                page_label=format_page_range(block.page_start, block.page_end),
            )
            selected.append(evidence)
            if block_id:
                seen_block_ids.add(block_id)
            seen_texts.add(text_key)
            total_chars_included = projected_chars
```

When creating stats, include:

```python
            min_blocks=self.settings.min_blocks,
            soft_target_blocks=self.settings.soft_target_blocks,
            max_blocks=self.settings.max_blocks,
            stop_reason=stop_reason,
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest gost-chat.tests.test_context_builder_adaptive -v
```

Expected: PASS.

- [ ] **Step 5: Run existing context smoke check**

Run:

```powershell
python scripts/context_builder_smoke.py
```

Expected: exits with code 0 and prints context builder smoke output.

- [ ] **Step 6: Commit**

```powershell
git add gost-chat/app/services/context_builder.py gost-chat/tests/test_context_builder_adaptive.py
git commit -m "feat: adapt context evidence selection"
```

---

### Task 3: Visual Evidence Models And Hint Extraction

**Files:**
- Create: `gost-chat/app/services/visual_evidence.py`
- Modify: `gost-chat/app/services/context_builder.py`
- Test: `gost-chat/tests/test_visual_evidence.py`

- [ ] **Step 1: Write the failing test**

Create `gost-chat/tests/test_visual_evidence.py`:

```python
import unittest

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.retrieval_types import RerankedBlock
from app.services.visual_evidence import (
    VisualEvidenceDecision,
    visual_ref_from_block,
    parse_visual_decision,
    guard_visual_decision,
)


def visual_block(block_id: str, score: float = 0.9) -> RerankedBlock:
    return RerankedBlock(
        block_id=block_id,
        text="Formula context",
        retrieval_text="Formula context",
        source_file="source.pdf",
        page=2,
        section_path=[],
        retrieval_score=score,
        rerank_score=score,
        payload={
            "has_visual_evidence": True,
            "bbox": [10.0, 20.0, 110.0, 120.0],
            "page_number": 2,
        },
        document_id="doc-1",
        page_start=2,
        page_end=2,
        block_type="formula_with_context",
        label="Formula 1",
    )


class VisualEvidenceTest(unittest.TestCase):
    def test_visual_ref_from_block_requires_metadata(self):
        ref = visual_ref_from_block(visual_block("v1"), text_preview="Formula context")

        self.assertIsNotNone(ref)
        self.assertEqual(ref.block_id, "v1")
        self.assertEqual(ref.document_id, "doc-1")
        self.assertEqual(ref.page_number, 2)
        self.assertEqual(ref.bbox, (10.0, 20.0, 110.0, 120.0))
        self.assertEqual(ref.block_type, "formula_with_context")
        self.assertEqual(ref.label, "Formula 1")

    def test_context_builder_returns_visual_hints(self):
        builder = ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2))

        built = builder.build("query", [visual_block("v1")])

        self.assertEqual(len(built.visual_hints.selected), 1)
        self.assertEqual(built.visual_hints.selected[0].block_id, "v1")
        self.assertEqual(built.visual_hints.total_count, 1)

    def test_parse_and_guard_decision_keeps_known_targets(self):
        decision = parse_visual_decision(
            '{"mode":"inspect_visual_and_show","target_block_ids":["v1","missing"],'
            '"show_in_sources":true,"show_in_answer":true,'
            '"needs_multimodal_followup":true,"reason":"Formula may be needed."}'
        )
        ref = visual_ref_from_block(visual_block("v1"), text_preview="Formula context")

        guarded = guard_visual_decision(decision, [ref], max_targets=1)

        self.assertEqual(guarded.mode, "inspect_visual_and_show")
        self.assertEqual(guarded.target_block_ids, ["v1"])
        self.assertTrue(guarded.show_in_sources)
        self.assertTrue(guarded.show_in_answer)

    def test_malformed_decision_falls_back_to_text_only(self):
        decision = parse_visual_decision("not json")

        self.assertEqual(decision, VisualEvidenceDecision.text_only("Invalid visual decision JSON."))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest gost-chat.tests.test_visual_evidence -v
```

Expected: FAIL because `visual_evidence.py` and `BuiltContext.visual_hints` do not exist.

- [ ] **Step 3: Write minimal implementation**

Create `gost-chat/app/services/visual_evidence.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.services.retrieval_types import RerankedBlock

VisualDecisionMode = Literal["text_only", "show_visual", "inspect_visual_and_show"]


@dataclass(frozen=True)
class VisualEvidenceRef:
    block_id: str
    document_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    block_type: str | None
    label: str | None
    source_file: str
    text_preview: str
    crop_path: str | None = None
    crop_status: str | None = None
    crop_width: int | None = None
    crop_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class ContextVisualHints:
    selected: list[VisualEvidenceRef]
    candidates: list[VisualEvidenceRef]

    @property
    def total_count(self) -> int:
        return len({ref.block_id for ref in [*self.selected, *self.candidates]})

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [ref.to_dict() for ref in self.selected],
            "candidates": [ref.to_dict() for ref in self.candidates],
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class VisualEvidenceDecision:
    mode: VisualDecisionMode
    target_block_ids: list[str]
    show_in_sources: bool
    show_in_answer: bool
    needs_multimodal_followup: bool
    reason: str

    @classmethod
    def text_only(cls, reason: str) -> "VisualEvidenceDecision":
        return cls(
            mode="text_only",
            target_block_ids=[],
            show_in_sources=False,
            show_in_answer=False,
            needs_multimodal_followup=False,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visual_ref_from_block(block: RerankedBlock, text_preview: str) -> VisualEvidenceRef | None:
    if block.payload.get("has_visual_evidence") is not True:
        return None
    bbox = _bbox(block.payload.get("bbox"))
    page_number = _int_value(block.payload.get("page_number")) or block.page_start or block.page
    if not block.block_id or not block.document_id or not page_number or bbox is None:
        return None
    return VisualEvidenceRef(
        block_id=block.block_id,
        document_id=block.document_id,
        page_number=page_number,
        bbox=bbox,
        block_type=block.block_type,
        label=block.label,
        source_file=block.source_file,
        text_preview=text_preview,
        crop_path=_str_or_none(block.payload.get("crop_path")),
        crop_status=_str_or_none(block.payload.get("crop_status")),
        crop_width=_int_value(block.payload.get("crop_width")),
        crop_height=_int_value(block.payload.get("crop_height")),
    )


def parse_visual_decision(raw: str) -> VisualEvidenceDecision:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return VisualEvidenceDecision.text_only("Invalid visual decision JSON.")
    if not isinstance(payload, dict):
        return VisualEvidenceDecision.text_only("Visual decision was not an object.")
    mode = payload.get("mode")
    if mode not in {"text_only", "show_visual", "inspect_visual_and_show"}:
        return VisualEvidenceDecision.text_only("Visual decision mode was invalid.")
    target_block_ids = [
        item.strip()
        for item in payload.get("target_block_ids", [])
        if isinstance(item, str) and item.strip()
    ]
    return VisualEvidenceDecision(
        mode=mode,
        target_block_ids=target_block_ids,
        show_in_sources=bool(payload.get("show_in_sources")),
        show_in_answer=bool(payload.get("show_in_answer")),
        needs_multimodal_followup=bool(payload.get("needs_multimodal_followup")),
        reason=_str_or_none(payload.get("reason")) or "",
    )


def guard_visual_decision(
    decision: VisualEvidenceDecision,
    available_refs: list[VisualEvidenceRef],
    max_targets: int,
) -> VisualEvidenceDecision:
    if decision.mode == "text_only":
        return decision
    available = {ref.block_id: ref for ref in available_refs}
    target_block_ids = [block_id for block_id in decision.target_block_ids if block_id in available]
    target_block_ids = target_block_ids[: max(0, max_targets)]
    if not target_block_ids:
        return VisualEvidenceDecision.text_only("No valid visual target blocks were requested.")
    return VisualEvidenceDecision(
        mode=decision.mode,
        target_block_ids=target_block_ids,
        show_in_sources=decision.show_in_sources,
        show_in_answer=decision.show_in_answer,
        needs_multimodal_followup=decision.needs_multimodal_followup,
        reason=decision.reason,
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _str_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
```

In `gost-chat/app/services/context_builder.py`, import `ContextVisualHints`, `VisualEvidenceRef`, and `visual_ref_from_block`. Add to `BuiltContext`:

```python
    visual_hints: ContextVisualHints
```

Build visual hints before returning:

```python
        visual_hints = self._build_visual_hints(selected, ranked_blocks)
```

Pass it into `BuiltContext`.

Add helper:

```python
    def _build_visual_hints(
        self,
        selected: list[ContextEvidence],
        ranked_blocks: list[RerankedBlock],
    ) -> ContextVisualHints:
        selected_refs = [
            ref
            for evidence in selected
            if (ref := visual_ref_from_block(evidence.block, evidence.evidence_preview))
        ]
        selected_ids = {ref.block_id for ref in selected_refs}
        candidate_refs: list[VisualEvidenceRef] = []
        for block in ranked_blocks:
            if block.block_id in selected_ids:
                continue
            preview = truncate_text(block.evidence_text, self.settings.evidence_preview_chars)
            ref = visual_ref_from_block(block, preview)
            if ref:
                candidate_refs.append(ref)
        return ContextVisualHints(selected=selected_refs, candidates=candidate_refs)
```

In `_with_context_info()` in `rag_service.py`, later tasks will include `built_context.visual_hints.to_dict()`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest gost-chat.tests.test_visual_evidence -v
```

Expected: PASS.

- [ ] **Step 5: Run context builder tests together**

Run:

```powershell
python -m unittest gost-chat.tests.test_context_builder_adaptive gost-chat.tests.test_visual_evidence -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add gost-chat/app/services/visual_evidence.py gost-chat/app/services/context_builder.py gost-chat/tests/test_visual_evidence.py
git commit -m "feat: extract visual evidence hints"
```

---

### Task 4: Broad Retrieval Pool

**Files:**
- Modify: `gost-chat/app/services/retrieval_pipeline.py`
- Test: `gost-chat/tests/test_retrieval_pipeline_pool.py`

- [ ] **Step 1: Write the failing test**

Create `gost-chat/tests/test_retrieval_pipeline_pool.py`:

```python
import unittest
from types import SimpleNamespace

from app.services.retrieval_pipeline import RetrievalPipeline
from app.services.retrieval_types import RetrievedBlock, make_reranked_block


def candidate(index: int) -> RetrievedBlock:
    return RetrievedBlock(
        block_id=f"b{index}",
        text=f"text {index}",
        retrieval_text=f"text {index}",
        source_file="source.pdf",
        page=1,
        section_path=[],
        retrieval_score=1.0 / index,
        payload={},
        document_id="doc-1",
        page_start=1,
        page_end=1,
    )


class FakeRetriever:
    def __init__(self):
        self.top_k = None

    def retrieve_blocks(self, query, top_k):
        self.top_k = top_k
        return [candidate(index) for index in range(1, top_k + 1)], {}


class FakeReranker:
    enabled = True

    def __init__(self):
        self.top_n = None

    def rerank(self, query, candidates, top_n):
        self.top_n = top_n
        return [make_reranked_block(item, 1.0 / (index + 1)) for index, item in enumerate(candidates[:top_n])]


class RetrievalPipelinePoolTest(unittest.TestCase):
    def test_reranker_receives_broad_pool_and_returns_context_pool(self):
        retriever = FakeRetriever()
        reranker = FakeReranker()
        settings = SimpleNamespace(
            retrieval_backend="json",
            reranker_top_k=40,
            reranker_top_n=12,
        )
        pipeline = RetrievalPipeline(retriever=retriever, settings=settings, reranker=reranker)

        result = pipeline.retrieve("query", top_k=12)

        self.assertEqual(retriever.top_k, 40)
        self.assertEqual(reranker.top_n, 12)
        self.assertEqual(len(result.candidates), 40)
        self.assertEqual(len(result.results), 12)
        self.assertEqual(result.info["candidate_pool_top_n"], 12)

    def test_requested_top_k_can_raise_retrieval_depth(self):
        retriever = FakeRetriever()
        reranker = FakeReranker()
        settings = SimpleNamespace(
            retrieval_backend="json",
            reranker_top_k=40,
            reranker_top_n=12,
        )
        pipeline = RetrievalPipeline(retriever=retriever, settings=settings, reranker=reranker)

        pipeline.retrieve("query", top_k=45)

        self.assertEqual(retriever.top_k, 45)
        self.assertEqual(reranker.top_n, 12)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest gost-chat.tests.test_retrieval_pipeline_pool -v
```

Expected: FAIL because `candidate_pool_top_n` is not present and the existing naming still calls the reranked pool `final_top_n`.

- [ ] **Step 3: Write minimal implementation**

In `gost-chat/app/services/retrieval_pipeline.py`, rename local variables for clarity:

```python
        candidate_pool_top_n = self._candidate_pool_top_n(top_k)
        retrieval_top_k = self._retrieval_top_k(top_k)
```

Set info:

```python
            "candidate_pool_top_n": candidate_pool_top_n,
```

Keep `"final_top_n"` temporarily for backward debug compatibility:

```python
            "final_top_n": candidate_pool_top_n,
```

Call reranker:

```python
        results = self._rerank_candidates(normalized_query, candidates, top_n=candidate_pool_top_n)
```

Rename `_final_top_n()` to `_candidate_pool_top_n()`:

```python
    def _candidate_pool_top_n(self, requested_top_k: int) -> int:
        if self._reranker and self._reranker.enabled:
            return max(1, min(self._settings.reranker_top_n, max(requested_top_k, self._settings.reranker_top_n)))
        return requested_top_k
```

This keeps reranked pool size controlled by `reranker_top_n`, while retrieval depth can be raised by either requested `top_k` or `reranker_top_k`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest gost-chat.tests.test_retrieval_pipeline_pool -v
```

Expected: PASS.

- [ ] **Step 5: Run retrieval smoke check**

Run:

```powershell
python scripts/retrieval_pipeline_smoke.py
```

Expected: exits with code 0.

- [ ] **Step 6: Commit**

```powershell
git add gost-chat/app/services/retrieval_pipeline.py gost-chat/tests/test_retrieval_pipeline_pool.py
git commit -m "feat: keep broad reranked candidate pools"
```

---

### Task 5: Lazy Visual Crop Service

**Files:**
- Create: `gost-chat/app/services/visual_crop_service.py`
- Test: `gost-chat/tests/test_visual_crop_service.py`

- [ ] **Step 1: Write the failing test**

Create `gost-chat/tests/test_visual_crop_service.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from app.services.visual_crop_service import VisualCropService, VisualCropSettings
from app.services.visual_evidence import VisualEvidenceRef


class VisualCropServiceTest(unittest.TestCase):
    def test_generates_deterministic_crop_from_registered_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "source.pdf"
            self._write_pdf(pdf_path)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "documents.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "source_path": str(pdf_path),
                                "file_name": "source.pdf",
                                "indexed_at": "now",
                                "stored_points": 1,
                                "file_size": pdf_path.stat().st_size,
                                "modified_at": "now",
                                "source_fingerprint": "abc",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            service = VisualCropService(
                VisualCropSettings(indexer_output_dir=root, crops_dir=root / "crops", dpi=120)
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(10.0, 10.0, 120.0, 80.0),
                block_type="figure",
                label="Figure 1",
                source_file="source.pdf",
                text_preview="Figure context",
            )

            crop = service.get_or_create_crop(ref)

            self.assertTrue(Path(crop.file_path).exists())
            self.assertEqual(crop.block_id, "block-1")
            self.assertEqual(crop.document_id, "doc-1")
            self.assertEqual(crop.format, "png")
            self.assertEqual(crop.dpi, 120)
            self.assertGreater(crop.width, 0)
            self.assertGreater(crop.height, 0)
            self.assertIn("/crops/doc-1/", crop.url_path.replace("\\", "/"))

    def test_invalid_bbox_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = VisualCropService(
                VisualCropSettings(indexer_output_dir=root, crops_dir=root / "crops", dpi=120)
            )
            ref = VisualEvidenceRef(
                block_id="block-1",
                document_id="doc-1",
                page_number=1,
                bbox=(20.0, 20.0, 10.0, 10.0),
                block_type="figure",
                label=None,
                source_file="source.pdf",
                text_preview="Figure context",
            )

            self.assertIsNone(service.get_or_create_crop(ref))

    def _write_pdf(self, path: Path) -> None:
        document = pymupdf.open()
        page = document.new_page(width=200, height=120)
        page.insert_text((20, 40), "Visual crop test")
        document.save(path)
        document.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest gost-chat.tests.test_visual_crop_service -v
```

Expected: FAIL because `visual_crop_service.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `gost-chat/app/services/visual_crop_service.py`:

```python
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
        if not crop_path.exists():
            if not self._render_crop(source_path, ref, crop_path):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest gost-chat.tests.test_visual_crop_service -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gost-chat/app/services/visual_crop_service.py gost-chat/tests/test_visual_crop_service.py
git commit -m "feat: lazily crop visual evidence"
```

---

### Task 6: RAG Visual Decision And Response Metadata

**Files:**
- Modify: `gost-chat/app/services/rag_service.py`
- Modify: `gost-chat/app/services/visual_evidence.py`
- Modify: `gost-chat/app/services/visual_crop_service.py`
- Modify: `gost-chat/app/schemas/ask.py`
- Modify: `gost-chat/app/api/ask.py`
- Modify: `gost-chat/app/main.py`
- Test: `gost-chat/tests/test_rag_visual_flow.py`
- Test: `gost-chat/tests/test_ask_schema_visuals.py`

- [ ] **Step 1: Write the failing RAG visual flow test**

Create `gost-chat/tests/test_rag_visual_flow.py`:

```python
import unittest

from app.services.context_builder import ContextBuilder, ContextBuilderSettings
from app.services.rag_service import RagService
from app.services.retrieval_pipeline import RetrievalPipelineResult
from app.services.retrieval_types import RetrievedBlock, RerankedBlock
from app.services.visual_crop_service import GeneratedCrop


class FakeLlmService:
    def __init__(self):
        self.calls = []

    async def chat(self, messages):
        self.calls.append(messages)
        if "Return only JSON" in messages[-1]["content"]:
            return (
                '{"mode":"inspect_visual_and_show","target_block_ids":["v1"],'
                '"show_in_sources":true,"show_in_answer":true,'
                '"needs_multimodal_followup":true,"reason":"Formula may be visual."}'
            )
        return "The text supports the answer [1]. A related visual fragment is attached."


class FakeRetrievalPipeline:
    def retrieve(self, query, top_k):
        visual_payload = {
            "has_visual_evidence": True,
            "bbox": [10.0, 20.0, 110.0, 120.0],
            "page_number": 2,
        }
        result = RerankedBlock(
            block_id="v1",
            text="Formula context",
            retrieval_text="Formula context",
            source_file="source.pdf",
            page=2,
            section_path=[],
            retrieval_score=0.9,
            rerank_score=0.95,
            payload=visual_payload,
            document_id="doc-1",
            page_start=2,
            page_end=2,
            block_type="formula_with_context",
            label="Formula 1",
        )
        candidate = RetrievedBlock(
            block_id=result.block_id,
            text=result.text,
            retrieval_text=result.retrieval_text,
            source_file=result.source_file,
            page=result.page,
            section_path=result.section_path,
            retrieval_score=result.retrieval_score,
            payload=result.payload,
            document_id=result.document_id,
            page_start=result.page_start,
            page_end=result.page_end,
            block_type=result.block_type,
            label=result.label,
        )
        return RetrievalPipelineResult(
            query=query,
            candidates=[candidate],
            results=[result],
            info={"backend": "test"},
        )


class FakeCropService:
    def get_or_create_crop(self, ref):
        return GeneratedCrop(
            block_id=ref.block_id,
            document_id=ref.document_id,
            file_path="data/crops/doc-1/page-2-v1.png",
            url_path="/crops/doc-1/page-2-v1.png",
            width=200,
            height=100,
            format="png",
            dpi=160,
        )


class RagVisualFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_visual_decision_generates_crop_and_adds_metadata(self):
        service = RagService(
            llm_service=FakeLlmService(),
            retrieval_pipeline=FakeRetrievalPipeline(),
            context_builder=ContextBuilder(ContextBuilderSettings(min_blocks=1, soft_target_blocks=1, max_blocks=2)),
            visual_crop_service=FakeCropService(),
            visual_decision_enabled=True,
            visual_max_crops_per_answer=1,
        )

        answer = await service.answer_question("query", top_k=12)

        self.assertEqual(len(answer.visual_evidence), 1)
        self.assertEqual(answer.visual_evidence[0].block_id, "v1")
        self.assertEqual(answer.visual_evidence[0].crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertTrue(answer.citations[0].has_visual_evidence)
        self.assertEqual(answer.citations[0].visual_evidence.crop_url, "/crops/doc-1/page-2-v1.png")
        self.assertEqual(answer.retrieval_info["visual"]["decision"]["mode"], "inspect_visual_and_show")
```

- [ ] **Step 2: Write the failing schema serialization test**

Create `gost-chat/tests/test_ask_schema_visuals.py`:

```python
import unittest

from app.schemas.ask import AskCitation, AskResponse, AskRetrievedChunk, AskVisualEvidence


class AskSchemaVisualsTest(unittest.TestCase):
    def test_response_serializes_visual_evidence(self):
        visual = AskVisualEvidence(
            block_id="v1",
            document_id="doc-1",
            source_file="source.pdf",
            page_number=2,
            block_type="figure",
            label="Figure 1",
            crop_path="data/crops/doc-1/page-2-v1.png",
            crop_url="/crops/doc-1/page-2-v1.png",
            width=200,
            height=100,
            format="png",
            dpi=160,
        )
        response = AskResponse(
            query="query",
            answer="answer",
            citations=[
                AskCitation(
                    document_id="doc-1",
                    file_name="source.pdf",
                    chunk_id="v1",
                    page_start=2,
                    page_end=2,
                    score=0.9,
                    evidence_preview="preview",
                    has_visual_evidence=True,
                    visual_evidence=visual,
                )
            ],
            retrieved_results_count=1,
            retrieval_used=True,
            retrieved_chunks=[
                AskRetrievedChunk(
                    document_id="doc-1",
                    file_name="source.pdf",
                    chunk_id="v1",
                    page_start=2,
                    page_end=2,
                    score=0.9,
                    text="preview",
                    has_visual_evidence=True,
                    visual_evidence=visual,
                )
            ],
            visual_evidence=[visual],
        )

        payload = response.model_dump()

        self.assertEqual(payload["visual_evidence"][0]["crop_url"], "/crops/doc-1/page-2-v1.png")
        self.assertEqual(payload["citations"][0]["visual_evidence"]["block_id"], "v1")
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
python -m unittest gost-chat.tests.test_rag_visual_flow gost-chat.tests.test_ask_schema_visuals -v
```

Expected: FAIL because RAG visual wiring and schema models do not exist.

- [ ] **Step 4: Implement visual response models**

In `gost-chat/app/services/rag_service.py`, add:

```python
@dataclass(frozen=True)
class RagVisualEvidence:
    block_id: str
    document_id: str
    source_file: str
    page_number: int
    block_type: str | None
    label: str | None
    crop_path: str
    crop_url: str
    width: int
    height: int
    format: str
    dpi: int
```

Add optional fields to `RagCitation` and `RagRetrievedChunk`:

```python
    block_type: str | None = None
    label: str | None = None
    has_visual_evidence: bool = False
    visual_evidence: RagVisualEvidence | None = None
```

Add to `RagAnswer`:

```python
    visual_evidence: list[RagVisualEvidence]
```

Update all existing `RagAnswer(...)` return paths with `visual_evidence=[]`.

- [ ] **Step 5: Implement decision and crop wiring**

In `RagService.__init__`, accept:

```python
        visual_crop_service: Any | None = None,
        visual_decision_enabled: bool = True,
        visual_max_crops_per_answer: int = 1,
```

Store those values.

Add helper:

```python
async def _decide_visual_evidence(self, built_context: BuiltContext) -> VisualEvidenceDecision:
    refs = [*built_context.visual_hints.selected, *built_context.visual_hints.candidates]
    if not self._visual_decision_enabled or not refs:
        return VisualEvidenceDecision.text_only("No visual decision was requested.")
    prompt = _build_visual_decision_prompt(built_context)
    raw = await self._llm_service.chat([{"role": "user", "content": prompt}])
    decision = parse_visual_decision(raw)
    return guard_visual_decision(decision, refs, max_targets=self._visual_max_crops_per_answer)
```

Add prompt builder:

```python
def _build_visual_decision_prompt(built_context: BuiltContext) -> str:
    return "\n\n".join(
        [
            "Return only JSON for whether visual evidence is needed.",
            "Allowed mode values: text_only, show_visual, inspect_visual_and_show.",
            "Use only the provided retrieved visual hints.",
            f"Question: {built_context.query}",
            "Visual hints:",
            str(built_context.visual_hints.to_dict()),
        ]
    )
```

After building context in `answer_question`, call the decision helper, crop selected refs if crop service is present, and convert crops into `RagVisualEvidence`.

Add helper:

```python
def _rag_visual_from_crop(ref: VisualEvidenceRef, crop: GeneratedCrop) -> RagVisualEvidence:
    return RagVisualEvidence(
        block_id=ref.block_id,
        document_id=ref.document_id,
        source_file=ref.source_file,
        page_number=ref.page_number,
        block_type=ref.block_type,
        label=ref.label,
        crop_path=crop.file_path,
        crop_url=crop.url_path,
        width=crop.width,
        height=crop.height,
        format=crop.format,
        dpi=crop.dpi,
    )
```

Update `_citation_from_evidence()` and `_retrieved_chunk_from_evidence()` to accept a `visual_by_block_id` map and attach visual metadata.

Update `_with_context_info()`:

```python
    merged["context"] = built_context.stats.to_dict()
    merged["visual"] = {
        "hints": built_context.visual_hints.to_dict(),
        "decision": visual_decision.to_dict(),
        "generated_count": len(visual_evidence),
    }
```

- [ ] **Step 6: Implement schema and API serialization**

In `gost-chat/app/schemas/ask.py`, add:

```python
class AskVisualEvidence(BaseModel):
    block_id: str
    document_id: str
    source_file: str
    page_number: int
    block_type: str | None = None
    label: str | None = None
    crop_path: str
    crop_url: str
    width: int
    height: int
    format: str
    dpi: int
```

Add to citation and chunk models:

```python
    block_type: str | None = None
    label: str | None = None
    has_visual_evidence: bool = False
    visual_evidence: AskVisualEvidence | None = None
```

Add to response:

```python
    visual_evidence: list[AskVisualEvidence] = Field(default_factory=list)
```

In `gost-chat/app/api/ask.py`, return:

```python
        visual_evidence=[AskVisualEvidence(**visual.__dict__) for visual in result.visual_evidence],
```

- [ ] **Step 7: Wire service construction**

In `gost-chat/app/main.py`, import:

```python
from app.services.visual_crop_service import VisualCropService, VisualCropSettings
```

Initialize before `RagService`:

```python
app.state.visual_crop_service = VisualCropService(
    VisualCropSettings(
        indexer_output_dir=settings.indexer_output_dir,
        crops_dir=settings.visual_crops_dir,
        dpi=settings.visual_crop_dpi,
    )
)
```

Pass to `RagService`:

```python
    visual_crop_service=app.state.visual_crop_service,
    visual_decision_enabled=settings.visual_enable_decision,
    visual_max_crops_per_answer=settings.visual_max_crops_per_answer,
```

Mount crop static files:

```python
settings.visual_crops_dir.mkdir(parents=True, exist_ok=True)
app.mount("/crops", StaticFiles(directory=settings.visual_crops_dir), name="crops")
```

- [ ] **Step 8: Run tests to verify they pass**

Run:

```powershell
python -m unittest gost-chat.tests.test_rag_visual_flow gost-chat.tests.test_ask_schema_visuals -v
```

Expected: PASS.

- [ ] **Step 9: Run related tests**

Run:

```powershell
python -m unittest gost-chat.tests.test_adaptive_settings gost-chat.tests.test_context_builder_adaptive gost-chat.tests.test_visual_evidence gost-chat.tests.test_visual_crop_service gost-chat.tests.test_rag_visual_flow gost-chat.tests.test_ask_schema_visuals -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add gost-chat/app/services/rag_service.py gost-chat/app/services/visual_evidence.py gost-chat/app/services/visual_crop_service.py gost-chat/app/schemas/ask.py gost-chat/app/api/ask.py gost-chat/app/main.py gost-chat/tests/test_rag_visual_flow.py gost-chat/tests/test_ask_schema_visuals.py
git commit -m "feat: route visual evidence through rag answers"
```

---

### Task 7: UI Visual Evidence Rendering

**Files:**
- Modify: `gost-chat/app/static/app.js`
- Modify: `gost-chat/app/static/style.css`
- Test: manual browser check

- [ ] **Step 1: Add UI rendering helpers**

In `gost-chat/app/static/app.js`, add helper:

```javascript
function createVisualEvidenceList(items) {
  const visuals = Array.isArray(items) ? items.filter((item) => item && item.crop_url) : [];
  const wrapper = document.createElement("div");
  wrapper.className = "visual-evidence-list";

  visuals.forEach((visual) => {
    const figure = document.createElement("figure");
    figure.className = "visual-evidence";

    const image = document.createElement("img");
    image.src = visual.crop_url;
    image.alt = formatVisualAltText(visual);
    image.loading = "lazy";

    const caption = document.createElement("figcaption");
    caption.textContent = formatVisualCaption(visual);

    figure.append(image, caption);
    wrapper.append(figure);
  });

  return wrapper;
}

function formatVisualAltText(visual) {
  const label = visual.label || visual.block_type || "visual evidence";
  return `${label} from ${visual.source_file}, page ${visual.page_number}`;
}

function formatVisualCaption(visual) {
  const label = visual.label ? `${visual.label} - ` : "";
  return `${label}${visual.source_file}, page ${visual.page_number}`;
}
```

In `replaceAssistantMessage()`, after citation summary:

```javascript
  if (Array.isArray(data.visual_evidence) && data.visual_evidence.length) {
    bodyNode.append(createVisualEvidenceList(data.visual_evidence));
  }
```

In `createCitationList()`, after preview:

```javascript
    if (citation.visual_evidence) {
      item.append(createVisualEvidenceList([citation.visual_evidence]));
    }
```

- [ ] **Step 2: Add CSS**

In `gost-chat/app/static/style.css`, add:

```css
.visual-evidence-list {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.visual-evidence {
  margin: 0;
  display: grid;
  gap: 6px;
}

.visual-evidence img {
  max-width: 100%;
  border: 1px solid #2d5788;
  border-radius: 8px;
  background: #ffffff;
}

.visual-evidence figcaption {
  color: #b7c9df;
  font-size: 0.8rem;
  overflow-wrap: anywhere;
}
```

- [ ] **Step 3: Manual browser check**

Start the chat backend the same way the project normally starts it:

```powershell
cd gost-chat
python -m uvicorn app.main:app --reload
```

Open the local chat page and submit a query that returns a response with `visual_evidence`. Expected: answer text remains readable, visual preview appears in the answer, source list keeps text metadata and preview together, and there are no console errors.

- [ ] **Step 4: Commit**

```powershell
git add gost-chat/app/static/app.js gost-chat/app/static/style.css
git commit -m "feat: show visual evidence in chat UI"
```

---

### Task 8: Final Verification

**Files:**
- Verify all changed files

- [ ] **Step 1: Run root tests**

Run from repository root:

```powershell
python -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 2: Run chat tests**

Run:

```powershell
python -m unittest discover -s gost-chat/tests
```

Expected: all tests pass.

- [ ] **Step 3: Run indexator tests**

Run:

```powershell
python -m unittest discover -s indexator/tests
```

Expected: all tests pass.

- [ ] **Step 4: Compile chat app**

Run from `gost-chat`:

```powershell
python -m compileall app tests
```

Expected: exits with code 0.

- [ ] **Step 5: Run smoke checks**

Run from repository root:

```powershell
python scripts/context_builder_smoke.py
python scripts/retrieval_pipeline_smoke.py
```

Expected: both commands exit with code 0.

- [ ] **Step 6: Inspect final diff**

Run:

```powershell
git diff --stat HEAD
git diff --check
```

Expected: diff only contains intended adaptive grounded visual evidence changes, and `git diff --check` has no output.

- [ ] **Step 7: Commit final verification note if needed**

If verification fixes required additional changes, stage the concrete files changed by those fixes. For example, if only the UI renderers changed:

```powershell
git add gost-chat/app/static/app.js gost-chat/app/static/style.css
git commit -m "test: verify adaptive visual evidence flow"
```

If no additional changes were required, do not create an empty commit.
