# RULES.md

## 1. General Rules

Follow these rules strictly before and during every task.

- Read the local project context first.
- Prefer the existing project structure over inventing a new one.
- Make the smallest reasonable change that solves the task well.
- Do not introduce unnecessary complexity.
- Do not expand scope without an explicit request.
- Keep the project aligned with the MVP goals of Indexator.

This is a Windows-first local desktop tool.
Respect that in all implementation choices.

---

## 2. Context7 Usage Rules

Use Context7 as a support tool, not as the main source of project truth.

### Use Context7 when:
- working with external library APIs
- checking version-sensitive usage
- confirming current best practices for dependencies
- integrating:
  - PySide6
  - PyMuPDF
  - Qdrant
  - transformers
  - sentence-transformers
  - PyInstaller
  - other external libraries

### Do not use Context7 when:
- local project files already define the needed behavior
- the question is about project architecture or business logic
- the answer should come from the repository itself
- naming, structure, block logic, or indexing philosophy are already defined in project files

### Priority order:
1. local project files
2. AGENTS.md
3. RULES.md
4. Context7 for external docs only

Do not let external documentation override explicit local project requirements.

---

## 3. Scope Control Rules

Do not turn the MVP into an overengineered platform.

Without explicit user request, do not add:

- full OCR pipelines
- multimodal image understanding
- graph retrieval systems
- reranking systems
- distributed infrastructure
- remote databases
- cloud orchestration
- user authentication
- web dashboards
- advanced async job systems
- microservices

Stay inside the Indexator MVP scope.

---

## 4. Code Change Rules

When making changes:

- prefer focused edits
- avoid large unrelated refactors
- do not rename files or modules without a strong reason
- do not rewrite working code unnecessarily
- preserve architecture clarity
- keep files reasonably sized
- create small reusable functions when appropriate

When changing behavior, explain clearly what changed.

---

## 5. Dependency Rules

Do not add a new dependency unless it is clearly justified.

Before adding a dependency:

- check whether the existing stack already covers the need
- prefer standard library or already-approved libraries when possible
- keep Windows compatibility in mind
- avoid heavy dependencies unless they clearly improve the MVP

Preferred core dependencies are already known:

- PySide6
- PyMuPDF
- qdrant-client
- requests
- torch
- transformers
- sentence-transformers

Be conservative with anything beyond this list.

---

## 6. Parsing and Indexing Rules

The indexing strategy must follow the project's middle-ground approach.

### Required principles
- do not use naive random fixed-size chunking as the main strategy
- preserve headings and section hierarchy
- paragraphs are the default text unit
- tables should stay whole when possible
- formulas should stay with nearby explanation
- figures should be full objects with caption/context
- appendices should be separate structured blocks

### Required block awareness
The pipeline should explicitly respect these block types:

- heading
- paragraph
- list_item
- table
- formula_with_context
- figure
- appendix_section

Do not collapse all content into one generic text flow.

---

## 7. Figure Rules

Figures are important for this project.

When implementing figure support:

- treat a figure as a full document object
- preserve its page and bbox
- preserve its caption if found
- preserve nearby context text if found
- keep the architecture ready for later preview/export

Do not reduce figures to random surrounding text only.

At MVP stage, figure retrieval may rely on:

- caption
- references in nearby text
- context text
- metadata

That is acceptable.

---

## 8. Qdrant Rules

Use **local Qdrant** as the default vector storage solution.

Preferred reasons:

- vector + payload storage together
- metadata filters
- better fit for structured document blocks
- future-ready retrieval architecture

Do not replace Qdrant with FAISS unless explicitly requested.

Design storage so that block metadata remains queryable and useful.

---

## 9. Embedding Backend Rules

The embedding layer must be replaceable.

Support these backends in architecture:

- API backend for BGE-M3
- Local HF backend for `deepvk/USER-bge-m3`

Do not hardcode the pipeline to a single provider.

Use a backend abstraction interface.

Avoid tightly coupling UI, pipeline, and embedding implementation.

---

## 10. UI Rules

The UI should be simple, practical, and desktop-friendly.

Priorities:

- usability
- clarity
- debugging value
- progress visibility

Do not spend time on visual polish beyond what is needed for a clean functional desktop app.

Prefer utility over decoration.

---

## 11. Windows Rules

This application targets Windows first.

Therefore:

- prefer Windows-friendly paths and behavior
- be careful with file path handling
- keep packaging with PyInstaller in mind
- avoid Linux-only assumptions
- avoid shell-specific workflows that are not Windows-friendly

Do not optimize for cross-platform behavior unless requested.

---

## 12. Language Rules

Use this language policy consistently:

- code: English
- comments in code: English
- filenames: English
- config keys: English
- architecture docs inside project: English
- direct responses to the user: Russian

Do not mix Russian into code structure unless the content itself is document data from PDFs.

---

## 13. Reporting Rules

After completing a task, report clearly:

- what was changed
- which files were created or modified
- what the new behavior is
- what still remains incomplete
- any important caveats

Keep reports practical and grounded.

Do not claim something works unless it was actually implemented.

---

## 14. Testing and Validation Rules

Prefer small real validations over abstract claims.

When possible:

- run the smallest useful test
- validate UI startup
- validate file scanning
- validate one real parsing path
- validate one embedding path
- validate one Qdrant write path

Prefer concrete evidence over optimistic assumptions.

---

## 15. Roadmap Discipline

Implement in layers.

Prefer this development sequence:

1. skeleton
2. UI
3. scan PDFs
4. parse PDFs
5. clean text
6. build text blocks
7. embeddings
8. Qdrant
9. full pipeline
10. preview/debug
11. richer support for tables, formulas, figures, appendices

Do not jump deep into advanced features before the base pipeline works.

---

## 16. Final Rule

This project is a **practical desktop Indexator for GOST PDFs**.

Every decision should be judged by this question:

Does this make the MVP more useful, more stable, and more maintainable without unnecessary complexity?

If not, do not do it.