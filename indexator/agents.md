# AGENTS.md

## Project Identity

You are the main coding agent for a local Python desktop application called **Indexator**.

Indexator is a separate application inside the user's local project folder.
It is a Windows-first desktop tool for indexing Russian GOST PDF documents into structured searchable blocks for future semantic retrieval and LLM-assisted question answering.

This application is not the chat app itself.
It is the **indexing tool** that prepares data for the future GOSTer system.

Before starting any implementation task, read `RULES.md` and follow it strictly.

---

## Mission

Build a practical MVP of Indexator with a **golden middle** approach:

- better than naive RAG chunking
- much simpler than a full document intelligence platform
- fast enough for real use
- modular enough to improve later

The goal is to index GOST PDFs into logical structured blocks that preserve document meaning and improve retrieval quality.

---

## Product Goal

Indexator should:

- scan a folder with PDF files
- parse GOST PDF documents
- extract logical blocks instead of naive random chunks
- generate embeddings using a switchable backend
- store vectors and metadata in local Qdrant
- provide a simple desktop interface
- run locally on Windows
- later be packable into `.exe`

---

## Scope of v0.1

The MVP should support:

- folder selection
- PDF scanning
- basic file list UI
- PDF parsing with PyMuPDF
- text cleanup
- logical block extraction
- embedding generation
- local Qdrant storage
- indexing progress and logs
- preview of extracted blocks

Do not overbuild beyond this scope unless explicitly requested.

---

## Core Indexing Philosophy

Do not use naive fixed-size random chunking.

Use a **structured middle-ground indexing strategy**:

- split regular text by paragraphs
- preserve headings and section hierarchy
- keep tables as whole objects
- keep figures as whole objects
- keep formulas together with nearby explanation
- keep appendices as separate structured sections

The system should preserve meaning and document structure without becoming overly complex.

---

## Supported Block Types

The main block types for v0.1 are:

- `heading`
- `paragraph`
- `list_item`
- `table`
- `formula_with_context`
- `figure`
- `appendix_section`

These are the main indexing units.

### Important notes

#### Paragraphs
Paragraphs are the default text units.
Short neighboring paragraphs may be merged inside the same section if it improves retrieval quality.

#### Tables
A table must be kept as a whole logical object whenever possible.
Do not split a table into random text chunks.

#### Formulas
A formula should not be indexed alone when nearby explanatory text exists.
Prefer `formula + explanation` as one logical block.

#### Figures
A figure must be treated as a full document object.
Preserve:

- page
- bbox
- label
- caption
- nearby context

Do not reduce figures to random fragments.

#### Appendices
Appendices such as `Приложение А`, `Приложение Б` should be detected and indexed as separate structured sections.

---

## Data and Storage Model

Each index block should preserve metadata needed for future retrieval.

Typical fields include:

- `id`
- `doc_id`
- `doc_title`
- `file_name`
- `file_path`
- `page_start`
- `page_end`
- `block_type`
- `section_path`
- `label`
- `text`
- `context_text`
- `bbox`
- `reading_order`
- `tokens_estimate`

Store vectors and payload metadata in **local Qdrant**.

---

## Embedding Backends

Indexator must support switchable embedding backends.

### Backend 1: API
Primary cloud embedding backend:

- BGE-M3 API
- intended for cheap and fast indexing
- should be configurable via settings/config

### Backend 2: Local HF
Primary local embedding backend:

- `deepvk/USER-bge-m3`
- intended for Russian semantic search
- should run through local Hugging Face / sentence-transformers style integration

Architecture must not depend on a single provider.

Use a backend abstraction layer.

---

## Core Tech Stack

Preferred stack:

- Python
- PySide6
- PyMuPDF
- Qdrant local
- requests
- transformers / sentence-transformers
- torch

This is a Windows-first desktop application.

---

## UI Expectations

The UI should stay simple and practical.

Main expected UI parts:

- folder picker
- file list table
- settings panel
- logs panel
- progress bar
- index button
- preview/debug area

This is a utility tool, not a polished consumer app.
Clean and functional is better than fancy.

---

## Architecture Expectations

Keep the architecture modular.

Recommended modules:

- UI
- core pipeline
- parsing
- embedding
- storage
- services
- utils

Prefer small focused modules over giant files.

---

## Development Style

Work incrementally.

Build the project in this order:

1. project skeleton
2. basic UI
3. file scanning
4. PDF parsing
5. text cleanup
6. block building
7. embedding backends
8. Qdrant integration
9. full indexing pipeline
10. preview and reporting
11. richer support for tables, formulas, figures, appendices

Do not try to implement every advanced feature at once.

---

## Quality Priorities

Prioritize the following:

1. correct architecture
2. stable indexing pipeline
3. readable code
4. debuggability
5. retrieval-friendly block structure
6. simple usable UI

Do not optimize prematurely.

---

## What Success Looks Like

A successful v0.1 means:

- the app launches locally on Windows
- the user can select a folder with PDFs
- the app can scan and index selected GOST PDFs
- the app creates structured blocks
- embeddings are generated through API or local backend
- vectors and metadata are stored in local Qdrant
- logs and progress are visible
- preview/debug information helps validate extraction quality

---

## Communication Style

When working on tasks:

- think and implement in English
- keep code, filenames, architecture, and comments in English
- respond to the user in Russian
- keep responses practical and grounded
- explain what was changed, where, and why
- mention limitations honestly

---

## Final Reminder

You are building a **practical Indexator MVP**, not a research prototype and not a full production platform.

Keep the solution:

- modular
- readable
- Windows-friendly
- easy to debug
- easy to extend later

Read `RULES.md` before making changes.