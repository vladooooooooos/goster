# Indexator

Indexator is a Windows-first Python desktop application for preparing Russian GOST PDF documents for future semantic retrieval in the GOSTer system.

This repository folder contains the standalone Indexator app. The current scope is a clean MVP foundation with the first local end-to-end indexing pipeline.

## Current status

- PySide6 desktop shell
- Folder selection control
- Readonly selected path field
- Non-recursive PDF folder scanning
- PDF table population with file name, path, size, page count, and status
- Raw PyMuPDF page and text block parsing
- Safe text cleanup layer
- First-pass structured block builder for headings, paragraphs, list items, and appendix sections
- Table of contents detection as a separate structured block type
- First-pass table detection for simple captioned tables
- First-pass figure detection for simple Russian figure and drawing captions
- First-pass formula detection for math-like blocks with nearby explanatory context
- Conservative raw text block refinement for obvious formula explanation, caption/body, appendix/body, and large numbered segment splits
- Simple parsed text and structured block debug preview in the log panel
- Landscape and rotated page notes in debug preview
- JSONL debug export for structured block previews
- Local BAAI/bge-m3 embedding preview for final structured blocks
- Compact embedding debug summary export without full vector dumps
- Local persistent Qdrant storage preview for structured blocks and vectors
- Compact Qdrant storage summary export
- Full local indexing pipeline connected to the Index selected action
- Compact indexing run summary export
- Shared index storage under `C:\goster\shared\data`
- Shared document registry for indexed PDF state
- Indexed-state detection for Ready, Indexed, Indexed (stale), Missing source, and Index error
- Source PDF fingerprint metadata for stale detection
- Reindex selected action for forced replacement of checked indexed documents
- Clear selected action for removing checked indexed documents from the shared index
- Clear all action for wiping only shared index-owned storage
- Log panel
- Progress bar
- Index selected action for the local end-to-end indexing pipeline
- JSON config loading
- Modular package structure for future pipeline work

## Project structure

```text
indexator/
  app/
    main.py
    ui/
    core/
    parsing/
    embedding/
    storage/
    services/
    utils/
  data/
  output/
  config.json
  requirements.txt
  README.md
```

## Run locally

Create and activate a virtual environment, install dependencies, then start the app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

Run the commands from the `indexator` app directory.

The local embedding backend uses automatic device selection by default. It uses CUDA only when the same virtual environment used to launch Indexator has a CUDA-enabled Torch build available; otherwise it falls back to CPU.

## Shared index storage

Indexator writes local retrieval data to the shared project storage:

```text
C:\goster\shared\data\
  qdrant\
  metadata\
    documents.json
    deletion_summaries\
  cache\
    indexator\
  debug\
    indexator\
```

Source PDFs are never part of clear semantics. Indexator only removes index-owned data inside this shared storage.

## Clearing indexed documents

- `Clear selected` removes indexed data for checked PDF rows. Files that were not indexed are skipped and reported in the log and summary.
- `Clear all` wipes the shared local index for the configured Qdrant collection, document registry, and Indexator-owned cache/debug artifacts.
- Both actions keep source PDFs untouched.
- Deletion uses stable `document_id` metadata derived from the resolved source path, not filename matching.
- Clear summaries are written under `C:\goster\shared\data\metadata\deletion_summaries`.

## Indexed states and reindexing

The file table merges the current PDF folder scan with the shared registry:

- `Ready`: the source PDF exists, but no registry entry exists for its `document_id`.
- `Indexed`: the source PDF exists and its stored size/modified timestamp match the registry.
- `Indexed (stale)`: the source PDF exists, but size or modified timestamp changed since indexing.
- `Missing source`: a registry entry exists for the selected folder, but the source PDF is no longer on disk.
- `Index error`: the registry records a failed indexing attempt.

`Reindex selected` is a force action. It works on checked available PDFs, removes old indexed data for each matching `document_id`, indexes the current source file again, and writes fresh registry metadata including `file_size`, `modified_at`, and a SHA-256 `source_fingerprint`.

`Index selected` is intended for checked `Ready` files. Use `Reindex selected` for unchanged indexed files or stale indexed files.

## Next MVP steps

1. Improve table, figure, and formula grouping heuristics with more real GOST samples.
2. Add a minimal retrieval debug path over indexed blocks.
