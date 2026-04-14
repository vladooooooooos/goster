# GOSTer

GOSTer is a document-grounded RAG chat for indexed PDF documents. The web app presents one primary chat experience: users ask questions in the main chat, the backend retrieves relevant indexed chunks locally, Polza.ai generates a grounded answer from the final prompt, and the UI displays backend-built citations.

PDF indexing remains a separate CLI step. Retrieval, reranking, citation building, and document context assembly stay local in the app. Only final LLM messages are sent to the configured remote generation provider.

## Features

- Single main web chat for grounded questions over indexed PDFs.
- FastAPI backend with REST endpoints:
  - `GET /health`
  - `GET /models`
  - `POST /ask`
  - `POST /search` for internal retrieval inspection
  - `POST /chat` for internal plain LLM chat/debug use
- OpenAI-compatible Polza.ai chat completion integration.
- Separate retrieval service for indexed PDF chunks.
- Dedicated RAG orchestration service for grounded document answers.
- Lexical chunk ranking over the current JSON indexer output.
- Backend-built citations from retrieved chunk metadata.
- Separate local CLI PDF indexer for preparing source documents.
- Configurable LLM provider, base URL, model, generation settings, and indexer output directory via environment variables.

## Project Structure

```text
gost-chat/
|-- app/
|   |-- main.py
|   |-- config.py
|   |-- api/
|   |   |-- ask.py
|   |   |-- chat.py
|   |   `-- search.py
|   |-- services/
|   |   |-- llm_service.py
|   |   |-- chat_service.py
|   |   |-- rag_service.py
|   |   `-- retriever.py
|   |-- schemas/
|   |   |-- ask.py
|   |   |-- chat.py
|   |   `-- search.py
|   |-- templates/
|   |   `-- index.html
|   `-- static/
|       |-- app.js
|       `-- style.css
|-- apps/
|   `-- indexer/
|       |-- main.py
|       |-- config.py
|       `-- services/
|           |-- chunker.py
|           |-- index_writer.py
|           |-- metadata_service.py
|           `-- pdf_loader.py
|-- docs/
|   `-- source PDF files
|-- data/
|   |-- index/
|   |   `-- chunks.jsonl
|   `-- metadata/
|       |-- documents.json
|       `-- indexing_summary.json
|-- requirements.txt
`-- README.md
```

The `docs/` and `data/` folders are runtime locations. They are created or used by local workflows and may be absent in a fresh checkout.

## Prerequisites

- Python 3.11 or newer.
- A Polza.ai API key for remote generation.
- The indexed document data already prepared under `data/`.

Example:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

If you use a different OpenAI-compatible provider later, set the `GOST_CHAT_LLM_*` values in `.env`.

## Setup

From the project directory:

```powershell
cd gost-chat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For NVIDIA GPU acceleration, replace the default CPU PyTorch wheel with a CUDA wheel after installing the base requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

When CUDA is available, `GOST_CHAT_EMBEDDING_DEVICE=auto` and `GOST_CHAT_RERANKER_DEVICE=auto` resolve to `cuda`. If the check prints `False`, the installed PyTorch wheel is CPU-only or the NVIDIA driver is not visible to Python.

Edit `.env` if needed:

```text
GOST_CHAT_APP_NAME=GOST Chat
GOST_CHAT_LLM_PROVIDER=polza
GOST_CHAT_LLM_BASE_URL=https://polza.ai/api/v1
GOST_CHAT_LLM_MODEL=google/gemma-4-31b-it
GOST_CHAT_LLM_TEMPERATURE=0.2
GOST_CHAT_LLM_MAX_TOKENS=1200
GOST_CHAT_LLM_REQUEST_TIMEOUT_SECONDS=120
GOST_CHAT_LLM_API_KEY_ENV_VAR=POLZA_API_KEY
GOST_CHAT_INDEXER_OUTPUT_DIR=data
GOST_CHAT_LOG_LEVEL=INFO
```

Set the API key outside committed config:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

## Full Local Flow

1. Put PDF files in `docs/`.
2. Build the local index:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data
```

3. Set the Polza.ai API key:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

4. Start the FastAPI app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

5. Open the web app:

```text
http://127.0.0.1:8000
```

6. Ask a question in the main chat. The chat submission calls the grounded RAG path and returns an answer with citations.

## Main Chat Flow

The main web chat is the primary user-facing experience.

1. The user submits a question in the single chat input.
2. `app/static/app.js` sends the question to `POST /ask` with a default `top_k` value.
3. `app/api/ask.py` calls `RagService.answer_question`.
4. `app/services/rag_service.py` calls `Retriever.search`.
5. `app/services/retriever.py` reads the JSON indexer output and returns the highest-scoring chunks.
6. If no chunks are found, the RAG service returns the configured Russian no-reliable-answer message without calling the remote LLM provider.
7. If chunks are found, the RAG service builds a grounded prompt and calls the configured OpenAI-compatible LLM service.
8. The response includes the answer, citations, retrieved chunk metadata, and retrieval information.
9. The UI appends the assistant answer to the same conversation stream and displays citations under the answer.

The RAG prompt instructs the assistant to answer only from retrieved indexed document context, avoid outside knowledge, state when evidence is insufficient, and avoid false certainty when retrieval is weak.

## Citations

Citations are built by the backend from retrieved chunk metadata, not from model-generated text. Each citation includes the fields available from the index:

- `file_name`
- `page_start`
- `page_end`
- `chunk_id`
- `score`
- `evidence_preview`

The web chat displays citations below each assistant response. Source snippets are shown in the citation list so the user can inspect the supporting evidence without leaving the conversation.

## CLI PDF Indexer

The indexer is a separate internal/admin command-line application. It scans a local folder for PDF files, extracts text page by page with PyMuPDF, splits the extracted text into overlapping chunks, and writes structured JSON output for retrieval by the chat app.

It intentionally does not call the LLM provider, implement the RAG prompt, serve the web UI, create embeddings, add a database, use a vector store, or call cloud services.

Run it from the project directory:

```powershell
python apps/indexer/main.py
```

By default, it reads PDFs from `docs/` and writes output under `data/`.

Useful flags:

```powershell
python apps/indexer/main.py --input-dir docs
python apps/indexer/main.py --input-dir C:\path\to\gost-pdfs --output-dir data
python apps/indexer/main.py --input-dir docs --reindex
python apps/indexer/main.py --output-dir data --clear
python apps/indexer/main.py --chunk-size 3000 --chunk-overlap 300
```

The same settings can be configured with environment variables:

```text
GOST_INDEXER_INPUT_DIR=docs
GOST_INDEXER_OUTPUT_DIR=data
GOST_INDEXER_CHUNK_SIZE=3000
GOST_INDEXER_CHUNK_OVERLAP=300
```

## Indexer Output

The indexer writes:

- `data/index/chunks.jsonl`: one JSON object per text chunk. Each chunk contains `document_id`, `file_name`, `chunk_id`, `page_start`, `page_end`, and `text`.
- `data/metadata/documents.json`: per-document metadata including `document_id`, `filename`, `source_path`, `file_hash`, `indexed_at`, `chunk_count`, and `status`.
- `data/metadata/indexing_summary.json`: summary of the latest run, including counts for found, indexed, skipped, and failed files plus output paths.

The retriever uses this format directly. It joins chunk records with document metadata by `document_id` when useful, but it does not rewrite or migrate indexer output.

## Retrieval

The retrieval service lives in `app/services/retriever.py`. It reads:

- `GOST_CHAT_INDEXER_OUTPUT_DIR/index/chunks.jsonl`
- `GOST_CHAT_INDEXER_OUTPUT_DIR/metadata/documents.json`
- `GOST_CHAT_INDEXER_OUTPUT_DIR/metadata/indexing_summary.json`, when present

The default `GOST_CHAT_INDEXER_OUTPUT_DIR` is `data`.

Retrieval currently uses a simple lexical baseline:

1. Tokenize the user query and chunk text with a Unicode word tokenizer.
2. Count term frequency per chunk.
3. Apply a small IDF-style weight across loaded chunks.
4. Add a coverage boost when more query terms match.
5. Add a small exact phrase bonus.
6. Return the highest-scoring chunks.

This is intentionally easy to replace later with embeddings, hybrid search, or reranking. No embeddings or vector database are added in this version.

The retriever caches the loaded index and reloads it when the chunk, document metadata, or summary files change by modified time or file size.

## Ask API

`POST /ask` is the grounded document question-answering endpoint used by the main chat.

Request:

```json
{
  "query": "What does the indexed document say about pipe pressure?",
  "top_k": 5
}
```

Response:

```json
{
  "query": "What does the indexed document say about pipe pressure?",
  "answer": "The indexed documents say ...",
  "citations": [
    {
      "document_id": "1b180454c521a2b6",
      "file_name": "sample.pdf",
      "chunk_id": "1b180454c521a2b6:00001",
      "page_start": 1,
      "page_end": 2,
      "score": 0.123456,
      "evidence_preview": "Extracted document text..."
    }
  ],
  "retrieved_results_count": 1,
  "retrieval_used": true,
  "retrieved_chunks": [
    {
      "document_id": "1b180454c521a2b6",
      "file_name": "sample.pdf",
      "chunk_id": "1b180454c521a2b6:00001",
      "page_start": 1,
      "page_end": 2,
      "score": 0.123456,
      "text": "Extracted document text..."
    }
  ],
  "retrieval_info": {
    "top_k": 5,
    "index_summary": {
      "schema_version": 1
    }
  }
}
```

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"query":"What does the indexed document say about pipe pressure?","top_k":5}'
```

Empty queries return `400`. Missing index files return `503`. Corrupted index data returns `500`. LLM provider configuration, availability, timeout, or model call errors return `503`.

## Internal And Debug Endpoints

`POST /search` remains available for internal retrieval inspection. It returns matching indexed chunks and does not call the LLM provider. It is not exposed as a separate panel in the main web UI.

`POST /chat` remains available for internal plain LLM chat/debug use through the existing chat service. It is no longer used by the main web chat.

A direct provider smoke test is available without retrieval:

```powershell
$env:POLZA_API_KEY="your-api-key"
python scripts/polza_llm_smoke.py
python scripts/polza_llm_smoke.py --prompt "Reply in Russian with one short sentence: did the test pass?"
```

## API Checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/models
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"query":"What does the indexed document say about pipe pressure?","top_k":5}'
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType "application/json" -Body '{"query":"steel pipe pressure","top_k":5}'
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body '{"message":"Hello"}'
python scripts/polza_llm_smoke.py
```

## Future RAG Improvements

The next RAG improvements should build on the current `RagService` and `Retriever` without changing the indexer format:

- Add embeddings or hybrid lexical plus vector retrieval behind the existing `Retriever` interface.
- Add reranking before prompt assembly.
- Add confidence or evidence-strength heuristics.
- Add tests around empty retrieval, corrupted indexes, and LLM provider failures.

## Notes

- The main web UI is a document-grounded RAG chat, not a general plain chat.
- The frontend calls the FastAPI backend only; it never calls the LLM provider directly.
- The default remote generation provider is Polza.ai at `https://polza.ai/api/v1`.
- The default remote generation model is `google/gemma-4-31b-it`.
- Retrieval is local-first and reads plain JSON files generated by the separate indexer.
- Indexing remains external via the CLI indexer.
