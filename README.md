# Goster Workspace

Goster is a local workspace for indexing Russian GOST PDF documents and asking document-grounded questions over the indexed content. The workspace currently contains multiple adjacent Python components rather than one single application.

The repository is intended to store source code, documentation, setup files, and lightweight examples. Local virtual environments, document corpora, generated indexes, vector stores, model caches, and machine-specific settings should stay outside Git.

## Repository Map

```text
.
|-- gost-chat/                 # FastAPI RAG chat web app and a JSON PDF indexer
|-- indexator/                 # Windows-first PySide6 desktop indexing app
|   |-- indexator/             # Indexator Python application package root
|   `-- run.bat                # Windows launcher for the desktop app
|-- shared/                    # Shared vector-store utilities used across components
|-- scripts/                   # Cross-component smoke checks
`-- docs/                      # Local PDF corpus location, not committed by default
```

Local-only folders that may appear during development:

```text
gost-chat/.venv/
gost-chat/data/
gost-chat/docs/
indexator/indexator/.venv/
indexator/indexator/data/
indexator/indexator/output/
indexator/indexator/testdocs/
shared/data/
goster_vector_store_smoke_*/
```

## Components

### Indexator

Purpose: desktop application for scanning PDF folders, parsing GOST documents, producing structured blocks, generating embeddings, and writing local Qdrant vector data.

Stack:

- Python
- PySide6
- PyMuPDF
- sentence-transformers / transformers / torch
- qdrant-client with local persistent storage

Entry points and important files:

- `indexator/indexator/app/main.py`
- `indexator/run.bat`
- `indexator/indexator/config.json`
- `indexator/indexator/requirements.txt`
- `indexator/indexator/tests/`

Local-only artifacts:

- `indexator/indexator/.venv/`
- `indexator/indexator/data/`
- `indexator/indexator/output/`
- `indexator/indexator/testdocs/`, unless a small, licensed fixture set is intentionally added
- `indexator/.test_tmp/`

### GOST Chat

Purpose: FastAPI web app for document-grounded chat over indexed content. It can use local retrieval data and an OpenAI-compatible LLM provider.

Stack:

- Python
- FastAPI / Uvicorn
- Jinja2, HTML, CSS, browser JavaScript
- PyMuPDF for the local JSON indexer
- qdrant-client, sentence-transformers, torch for vector retrieval and reranking paths
- pydantic-settings for configuration

Entry points and important files:

- `gost-chat/app/main.py`
- `gost-chat/apps/indexer/main.py`
- `gost-chat/app/templates/index.html`
- `gost-chat/app/static/`
- `gost-chat/requirements.txt`
- `gost-chat/.env.example`

Local-only artifacts:

- `gost-chat/.venv/`
- `gost-chat/.env`
- `gost-chat/docs/`
- `gost-chat/data/`

### Shared

Purpose: shared Python utilities for Qdrant vector storage configuration, payloads, and models.

Stack:

- Python
- qdrant-client integration code

Entry points and important files:

- `shared/vector_store/config.py`
- `shared/vector_store/qdrant_store.py`
- `shared/vector_store/models.py`
- `shared/vector_store/payloads.py`

Local-only artifacts:

- `shared/data/`

### Scripts

Purpose: smoke checks that validate retrieval, context building, and shared vector-store behavior across components.

Stack:

- Python
- component dependencies from `gost-chat` and `indexator`

Entry points:

- `scripts/vector_store_smoke.py`
- `scripts/retrieval_pipeline_smoke.py`
- `scripts/context_builder_smoke.py`
- `gost-chat/scripts/polza_llm_smoke.py`

## File Classification

Keep in Git:

- Handwritten Python source under `gost-chat/app/`, `gost-chat/apps/`, `indexator/indexator/app/`, `shared/`, and `scripts/`
- Tests under `indexator/indexator/tests/`
- Requirements files
- README and project documentation
- `.env.example`
- `indexator/indexator/config.json`, because it is a reproducible default config without secrets
- Lightweight launch scripts such as `indexator/run.bat`

Ignore from Git:

- `.venv/`, `venv/`, Python caches, test caches
- `.env` and other local environment files
- Qdrant local storage, SQLite databases, lock files
- Generated JSON/JSONL indexing output
- Local smoke-test folders
- Local model caches and downloaded dependency caches
- Raw PDF corpora under `docs/`, `gost-chat/docs/`, and `indexator/indexator/testdocs/`

Optional / maybe keep:

- A tiny, licensed sample PDF fixture, preferably under a dedicated `samples/` or `fixtures/` folder
- Small expected-output fixtures for tests if they are stable and intentionally reviewed
- Developer notes such as `FRIEND_README.md`, `RULES.md`, and `agents.md` if they are useful to future maintainers

Replace with example/template:

- `gost-chat/.env` should stay local; commit `gost-chat/.env.example`
- Any future secret-bearing provider config should be committed only as `*.example`

Externalize from repo and document:

- Full GOST PDF corpus
- Qdrant/vector store data
- Generated chunk indexes and metadata
- Downloaded ML models and caches
- Large debug exports

## Fresh Machine Recovery

### 1. Clone and inspect

```powershell
git clone <repo-url> goster
cd goster
```

Restore external artifacts outside Git if you need the same local state:

- PDF corpus into `docs/` or a chosen external folder
- Any saved Qdrant store into `shared/data/qdrant/`
- Any saved JSON index into `gost-chat/data/`

Generated artifacts can also be rebuilt from PDFs.

### 2. Set up Indexator

```powershell
cd indexator\indexator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

Indexator reads `config.json`. By default it writes shared Qdrant data to `shared/data/qdrant` relative to the app directory.

On Windows, the launcher can also be used:

```powershell
..\run.bat
```

### 3. Set up GOST Chat

```powershell
cd ..\..\gost-chat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` for local settings. Keep real API keys in the shell or a secret manager:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

Build a JSON index from local PDFs when using the JSON retrieval path:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data
```

Run the web app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

### 4. Run Smoke Checks

From the workspace root:

```powershell
python scripts/vector_store_smoke.py
python scripts/retrieval_pipeline_smoke.py
python scripts/context_builder_smoke.py
```

From `gost-chat`, with the API key configured:

```powershell
python scripts/polza_llm_smoke.py
```

## Safe Git Cleanup

If this folder is initialized as a Git repository and generated/local files were already tracked, untrack them without deleting local copies:

```powershell
git rm --cached -r gost-chat/.venv indexator/indexator/.venv
git rm --cached -r gost-chat/data indexator/indexator/data indexator/indexator/output shared/data
git rm --cached -r docs gost-chat/docs indexator/indexator/testdocs
git rm --cached -r goster_vector_store_smoke_kog7mk5k goster_vector_store_smoke_tmyo2suz indexator/.test_tmp
git rm --cached gost-chat/.env
```

Use only the commands for paths that are actually tracked. Check first:

```powershell
git ls-files gost-chat/.venv indexator/indexator/.venv gost-chat/data indexator/indexator/data indexator/indexator/output shared/data docs gost-chat/docs indexator/indexator/testdocs gost-chat/.env
```

## Commit and Push

```powershell
git status --short
git add .gitignore README.md indexator/run.bat
git add gost-chat/README.md gost-chat/.env.example indexator/indexator/README.md indexator/indexator/config.json
git add gost-chat/app gost-chat/apps gost-chat/scripts indexator/indexator/app indexator/indexator/tests shared scripts
git commit -m "Prepare Goster workspace for reproducible development"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

If the remote already exists, skip `git remote add origin` or update it with:

```powershell
git remote set-url origin <repo-url>
```

## Disaster Recovery Checklist

- Clone the repository.
- Recreate Python virtual environments from each component's `requirements.txt`.
- Copy `.env.example` to `.env` where needed and configure local settings.
- Restore API keys through environment variables or a secret manager.
- Restore PDFs from external storage, or obtain a minimal licensed sample set.
- Rebuild `gost-chat/data/` with `gost-chat/apps/indexer/main.py` if using JSON retrieval.
- Rebuild or restore `shared/data/qdrant/` if using vector retrieval.
- Launch Indexator and confirm it can scan a PDF folder.
- Launch GOST Chat and check `GET /health`.
- Run smoke checks from `scripts/`.
