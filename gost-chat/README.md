# GOSTer Chat

GOSTer Chat — document-grounded RAG chat для проиндексированных PDF-документов. Web app даёт один основной chat flow: пользователь задаёт вопрос, backend локально извлекает релевантные chunks или vector blocks, LLM provider генерирует grounded answer, а UI показывает citations, построенные backend-ом.

Indexing остаётся отдельным шагом. Retrieval, reranking, citation building и сборка document context выполняются локально. Во внешний LLM provider отправляются только финальные LLM messages.

## Возможности

- Основной web chat для grounded questions по indexed PDFs.
- FastAPI backend с REST endpoints:
  - `GET /health`
  - `GET /models`
  - `POST /ask`
  - `POST /search` для internal retrieval inspection
  - `POST /chat` для internal plain LLM chat/debug
- OpenAI-compatible Polza.ai chat completion integration.
- Локальный JSON retrieval fallback по `data/index/chunks.jsonl`.
- Vector retrieval через shared Qdrant store.
- Local embeddings через `sentence-transformers`.
- Optional local reranking.
- Dedicated RAG orchestration service для grounded document answers.
- Backend-built citations из retrieved metadata.
- Внутренний lite JSON PDF indexer в `apps/indexer`.
- Конфигурация через `.env` и environment variables.

## Структура проекта

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
|   |   |-- retriever.py
|   |   |-- qdrant_retriever.py
|   |   |-- retrieval_pipeline.py
|   |   `-- reranker_service.py
|   |-- schemas/
|   |-- templates/
|   `-- static/
|-- apps/
|   `-- indexer/
|       |-- main.py
|       |-- config.py
|       `-- services/
|-- docs/
|-- data/
|-- requirements.txt
`-- README.md
```

`docs/` и `data/` — runtime locations. Они создаются или используются локальными workflows и могут отсутствовать в fresh checkout.

## Требования

- Python 3.11+.
- Polza.ai API key или другой OpenAI-compatible provider для remote generation.
- Подготовленные indexed data:
  - либо JSON index в `gost-chat/data/`;
  - либо shared Qdrant store в `shared/data/qdrant/`, обычно созданный Indexator.

## Рекомендуемая установка через корневую среду

Основной сценарий workspace — единое окружение `C:\goster\.venv`.

Из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Проверить CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Скопировать config example:

```powershell
cd gost-chat
Copy-Item .env.example .env
```

Реальный API key храните вне Git:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

## Установка только для GOST Chat

Если нужен только chat component:

```powershell
cd gost-chat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Для NVIDIA GPU acceleration можно заменить default CPU PyTorch wheel на CUDA wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Когда CUDA доступна, `GOST_CHAT_EMBEDDING_DEVICE=auto` и `GOST_CHAT_RERANKER_DEVICE=auto` resolve to `cuda`. Если check печатает `False`, установлен CPU-only Torch или NVIDIA driver не виден Python.

## Desktop window launcher

The desktop launcher is an optional PyWebView shell around the same local FastAPI web app. It starts the existing backend command, waits for `http://127.0.0.1:8000/health`, and opens the same UI in a native desktop window. The browser workflow remains unchanged.

Install the optional dependency:

```powershell
python -m pip install pywebview
```

Launch in browser mode:

```powershell
cd gost-chat
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Launch in desktop-window mode:

```powershell
cd gost-chat
python run_desktop.py
```

The desktop window displays the same local URL, so future backend and frontend changes are picked up automatically without a separate desktop UI codepath.

## Основные настройки `.env`

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
GOST_CHAT_RETRIEVAL_BACKEND=auto
GOST_CHAT_QDRANT_LOCAL_PATH=../shared/data/qdrant
GOST_CHAT_QDRANT_COLLECTION_NAME=gost_blocks
GOST_CHAT_EMBEDDING_MODEL_NAME=BAAI/bge-m3
GOST_CHAT_EMBEDDING_DEVICE=auto
GOST_CHAT_RERANKER_ENABLED=true
```

## Полный локальный сценарий

### Вариант A: основной сценарий через Indexator и Qdrant

1. Запустить Indexator из корня:

```powershell
.\indexator\run.bat
```

2. Выбрать папку с PDFs и выполнить indexing.
3. Убедиться, что shared Qdrant data лежит в `shared/data/qdrant/`.
4. В `gost-chat/.env` оставить:

```text
GOST_CHAT_RETRIEVAL_BACKEND=auto
GOST_CHAT_QDRANT_LOCAL_PATH=../shared/data/qdrant
GOST_CHAT_QDRANT_COLLECTION_NAME=gost_blocks
```

5. Запустить web app:

```powershell
cd gost-chat
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

6. Открыть:

```text
http://127.0.0.1:8000
```

### Вариант B: fallback через lite JSON indexer

Этот вариант полезен для demo, smoke/debug или если vector store временно не нужен.

1. Положить PDF в `gost-chat/docs/`.
2. Собрать JSON index:

```powershell
cd gost-chat
python apps/indexer/main.py --input-dir docs --output-dir data
```

3. Запустить web app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Основной chat flow

1. Пользователь отправляет вопрос в single chat input.
2. `app/static/app.js` отправляет запрос в `POST /ask`.
3. `app/api/ask.py` вызывает `RagService.answer_question`.
4. `app/services/retrieval_pipeline.py` выбирает retrieval backend и запускает retrieval/reranking.
5. `app/services/context_builder.py` собирает context из выбранных evidence blocks.
6. Если evidence недостаточно, RAG service возвращает configured no-reliable-answer message без remote LLM call.
7. Если context есть, RAG service строит grounded prompt и вызывает configured OpenAI-compatible LLM service.
8. Response содержит answer, citations, retrieved chunk metadata и retrieval info.
9. UI добавляет assistant answer в conversation stream и показывает citations под ответом.

RAG prompt требует отвечать только по retrieved document context, не использовать outside knowledge, явно говорить о недостатке evidence и не выдумывать unsupported claims.

## Citations

Citations строятся backend-ом из retrieved metadata, а не из model-generated text. Citation может включать:

- `file_name`
- `page_start`
- `page_end`
- `chunk_id` или `block_id`
- `score`
- `retrieval_score`
- `rerank_score`
- `evidence_preview`

Web chat показывает citations под каждым assistant response.

## Lite CLI PDF Indexer

`apps/indexer` — внутреннее административное CLI-приложение. Оно сканирует локальную папку с PDF, извлекает текст через PyMuPDF, режет его на overlapping chunks и пишет JSON output для chat app.

Он намеренно не вызывает LLM provider, не строит RAG prompt, не обслуживает web UI, не создаёт embeddings, не использует Qdrant и не обращается к cloud services.

Запуск из `gost-chat`:

```powershell
python apps/indexer/main.py
```

Полезные flags:

```powershell
python apps/indexer/main.py --input-dir docs
python apps/indexer/main.py --input-dir C:\path\to\gost-pdfs --output-dir data
python apps/indexer/main.py --input-dir docs --reindex
python apps/indexer/main.py --output-dir data --clear
python apps/indexer/main.py --chunk-size 3000 --chunk-overlap 300
```

Переменные окружения:

```text
GOST_INDEXER_INPUT_DIR=docs
GOST_INDEXER_OUTPUT_DIR=data
GOST_INDEXER_CHUNK_SIZE=3000
GOST_INDEXER_CHUNK_OVERLAP=300
```

## Выходные файлы JSON indexer

Lite indexer пишет:

- `data/index/chunks.jsonl`: один JSON object на каждый text chunk.
- `data/metadata/documents.json`: metadata по документам.
- `data/metadata/indexing_summary.json`: summary последнего run.

JSON retriever читает этот формат напрямую. Vector/Qdrant retrieval использует shared Qdrant store, обычно подготовленный Indexator.

## API

`POST /ask` — основной grounded document question-answering endpoint.

Request:

```json
{
  "query": "Что документы говорят о давлении в трубопроводе?",
  "top_k": 5
}
```

Response:

```json
{
  "query": "Что документы говорят о давлении в трубопроводе?",
  "answer": "Ответ по проиндексированным документам...",
  "citations": [],
  "retrieved_results_count": 0,
  "retrieval_used": false,
  "retrieved_chunks": [],
  "retrieval_info": {}
}
```

Empty queries return `400`. Missing index files return `503`. Corrupted index data returns `500`. LLM provider configuration, availability, timeout или model call errors return `503`.

## Внутренние и debug endpoints

`POST /search` остаётся доступен для internal retrieval inspection. Он возвращает matching indexed chunks/blocks и не вызывает LLM provider.

`POST /chat` остаётся доступен для internal plain LLM chat/debug через existing chat service. Main web chat его не использует.

Direct provider smoke test:

```powershell
$env:POLZA_API_KEY="your-api-key"
python scripts/polza_llm_smoke.py
python scripts/polza_llm_smoke.py --prompt "Ответь по-русски одним коротким предложением: тест прошёл?"
```

## API checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/models
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"query":"What does the indexed document say about pipe pressure?","top_k":5}'
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType "application/json" -Body '{"query":"steel pipe pressure","top_k":5}'
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body '{"message":"Hello"}'
python scripts/polza_llm_smoke.py
```

## Будущие улучшения RAG

- Улучшить hybrid retrieval: lexical + vector.
- Добавить confidence/evidence-strength heuristics.
- Расширить tests around empty retrieval, corrupted indexes и LLM provider failures.
- Постепенно сделать Indexator основным indexing path, оставив lite JSON indexer как fallback/demo path.

## Примечания

- Main web UI — document-grounded RAG chat, а не general plain chat.
- Frontend вызывает только FastAPI backend и никогда не обращается к LLM provider напрямую.
- Default remote generation provider — Polza.ai at `https://polza.ai/api/v1`.
- Default remote generation model — `google/gemma-4-31b-it`.
- Retrieval local-first: JSON fallback читает `gost-chat/data`, vector path читает `shared/data/qdrant`.
- Indexing остаётся external step: основной сценарий через Indexator, fallback через `apps/indexer`.
