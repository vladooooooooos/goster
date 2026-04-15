# Рабочая папка Goster

Goster — локальная рабочая папка для индексирования русскоязычных PDF-документов ГОСТ и вопросов по проиндексированному содержимому. Сейчас это не одно приложение, а несколько соседних Python-компонентов.

В Git стоит хранить исходный код, документацию, setup-файлы и лёгкие примеры. Локальные виртуальные окружения, корпуса документов, сгенерированные индексы, векторные хранилища, кэши моделей и настройки конкретной машины должны оставаться вне Git.

## Карта репозитория

```text
.
|-- gost-chat/                 # FastAPI RAG chat и JSON PDF indexer
|-- indexator/                 # Windows-first PySide6 desktop app для индексирования
|   |-- indexator/             # Корень Python-приложения Indexator
|   `-- run.bat                # Windows launcher для desktop app
|-- shared/                    # Общие vector-store utilities для компонентов
|-- scripts/                   # Cross-component smoke checks
`-- docs/                      # Локальный PDF corpus, по умолчанию не коммитится
```

Локальные папки, которые могут появляться при разработке:

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

## Компоненты

### Indexator

Назначение: desktop-приложение для сканирования папок с PDF, разбора ГОСТ-документов, построения structured blocks, генерации embeddings и записи локальных Qdrant vector data.

Стек:

- Python
- PySide6
- PyMuPDF
- sentence-transformers / transformers / torch
- qdrant-client с локальным persistent storage

Точки входа и важные файлы:

- `indexator/indexator/app/main.py`
- `indexator/run.bat`
- `indexator/indexator/config.json`
- `indexator/indexator/requirements.txt`
- `indexator/indexator/tests/`

Локальные артефакты:

- `indexator/indexator/.venv/`
- `indexator/indexator/data/`
- `indexator/indexator/output/`
- `indexator/indexator/testdocs/`, если только это не маленький намеренно добавленный набор лицензированных fixtures
- `indexator/.test_tmp/`

### GOST Chat

Назначение: FastAPI web app для document-grounded chat по проиндексированному содержимому. Может использовать локальные retrieval data и OpenAI-compatible LLM provider.

Стек:

- Python
- FastAPI / Uvicorn
- Jinja2, HTML, CSS, browser JavaScript
- PyMuPDF для локального JSON indexer
- qdrant-client, sentence-transformers, torch для vector retrieval и reranking
- pydantic-settings для конфигурации

Точки входа и важные файлы:

- `gost-chat/app/main.py`
- `gost-chat/apps/indexer/main.py`
- `gost-chat/app/templates/index.html`
- `gost-chat/app/static/`
- `gost-chat/requirements.txt`
- `gost-chat/.env.example`

Локальные артефакты:

- `gost-chat/.venv/`
- `gost-chat/.env`
- `gost-chat/docs/`
- `gost-chat/data/`

### Shared

Назначение: общие Python utilities для конфигурации Qdrant vector storage, payloads и models.

Стек:

- Python
- integration code для qdrant-client

Точки входа и важные файлы:

- `shared/vector_store/config.py`
- `shared/vector_store/qdrant_store.py`
- `shared/vector_store/models.py`
- `shared/vector_store/payloads.py`

Локальные артефакты:

- `shared/data/`

### Scripts

Назначение: smoke checks, которые проверяют retrieval, context building и поведение shared vector store между компонентами.

Стек:

- Python
- зависимости компонентов `gost-chat` и `indexator`

Точки входа:

- `scripts/vector_store_smoke.py`
- `scripts/retrieval_pipeline_smoke.py`
- `scripts/context_builder_smoke.py`
- `gost-chat/scripts/polza_llm_smoke.py`

## Классификация файлов

Хранить в Git:

- Рукописный Python source в `gost-chat/app/`, `gost-chat/apps/`, `indexator/indexator/app/`, `shared/` и `scripts/`
- Tests в `indexator/indexator/tests/`
- Requirements files
- README и project documentation
- `.env.example`
- `indexator/indexator/config.json`, потому что это воспроизводимый default config без секретов
- Lightweight launch scripts, например `indexator/run.bat`

Игнорировать в Git:

- `.venv/`, `venv/`, Python caches и test caches
- `.env` и другие local environment files
- Локальное Qdrant storage, SQLite databases и lock files
- Сгенерированный JSON/JSONL indexing output
- Локальные smoke-test folders
- Локальные model caches и downloaded dependency caches
- Raw PDF corpora в `docs/`, `gost-chat/docs/` и `indexator/indexator/testdocs/`

Опционально / возможно хранить:

- Маленький лицензированный sample PDF fixture, желательно в отдельной папке `samples/` или `fixtures/`
- Маленькие expected-output fixtures для tests, если они стабильны и намеренно проверены
- Developer notes вроде `FRIEND_README.md`, `RULES.md` и `agents.md`, если они полезны будущим maintainers

Заменять example/template-файлами:

- `gost-chat/.env` должен оставаться локальным; в Git хранить `gost-chat/.env.example`
- Любой будущий config с секретами коммитить только как `*.example`

Вынести из репозитория и явно документировать:

- Полный GOST PDF corpus
- Qdrant/vector store data
- Сгенерированные chunk indexes и metadata
- Скачанные ML models и caches
- Большие debug exports

## Восстановление на чистой машине

### 1. Клонировать и осмотреть

```powershell
git clone <repo-url> goster
cd goster
```

Если нужно восстановить тот же локальный state, верните external artifacts вне Git:

- PDF corpus в `docs/` или выбранную внешнюю папку
- Сохранённое Qdrant store в `shared/data/qdrant/`
- Сохранённый JSON index в `gost-chat/data/`

Generated artifacts можно также пересобрать из PDF.

### 2. Настроить Indexator

```powershell
cd indexator\indexator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

Indexator читает `config.json`. По умолчанию он пишет shared Qdrant data в `shared/data/qdrant` относительно директории приложения.

На Windows можно также использовать launcher:

```powershell
..\run.bat
```

### 3. Настроить GOST Chat

```powershell
cd ..\..\gost-chat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Отредактируйте `.env` под локальные настройки. Реальные API keys храните в shell или secret manager:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

Соберите JSON index из локальных PDF, если используете JSON retrieval path:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data
```

Запустите web app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Откройте:

```text
http://127.0.0.1:8000
```

### 4. Запустить smoke checks

Из корня рабочей папки:

```powershell
python scripts/vector_store_smoke.py
python scripts/retrieval_pipeline_smoke.py
python scripts/context_builder_smoke.py
```

Из `gost-chat`, с настроенным API key:

```powershell
python scripts/polza_llm_smoke.py
```

## Безопасная очистка Git

Если эта папка инициализирована как Git repository и generated/local files уже попали в tracked files, уберите их из индекса без удаления локальных копий:

```powershell
git rm --cached -r gost-chat/.venv indexator/indexator/.venv
git rm --cached -r gost-chat/data indexator/indexator/data indexator/indexator/output shared/data
git rm --cached -r docs gost-chat/docs indexator/indexator/testdocs
git rm --cached -r goster_vector_store_smoke_kog7mk5k goster_vector_store_smoke_tmyo2suz indexator/.test_tmp
git rm --cached gost-chat/.env
```

Выполняйте только команды для путей, которые действительно tracked. Сначала проверьте:

```powershell
git ls-files gost-chat/.venv indexator/indexator/.venv gost-chat/data indexator/indexator/data indexator/indexator/output shared/data docs gost-chat/docs indexator/indexator/testdocs gost-chat/.env
```

## Коммит и push

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

Если remote уже существует, пропустите `git remote add origin` или обновите его:

```powershell
git remote set-url origin <repo-url>
```

## Контрольный список восстановления

- Клонировать репозиторий.
- Пересоздать Python virtual environments из `requirements.txt` каждого компонента.
- Скопировать `.env.example` в `.env` там, где это нужно, и настроить local settings.
- Восстановить API keys через environment variables или secret manager.
- Восстановить PDFs из external storage или взять минимальный лицензированный sample set.
- Пересобрать `gost-chat/data/` через `gost-chat/apps/indexer/main.py`, если используется JSON retrieval.
- Пересобрать или восстановить `shared/data/qdrant/`, если используется vector retrieval.
- Запустить Indexator и проверить, что он сканирует папку с PDF.
- Запустить GOST Chat и проверить `GET /health`.
- Запустить smoke checks из `scripts/`.
