# Рабочая папка Goster

Goster — локальная рабочая папка для индексирования русскоязычных PDF-документов ГОСТ и вопросов по проиндексированному содержимому. Сейчас это не одно приложение, а несколько соседних Python-компонентов.

В Git стоит хранить исходный код, документацию, setup-файлы и лёгкие примеры. Локальные виртуальные окружения, корпуса документов, сгенерированные индексы, векторные хранилища, кэши моделей и настройки конкретной машины должны оставаться вне Git.

## Карта репозитория

```text
.
|-- gost-chat/                 # FastAPI RAG chat и JSON PDF indexer
|-- indexator/                 # Windows-first PySide6 desktop app для индексирования
|   |-- app/                   # Корень Python-приложения Indexator
|   `-- launch_indexator.cmd   # Переносимый Windows-лаунчер для desktop-приложения
|-- shared/                    # Общие vector-store utilities для компонентов
|-- scripts/                   # Cross-component smoke checks
`-- docs/                      # Локальный PDF corpus, по умолчанию не коммитится
```

Локальные папки, которые могут появляться при разработке:

```text
.venv/
gost-chat/data/
gost-chat/docs/
indexator/data/
indexator/output/
indexator/testdocs/
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
- qdrant-client connected to a local Qdrant server

Точки входа и важные файлы:

- `indexator/app/main.py`
- `indexator/launch_indexator.cmd`
- `indexator/config.json`
- `indexator/requirements.txt`
- `indexator/tests/`

Локальные артефакты:

- `indexator/data/`
- `indexator/output/`
- `indexator/testdocs/`, если только это не маленький намеренно добавленный набор лицензированных fixtures
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

- Рукописный Python source в `gost-chat/app/`, `gost-chat/apps/`, `indexator/app/`, `shared/` и `scripts/`
- Tests в `indexator/tests/`
- Файлы зависимостей
- Корневой `requirements-dev.txt` для общей workspace/dev-среды
- README и документация проекта
- `.env.example`
- `indexator/config.json`, потому что это воспроизводимый default config без секретов
- Лёгкие лаунчеры, например `indexator/launch_indexator.cmd`

Игнорировать в Git:

- `.venv/`, `venv/`, Python caches и test caches
- `.env` и другие local environment files
- Codex-only файлы `indexator/agents.md` и `indexator/RULES.md`
- Локальное Qdrant storage, SQLite databases и lock files
- Сгенерированный JSON/JSONL indexing output
- Локальные smoke-test folders
- Локальные model caches и downloaded dependency caches
- Raw PDF corpora в `docs/`, `gost-chat/docs/` и `indexator/testdocs/`

Опционально / возможно хранить:

- Маленький лицензированный sample PDF fixture, желательно в отдельной папке `samples/` или `fixtures/`
- Маленькие expected-output fixtures для tests, если они стабильны и намеренно проверены
- Developer notes, если они полезны будущим maintainers и не являются локальными Codex-only инструкциями

Заменять example/template-файлами:

- `gost-chat/.env` должен оставаться локальным; в Git хранить `gost-chat/.env.example`
- Любой будущий config с секретами коммитить только как `*.example`

Вынести из репозитория и явно документировать:

- Полный GOST PDF corpus
- Qdrant/vector store data
- Сгенерированные chunk indexes и metadata
- Скачанные ML models и caches
- Большие debug exports

## Примерный размер зависимостей

Размер установленного `.venv` зависит от версии Python, платформы и варианта `torch`.

Ориентиры по старой локальной установке с двумя component `.venv`:

- `gost-chat/.venv`: около 4.7 GB, потому что установлен CUDA-вариант `torch`; сам пакет `torch` занимает около 4.2 GB.
- старый component `.venv` для Indexator: около 1.6 GB; заметную часть занимали `PySide6`, `torch`, `transformers` и ML-зависимости.
- Два отдельных окружения вместе могут занимать около 6.3 GB или больше.
- Одно корневое dev-окружение через `requirements-dev.txt` обычно будет компактнее двух отдельных окружений, потому что общие зависимости ставятся один раз. Для CPU-only `torch` ожидайте примерно 2-3 GB, для CUDA-варианта `torch` — примерно 5-7 GB.

Если важен минимальный размер, создавайте окружение только для нужного компонента:

```powershell
python -m pip install -r gost-chat/requirements.txt
```

или:

```powershell
python -m pip install -r indexator/requirements.txt
```

Если нужно запускать smoke checks из корня и работать сразу с обоими компонентами, используйте:

```powershell
python -m pip install -r requirements-dev.txt
```

Если нужен CUDA-enabled `torch` в едином корневом окружении, после установки dev requirements переустановите `torch` из CUDA wheel index:

```powershell
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## Восстановление на чистой машине

### 1. Клонировать и осмотреть

```powershell
git clone <repo-url> goster
cd goster
```

Если нужно восстановить тот же локальный state, верните external artifacts вне Git:

- PDF corpus в `docs/` или выбранную внешнюю папку
- Local Qdrant server data managed by Qdrant, usually through `docker compose up qdrant`
- Сохранённый JSON index в `gost-chat/data/`

Сгенерированные артефакты можно также пересобрать из PDF.

### 2. Создать единую корневую Python-среду

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

### 3. Запустить Indexator

Indexator reads `config.json`. By default it writes vectors to the local Qdrant server at `http://127.0.0.1:6333`, collection `gost_blocks`.

The chat and Indexator launchers check this server on startup. If it is not reachable, they run Docker Compose and wait for Qdrant to become available:

```powershell
docker compose up -d qdrant
```

To start or stop the local Qdrant server manually with a double click, run:

```powershell
.\start_qdrant_server.cmd
.\stop_qdrant_server.cmd
```

На Windows можно также использовать переносимый лаунчер:

```powershell
.\indexator\launch_indexator.cmd
```

Или вручную из активированной корневой среды:

```powershell
cd indexator
python -m app.main
```

### 4. Настроить GOST Chat

```powershell
cd gost-chat
Copy-Item .env.example .env
```

Отредактируйте `.env` под локальные настройки. В `.env.example` уже указано `GOST_CHAT_LLM_API_KEY_ENV_VAR=POLZA_API_KEY`, поэтому GOST Chat будет искать ключ Polza.ai именно в переменной среды `POLZA_API_KEY`.

Для текущего окна PowerShell можно задать её так:

```powershell
$env:POLZA_API_KEY="your-api-key"
```

Чтобы ключ Polza.ai был доступен после перезапуска PowerShell и при запуске через `.cmd`, создайте пользовательскую переменную среды Windows:

```powershell
[Environment]::SetEnvironmentVariable("POLZA_API_KEY", "your-api-key", "User")
```

После этого откройте новое окно PowerShell или перезапустите приложение, чтобы Windows передала новую переменную процессам.

То же самое можно сделать вручную через интерфейс Windows: откройте настройки переменных среды пользователя, добавьте переменную `POLZA_API_KEY` и укажите в ней ваш API key Polza.ai. После сохранения также откройте новое окно PowerShell или перезапустите приложение.

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

### 5. Запустить smoke checks

Из корня рабочей папки:

```powershell
.\.venv\Scripts\Activate.ps1
```

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

Если эта папка инициализирована как Git repository и сгенерированные/локальные файлы уже попали в tracked files, уберите их из индекса без удаления локальных копий:

```powershell
git rm --cached -r .venv gost-chat/.venv indexator/.venv
git rm --cached -r gost-chat/data indexator/data indexator/output shared/data
git rm --cached -r docs gost-chat/docs indexator/testdocs
git rm --cached -r goster_vector_store_smoke_kog7mk5k goster_vector_store_smoke_tmyo2suz indexator/.test_tmp
git rm --cached indexator/agents.md indexator/RULES.md
git rm --cached gost-chat/.env
```

Выполняйте только команды для путей, которые действительно tracked. Сначала проверьте:

```powershell
git ls-files .venv gost-chat/.venv indexator/.venv gost-chat/data indexator/data indexator/output shared/data docs gost-chat/docs indexator/testdocs indexator/agents.md indexator/RULES.md gost-chat/.env
```

## Коммит и push

```powershell
git status --short
git add .gitignore README.md indexator/launch_indexator.cmd
git add gost-chat/README.md gost-chat/.env.example indexator/README.md indexator/config.json
git add gost-chat/app gost-chat/apps gost-chat/scripts indexator/app indexator/tests shared scripts
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
- Пересоздать единую корневую Python-среду через `requirements-dev.txt`.
- Скопировать `.env.example` в `.env` там, где это нужно, и настроить локальные параметры.
- Восстановить API keys через environment variables или secret manager.
- Восстановить PDFs из external storage или взять минимальный лицензированный sample set.
- Пересобрать `gost-chat/data/` через `gost-chat/apps/indexer/main.py`, если используется JSON retrieval.
- Start the local Qdrant server and reindex PDFs with Indexator when vector retrieval is used.
- Запустить Indexator и проверить, что он сканирует папку с PDF.
- Запустить GOST Chat и проверить `GET /health`.
- Запустить smoke checks из `scripts/`.
