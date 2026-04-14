# CLI Indexer

`apps/indexer` — это отдельное внутреннее CLI-приложение для подготовки PDF-документов к будущему RAG-поиску в чат-приложении.

Индексатор работает локально: читает PDF-файлы из указанной папки, извлекает текст, режет его на chunks и сохраняет структурированные JSON-файлы на диск. Он не вызывает Ollama, не создаёт embeddings, не выполняет retrieval, не собирает RAG-промпты и не использует базу данных или облачные сервисы.

## Кто выполняет индексирование

Индексирование выполняет локальный Python-скрипт:

```powershell
python apps/indexer/main.py
```

Это административный инструмент. Его нужно запускать вручную или позже подключить к отдельному внутреннему расписанию/пайплайну. Существующее чат-приложение в `app/` сейчас не запускает индексатор автоматически.

## Как работает индексатор

1. Загружает настройки из CLI-аргументов и переменных окружения.
2. Ищет PDF-файлы в папке `docs/` или в папке из `--input-dir`.
3. Игнорирует все файлы, которые не заканчиваются на `.pdf`.
4. Для каждого PDF считает SHA-256 hash файла.
5. Строит стабильный `document_id` на основе пути к файлу.
6. Сравнивает hash с уже сохранёнными metadata.
7. Если файл уже индексировался и hash не изменился, пропускает его.
8. Если файл новый, изменился или передан флаг `--reindex`, извлекает текст через PyMuPDF.
9. Извлекает текст постранично, чтобы сохранить ссылки на страницы.
10. Режет текст на chunks с настраиваемым размером и overlap.
11. Сохраняет chunks и metadata в локальные JSON-файлы.
12. Печатает summary в консоль.

Если один PDF не удалось обработать, индексатор логирует ошибку, помечает документ как `failed` и продолжает обработку остальных PDF.

## Установка зависимости

Индексатор использует PyMuPDF. Установи зависимости проекта:

```powershell
pip install -r requirements.txt
```

Если используешь проектное виртуальное окружение:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если команда `python apps/indexer/main.py` падает с ошибкой `ModuleNotFoundError: No module named 'fitz'`, значит используется Python-окружение без PyMuPDF. Запусти индексатор через проектное окружение:

```powershell
.\.venv\Scripts\python.exe apps/indexer/main.py
```

Или активируй окружение перед запуском:

```powershell
.\.venv\Scripts\Activate.ps1
python apps/indexer/main.py
```

## Основные команды

Запуск с настройками по умолчанию:

```powershell
python apps/indexer/main.py
```

По умолчанию индексатор читает PDF из `docs/` и пишет результат в `data/`.

Указать входную папку:

```powershell
python apps/indexer/main.py --input-dir docs
```

Указать входную и выходную папки:

```powershell
python apps/indexer/main.py --input-dir C:\path\to\gost-pdfs --output-dir data
```

Принудительно переиндексировать все PDF, даже если hash не изменился:

```powershell
python apps/indexer/main.py --input-dir docs --reindex
```

Настроить размер chunk и overlap:

```powershell
python apps/indexer/main.py --chunk-size 3000 --chunk-overlap 300
```

Очистить файлы индекса:

```powershell
python apps/indexer/main.py --output-dir data --clear
```

Показать справку:

```powershell
python apps/indexer/main.py --help
```

## Переменные окружения

Вместо CLI-аргументов можно использовать переменные окружения:

```text
GOST_INDEXER_INPUT_DIR=docs
GOST_INDEXER_OUTPUT_DIR=data
GOST_INDEXER_CHUNK_SIZE=3000
GOST_INDEXER_CHUNK_OVERLAP=300
```

CLI-аргументы удобнее для разовых запусков. Переменные окружения удобнее для повторяемого локального сценария.

## Выходные файлы

Индексатор создаёт такие файлы:

```text
data/
├─ index/
│  └─ chunks.jsonl
└─ metadata/
   ├─ documents.json
   └─ indexing_summary.json
```

### `data/index/chunks.jsonl`

Файл с chunks. Каждая строка — отдельный JSON-объект:

```json
{
  "document_id": "1b180454c521a2b6",
  "file_name": "sample.pdf",
  "chunk_id": "1b180454c521a2b6:00001",
  "page_start": 1,
  "page_end": 1,
  "text": "Extracted document text..."
}
```

Этот файл должен стать основным источником подготовленного текста для будущего retrieval/embedding шага.

### `data/metadata/documents.json`

Файл с metadata по документам:

```json
{
  "schema_version": 1,
  "documents": [
    {
      "document_id": "1b180454c521a2b6",
      "filename": "sample.pdf",
      "source_path": "C:\\path\\to\\sample.pdf",
      "file_hash": "sha256...",
      "indexed_at": "2026-04-12T19:25:24.219067+00:00",
      "chunk_count": 12,
      "status": "indexed"
    }
  ]
}
```

Этот файл нужен, чтобы понимать состояние каждого PDF: был ли он проиндексирован, сколько chunks создано, какой hash был у файла и когда выполнялась индексация.

### `data/metadata/indexing_summary.json`

Файл со summary последнего запуска:

```json
{
  "schema_version": 1,
  "indexed_at": "2026-04-12T19:25:28.601801+00:00",
  "input_dir": "C:\\path\\to\\docs",
  "output_dir": "C:\\path\\to\\data",
  "chunk_size": 3000,
  "chunk_overlap": 300,
  "reindex": false,
  "clear": false,
  "files_found": 10,
  "files_indexed": 2,
  "files_skipped": 8,
  "files_failed": 0,
  "errors": []
}
```

Этот файл удобен для проверки результата последнего запуска.

## Как работает `--clear`

Команда:

```powershell
python apps/indexer/main.py --output-dir data --clear
```

Удаляет только файлы, созданные индексатором:

- `data/index/chunks.jsonl`
- `data/metadata/documents.json`
- `data/metadata/indexing_summary.json`

Она не удаляет исходные PDF и не удаляет папку `docs/`. После очистки можно запустить индексатор снова, и он создаст выходные файлы заново.

## Пример рабочего сценария

1. Положить PDF-файлы в папку `docs/`.
2. Запустить индексирование:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data
```

3. Проверить summary в консоли.
4. Проверить chunks:

```powershell
Get-Content data\index\chunks.jsonl -TotalCount 3
```

5. Добавить или заменить PDF-файлы в `docs/`.
6. Запустить индексатор снова:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data
```

Неизменённые PDF будут пропущены, изменённые будут переиндексированы.

## Как будущий chat app должен использовать результат

Будущий retrieval слой должен читать `data/index/chunks.jsonl` как источник chunks. Для каждого chunk можно использовать:

- `chunk_id` как стабильный ключ chunk.
- `document_id` для связи с `documents.json`.
- `text` для embeddings и поиска.
- `file_name`, `page_start`, `page_end` для ссылок на источник.

Следующий этап может создать embeddings для каждого chunk и сохранить их в отдельном локальном формате или vector store. Текущий indexer намеренно этого не делает.

## Ограничения текущей версии

- Удалённые из `docs/` PDF пока не удаляются автоматически из уже сохранённого индекса.
- Duplicate handling ограничен hash-проверкой для уже известных файлов.
- Chunking простой и основан на тексте, размере chunk и overlap.
- Нет embeddings, retrieval, RAG, базы данных, vector DB или cloud API.
