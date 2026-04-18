# Indexator

Indexator — Windows-first desktop-приложение на Python для подготовки русскоязычных ГОСТ PDF-документов к семантическому поиску в системе Goster.

Папка `indexator/` содержит самостоятельное приложение Indexator. Текущий scope — чистая MVP-основа с локальным end-to-end indexing pipeline.

## Текущий статус

- PySide6 desktop shell.
- Выбор папки с PDF.
- Readonly-поле выбранного пути.
- Нерекурсивное сканирование PDF-папки.
- Таблица PDF с именем файла, путём, размером, количеством страниц и статусом.
- Raw parsing страниц и text blocks через PyMuPDF.
- Safe text cleanup layer.
- First-pass structured block builder для headings, paragraphs, list items и appendix sections.
- Table of contents detection как отдельный structured block type.
- First-pass table detection для простых captioned tables.
- First-pass figure detection для простых русскоязычных captions рисунков и чертежей.
- First-pass formula detection для math-like blocks с ближайшим explanatory context.
- Conservative raw text block refinement для очевидных formula explanation, caption/body, appendix/body и больших numbered segment splits.
- Превью parsed text и structured blocks в log panel.
- Landscape и rotated page notes в debug preview.
- JSONL debug export для structured block previews.
- Local BAAI/bge-m3 embedding preview для final structured blocks.
- Compact embedding debug summary export без full vector dumps.
- Local persistent Qdrant storage preview для structured blocks и vectors.
- Compact Qdrant storage summary export.
- Full local indexing pipeline, подключённый к действию `Index selected`.
- Compact indexing run summary export.
- Shared index storage в `C:\goster\shared\data`.
- Shared document registry для indexed PDF state.
- Indexed-state detection: `Ready`, `Indexed`, `Indexed (stale)`, `Missing source`, `Index error`.
- Source PDF fingerprint metadata для stale detection.
- `Reindex selected` для forced replacement выбранных indexed documents.
- `Clear selected` для удаления выбранных indexed documents из shared index.
- `Clear all` для очистки только shared index-owned storage.
- Log panel.
- Progress bar.
- JSON config loading.
- Modular package structure для будущей pipeline-работы.

## Структура проекта

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
  tests/
  data/
  output/
  config.json
  requirements.txt
  README.md
  launch_indexator.cmd
  run.bat
```

`data/`, `output/` и `testdocs/` — локальные runtime/debug папки. По умолчанию они не должны попадать в Git.

## Запуск из корневой среды

Основной сценарий в этом workspace — единое корневое окружение `C:\goster\.venv`.

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

Запустить Indexator через переносимый лаунчер:

```powershell
.\indexator\launch_indexator.cmd
```

Или вручную:

```powershell
cd indexator
python -m app.main
```

## Запуск как отдельного компонента

Если нужен только Indexator без всего workspace:

```powershell
cd indexator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

Local embedding backend по умолчанию использует automatic device selection. CUDA будет использоваться только если в том же virtual environment установлен CUDA-enabled Torch build; иначе будет CPU fallback.

## Shared index storage

Indexator пишет локальные retrieval data в shared storage проекта:

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

Source PDFs не входят в clear semantics. Indexator удаляет только index-owned data внутри shared storage.

## Очистка indexed documents

- `Clear selected` удаляет indexed data для отмеченных PDF rows. Неиндексированные файлы пропускаются и отражаются в log и summary.
- `Clear all` очищает shared local index для configured Qdrant collection, document registry и Indexator-owned cache/debug artifacts.
- Оба действия не трогают source PDFs.
- Deletion использует стабильный `document_id`, derived from resolved source path, а не filename matching.
- Clear summaries пишутся в `C:\goster\shared\data\metadata\deletion_summaries`.

## Indexed states и reindexing

Таблица файлов объединяет текущий PDF folder scan и shared registry:

- `Ready`: source PDF существует, но registry entry для его `document_id` нет.
- `Indexed`: source PDF существует, а сохранённые size/modified timestamp совпадают с registry.
- `Indexed (stale)`: source PDF существует, но size или modified timestamp изменились после indexing.
- `Missing source`: registry entry есть для выбранной папки, но source PDF больше нет на диске.
- `Index error`: registry содержит failed indexing attempt.

`Reindex selected` — force action. Он работает с отмеченными доступными PDF, удаляет старые indexed data для каждого matching `document_id`, заново индексирует текущий source file и пишет свежие registry metadata, включая `file_size`, `modified_at` и SHA-256 `source_fingerprint`.

`Index selected` предназначен для отмеченных `Ready` files. Для unchanged indexed files или stale indexed files используйте `Reindex selected`.

## Проверки

Из корня workspace:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="C:\goster\indexator"
python -m unittest discover -s indexator\tests
docker compose up -d qdrant
python scripts\vector_store_smoke.py
```

## Следующие MVP-шаги

1. Улучшить table, figure и formula grouping heuristics на большем наборе реальных GOST samples.
2. Добавить минимальный retrieval debug path поверх indexed blocks.
