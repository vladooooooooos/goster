# GOSTer: быстрая инструкция для запуска

В архиве лежит локальное FastAPI-приложение с RAG-чатом по проиндексированным PDF-документам.

## Что нужно установить

- Python 3.11 или новее.
- Ollama, установленная и запущенная локально.
- Локально загруженная модель Ollama.

По умолчанию в `.env.example` указана модель:

```text
gemma4:e4b
```

Если хочешь использовать другую модель Ollama, после создания `.env` измени значение `GOST_CHAT_OLLAMA_MODEL`.

## Как запустить

Открой PowerShell в распакованной папке `gost-chat` и выполни:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull gemma4:e4b
```

Запусти Ollama, если она еще не запущена:

```powershell
ollama serve
```

В другом окне PowerShell, из той же папки `gost-chat`, выполни:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Открой приложение в браузере:

```text
http://127.0.0.1:8000
```

## Документы и индекс

В архиве могут быть папки `docs/` и `data/`. Если файл `data/index/chunks.jsonl` уже есть, приложение сразу сможет использовать готовый индекс.

Чтобы пересобрать индекс из PDF-файлов в папке `docs/`, выполни:

```powershell
python apps/indexer/main.py --input-dir docs --output-dir data --reindex
```

После этого перезапусти FastAPI-приложение.

## Если что-то не работает

- Если приложение возвращает ошибку Ollama, проверь, что запущена команда `ollama serve`.
- Если модель не найдена, выполни `ollama pull gemma4:e4b` или проверь, что модель доступна в Ollama именно под тегом `gemma4:e4b`.
- Если приложение пишет, что индекс не найден, пересобери индекс командой из раздела выше.
