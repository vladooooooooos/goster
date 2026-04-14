import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.llm_service import LlmServiceError, create_llm_service


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct Polza.ai LLM smoke test.")
    parser.add_argument(
        "--prompt",
        default="Ответь одним коротким предложением на русском языке: подключение к Polza.ai работает?",
        help="Direct prompt to send to the configured LLM provider.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    llm_service = create_llm_service(settings)

    try:
        answer = await llm_service.chat(
            [
                {
                    "role": "system",
                    "content": "Отвечай на русском языке, кратко и по существу.",
                },
                {"role": "user", "content": args.prompt},
            ]
        )
    except LlmServiceError as exc:
        print(f"Smoke test failed: {exc}")
        return 1

    print(f"Provider: {llm_service.provider}")
    print(f"Model: {llm_service.model}")
    print(f"Answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
