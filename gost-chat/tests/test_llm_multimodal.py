import unittest
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import Settings
from app.services.llm_service import PolzaLlmService


class PolzaMultimodalPayloadTest(unittest.TestCase):
    def test_builds_openai_compatible_image_content(self):
        service = PolzaLlmService(Settings())

        payload = service.build_chat_payload(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this figure."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ]
        )

        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,abc")


if __name__ == "__main__":
    unittest.main()
