import unittest
from pathlib import Path


class FrontendSessionApiTest(unittest.TestCase):
    def test_frontend_posts_messages_to_session_api_instead_of_ask(self):
        source = Path("gost-chat/app/static/app.js").read_text(encoding="utf-8")

        self.assertIn('fetch("/chat/sessions"', source)
        self.assertIn("backendSessionId", source)
        self.assertNotIn('fetch("/ask"', source)


if __name__ == "__main__":
    unittest.main()
