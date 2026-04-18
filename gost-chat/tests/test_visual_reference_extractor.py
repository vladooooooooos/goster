import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.visual_reference_extractor import VisualReferenceExtractor


class VisualReferenceExtractorTest(unittest.TestCase):
    def test_extracts_russian_and_english_graphic_references(self):
        mentions = VisualReferenceExtractor().extract(
            "See Figure 1, Drawing 2, Table E.2, formula (1), "
            "\u0440\u0438\u0441. 3.3.4, "
            "\u0420\u0438\u0441\u0443\u043d\u043e\u043a 1, "
            "\u0427\u0435\u0440\u0442. 2, "
            "\u0422\u0430\u0431\u043b\u0438\u0446\u0430 \u0415.2, "
            "\u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 2-2"
        )

        labels = {mention.normalized_label for mention in mentions}

        self.assertIn("figure 1", labels)
        self.assertIn("drawing 2", labels)
        self.assertIn("table e.2", labels)
        self.assertIn("formula 1", labels)
        self.assertIn("figure 3.3.4", labels)
        self.assertIn("appendix 2-2", labels)

    def test_text_without_graphic_reference_has_no_mentions(self):
        mentions = VisualReferenceExtractor().extract(
            "\u041e\u0431\u043e\u0440\u0443\u0434\u043e\u0432\u0430\u043d\u0438\u0435 "
            "\u0434\u043e\u043b\u0436\u043d\u043e \u0431\u044b\u0442\u044c "
            "\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e \u0434\u043b\u044f "
            "\u0440\u0430\u0431\u043e\u0447\u0435\u0433\u043e \u043c\u0435\u0441\u0442\u0430."
        )

        self.assertEqual(mentions, [])


if __name__ == "__main__":
    unittest.main()
