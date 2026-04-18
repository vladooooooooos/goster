import unittest

from app.config import Settings
from app.schemas.ask import AskRequest
from app.services.context_builder import ContextBuilderSettings


class AdaptiveSettingsTest(unittest.TestCase):
    def test_ask_request_uses_wider_default_top_k(self):
        request = AskRequest(query="What is required?")

        self.assertEqual(request.top_k, 12)

    def test_settings_expose_adaptive_context_defaults(self):
        settings = Settings()

        self.assertEqual(settings.context_min_blocks, 2)
        self.assertEqual(settings.context_soft_target_blocks, 5)
        self.assertEqual(settings.context_max_blocks, 10)
        self.assertEqual(settings.context_max_chars, 18000)
        self.assertEqual(settings.context_adaptive_score_threshold, 0.12)

    def test_settings_expose_visual_defaults(self):
        settings = Settings()

        self.assertTrue(settings.visual_enable_decision)
        self.assertEqual(settings.visual_crops_dir.as_posix(), "data/crops")
        self.assertEqual(settings.visual_crop_dpi, 160)
        self.assertEqual(settings.visual_max_crops_per_answer, 1)

    def test_context_builder_settings_accept_adaptive_values(self):
        settings = ContextBuilderSettings(
            min_blocks=2,
            soft_target_blocks=5,
            max_blocks=10,
            max_context_chars=18000,
            adaptive_score_threshold=0.12,
        )

        self.assertEqual(settings.min_blocks, 2)
        self.assertEqual(settings.soft_target_blocks, 5)
        self.assertEqual(settings.max_blocks, 10)
        self.assertEqual(settings.max_context_chars, 18000)
        self.assertEqual(settings.adaptive_score_threshold, 0.12)
