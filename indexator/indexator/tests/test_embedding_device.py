from __future__ import annotations

import sys
import types
import unittest

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.embedding.local_backend import resolve_device


class EmbeddingDeviceResolutionTest(unittest.TestCase):
    def test_explicit_cuda_falls_back_to_cpu_when_torch_cuda_is_unavailable(self) -> None:
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = make_fake_torch(cuda_available=False)
        try:
            self.assertEqual(resolve_device("cuda"), "cpu")
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

    def test_auto_uses_cuda_when_torch_cuda_is_available(self) -> None:
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = make_fake_torch(cuda_available=True)
        try:
            self.assertEqual(resolve_device("auto"), "cuda")
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch


def make_fake_torch(cuda_available: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
    )


if __name__ == "__main__":
    unittest.main()
