from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="session")
def _stub_sherpa_onnx():
    """Inject a mock sherpa_onnx into sys.modules for the entire test session."""
    mock = MagicMock()
    mock.OfflineRecognizer = MagicMock()
    mock.OfflineRecognizer.from_moonshine_v2 = MagicMock()
    mock.VadModelConfig = MagicMock()
    mock.VoiceActivityDetector = MagicMock()
    sys.modules["sherpa_onnx"] = mock
    yield
    sys.modules.pop("sherpa_onnx", None)
