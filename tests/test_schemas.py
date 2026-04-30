from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_health_response_model():
    from app.schemas.responses import HealthResponse

    resp = HealthResponse()
    assert resp.status == "ok"
    assert resp.model == "moonshine-v2"
    assert resp.version == "1.0.0"


def test_transcription_response_model():
    from app.schemas.responses import TranscriptionResponse

    resp = TranscriptionResponse(text="Hello world")
    assert resp.text == "Hello world"
    assert "text" in resp.model_dump()


def test_model_list_response():
    from app.schemas.responses import ModelInfo, ModelListResponse

    resp = ModelListResponse(data=[ModelInfo(id="test")])
    assert resp.object == "list"
    assert len(resp.data) == 1
    assert resp.data[0].id == "test"


def test_recognizer_init():
    from app.core.recognizer import MoonshineRecognizer

    with patch.object(
        MoonshineRecognizer, "_resolve_file", return_value="/tmp/fake"
    ):
        recognizer = MoonshineRecognizer(
            model_path="/tmp/models", num_threads=2
        )
        assert recognizer is not None


def test_transcribe_returns_text():
    from app.core.recognizer import MoonshineRecognizer

    mock_stream = MagicMock()
    mock_stream.result.text = "  hello world  "
    mock_recognizer_obj = MagicMock()
    mock_recognizer_obj.create_stream.return_value = mock_stream
    sherpa = sys.modules["sherpa_onnx"]
    sherpa.OfflineRecognizer.from_moonshine_v2.return_value = (
        mock_recognizer_obj
    )

    with patch.object(
        MoonshineRecognizer, "_resolve_file", return_value="/tmp/fake"
    ):
        recognizer = MoonshineRecognizer(model_path="/tmp/models")
        result = recognizer.transcribe(
            np.zeros(16000, dtype=np.float32), 16000
        )
        assert result == "hello world"


def test_transcribe_2d_audio():
    from app.core.recognizer import MoonshineRecognizer

    mock_stream = MagicMock()
    mock_stream.result.text = "stereo test"
    mock_recognizer_obj = MagicMock()
    mock_recognizer_obj.create_stream.return_value = mock_stream
    sherpa = sys.modules["sherpa_onnx"]
    sherpa.OfflineRecognizer.from_moonshine_v2.return_value = (
        mock_recognizer_obj
    )

    with patch.object(
        MoonshineRecognizer, "_resolve_file", return_value="/tmp/fake"
    ):
        recognizer = MoonshineRecognizer(model_path="/tmp/models")
        stereo = np.zeros((16000, 2), dtype=np.float32)
        result = recognizer.transcribe(stereo, 16000)
        assert result == "stereo test"


def test_stream_delta_event():
    from app.schemas.responses import StreamDeltaEvent

    event = StreamDeltaEvent(delta="hello")
    assert event.type == "transcript.text.delta"
    assert event.delta == "hello"


def test_stream_done_event():
    from app.schemas.responses import StreamDoneEvent

    event = StreamDoneEvent(text="hello world")
    assert event.type == "transcript.text.done"
    assert event.text == "hello world"
