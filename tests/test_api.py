from __future__ import annotations

import io
import struct
import sys
import wave
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_recognizer():
    mock = MagicMock()
    mock.transcribe.return_value = "Hello world"
    mock.transcribe_with_timestamps.return_value = {
        "text": "Hello world",
        "duration": 1.0,
        "language": None,
        "segments": [],
        "words": [],
    }
    return mock


@pytest.fixture()
def client(mock_recognizer):
    for mod_name in list(sys.modules):
        if mod_name.startswith("app."):
            del sys.modules[mod_name]

    with patch.dict("os.environ", {"MODEL_PATH": "/tmp/models"}):
        from app.main import app as test_app

        test_app.router.on_startup.clear()
        test_app.router.on_shutdown.clear()

        with (
            patch(
                "app.core.recognizer.get_recognizer",
                return_value=mock_recognizer,
            ),
            patch("app.core.vad.get_vad", return_value=None),
        ):
            yield TestClient(test_app)


def _make_wav(duration_samples: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(
            struct.pack(f"<{duration_samples}h", *([0] * duration_samples))
        )
    return buf.getvalue()


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Moonshine v2 STT Server"
    assert "docs" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "moonshine-v2"


def test_transcribe_empty_file(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2"},
        files={"file": ("test.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_json(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2", "response_format": "json"},
        files={"file": ("test.wav", _make_wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Hello world"


def test_transcribe_text_format(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2", "response_format": "text"},
        files={"file": ("test.wav", _make_wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.text == "Hello world"


def test_transcribe_verbose_json(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2", "response_format": "verbose_json"},
        files={"file": ("test.wav", _make_wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "duration" in data
