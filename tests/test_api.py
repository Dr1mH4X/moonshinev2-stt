from __future__ import annotations

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
    with patch("app.api.transcriptions.get_recognizer", return_value=mock_recognizer), \
         patch("app.api.health.get_recognizer", return_value=mock_recognizer), \
         patch("app.main.get_recognizer", return_value=mock_recognizer), \
         patch("app.main.get_vad", return_value=None):
        from app.main import app
        yield TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Moonshine v2 STT Server"
    assert "docs" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "moonshine-v2"


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "moonshine-v2"


def test_transcribe_empty_file(client):
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2"},
        files={"file": ("test.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_transcribe_json(client, mock_recognizer):
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
    wav_bytes = buf.getvalue()

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2", "response_format": "json"},
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["text"] == "Hello world"


def test_transcribe_text_format(client, mock_recognizer):
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
    wav_bytes = buf.getvalue()

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "moonshine-v2", "response_format": "text"},
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.text == "Hello world"
