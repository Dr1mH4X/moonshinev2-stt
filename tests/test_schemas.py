from __future__ import annotations


def test_imports():
    from app.core.recognizer import MoonshineRecognizer
    from app.core.vad import VADManager
    from app.schemas.responses import HealthResponse, TranscriptionResponse
    assert MoonshineRecognizer is not None
    assert VADManager is not None
    assert HealthResponse is not None
    assert TranscriptionResponse is not None


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
