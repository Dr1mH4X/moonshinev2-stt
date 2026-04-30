from __future__ import annotations

from pydantic import BaseModel


class WordInfo(BaseModel):
    word: str
    start: float = 0.0
    end: float = 0.0


class SegmentInfo(BaseModel):
    id: int
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponse(BaseModel):
    text: str


class TranscriptionVerboseResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float = 0.0
    segments: list[SegmentInfo] = []
    words: list[WordInfo] = []


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "moonshine-ai"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = "moonshine-v2"
    version: str = "1.0.0"


class StreamDeltaEvent(BaseModel):
    type: str = "transcript.text.delta"
    delta: str


class StreamDoneEvent(BaseModel):
    type: str = "transcript.text.done"
    text: str
