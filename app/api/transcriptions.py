from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.core.recognizer import get_recognizer
from app.schemas.responses import (
    TranscriptionResponse,
    TranscriptionVerboseResponse,
)
from app.utils.audio import convert_to_wav_bytes, format_srt, format_vtt

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),  # noqa: B008
    model: str = Form("moonshine-v2"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: str | None = Form(None),
    stream: bool = Form(False),
) -> Response:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio_data, sample_rate = convert_to_wav_bytes(
            audio_bytes, filename=file.filename or ""
        )
    except Exception as e:
        logger.error("Audio decode error: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Failed to decode audio: {e}"
        ) from e

    recognizer = get_recognizer()

    if stream:
        return StreamingResponse(
            _stream_transcription(recognizer, audio_data, sample_rate, response_format),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    start_time = time.monotonic()
    text = recognizer.transcribe(audio_data, sample_rate)
    duration = time.monotonic() - start_time

    if response_format == "text":
        return Response(content=text, media_type="text/plain")

    if response_format == "verbose_json":
        result = TranscriptionVerboseResponse(
            text=text,
            language=language,
            duration=duration,
            words=[],
        )
        return Response(
            content=result.model_dump_json(),
            media_type="application/json",
        )

    if response_format == "srt":
        segments = [{"start": 0.0, "end": duration, "text": text}]
        return Response(content=format_srt(segments), media_type="text/plain")

    if response_format == "vtt":
        segments = [{"start": 0.0, "end": duration, "text": text}]
        return Response(content=format_vtt(segments), media_type="text/plain")

    result = TranscriptionResponse(text=text)
    return Response(
        content=result.model_dump_json(),
        media_type="application/json",
    )


def _stream_transcription(
    recognizer,
    audio_data,
    sample_rate: int,
    response_format: str,
) -> Generator[str, None, None]:
    chunk_size = int(sample_rate * 0.5)
    accumulated_text = ""

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if len(chunk) == 0:
            continue

        text = recognizer.transcribe(chunk, sample_rate)
        if text:
            delta = text[len(accumulated_text):]
            if delta:
                event = json.dumps({"type": "transcript.text.delta", "delta": delta})
                yield f"data: {event}\n\n"
                accumulated_text = text

    final_event = json.dumps({"type": "transcript.text.done", "text": accumulated_text})
    yield f"data: {final_event}\n\n"
