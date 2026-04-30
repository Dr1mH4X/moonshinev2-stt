from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.recognizer import get_recognizer
from app.core.vad import get_vad

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/v1/audio/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    vad = get_vad()
    if vad is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    "WebSocket streaming requires silero_vad.onnx "
                    "in the model directory. "
                    "Download from: "
                    "https://github.com/k2-fsa/sherpa-onnx/releases/"
                    "download/asr-models/silero_vad.onnx"
                ),
            }
        )
        await websocket.close()
        return

    recognizer = get_recognizer()
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            samples = np.frombuffer(data, dtype=np.float32)

            speech_segments = vad.process_chunk(samples)
            for segment in speech_segments:
                if len(segment) < 1600:
                    continue
                text = recognizer.transcribe(segment, 16000)
                if text:
                    await websocket.send_json(
                        {
                            "type": "transcript.text.delta",
                            "delta": text,
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "transcript.text.done",
                            "text": text,
                        }
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        vad.reset()
