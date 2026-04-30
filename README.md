# Moonshine v2 STT Server

OpenAI-compatible Speech-to-Text API server powered by [Moonshine v2](https://github.com/moonshine-ai/moonshine) models via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

## Features

- **OpenAI Compatible** — Drop-in replacement for `/v1/audio/transcriptions`
- **WebSocket Streaming** — Real-time transcription via `/v1/audio/stream`
- **SSE Streaming** — Server-Sent Events with `stream=true`
- **Multi-format** — wav, mp3, flac, ogg, webm, m4a, and more
- **Multi-language** — en, zh, ja, ko, ar, es, uk, vi
- **Docker Ready** — Single container, mount models as volume
- **CPU Only** — No GPU required

## Supported Models

Models use `.ort` + `tokens.txt` format from [csukuangfj2 on HuggingFace](https://huggingface.co/csukuangfj2):

| Model | Language | Size | WER |
|-------|----------|------|-----|
| `sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27` | English | 44 MB | 12.0% |
| `sherpa-onnx-moonshine-base-en-quantized-2026-02-27` | English | 142 MB | 7.8% |
| `sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27` | Japanese | 73 MB | — |
| `sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27` | Korean | 72 MB | — |
| `sherpa-onnx-moonshine-base-zh-quantized-2026-02-27` | Chinese | 142 MB | — |
| `sherpa-onnx-moonshine-base-ar-quantized-2026-02-27` | Arabic | 142 MB | — |
| `sherpa-onnx-moonshine-base-ja-quantized-2026-02-27` | Japanese | 142 MB | — |
| `sherpa-onnx-moonshine-base-ko-quantized-2026-02-27` | Korean | 142 MB | — |
| `sherpa-onnx-moonshine-base-uk-quantized-2026-02-27` | Ukrainian | 142 MB | — |
| `sherpa-onnx-moonshine-base-vi-quantized-2026-02-27` | Vietnamese | 142 MB | — |
| `sherpa-onnx-moonshine-base-es-quantized-2026-02-27` | Spanish | 65 MB | — |

## Quick Start

### 1. Download Model

```bash
mkdir -p models

# English tiny model (~44 MB)
wget -P models https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/resolve/main/encoder_model.ort
wget -P models https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/resolve/main/decoder_model_merged.ort
wget -P models https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/resolve/main/tokens.txt

# (Optional) VAD model for WebSocket streaming
wget -P models https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

### 2. Run with Docker

```bash
docker run -d \
  --name moonshine-stt \
  -p 8000:8000 \
  -v $(pwd)/models:/models:ro \
  -e MODEL_PATH=/models \
  -e NUM_THREADS=4 \
  moonshinev2-stt:latest
```

### 3. Test

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=moonshine-v2"
```

## Docker Compose

```yaml
services:
  moonshine-stt:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    environment:
      MODEL_PATH: /models
      MODEL_NAME: moonshine-v2
      NUM_THREADS: "4"
    restart: unless-stopped
```

```bash
docker compose up -d
```

## API Reference

### POST /v1/audio/transcriptions

OpenAI-compatible transcription endpoint.

**Request** (`multipart/form-data`):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | File | Yes | — | Audio file |
| `model` | string | Yes | — | Model name |
| `language` | string | No | null | ISO-639-1 code |
| `response_format` | string | No | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | No | 0.0 | Unused (compatibility) |
| `stream` | bool | No | false | Enable SSE streaming |

**Response** (`json`):

```json
{"text": "Hello world"}
```

**Response** (`verbose_json`):

```json
{
  "text": "Hello world",
  "language": "en",
  "duration": 1.23,
  "segments": [],
  "words": []
}
```

**SSE Streaming** (`stream=true`):

```
data: {"type": "transcript.text.delta", "delta": "Hello "}

data: {"type": "transcript.text.delta", "delta": "world"}

data: {"type": "transcript.text.done", "text": "Hello world"}
```

### WS /v1/audio/stream

WebSocket endpoint for real-time streaming transcription.

**Protocol:**

1. Client connects to `ws://host:8000/v1/audio/stream`
2. Client sends binary frames (float32 PCM audio, 16kHz mono)
3. Server returns JSON events when speech is detected:

```json
{"type": "transcript.text.delta", "delta": "Hello"}
{"type": "transcript.text.done", "text": "Hello world"}
```

**Python example:**

```python
import asyncio
import numpy as np
import sounddevice as sd
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/v1/audio/stream"
    async with websockets.connect(uri) as ws:
        with sd.InputStream(channels=1, samplerate=16000, dtype="float32") as s:
            while True:
                data, _ = s.read(1600)  # 100ms chunks
                await ws.send(data.tobytes())
                result = await ws.recv()
                print(result)

asyncio.run(stream_audio())
```

### GET /health

```json
{"status": "ok", "model": "moonshine-v2", "version": "1.0.0"}
```

### GET /v1/models

```json
{
  "object": "list",
  "data": [{"id": "moonshine-v2", "object": "model", "owned_by": "moonshine-ai"}]
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models` | Path to model directory |
| `MODEL_NAME` | `moonshine-v2` | Model name returned by API |
| `NUM_THREADS` | `4` | Inference threads |
| `DEBUG` | `false` | Enable debug logging |

## OpenAI Python SDK Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="moonshine-v2",
        file=f,
        response_format="verbose_json",
    )

print(transcript.text)
```

## Development

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx ruff

# Run server
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Lint
ruff check app/ tests/
```

## License

MIT
