from __future__ import annotations

import io
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm",
    ".opus",
    ".aac",
    ".wma",
}


def convert_to_wav_bytes(
    audio_bytes: bytes, filename: str = ""
) -> tuple[np.ndarray, int]:
    """Convert any audio format to mono float32 PCM using ffmpeg."""
    ext = Path(filename).suffix.lower() if filename else ""
    if ext == ".wav" or ext == "":
        return _decode_wav_bytes(audio_bytes)
    return _decode_with_ffmpeg(audio_bytes)


def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio[:, 0]
        return audio, sr
    except Exception:
        return _decode_with_ffmpeg(audio_bytes)


def _decode_with_ffmpeg(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as inf:
        inf.write(audio_bytes)
        input_path = inf.name

    output_path = input_path + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                "-acodec",
                "pcm_s16le",
                output_path,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        import soundfile as sf

        audio, sr = sf.read(output_path, dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            audio = audio[:, 0]
        return audio, sr
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed: %s", e.stderr.decode())
        raise ValueError(f"Failed to decode audio: {e.stderr.decode()}") from e
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg.get("start", 0.0))
        end = _format_srt_time(seg.get("end", 0.0))
        text = seg.get("text", "")
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = _format_vtt_time(seg.get("start", 0.0))
        end = _format_vtt_time(seg.get("end", 0.0))
        text = seg.get("text", "")
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
