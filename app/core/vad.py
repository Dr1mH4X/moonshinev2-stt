from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SIZE = 512


class VADManager:
    """Wraps silero-vad via sherpa-onnx for voice activity detection."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        min_silence_duration: float = 0.25,
        min_speech_duration: float = 0.25,
        buffer_size_seconds: float = 60.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration

        vad_onnx_path = self._resolve_vad_model()
        logger.info("Loading VAD model: %s", vad_onnx_path)

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = vad_onnx_path
        config.silero_vad.min_silence_duration = self.min_silence_duration
        config.silero_vad.min_speech_duration = self.min_speech_duration
        config.silero_vad.window_size = DEFAULT_WINDOW_SIZE
        config.sample_rate = self.sample_rate

        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=buffer_size_seconds
        )
        self._window_size = DEFAULT_WINDOW_SIZE

    def _resolve_vad_model(self) -> str:
        candidate = self.model_path / "silero_vad.onnx"
        if candidate.exists():
            return str(candidate)
        matches = list(self.model_path.rglob("silero_vad.onnx"))
        if matches:
            return str(matches[0])
        raise FileNotFoundError(
            f"silero_vad.onnx not found under {self.model_path}. "
            "Download from: https://github.com/k2-fsa/sherpa-onnx/"
            "releases/download/asr-models/silero_vad.onnx"
        )

    def process_chunk(self, samples: np.ndarray) -> list[np.ndarray]:
        segments: list[np.ndarray] = []
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        samples = samples.astype(np.float32)

        idx = 0
        while idx + self._window_size <= len(samples):
            self._vad.accept_waveform(samples[idx : idx + self._window_size])
            idx += self._window_size

        if idx < len(samples):
            self._vad.accept_waveform(samples[idx:])

        while not self._vad.empty():
            segments.append(self._vad.front.samples.copy())
            self._vad.pop()

        return segments

    def reset(self) -> None:
        self._vad.reset()


_vad: VADManager | None = None


def get_vad() -> VADManager | None:
    global _vad
    if _vad is None:
        model_path = os.environ.get("MODEL_PATH", "/models")
        if not any(Path(model_path).rglob("silero_vad.onnx")):
            logger.warning(
                "silero_vad.onnx not found in %s, WebSocket streaming disabled",
                model_path,
            )
            return None
        _vad = VADManager(model_path=model_path)
    return _vad


def reset_vad() -> None:
    global _vad
    _vad = None
