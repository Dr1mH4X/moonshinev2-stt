from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


class MoonshineRecognizer:
    """Wraps sherpa-onnx OfflineRecognizer for Moonshine v2 models."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        debug: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.num_threads = num_threads
        self.debug = debug

        self._encoder_path = self._resolve_file("encoder_model.ort")
        self._decoder_path = self._resolve_file("decoder_model_merged.ort")
        self._tokens_path = self._resolve_file("tokens.txt")

        self._recognizer = self._build_recognizer()
        logger.info(
            "MoonshineRecognizer loaded: encoder=%s decoder=%s tokens=%s threads=%d",
            self._encoder_path,
            self._decoder_path,
            self._tokens_path,
            self.num_threads,
        )

    def _resolve_file(self, name: str) -> str:
        candidate = self.model_path / name
        if candidate.exists():
            return str(candidate)
        matches = list(self.model_path.rglob(name))
        if matches:
            return str(matches[0])
        raise FileNotFoundError(f"Model file '{name}' not found under {self.model_path}")

    def _build_recognizer(self) -> sherpa_onnx.OfflineRecognizer:
        return sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=self._encoder_path,
            decoder=self._decoder_path,
            tokens=self._tokens_path,
            num_threads=self.num_threads,
            debug=self.debug,
            provider="cpu",
            decoding_method="greedy_search",
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> str:
        audio = self._ensure_mono_float32(audio)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        text = self.transcribe(audio, sample_rate)
        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
        return {
            "text": text,
            "duration": duration,
            "language": None,
            "segments": [],
            "words": [],
        }

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> str:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        return self.transcribe(audio, sample_rate)

    @staticmethod
    def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio[:, 0]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio


_recognizer: MoonshineRecognizer | None = None


def get_recognizer() -> MoonshineRecognizer:
    global _recognizer
    if _recognizer is None:
        model_path = os.environ.get("MODEL_PATH", "/models")
        num_threads = int(os.environ.get("NUM_THREADS", "4"))
        debug = os.environ.get("DEBUG", "false").lower() == "true"
        _recognizer = MoonshineRecognizer(
            model_path=model_path,
            num_threads=num_threads,
            debug=debug,
        )
    return _recognizer


def reset_recognizer() -> None:
    global _recognizer
    _recognizer = None
