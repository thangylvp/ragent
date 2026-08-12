"""Upstream PyTorch FireRedVAD streaming-model adapter."""

from __future__ import annotations

from pathlib import Path

from .base import VadSegment, distribution_version


class FireRedBackend:
    name = "firered"

    def __init__(self, model_dir: str):
        try:
            from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        except ImportError as exc:
            raise RuntimeError("install `fireredvad` in a PyTorch environment") from exc
        self._version = distribution_version("fireredvad")
        self._model = FireRedStreamVad.from_pretrained(
            model_dir,
            FireRedStreamVadConfig(
                use_gpu=False,
                speech_threshold=0.5,
                smooth_window_size=5,
                pad_start_frame=5,
                min_speech_frame=8,
                max_speech_frame=2000,
                min_silence_frame=20,
            ),
        )

    @property
    def info(self) -> dict:
        return {
            "runtime": "pytorch_cpu",
            "model": "FireRedVAD/Stream-VAD",
            "package_version": self._version,
        }

    def detect(self, wav_path: str | Path) -> list[VadSegment]:
        _frames, result = self._model.detect_full(str(wav_path))
        return [VadSegment(float(start), float(end)) for start, end in result["timestamps"]]
