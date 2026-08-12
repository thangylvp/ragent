"""OmniVAD-Kit ncnn deployment of the FireRed streaming model."""

from __future__ import annotations

from pathlib import Path

from .base import VadSegment, distribution_version


class OmniVadBackend:
    name = "omnivad"

    def __init__(self):
        try:
            from omnivad import OmniStreamVAD
        except ImportError as exc:
            raise RuntimeError("install `omnivad` to use the ncnn backend") from exc
        self._version = distribution_version("omnivad")
        self._model = OmniStreamVAD(
            threshold=0.5,
            smooth_window_size=5,
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=20,
        )

    @property
    def info(self) -> dict:
        return {
            "runtime": "ncnn_cpu",
            "model": "FireRedVAD/Stream-VAD",
            "package_version": self._version,
        }

    def detect(self, wav_path: str | Path) -> list[VadSegment]:
        timestamps = self._model.detect_segments(str(wav_path))
        return [VadSegment(float(start), float(end)) for start, end in timestamps]
