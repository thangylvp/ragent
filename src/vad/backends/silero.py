"""Silero VAD PyTorch CPU adapter."""

from __future__ import annotations

from pathlib import Path

from .audio import read_pcm16_mono
from .base import VadSegment, distribution_version


class SileroBackend:
    name = "silero"

    def __init__(self):
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError("install `silero-vad` in a PyTorch environment") from exc
        self._torch = torch
        self._model = load_silero_vad(onnx=False)

    @property
    def info(self) -> dict:
        return {
            "runtime": "pytorch_cpu",
            "model": "silero_vad_v6",
            "package_version": distribution_version("silero-vad"),
        }

    def detect(self, wav_path: str | Path) -> list[VadSegment]:
        from silero_vad import get_speech_timestamps

        samples, sample_rate = read_pcm16_mono(wav_path)
        if sample_rate != 16_000:
            raise ValueError(f"expected 16000 Hz, got {sample_rate}: {wav_path}")
        audio = self._torch.tensor(samples, dtype=self._torch.float32) / 32768.0
        timestamps = get_speech_timestamps(
            audio,
            self._model,
            threshold=0.5,
            sampling_rate=sample_rate,
            min_speech_duration_ms=80,
            min_silence_duration_ms=200,
            speech_pad_ms=50,
            return_seconds=True,
        )
        return [
            VadSegment(float(item["start"]), float(item["end"])) for item in timestamps
        ]
