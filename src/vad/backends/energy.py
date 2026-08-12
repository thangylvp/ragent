"""Batch adapter for the repository's dependency-free streaming baseline."""

from __future__ import annotations

from pathlib import Path

from vad import EnergyVad, VadEventKind

from .audio import read_pcm16_mono
from .base import VadSegment


class EnergyBackend:
    name = "energy"

    def __init__(self):
        self.engine = EnergyVad()

    @property
    def info(self) -> dict:
        return {
            "runtime": "python_stdlib",
            "model": "adaptive_energy_baseline",
            "frame_ms": self.engine.config.frame_ms,
        }

    def detect(self, wav_path: str | Path) -> list[VadSegment]:
        samples, sample_rate = read_pcm16_mono(wav_path)
        if sample_rate != self.engine.config.sample_rate:
            raise ValueError(f"expected 16000 Hz, got {sample_rate}: {wav_path}")
        self.engine.reset()
        frame_samples = self.engine.config.frame_samples
        segments: list[VadSegment] = []
        pcm = samples.tobytes()
        frame_bytes = frame_samples * 2
        full_bytes = len(pcm) - (len(pcm) % frame_bytes)
        for offset in range(0, full_bytes, frame_bytes):
            segments.extend(self._segments(self.engine.process_frame(pcm[offset : offset + frame_bytes])))

        # A batch file has an explicit end. Add enough silence to exercise the
        # same endpoint rule as a live stream instead of dropping an open turn.
        silence = b"\x00" * frame_bytes
        end_frames = self.engine.config.end_silence_ms // self.engine.config.frame_ms
        for _ in range(end_frames):
            segments.extend(self._segments(self.engine.process_frame(silence)))
        return segments

    def _segments(self, events) -> list[VadSegment]:
        out = []
        sample_rate = self.engine.config.sample_rate
        for event in events:
            if event.kind not in {
                VadEventKind.SPEECH_ENDED,
                VadEventKind.MAX_DURATION_REACHED,
            }:
                continue
            if event.utterance_start_sample is None or event.utterance_end_sample is None:
                continue
            out.append(
                VadSegment(
                    event.utterance_start_sample / sample_rate,
                    event.utterance_end_sample / sample_rate,
                )
            )
        return out
