"""WebRTC VAD with a deterministic frame collector for segment output."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from .audio import read_pcm16_mono
from .base import VadSegment, distribution_version


class WebRtcBackend:
    name = "webrtc"

    def __init__(self, *, mode: int = 2, frame_ms: int = 30, padding_ms: int = 300):
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("install `webrtcvad-wheels`") from exc
        self._vad = webrtcvad.Vad(mode)
        self.mode = mode
        self.frame_ms = frame_ms
        self.padding_ms = padding_ms

    @property
    def info(self) -> dict:
        return {
            "runtime": "native_cpu",
            "model": "webrtc_vad",
            "package_version": distribution_version("webrtcvad-wheels"),
            "aggressiveness": self.mode,
        }

    def detect(self, wav_path: str | Path) -> list[VadSegment]:
        samples, sample_rate = read_pcm16_mono(wav_path)
        if sample_rate not in {8_000, 16_000, 32_000, 48_000}:
            raise ValueError(f"WebRTC VAD does not support {sample_rate} Hz")
        frame_samples = sample_rate * self.frame_ms // 1000
        frame_bytes = frame_samples * 2
        raw = samples.tobytes()
        frames = [
            raw[offset : offset + frame_bytes]
            for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes)
        ]
        flags = [self._vad.is_speech(frame, sample_rate) for frame in frames]
        return self._collect(flags)

    def _collect(self, flags: list[bool]) -> list[VadSegment]:
        padding_frames = self.padding_ms // self.frame_ms
        ring: deque[tuple[int, bool]] = deque(maxlen=padding_frames)
        triggered = False
        start_frame = 0
        segments: list[VadSegment] = []
        for index, is_speech in enumerate(flags):
            ring.append((index, is_speech))
            voiced = sum(flag for _, flag in ring)
            if not triggered and len(ring) == padding_frames and voiced > 0.9 * len(ring):
                triggered = True
                start_frame = ring[0][0]
                ring.clear()
            elif triggered and len(ring) == padding_frames:
                unvoiced = len(ring) - voiced
                if unvoiced > 0.9 * len(ring):
                    end_frame = ring[-1][0] + 1
                    segments.append(self._segment(start_frame, end_frame))
                    triggered = False
                    ring.clear()
        if triggered:
            segments.append(self._segment(start_frame, len(flags)))
        return segments

    def _segment(self, start_frame: int, end_frame: int) -> VadSegment:
        scale = self.frame_ms / 1000.0
        return VadSegment(start_frame * scale, end_frame * scale)
