"""Dependency-free adaptive-energy VAD baseline.

This is intentionally a baseline, not a claim that energy alone will solve
far-field robot audio. It owns buffering and endpointing semantics behind the
same interface that a future neural speech classifier can implement.
"""

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass

from .base import VadEvent, VadEventKind, VadState


@dataclass(frozen=True, slots=True)
class EnergyVadConfig:
    """Versionable energy-VAD and endpointing parameters."""

    sample_rate: int = 16_000
    frame_ms: int = 20
    pre_roll_ms: int = 300
    start_confirm_ms: int = 100
    min_speech_ms: int = 200
    end_silence_ms: int = 600
    trailing_silence_ms: int = 200
    max_utterance_ms: int = 15_000
    initial_noise_dbfs: float = -60.0
    minimum_start_dbfs: float = -42.0
    minimum_continue_dbfs: float = -48.0
    start_margin_db: float = 12.0
    continue_margin_db: float = 6.0
    noise_ema: float = 0.95

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive")
        if self.sample_rate * self.frame_ms % 1000:
            raise ValueError("frame_ms must represent a whole number of samples")
        duration_fields = {
            "pre_roll_ms": self.pre_roll_ms,
            "start_confirm_ms": self.start_confirm_ms,
            "min_speech_ms": self.min_speech_ms,
            "end_silence_ms": self.end_silence_ms,
            "trailing_silence_ms": self.trailing_silence_ms,
            "max_utterance_ms": self.max_utterance_ms,
        }
        for name, value in duration_fields.items():
            if value < 0 or value % self.frame_ms:
                raise ValueError(f"{name} must be non-negative and divisible by frame_ms")
        if self.start_confirm_ms == 0 or self.end_silence_ms == 0:
            raise ValueError("start_confirm_ms and end_silence_ms must be positive")
        if self.pre_roll_ms < self.start_confirm_ms:
            raise ValueError("pre_roll_ms must be at least start_confirm_ms")
        if self.min_speech_ms < self.start_confirm_ms:
            raise ValueError("min_speech_ms must be at least start_confirm_ms")
        if self.trailing_silence_ms > self.end_silence_ms:
            raise ValueError("trailing_silence_ms cannot exceed end_silence_ms")
        if self.max_utterance_ms < self.min_speech_ms:
            raise ValueError("max_utterance_ms must be at least min_speech_ms")
        if not 0.0 <= self.noise_ema < 1.0:
            raise ValueError("noise_ema must be in [0, 1)")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


class EnergyVad:
    """Streaming adaptive-energy detector with endpoint buffering."""

    def __init__(self, config: EnergyVadConfig | None = None):
        self.config = config or EnergyVadConfig()
        self._pre_roll_frames = self._frames(self.config.pre_roll_ms)
        self._start_frames = self._frames(self.config.start_confirm_ms)
        self._min_speech_frames = self._frames(self.config.min_speech_ms)
        self._end_frames = self._frames(self.config.end_silence_ms)
        self._trailing_frames = self._frames(self.config.trailing_silence_ms)
        self._max_frames = self._frames(self.config.max_utterance_ms)
        self._enabled = True
        self._total_samples = 0
        self._noise_floor_dbfs = self.config.initial_noise_dbfs
        self._state = VadState.IDLE
        self._pre_roll: deque[bytes] = deque(maxlen=self._pre_roll_frames or 1)
        self._candidate_frames = 0
        self._segment: list[bytes] = []
        self._segment_start_sample: int | None = None
        self._active_frames = 0
        self._voiced_frames = 0
        self._silence_frames = 0
        self._cooldown_silence_frames = 0

    @property
    def state(self) -> VadState:
        return self._state

    @property
    def noise_floor_dbfs(self) -> float:
        return self._noise_floor_dbfs

    @property
    def listening(self) -> bool:
        return self._enabled

    def process_frame(self, pcm16le: bytes) -> list[VadEvent]:
        """Consume exactly one configured PCM16LE frame."""

        frame = bytes(pcm16le)
        expected_bytes = self.config.frame_samples * 2
        if len(frame) != expected_bytes:
            raise ValueError(
                f"expected {expected_bytes} PCM16LE bytes per frame, got {len(frame)}"
            )

        self._total_samples += self.config.frame_samples
        if not self._enabled:
            return []

        level_dbfs = self._dbfs(frame)
        start_threshold = max(
            self.config.minimum_start_dbfs,
            self._noise_floor_dbfs + self.config.start_margin_db,
        )
        continue_threshold = max(
            self.config.minimum_continue_dbfs,
            self._noise_floor_dbfs + self.config.continue_margin_db,
        )

        if self._state is VadState.COOLDOWN:
            if level_dbfs < continue_threshold:
                self._cooldown_silence_frames += 1
                self._remember(frame)
            else:
                self._cooldown_silence_frames = 0
                self._pre_roll.clear()
            if self._cooldown_silence_frames >= self._end_frames:
                self._state = VadState.IDLE
                self._candidate_frames = 0
                self._cooldown_silence_frames = 0
            return []

        if self._state is VadState.IDLE:
            self._remember(frame)
            if level_dbfs >= start_threshold:
                self._candidate_frames += 1
            else:
                self._candidate_frames = 0
                self._adapt_noise_floor(level_dbfs)

            if self._candidate_frames < self._start_frames:
                return []

            self._state = VadState.SPEECH
            self._segment = list(self._pre_roll)
            self._segment_start_sample = self._total_samples - (
                len(self._segment) * self.config.frame_samples
            )
            self._active_frames = self._candidate_frames
            self._voiced_frames = self._candidate_frames
            self._silence_frames = 0
            return [
                self._event(
                    VadEventKind.SPEECH_STARTED,
                    sample_index=self._total_samples,
                    level_dbfs=level_dbfs,
                )
            ]

        self._segment.append(frame)
        self._active_frames += 1
        if level_dbfs >= continue_threshold:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._active_frames >= self._max_frames:
            kind = (
                VadEventKind.MAX_DURATION_REACHED
                if self._voiced_frames >= self._min_speech_frames
                else VadEventKind.SEGMENT_REJECTED
            )
            event = self._finalize(
                kind,
                reason=(
                    "max_duration"
                    if kind is VadEventKind.MAX_DURATION_REACHED
                    else "too_short_at_max_duration"
                ),
                level_dbfs=level_dbfs,
                enter_cooldown=True,
            )
            return [event]

        if self._silence_frames >= self._end_frames:
            kind = (
                VadEventKind.SPEECH_ENDED
                if self._voiced_frames >= self._min_speech_frames
                else VadEventKind.SEGMENT_REJECTED
            )
            reason = "end_silence" if kind is VadEventKind.SPEECH_ENDED else "too_short"
            return [self._finalize(kind, reason=reason, level_dbfs=level_dbfs)]

        return []

    def set_listening(self, enabled: bool) -> list[VadEvent]:
        """Open or close the half-duplex input gate.

        Closing the gate rejects any partial segment so robot TTS can never
        turn into an executable model request.
        """

        if enabled == self._enabled:
            return []
        self._enabled = enabled
        events: list[VadEvent] = []
        if not enabled and self._state is VadState.SPEECH:
            events.append(
                self._finalize(
                    VadEventKind.SEGMENT_REJECTED,
                    reason="listening_gate_closed",
                    level_dbfs=-120.0,
                )
            )
        self._clear_endpoint_state()
        self._noise_floor_dbfs = self.config.initial_noise_dbfs
        return events

    def reset(self) -> None:
        self._enabled = True
        self._total_samples = 0
        self._noise_floor_dbfs = self.config.initial_noise_dbfs
        self._clear_endpoint_state()

    def _frames(self, duration_ms: int) -> int:
        return duration_ms // self.config.frame_ms

    def _remember(self, frame: bytes) -> None:
        if self._pre_roll_frames:
            self._pre_roll.append(frame)

    def _adapt_noise_floor(self, level_dbfs: float) -> None:
        alpha = self.config.noise_ema
        self._noise_floor_dbfs = (
            alpha * self._noise_floor_dbfs + (1.0 - alpha) * level_dbfs
        )

    def _finalize(
        self,
        kind: VadEventKind,
        *,
        reason: str,
        level_dbfs: float,
        enter_cooldown: bool = False,
    ) -> VadEvent:
        removed_frames = 0
        if reason in {"end_silence", "too_short"}:
            removed_frames = max(0, self._silence_frames - self._trailing_frames)
        retained = self._segment[:-removed_frames] if removed_frames else self._segment
        end_sample = self._total_samples - removed_frames * self.config.frame_samples
        event = self._event(
            kind,
            sample_index=self._total_samples,
            end_sample=end_sample,
            audio=b"".join(retained),
            reason=reason,
            level_dbfs=level_dbfs,
        )
        quiet_tail_count = min(self._silence_frames, self._pre_roll_frames)
        quiet_tail = self._segment[-quiet_tail_count:] if quiet_tail_count else []
        self._clear_endpoint_state()
        if enter_cooldown:
            self._state = VadState.COOLDOWN
        else:
            for frame in quiet_tail:
                self._remember(frame)
        return event

    def _event(
        self,
        kind: VadEventKind,
        *,
        sample_index: int,
        level_dbfs: float,
        end_sample: int | None = None,
        audio: bytes | None = None,
        reason: str | None = None,
    ) -> VadEvent:
        buffered_ms = 0
        if audio is not None:
            buffered_ms = len(audio) // 2 * 1000 // self.config.sample_rate
        return VadEvent(
            kind=kind,
            state=self._state,
            sample_index=sample_index,
            utterance_start_sample=self._segment_start_sample,
            utterance_end_sample=end_sample,
            audio_pcm16le=audio,
            reason=reason,
            speech_ms=self._voiced_frames * self.config.frame_ms,
            buffered_ms=buffered_ms,
            level_dbfs=level_dbfs,
            noise_floor_dbfs=self._noise_floor_dbfs,
        )

    def _clear_endpoint_state(self) -> None:
        self._state = VadState.IDLE
        self._pre_roll.clear()
        self._candidate_frames = 0
        self._segment = []
        self._segment_start_sample = None
        self._active_frames = 0
        self._voiced_frames = 0
        self._silence_frames = 0
        self._cooldown_silence_frames = 0

    @staticmethod
    def _dbfs(frame: bytes) -> float:
        sample_count = len(frame) // 2
        square_sum = sum(sample * sample for (sample,) in struct.iter_unpack("<h", frame))
        if not square_sum:
            return -120.0
        rms = math.sqrt(square_sum / sample_count)
        return max(-120.0, 20.0 * math.log10(rms / 32768.0))
