"""Public VAD types shared by the harness, robot adapters and web demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class VadState(str, Enum):
    """Streaming endpoint state."""

    IDLE = "idle"
    SPEECH = "speech"
    COOLDOWN = "cooldown"


class VadEventKind(str, Enum):
    """Events emitted at meaningful utterance boundaries."""

    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SEGMENT_REJECTED = "segment_rejected"
    MAX_DURATION_REACHED = "max_duration_reached"


@dataclass(frozen=True, slots=True)
class VadEvent:
    """One timestamped VAD transition or finalized segment.

    Sample positions refer to the input stream and therefore remain stable
    when processing is faster or slower than real time.
    """

    kind: VadEventKind
    state: VadState
    sample_index: int
    utterance_start_sample: int | None = None
    utterance_end_sample: int | None = None
    audio_pcm16le: bytes | None = None
    reason: str | None = None
    speech_ms: int = 0
    buffered_ms: int = 0
    level_dbfs: float = -120.0
    noise_floor_dbfs: float = -60.0


class VadEngine(Protocol):
    """Stable streaming interface; classifiers may change behind it."""

    @property
    def state(self) -> VadState: ...

    def process_frame(self, pcm16le: bytes) -> list[VadEvent]: ...

    def set_listening(self, enabled: bool) -> list[VadEvent]: ...

    def reset(self) -> None: ...
