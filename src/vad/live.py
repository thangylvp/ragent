"""Live PCM sessions for comparing VAD classifiers behind one event contract.

These adapters are intentionally small and stateful: one instance owns one
microphone stream. They normalize optional classifiers to ``VadEvent`` so a
web test or future audio transport can swap detectors without changing its
utterance handling.
"""

from __future__ import annotations

import array
import importlib.util
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .base import VadEvent, VadEventKind, VadState
from .energy import EnergyVad


# Production-oriented defaults for the FireRed streaming acoustic model. Keep
# these identical between the ncnn and PyTorch runtimes so backend comparisons
# measure the runtime rather than different endpoint policies.
FIRERED_STREAM_THRESHOLD = 0.65
FIRERED_STREAM_SMOOTH_FRAMES = 5
FIRERED_STREAM_PAD_START_FRAMES = 8
FIRERED_STREAM_MIN_SPEECH_FRAMES = 15
FIRERED_STREAM_MAX_SPEECH_FRAMES = 2_000
FIRERED_STREAM_MIN_SILENCE_FRAMES = 30


@dataclass(frozen=True, slots=True)
class LiveVadUpdate:
    state: VadState
    confidence: float | None
    is_speech: bool | None
    events: tuple[VadEvent, ...] = ()


class LiveVadSession(Protocol):
    name: str
    sample_rate: int
    frame_samples: int

    def process_frame(self, pcm16le: bytes) -> LiveVadUpdate: ...

    def reset(self) -> None: ...


class EnergyLiveVad:
    name = "energy"

    def __init__(self):
        self.engine = EnergyVad()
        self.sample_rate = self.engine.config.sample_rate
        self.frame_samples = self.engine.config.frame_samples

    def process_frame(self, pcm16le: bytes) -> LiveVadUpdate:
        events = tuple(self.engine.process_frame(pcm16le))
        return LiveVadUpdate(
            state=self.engine.state,
            confidence=None,
            is_speech=self.engine.state is VadState.SPEECH,
            events=events,
        )

    def reset(self) -> None:
        self.engine.reset()


class _EventModel(Protocol):
    def process(self, pcm16le: bytes): ...

    def reset(self) -> None: ...


class EventModelLiveVad:
    """Buffer audio around FireRed-compatible start/end frame events."""

    sample_rate = 16_000
    frame_samples = 160

    def __init__(
        self,
        name: str,
        model: _EventModel,
        *,
        history_frames: int = 2_300,
    ):
        self.name = name
        self.model = model
        self._history: deque[tuple[int, bytes]] = deque(maxlen=history_frames)
        self._input_frame = 0
        self._active_start_frame: int | None = None

    def process_frame(self, pcm16le: bytes) -> LiveVadUpdate:
        self._validate_frame(pcm16le)
        self._input_frame += 1
        self._history.append((self._input_frame, bytes(pcm16le)))
        result = self.model.process(pcm16le)
        if result is None:
            return LiveVadUpdate(self._state, None, None)

        events = []
        if bool(result.is_speech_start):
            self._active_start_frame = max(1, int(result.speech_start_frame))
            events.append(
                VadEvent(
                    kind=VadEventKind.SPEECH_STARTED,
                    state=VadState.SPEECH,
                    sample_index=self._input_frame * self.frame_samples,
                    utterance_start_sample=(self._active_start_frame - 1)
                    * self.frame_samples,
                )
            )

        if bool(result.is_speech_end) and self._active_start_frame is not None:
            end_frame = min(self._input_frame, int(result.speech_end_frame))
            start_frame = self._active_start_frame
            selected = [
                frame
                for index, frame in self._history
                if start_frame <= index <= end_frame
            ]
            start_sample = (start_frame - 1) * self.frame_samples
            end_sample = end_frame * self.frame_samples
            duration_ms = max(0, end_sample - start_sample) * 1000 // self.sample_rate
            events.append(
                VadEvent(
                    kind=VadEventKind.SPEECH_ENDED,
                    state=VadState.SPEECH,
                    sample_index=self._input_frame * self.frame_samples,
                    utterance_start_sample=start_sample,
                    utterance_end_sample=end_sample,
                    audio_pcm16le=b"".join(selected),
                    reason="model_end_silence",
                    speech_ms=duration_ms,
                    buffered_ms=duration_ms,
                )
            )
            self._active_start_frame = None

        confidence = getattr(result, "smoothed_prob", None)
        if confidence is None:
            confidence = getattr(result, "confidence", None)
        return LiveVadUpdate(
            state=self._state,
            confidence=float(confidence) if confidence is not None else None,
            is_speech=bool(result.is_speech),
            events=tuple(events),
        )

    @property
    def _state(self) -> VadState:
        return VadState.SPEECH if self._active_start_frame is not None else VadState.IDLE

    def reset(self) -> None:
        self.model.reset()
        self._history.clear()
        self._input_frame = 0
        self._active_start_frame = None

    def _validate_frame(self, pcm16le: bytes) -> None:
        expected = self.frame_samples * 2
        if len(pcm16le) != expected:
            raise ValueError(f"expected {expected} PCM16LE bytes, got {len(pcm16le)}")


class _BinaryLiveVad:
    """Endpoint a per-frame speech probability while retaining PCM audio."""

    sample_rate = 16_000

    def __init__(
        self,
        name: str,
        frame_samples: int,
        classify: Callable[[bytes], tuple[bool, float | None]],
        reset_classifier: Callable[[], None],
        *,
        start_ms: int = 80,
        end_ms: int = 200,
        pre_roll_ms: int = 50,
        trailing_ms: int = 50,
        max_speech_ms: int = 20_000,
    ):
        self.name = name
        self.frame_samples = frame_samples
        self._classify = classify
        self._reset_classifier = reset_classifier
        frame_ms = frame_samples * 1000 / self.sample_rate
        self._start_frames = max(1, math.ceil(start_ms / frame_ms))
        self._end_frames = max(1, math.ceil(end_ms / frame_ms))
        self._pre_roll_frames = max(0, math.ceil(pre_roll_ms / frame_ms))
        self._trailing_frames = max(0, math.ceil(trailing_ms / frame_ms))
        self._max_frames = max(1, math.ceil(max_speech_ms / frame_ms))
        self._history: deque[tuple[int, bytes]] = deque(
            maxlen=self._max_frames + self._pre_roll_frames + self._end_frames + 8
        )
        self._frame_index = 0
        self._candidate_frames = 0
        self._silence_frames = 0
        self._active_start_frame: int | None = None

    def process_frame(self, pcm16le: bytes) -> LiveVadUpdate:
        expected = self.frame_samples * 2
        if len(pcm16le) != expected:
            raise ValueError(f"expected {expected} PCM16LE bytes, got {len(pcm16le)}")
        self._frame_index += 1
        self._history.append((self._frame_index, bytes(pcm16le)))
        is_speech, confidence = self._classify(pcm16le)
        events = []

        if self._active_start_frame is None:
            self._candidate_frames = self._candidate_frames + 1 if is_speech else 0
            if self._candidate_frames >= self._start_frames:
                self._active_start_frame = max(
                    1,
                    self._frame_index
                    - self._candidate_frames
                    + 1
                    - self._pre_roll_frames,
                )
                self._silence_frames = 0
                events.append(
                    VadEvent(
                        kind=VadEventKind.SPEECH_STARTED,
                        state=VadState.SPEECH,
                        sample_index=self._frame_index * self.frame_samples,
                        utterance_start_sample=(self._active_start_frame - 1)
                        * self.frame_samples,
                    )
                )
        else:
            self._silence_frames = 0 if is_speech else self._silence_frames + 1
            active_frames = self._frame_index - self._active_start_frame + 1
            if self._silence_frames >= self._end_frames:
                end_frame = min(
                    self._frame_index,
                    self._frame_index - self._end_frames + self._trailing_frames,
                )
                events.append(self._finish(end_frame, VadEventKind.SPEECH_ENDED, "end_silence"))
            elif active_frames >= self._max_frames:
                events.append(
                    self._finish(
                        self._frame_index,
                        VadEventKind.MAX_DURATION_REACHED,
                        "max_duration",
                    )
                )

        return LiveVadUpdate(
            state=(
                VadState.SPEECH
                if self._active_start_frame is not None
                else VadState.IDLE
            ),
            confidence=confidence,
            is_speech=is_speech,
            events=tuple(events),
        )

    def _finish(
        self,
        end_frame: int,
        kind: VadEventKind,
        reason: str,
    ) -> VadEvent:
        assert self._active_start_frame is not None
        start_frame = self._active_start_frame
        selected = [
            frame
            for index, frame in self._history
            if start_frame <= index <= end_frame
        ]
        start_sample = (start_frame - 1) * self.frame_samples
        end_sample = end_frame * self.frame_samples
        duration_ms = (end_sample - start_sample) * 1000 // self.sample_rate
        event = VadEvent(
            kind=kind,
            state=VadState.SPEECH,
            sample_index=self._frame_index * self.frame_samples,
            utterance_start_sample=start_sample,
            utterance_end_sample=end_sample,
            audio_pcm16le=b"".join(selected),
            reason=reason,
            speech_ms=duration_ms,
            buffered_ms=duration_ms,
        )
        self._active_start_frame = None
        self._candidate_frames = 0
        self._silence_frames = 0
        return event

    def reset(self) -> None:
        self._reset_classifier()
        self._history.clear()
        self._frame_index = 0
        self._candidate_frames = 0
        self._silence_frames = 0
        self._active_start_frame = None


class _OmniModel:
    def __init__(self):
        import numpy as np
        from omnivad import OmniStreamVAD

        self.np = np
        self.model = OmniStreamVAD(
            threshold=FIRERED_STREAM_THRESHOLD,
            smooth_window_size=FIRERED_STREAM_SMOOTH_FRAMES,
            pad_start_frame=FIRERED_STREAM_PAD_START_FRAMES,
            min_speech_frame=FIRERED_STREAM_MIN_SPEECH_FRAMES,
            max_speech_frame=FIRERED_STREAM_MAX_SPEECH_FRAMES,
            min_silence_frame=FIRERED_STREAM_MIN_SILENCE_FRAMES,
        )

    def process(self, pcm16le: bytes):
        return self.model.process(self.np.frombuffer(pcm16le, dtype="<i2"))

    def reset(self) -> None:
        self.model.reset()


class _FireRedModel:
    def __init__(self, model_dir: str | Path):
        import numpy as np
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        self.np = np
        self._window = bytearray()
        self.model = FireRedStreamVad.from_pretrained(
            str(model_dir),
            FireRedStreamVadConfig(
                use_gpu=False,
                speech_threshold=FIRERED_STREAM_THRESHOLD,
                smooth_window_size=FIRERED_STREAM_SMOOTH_FRAMES,
                pad_start_frame=FIRERED_STREAM_PAD_START_FRAMES,
                min_speech_frame=FIRERED_STREAM_MIN_SPEECH_FRAMES,
                max_speech_frame=FIRERED_STREAM_MAX_SPEECH_FRAMES,
                min_silence_frame=FIRERED_STREAM_MIN_SILENCE_FRAMES,
            ),
        )

    def process(self, pcm16le: bytes):
        # Upstream FireRed consumes overlapping 25 ms windows on a 10 ms hop.
        self._window.extend(pcm16le)
        frame_bytes = 400 * 2
        if len(self._window) < frame_bytes:
            return None
        frame = bytes(self._window[:frame_bytes])
        del self._window[: 160 * 2]
        return self.model.detect_frame(self.np.frombuffer(frame, dtype="<i2"))

    def reset(self) -> None:
        self._window.clear()
        self.model.reset()


def _create_webrtc() -> LiveVadSession:
    import webrtcvad

    model = webrtcvad.Vad(2)

    def classify(frame: bytes) -> tuple[bool, float | None]:
        value = bool(model.is_speech(frame, 16_000))
        return value, 1.0 if value else 0.0

    return _BinaryLiveVad("webrtc", 320, classify, lambda: None)


def _create_silero() -> LiveVadSession:
    import numpy as np
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=False)

    def classify(frame: bytes) -> tuple[bool, float | None]:
        samples = np.frombuffer(frame, dtype="<i2").astype("float32") / 32768.0
        confidence = float(model(torch.from_numpy(samples), 16_000).item())
        return confidence >= 0.5, confidence

    return _BinaryLiveVad("silero", 512, classify, model.reset_states)


def live_vad_catalog(firered_model_dir: str | Path | None = None) -> list[dict]:
    specs = [
        ("energy", "Adaptive energy baseline", True, None),
        (
            "omnivad",
            "FireRed Stream-VAD on ncnn",
            importlib.util.find_spec("omnivad") is not None,
            "install omnivad",
        ),
        (
            "firered",
            "FireRed Stream-VAD on PyTorch CPU",
            importlib.util.find_spec("fireredvad") is not None
            and bool(firered_model_dir)
            and Path(firered_model_dir).is_dir(),
            "install fireredvad and configure WEBTEST_FIRERED_MODEL_DIR",
        ),
        (
            "silero",
            "Silero VAD v6 on PyTorch CPU",
            importlib.util.find_spec("silero_vad") is not None,
            "install silero-vad",
        ),
        (
            "webrtc",
            "WebRTC GMM VAD, mode 2",
            importlib.util.find_spec("webrtcvad") is not None,
            "install webrtcvad-wheels",
        ),
    ]
    return [
        {
            "name": name,
            "description": description,
            "available": available,
            "unavailable_reason": None if available else reason,
        }
        for name, description, available, reason in specs
    ]


def create_live_vad(
    name: str,
    *,
    firered_model_dir: str | Path | None = None,
) -> LiveVadSession:
    normalized = name.strip().lower().replace("-", "")
    if normalized == "energy":
        return EnergyLiveVad()
    if normalized in {"omnivad", "omni"}:
        return EventModelLiveVad("omnivad", _OmniModel())
    if normalized in {"firered", "fireredvad"}:
        if not firered_model_dir:
            raise ValueError("FireRed requires a Stream-VAD model directory")
        return EventModelLiveVad("firered", _FireRedModel(firered_model_dir))
    if normalized == "silero":
        return _create_silero()
    if normalized in {"webrtc", "webrtcvad"}:
        return _create_webrtc()
    raise ValueError(f"unknown live VAD backend: {name!r}")


def pcm_dbfs(pcm16le: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm16le)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return -120.0
    square_mean = sum(float(value) * value for value in samples) / len(samples)
    if square_mean <= 0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(math.sqrt(square_mean) / 32768.0))
