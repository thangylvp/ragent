"""Common segment contract and lazy factory for independently installed VADs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VadSegment:
    start_sec: float
    end_sec: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError(f"invalid VAD segment: {self.start_sec}..{self.end_sec}")


class VadBackend(Protocol):
    name: str

    @property
    def info(self) -> dict: ...

    def detect(self, wav_path: str | Path) -> list[VadSegment]: ...


def distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def create_backend(name: str, *, model_dir: str | None = None) -> VadBackend:
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "energy":
        from .energy import EnergyBackend

        return EnergyBackend()
    if normalized in {"firered", "fireredvad"}:
        if not model_dir:
            raise ValueError("FireRedVAD requires --model-dir pointing to Stream-VAD")
        from .firered import FireRedBackend

        return FireRedBackend(model_dir)
    if normalized in {"omnivad", "omnivad_kit"}:
        from .omnivad import OmniVadBackend

        return OmniVadBackend()
    if normalized == "silero":
        from .silero import SileroBackend

        return SileroBackend()
    if normalized in {"webrtc", "webrtcvad"}:
        from .webrtc import WebRtcBackend

        return WebRtcBackend()
    raise ValueError(
        f"unknown VAD backend {name!r}; expected energy, firered, omnivad, silero, or webrtc"
    )
