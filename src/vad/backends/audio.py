"""Small standard-library WAV helpers shared by VAD evaluation adapters."""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path


def read_pcm16_mono(wav_path: str | Path) -> tuple[array.array, int]:
    path = Path(wav_path)
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(
            f"{path} must be mono PCM16 WAV, got channels={channels}, sample_width={sample_width}"
        )
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples, sample_rate


def duration_sec(wav_path: str | Path) -> float:
    with wave.open(str(wav_path), "rb") as reader:
        return reader.getnframes() / reader.getframerate()
