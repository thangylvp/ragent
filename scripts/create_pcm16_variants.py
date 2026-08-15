#!/usr/bin/env python3
"""Create acoustically equivalent PCM16 WAV variants for cache-safe benchmarks."""

from __future__ import annotations

import argparse
import array
import sys
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", type=int, required=True)
    return parser.parse_args()


def create_variant(source: Path, destination: Path, offset: int) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (
        1,
        2,
        16_000,
        "NONE",
    ):
        raise ValueError(f"expected mono 16 kHz PCM16 WAV: {source}")
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError(f"empty WAV: {source}")
    index = max(0, len(samples) - 32)
    value = int(samples[index])
    samples[index] = max(-32768, min(32767, value + offset))
    if sys.byteorder != "little":
        samples.byteswap()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(samples.tobytes())


def main() -> int:
    args = parse_args()
    if args.variants < 1:
        raise ValueError("--variants must be positive")
    for source in args.wav:
        if not source.is_file():
            raise FileNotFoundError(source)
    for variant in range(1, args.variants + 1):
        for source in args.wav:
            create_variant(
                source,
                args.output_root / f"r{variant}" / source.name,
                variant,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
