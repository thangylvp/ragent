#!/usr/bin/env python3
"""Evaluate one VAD backend on the existing robot speech/background corpus.

Run each optional backend with the Python environment that owns its runtime.
Outputs use one JSON schema so reports can be compared without importing all
VAD packages into one environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vad.backends import create_backend  # noqa: E402
from vad.backends.audio import duration_sec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=["energy", "firered", "omnivad", "silero", "webrtc"],
    )
    parser.add_argument(
        "--corpus-root",
        default=str(REPO_ROOT.parent / "data_test_robot"),
    )
    parser.add_argument("--model-dir", help="FireRedVAD Stream-VAD directory")
    parser.add_argument("--output", help="JSON output; defaults under outputs/vad/benchmarks")
    parser.add_argument("--limit", type=int, help="debug: cap files after stable sorting")
    return parser.parse_args()


def category(path: Path, corpus_root: Path) -> tuple[str, str]:
    relative = path.relative_to(corpus_root)
    parts = relative.parts
    if parts[0] == "recording_2404_filter_16k":
        condition = "moving" if "robot_di_chuyen" in parts else "stationary"
        return f"speech_robot_{condition}", "speech"
    if parts[0] == "vmo_1703_filter_16k":
        return "speech_vmo", "speech"
    if parts[0] == "vmo_2305_16k" and len(parts) > 1:
        name = parts[1]
        normalized = "quiet" if name == "silence" else name
        return f"speech_noise_{normalized}", "speech"
    return "unclassified", "unknown"


def clipped_coverage(segments, duration: float) -> float:
    intervals = []
    for segment in sorted(segments, key=lambda item: item.start_sec):
        start = max(0.0, min(duration, segment.start_sec))
        end = max(start, min(duration, segment.end_sec))
        if not intervals or start > intervals[-1][1]:
            intervals.append([start, end])
        else:
            intervals[-1][1] = max(intervals[-1][1], end)
    return sum(end - start for start, end in intervals)


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    summary = {}
    for name, items in sorted(grouped.items()):
        duration = sum(item["duration_sec"] for item in items)
        detected = sum(item["detected_sec"] for item in items)
        latency = sum(item["latency_ms"] for item in items)
        summary[name] = {
            "label": items[0]["label"],
            "files": len(items),
            "audio_sec": round(duration, 3),
            "activated_files": sum(bool(item["segments"]) for item in items),
            "activation_rate": round(
                sum(bool(item["segments"]) for item in items) / len(items), 4
            ),
            "detected_audio_ratio": round(detected / duration, 4) if duration else 0.0,
            "mean_latency_ms": round(latency / len(items), 3),
            "rtf": round((latency / 1000.0) / duration, 6) if duration else 0.0,
        }
    return summary


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def main() -> None:
    args = parse_args()
    corpus_root = Path(args.corpus_root).resolve()
    paths = sorted(corpus_root.rglob("*.wav"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no WAV files found under {corpus_root}")

    load_started = time.perf_counter()
    backend = create_backend(args.backend, model_dir=args.model_dir)
    load_ms = (time.perf_counter() - load_started) * 1000

    rows = []
    started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        duration = duration_sec(path)
        file_started = time.perf_counter()
        segments = backend.detect(path)
        latency_ms = (time.perf_counter() - file_started) * 1000
        group, label = category(path, corpus_root)
        rows.append(
            {
                "path": str(path.relative_to(corpus_root)),
                "category": group,
                "label": label,
                "duration_sec": round(duration, 6),
                "detected_sec": round(clipped_coverage(segments, duration), 6),
                "latency_ms": round(latency_ms, 3),
                "segments": [
                    {
                        "start_sec": round(segment.start_sec, 4),
                        "end_sec": round(segment.end_sec, 4),
                        "confidence": segment.confidence,
                    }
                    for segment in segments
                ],
            }
        )
        if index % 25 == 0 or index == len(paths):
            print(f"[{backend.name}] {index}/{len(paths)}", file=sys.stderr)

    elapsed_sec = time.perf_counter() - started
    audio_sec = sum(item["duration_sec"] for item in rows)
    report = {
        "schema_version": 1,
        "backend": backend.name,
        "backend_info": backend.info,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "load_ms": round(load_ms, 3),
            "wall_sec": round(elapsed_sec, 3),
            "audio_sec": round(audio_sec, 3),
            "throughput_x_realtime": round(audio_sec / elapsed_sec, 3),
            "process_peak_rss_mb": round(peak_rss_mb(), 3),
        },
        "corpus": {
            "root": str(corpus_root),
            "files": len(rows),
            "note": (
                "All current files contain user speech, including vmo_2305 files grouped by "
                "background condition. No false-alarm or boundary-error metric is reported "
                "without manually labeled continuous positive and negative recordings."
            ),
        },
        "summary": summarize(rows),
        "files": rows,
    }

    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "outputs" / "vad" / "benchmarks" / f"{backend.name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    print(output)


if __name__ == "__main__":
    main()
