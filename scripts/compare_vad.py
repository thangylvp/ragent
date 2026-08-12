#!/usr/bin/env python3
"""Render comparable VAD JSON reports as a compact Markdown table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _speech_mask(row: dict, frame_sec: float = 0.01) -> bytearray:
    frame_count = max(1, math.ceil(row["duration_sec"] / frame_sec))
    mask = bytearray(frame_count)
    for segment in row["segments"]:
        start = min(
            frame_count,
            max(0, math.floor(segment["start_sec"] / frame_sec)),
        )
        end = min(frame_count, math.ceil(segment["end_sec"] / frame_sec))
        mask[start:end] = b"\x01" * max(0, end - start)
    return mask


def _firered_omni_agreement(reports: list[dict]) -> list[str]:
    by_name = {report["backend"]: report for report in reports}
    if "firered" not in by_name or "omnivad" not in by_name:
        return []

    firered = {row["path"]: row for row in by_name["firered"]["files"]}
    omnivad = {row["path"]: row for row in by_name["omnivad"]["files"]}
    common = sorted(firered.keys() & omnivad.keys())
    if not common:
        return []

    activation_matches = 0
    segment_count_matches = 0
    mask_ious = []
    detected_deltas_ms = []
    for path in common:
        first = firered[path]
        second = omnivad[path]
        activation_matches += bool(first["segments"]) == bool(second["segments"])
        segment_count_matches += len(first["segments"]) == len(second["segments"])
        first_mask = _speech_mask(first)
        second_mask = _speech_mask(second)
        intersection = sum(a and b for a, b in zip(first_mask, second_mask))
        union = sum(a or b for a, b in zip(first_mask, second_mask))
        mask_ious.append(intersection / union if union else 1.0)
        detected_deltas_ms.append(
            abs(first["detected_sec"] - second["detected_sec"]) * 1000
        )

    return [
        "",
        "## FireRed / OmniVAD edge-port agreement",
        "",
        f"Compared on {len(common)} common files at a 10 ms frame grid "
        "(no ground-truth boundaries):",
        "",
        f"- activation agreement: {100 * activation_matches / len(common):.2f}%",
        "- exact segment-count agreement: "
        f"{100 * segment_count_matches / len(common):.2f}%",
        "- mean speech-mask intersection-over-union: "
        f"{100 * sum(mask_ious) / len(mask_ious):.2f}%",
        "- mean absolute detected-duration difference: "
        f"{sum(detected_deltas_ms) / len(detected_deltas_ms):.1f} ms/file",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    backend_names = [report["backend"] for report in reports]
    duplicates = sorted(
        {name for name in backend_names if backend_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(f"pass exactly one report per backend; duplicates: {duplicates}")
    categories = sorted({name for report in reports for name in report["summary"]})

    lines = [
        "# VAD component benchmark",
        "",
        "All current categories contain user speech. Activation is desirable; false-alarm "
        "evaluation requires a separately labeled negative corpus.",
        "",
        "| Backend | Runtime | Load ms | Realtime × | Peak RSS MB |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for report in reports:
        runtime = report["runtime"]
        lines.append(
            f"| {report['backend']} | {report['backend_info']['runtime']} | "
            f"{runtime['load_ms']:.1f} | {runtime['throughput_x_realtime']:.1f} | "
            f"{runtime['process_peak_rss_mb']:.1f} |"
        )

    lines.extend(
        [
            "",
            "| Category | " + " | ".join(r["backend"] for r in reports) + " |",
            "| --- | " + " | ".join("---:" for _ in reports) + " |",
        ]
    )
    for category in categories:
        values = []
        for report in reports:
            item = report["summary"].get(category)
            values.append(f"{100 * item['activation_rate']:.1f}%" if item else "—")
        lines.append(f"| {category} activation | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "| Category | " + " | ".join(r["backend"] for r in reports) + " |",
            "| --- | " + " | ".join("---:" for _ in reports) + " |",
        ]
    )
    for category in categories:
        values = []
        for report in reports:
            item = report["summary"].get(category)
            values.append(f"{100 * item['detected_audio_ratio']:.1f}%" if item else "—")
        lines.append(f"| {category} detected audio | " + " | ".join(values) + " |")

    lines.extend(_firered_omni_agreement(reports))

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
