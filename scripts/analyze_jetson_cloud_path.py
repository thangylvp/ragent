#!/usr/bin/env python3
"""Aggregate cloud-path JSONL and Jetson tegrastats into report-ready JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_jetson_full_system import distribution, parse_tegrastats


LATENCY_METRICS = (
    "speech_end_to_first_slm_token_ms",
    "speech_end_to_last_slm_token_ms",
    "speech_end_to_cloud_llm_ms",
    "speech_end_to_first_audio_byte_ms",
    "audio_ready_to_first_slm_token_ms",
    "audio_ready_to_last_slm_token_ms",
    "audio_ready_to_cloud_llm_ms",
    "audio_ready_to_first_audio_byte_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--conditions", nargs="+", type=int, default=[100, 70, 50, 30])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dist(rows: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float]) -> dict[str, float] | None:
    values = [float(getter(row)) for row in rows]
    return distribution(values) if values else None


def main() -> int:
    args = parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    output: dict[str, Any] = {"schema_version": 1, "corpus": corpus, "conditions": {}}
    for condition in args.conditions:
        directory = args.result_root / f"p{condition}"
        records = [json.loads(line) for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        measured = [row for row in records if row["measured"]]
        valid = [row for row in measured if row["valid_cloud_path"]]
        by_sample = {}
        for sample in corpus["samples"]:
            rows = [row for row in valid if row["sample"] == sample["file"]]
            by_sample[sample["file"]] = {
                "duration_seconds": sample["duration_seconds"],
                "runs": len(rows),
                "latency_ms": {metric: dist(rows, lambda row, key=metric: row[key]) for metric in LATENCY_METRICS},
                "cloud_ms": dist(rows, lambda row: row["cloud"]["latency_ms"]),
                "tts_ms": dist(rows, lambda row: row["tts"]["synthesis_ms"]),
                "tts_rtf": dist(rows, lambda row: row["tts"]["rtf"]),
            }
        components = {
            "vad_endpoint_ms": dist(valid, lambda row: row["speech_end_to_vad_finalized_ms"]),
            "slm_request_to_first_token_ms": dist(valid, lambda row: row["slm"]["timings"]["request_to_first_token_ms"]),
            "slm_first_to_last_token_ms": dist(valid, lambda row: row["slm"]["timings"]["first_to_last_token_ms"]),
            "model_adapter_after_last_token_ms": dist(valid, lambda row: row["slm"]["timings"]["adapter_total_ms"] - row["slm"]["timings"]["request_to_last_token_ms"]),
            "cloud_ms": dist(valid, lambda row: row["cloud"]["latency_ms"]),
            "tts_synthesis_ms": dist(valid, lambda row: row["tts"]["synthesis_ms"]),
            "tts_rtf": dist(valid, lambda row: row["tts"]["rtf"]),
            "response_ready_to_first_audio_byte_ms": dist(valid, lambda row: row["audio_ready_to_first_audio_byte_ms"] - row["audio_ready_to_response_ready_ms"]),
            "cloud_response_chars": dist(valid, lambda row: row["cloud"]["response_chars"]),
            "tts_output_duration_s": dist(valid, lambda row: row["tts"]["output_duration_s"]),
        }
        output["conditions"][str(condition)] = {
            "measured_runs": len(measured),
            "valid_cloud_runs": len(valid),
            "success_rate": round(len(valid) / len(measured), 6) if measured else 0,
            "latency_ms": {metric: dist(valid, lambda row, key=metric: row[key]) for metric in LATENCY_METRICS},
            "components": components,
            "by_sample": by_sample,
            "jetson_resources": parse_tegrastats(directory / "tegrastats.log"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
