#!/usr/bin/env python3
"""Benchmark non_tool through Gemini and uncached laptop OmniVoice TTS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.benchmark_jetson_full_system import (  # noqa: E402
    distribution,
    run_once,
    selected_variant,
)


METRICS = (
    "speech_end_to_vad_finalized_ms",
    "speech_end_to_first_slm_token_ms",
    "speech_end_to_last_slm_token_ms",
    "speech_end_to_cloud_llm_ms",
    "speech_end_to_first_audio_byte_ms",
    "audio_ready_to_first_slm_token_ms",
    "audio_ready_to_last_slm_token_ms",
    "audio_ready_to_cloud_llm_ms",
    "audio_ready_to_response_ready_ms",
    "audio_ready_to_first_audio_byte_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--backend", default="omnivad")
    parser.add_argument("--enhancer", default="fastenhancer_s")
    parser.add_argument("--prefix-ms", type=int, default=500)
    parser.add_argument("--suffix-ms", type=int, default=1000)
    parser.add_argument("--packet-ms", type=int, default=20)
    parser.add_argument("--gpu-limit", type=int, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    response_timing = record["harness"]["timings"]
    endpoint_ms = float(record["speech_end_to_vad_finalized_ms"])
    cloud_ms = response_timing.get("audio_to_cloud_llm_ms")
    record["audio_ready_to_cloud_llm_ms"] = (
        float(cloud_ms) if cloud_ms is not None else None
    )
    record["speech_end_to_cloud_llm_ms"] = (
        round(endpoint_ms + float(cloud_ms), 3) if cloud_ms is not None else None
    )
    clip_id = record["audio"]["id"]
    audio_mode = response_timing.get("audio_mode")
    if audio_mode == "dynamic_cloud_tts":
        audio_path = Path(os.environ["DEMO_VOICE_CACHE_DIR"]) / f"{clip_id}.wav"
    else:
        audio_path = Path(os.environ["DEMO_STATIC_AUDIO_MANIFEST"]).parent / f"{clip_id}.wav"
    output_duration_s = wav_duration(audio_path)
    tts_ms = float(response_timing["tts_synthesis_ms"])
    valid = (
        record["slm"]["route"] == "non_tool"
        and record["harness"]["route"] == "cloud"
        and cloud_ms is not None
        and endpoint_ms >= 0
        and audio_mode == "dynamic_cloud_tts"
        and record["audio"]["cache_hit"] is False
        and not record["harness"]["errors"]
    )
    record["valid_cloud_path"] = valid
    record["cloud"] = {
        "model": record["harness"].get("cloud_model"),
        "latency_ms": response_timing["cloud_ms"],
        "response_chars": len(record["harness"]["assistant_text"]),
    }
    record["tts"] = {
        "provider": "omnivoice",
        "synthesis_ms": tts_ms,
        "harness_tts_ms": response_timing["tts_ms"],
        "output_duration_s": round(output_duration_s, 6),
        "rtf": (
            round(tts_ms / 1000 / output_duration_s, 6)
            if audio_mode == "dynamic_cloud_tts"
            else None
        ),
        "forced_uncached": audio_mode == "dynamic_cloud_tts",
        "source_fully_reached_speech_end": endpoint_ms >= 0,
    }
    return record


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    measured = [row for row in records if row["measured"]]
    valid = [row for row in measured if row["valid_cloud_path"]]
    samples: dict[str, Any] = {}
    for sample in sorted({row["sample"] for row in measured}):
        rows = [row for row in measured if row["sample"] == sample]
        good = [row for row in rows if row["valid_cloud_path"]]
        samples[sample] = {
            "source_duration_s": rows[0]["source_duration_s"],
            "runs": len(rows),
            "valid_cloud_runs": len(good),
            "metrics_ms": {
                metric: distribution([float(row[metric]) for row in good])
                for metric in METRICS
            } if good else {},
            "cloud_ms": distribution([float(row["cloud"]["latency_ms"]) for row in good]) if good else None,
            "tts_ms": distribution([float(row["tts"]["synthesis_ms"]) for row in good]) if good else None,
            "tts_rtf": distribution([float(row["tts"]["rtf"]) for row in good]) if good else None,
            "routes": sorted({row["harness"]["route"] for row in rows}),
        }
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu_limit_percent": args.gpu_limit,
        "warmup_cycles": args.warmup_cycles,
        "measured_cycles": args.runs,
        "measured_runs": len(measured),
        "valid_cloud_runs": len(valid),
        "cloud_path_success_rate": round(len(valid) / len(measured), 6) if measured else 0,
        "forced_uncached_tts": True,
        "overall_valid_metrics_ms": {
            metric: distribution([float(row[metric]) for row in valid])
            for metric in METRICS
        } if valid else {},
        "cloud_ms": distribution([float(row["cloud"]["latency_ms"]) for row in valid]) if valid else None,
        "tts_ms": distribution([float(row["tts"]["synthesis_ms"]) for row in valid]) if valid else None,
        "tts_rtf": distribution([float(row["tts"]["rtf"]) for row in valid]) if valid else None,
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmup_cycles < 1:
        raise ValueError("runs and warmup cycles must be positive")
    for path in args.wav:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.setdefault("WEBTEST_MODEL_MODE", "vllm")
    os.environ.setdefault("DEMO_CLOUD_ENABLED", "1")
    os.environ.setdefault("DEMO_TTS_ENABLED", "1")
    os.environ.setdefault("DEMO_TTS_PROVIDER", "omnivoice")
    os.environ.setdefault("DEMO_OMNIVOICE_FORCE_SYNTHESIS", "1")

    from fastapi.testclient import TestClient
    from demo.backend.app import app

    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_cycles = args.warmup_cycles + args.runs
    client = TestClient(app)
    with args.raw_jsonl.open("w", encoding="utf-8", buffering=1) as raw:
        for cycle in range(total_cycles):
            measured = cycle >= args.warmup_cycles
            ordered = args.wav[cycle % len(args.wav):] + args.wav[:cycle % len(args.wav)]
            for source in ordered:
                record = run_once(
                    client,
                    source,
                    selected_variant(source, args.variants_root, cycle),
                    args,
                    cycle=cycle,
                    measured=measured,
                )
                record = enrich(record)
                records.append(record)
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps({
                    "gpu": args.gpu_limit,
                    "cycle": cycle + 1,
                    "measured": measured,
                    "sample": source.name,
                    "valid": record["valid_cloud_path"],
                    "slm_last_ms": record["speech_end_to_last_slm_token_ms"],
                    "cloud_done_ms": record["speech_end_to_cloud_llm_ms"],
                    "first_audio_ms": record["speech_end_to_first_audio_byte_ms"],
                    "cloud_ms": record["cloud"]["latency_ms"],
                    "tts_ms": record["tts"]["synthesis_ms"],
                }, ensure_ascii=False), flush=True)
    summary = summarize(records, args)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["cloud_path_success_rate"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
