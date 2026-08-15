#!/usr/bin/env python3
"""Benchmark the local tool path through VAD, SLM, execution, and cached audio.

The WAVs are sent through the real WebSocket application at microphone pace.
Each measured run writes one lossless JSONL record. Two latency clocks are
retained deliberately:

* ``speech_end_to_*`` starts at OmniVAD's acoustic speech-end frame and includes
  endpointing delay. This is closest to what a user perceives after speaking.
* ``audio_ready_to_*`` starts when VAD has finalized the waveform handed to the
  SLM. This is the clock shown by the web UI's ``Audio -> ...`` diagnostics.

"Time to first audio" is the first byte returned by the cached-WAV endpoint.
It excludes browser audio decoding, output buffering, and speaker hardware.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


METRICS = (
    "speech_end_to_vad_finalized_ms",
    "speech_end_to_first_slm_token_ms",
    "speech_end_to_last_slm_token_ms",
    "speech_end_to_first_audio_byte_ms",
    "audio_ready_to_first_slm_token_ms",
    "audio_ready_to_last_slm_token_ms",
    "audio_ready_to_first_audio_byte_ms",
    "audio_ready_to_response_ready_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--variants-root", type=Path)
    parser.add_argument("--runs", type=int, default=8, help="measured cycles")
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


def speech_end_from_energy(
    pcm16le: bytes,
    *,
    threshold_dbfs: float = -42.0,
    frame_ms: int = 10,
) -> float:
    """Annotate the last active source frame independently of the live VAD."""

    samples = array.array("h")
    samples.frombytes(pcm16le)
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = 16_000 * frame_ms // 1000
    last_active_end = 0
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        if not frame:
            continue
        square_mean = sum(float(value) * value for value in frame) / len(frame)
        dbfs = (
            -120.0
            if square_mean <= 0
            else 20.0 * math.log10(math.sqrt(square_mean) / 32768.0)
        )
        if dbfs >= threshold_dbfs:
            last_active_end = min(len(samples), offset + len(frame))
    if last_active_end == 0:
        raise ValueError("source WAV has no frame above the speech annotation threshold")
    return last_active_end * 1000 / 16_000


def read_pcm(path: Path) -> tuple[bytes, float, float]:
    with wave.open(str(path), "rb") as reader:
        properties = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.getcomptype(),
        )
        if properties != (1, 2, 16_000, "NONE"):
            raise ValueError(f"expected mono 16 kHz PCM16 WAV, got {properties}: {path}")
        frames = reader.readframes(reader.getnframes())
    return (
        frames,
        len(frames) / 2 / 16_000,
        speech_end_from_energy(frames),
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "stddev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def selected_variant(source: Path, variants_root: Path | None, cycle: int) -> Path:
    if variants_root is None:
        return source
    candidate = variants_root / f"r{cycle + 1}" / source.name
    if not candidate.is_file():
        raise FileNotFoundError(f"missing byte-unique benchmark variant: {candidate}")
    return candidate


def first_audio_byte(client, url: str) -> tuple[float, int, str]:
    first_epoch_ms: float | None = None
    total = 0
    content_type = ""
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            if first_epoch_ms is None:
                first_epoch_ms = time.time() * 1000
                if not chunk.startswith(b"RIFF"):
                    raise RuntimeError("cached response did not begin with a RIFF WAV header")
            total += len(chunk)
    if first_epoch_ms is None:
        raise RuntimeError("cached response endpoint returned no body")
    return first_epoch_ms, total, content_type


def run_once(
    client,
    source: Path,
    streamed_path: Path,
    args: argparse.Namespace,
    *,
    cycle: int,
    measured: bool,
) -> dict[str, Any]:
    pcm, source_duration_s, reference_speech_end_ms = read_pcm(streamed_path)
    packet_bytes = 16_000 * args.packet_ms // 1000 * 2
    stream = bytes(16_000 * args.prefix_ms // 1000 * 2) + pcm
    stream += bytes(16_000 * args.suffix_ms // 1000 * 2)
    stop_sending = threading.Event()
    sender_error: list[str] = []
    audio_start_epoch_ms: list[float] = []
    client.post("/api/hardware/reset").raise_for_status()

    with client.websocket_connect("/api/audio/stream") as websocket:
        websocket.send_json(
            {
                "event": "start_stream",
                "backend": args.backend,
                "enhancer": args.enhancer,
                "sample_rate": 16_000,
                "channels": 1,
                "encoding": "pcm16le",
            }
        )
        started = websocket.receive_json()
        if started.get("event") != "stream_started":
            raise RuntimeError(f"expected stream_started, got {started}")

        def send_realtime() -> None:
            begin = time.perf_counter()
            audio_start_epoch_ms.append(time.time() * 1000)
            try:
                for offset in range(0, len(stream), packet_bytes):
                    if stop_sending.is_set():
                        return
                    deadline = begin + ((offset // packet_bytes) + 1) * args.packet_ms / 1000
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    websocket.send_bytes(stream[offset : offset + packet_bytes])
            except Exception as exc:  # normal disconnects are suppressed below
                if not stop_sending.is_set():
                    sender_error.append(f"{type(exc).__name__}: {exc}")

        sender = threading.Thread(target=send_realtime, daemon=True)
        sender.start()
        events: list[dict[str, Any]] = [
            {**started, "client_received_epoch_ms": round(time.time() * 1000, 3)}
        ]
        by_name: dict[str, dict[str, Any]] = {"stream_started": started}
        while "assistant_response" not in by_name:
            event = websocket.receive_json()
            event["client_received_epoch_ms"] = round(time.time() * 1000, 3)
            events.append(event)
            name = event.get("event")
            if name == "error":
                raise RuntimeError(event.get("message", "unknown server error"))
            if name in {
                "utterance_started",
                "utterance_finalized",
                "model_started",
                "model_result",
                "assistant_response",
            }:
                by_name[name] = event
            if name == "utterance_finalized":
                stop_sending.set()

        stop_sending.set()
        sender.join(timeout=2)
        if sender_error:
            raise RuntimeError(sender_error[0])
        if not audio_start_epoch_ms:
            raise RuntimeError("audio sender did not start")

        finalized = by_name["utterance_finalized"]
        model_event = by_name["model_result"]
        assistant_event = by_name["assistant_response"]
        response = assistant_event["response"]
        audio_items = response.get("audio") or []
        if not audio_items or not audio_items[0].get("url"):
            raise RuntimeError(f"harness returned no playable cached audio: {response}")
        first_audio_epoch_ms, audio_bytes, content_type = first_audio_byte(
            client, audio_items[0]["url"]
        )
        audio_from_vad_ms = first_audio_epoch_ms - assistant_event[
            "audio_baseline_timestamp_ms"
        ]
        websocket.send_json(
            {
                "event": "audio_playback_started",
                "turn_id": assistant_event["turn_id"],
                "audio_from_vad_ms": round(audio_from_vad_ms, 3),
            }
        )
        websocket.send_json(
            {"event": "playback_finished", "turn_id": assistant_event["turn_id"]}
        )
        while True:
            event = websocket.receive_json()
            event["client_received_epoch_ms"] = round(time.time() * 1000, 3)
            events.append(event)
            if event.get("event") == "error":
                raise RuntimeError(event.get("message", "unknown server error"))
            if event.get("event") == "input_gate" and event.get("state") == "open":
                break

    model_result = model_event["result"]
    model_timing = model_event["timings"]
    response_timing = response["timings"]
    speech_end_epoch_ms = (
        audio_start_epoch_ms[0] + args.prefix_ms + reference_speech_end_ms
    )
    vad_epoch_ms = float(finalized["vad_finalized_timestamp_ms"])
    endpoint_ms = vad_epoch_ms - speech_end_epoch_ms
    executions = response.get("executions") or []
    valid_tool_path = (
        model_result.get("route") == "tool"
        and response.get("route") == "executed"
        and response_timing.get("audio_mode") == "static_cache"
        and len(executions) == 1
        and executions[0].get("status") == "success"
    )
    record = {
        "schema_version": 1,
        "gpu_limit_percent": args.gpu_limit,
        "backend": args.backend,
        "enhancer": args.enhancer,
        "sample": source.name,
        "streamed_wav": str(streamed_path),
        "source_duration_s": round(source_duration_s, 6),
        "reference_speech_end_ms": round(reference_speech_end_ms, 3),
        "reference_speech_end_method": "last 10 ms PCM frame at or above -42 dBFS",
        "cycle": cycle + 1,
        "measured": measured,
        "valid_tool_path": valid_tool_path,
        "speech_end_to_vad_finalized_ms": round(endpoint_ms, 3),
        "speech_end_to_first_slm_token_ms": round(
            endpoint_ms + model_timing["audio_to_first_llm_token_ms"], 3
        ),
        "speech_end_to_last_slm_token_ms": round(
            endpoint_ms + model_timing["audio_to_last_llm_token_ms"], 3
        ),
        "speech_end_to_first_audio_byte_ms": round(
            first_audio_epoch_ms - speech_end_epoch_ms, 3
        ),
        "audio_ready_to_first_slm_token_ms": model_timing[
            "audio_to_first_llm_token_ms"
        ],
        "audio_ready_to_last_slm_token_ms": model_timing[
            "audio_to_last_llm_token_ms"
        ],
        "audio_ready_to_first_audio_byte_ms": round(audio_from_vad_ms, 3),
        "audio_ready_to_response_ready_ms": response_timing[
            "audio_to_response_ready_ms"
        ],
        "vad": {
            "captured_audio_ms": finalized["duration_ms"],
            "utterance_start_audio_ms": finalized["utterance_start_audio_ms"],
            "utterance_end_audio_ms": finalized["utterance_end_audio_ms"],
            "endpoint_audio_ms": finalized["endpoint_audio_ms"],
            "frames": finalized["vad_frames_processed"],
            "compute_total_ms": finalized["vad_process_total_ms"],
            "compute_mean_ms": finalized["vad_process_mean_ms"],
            "compute_max_ms": finalized["vad_process_max_ms"],
        },
        "enhancement": {
            "frames": finalized["enhancement_frames_processed"],
            "compute_total_ms": finalized["enhancement_compute_total_ms"],
            "compute_mean_ms": finalized["enhancement_compute_mean_ms"],
            "compute_max_ms": finalized["enhancement_compute_max_ms"],
            "algorithmic_delay_ms": finalized["enhancement_algorithmic_delay_ms"],
        },
        "slm": {
            "route": model_result.get("route"),
            "raw": model_result.get("raw"),
            "calls": model_result.get("calls"),
            "output_tokens": model_result.get("output_tokens"),
            "adapter_latency_ms": model_result.get("latency_ms"),
            "timings": model_result.get("timings"),
            "pipeline_timings": model_timing,
        },
        "harness": {
            "route": response.get("route"),
            "assistant_text": response.get("assistant_text"),
            "cloud_model": response.get("cloud_model"),
            "executions": executions,
            "timings": response_timing,
            "errors": response.get("errors"),
        },
        "audio": {
            "url": audio_items[0]["url"],
            "id": audio_items[0].get("id"),
            "bytes": audio_bytes,
            "content_type": content_type,
            "cache_hit": audio_items[0].get("cache_hit"),
        },
        "events": events,
    }
    return record


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    measured = [item for item in records if item["measured"]]
    samples: dict[str, Any] = {}
    for sample in sorted({item["sample"] for item in measured}):
        rows = [item for item in measured if item["sample"] == sample]
        valid = [item for item in rows if item["valid_tool_path"]]
        samples[sample] = {
            "source_duration_s": rows[0]["source_duration_s"],
            "runs": len(rows),
            "valid_tool_runs": len(valid),
            "tool_path_success_rate": round(len(valid) / len(rows), 6),
            "metrics_ms": {
                metric: distribution([float(row[metric]) for row in valid])
                for metric in METRICS
            }
            if valid
            else {},
            "routes": sorted({row["harness"]["route"] for row in rows}),
            "slm_outputs": sorted({row["slm"]["raw"] for row in rows}),
        }
    valid_all = [item for item in measured if item["valid_tool_path"]]
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu_limit_percent": args.gpu_limit,
        "backend": args.backend,
        "enhancer": args.enhancer,
        "packet_ms": args.packet_ms,
        "prefix_ms": args.prefix_ms,
        "suffix_ms": args.suffix_ms,
        "warmup_cycles": args.warmup_cycles,
        "measured_cycles": args.runs,
        "measured_runs": len(measured),
        "valid_tool_runs": len(valid_all),
        "tool_path_success_rate": round(len(valid_all) / len(measured), 6),
        "metric_definitions": {
            "speech_end": "OmniVAD acoustic speech-end frame; includes endpoint delay",
            "audio_ready": "VAD-finalized WAV is ready for SLM dispatch",
            "first_audio": "first byte from local cached-WAV HTTP endpoint",
        },
        "overall_valid_metrics_ms": {
            metric: distribution([float(row[metric]) for row in valid_all])
            for metric in METRICS
        }
        if valid_all
        else {},
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup_cycles < 1:
        raise ValueError("--warmup-cycles must be at least 1")
    for path in args.wav:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.setdefault("WEBTEST_MODEL_MODE", "vllm")
    os.environ.setdefault("DEMO_CLOUD_ENABLED", "0")
    os.environ.setdefault("DEMO_TTS_ENABLED", "0")

    from fastapi.testclient import TestClient
    from demo.backend.app import app

    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_cycles = args.warmup_cycles + args.runs
    variant_count = total_cycles
    if args.variants_root:
        available = sorted(
            int(path.name[1:])
            for path in args.variants_root.glob("r*")
            if path.is_dir() and path.name[1:].isdigit()
        )
        if len(available) < variant_count:
            raise ValueError(
                f"need {variant_count} variant directories, found {len(available)}"
            )

    client = TestClient(app)
    with args.raw_jsonl.open("w", encoding="utf-8", buffering=1) as raw:
        for cycle in range(total_cycles):
            measured = cycle >= args.warmup_cycles
            # Rotate sample order across cycles so thermal drift is not tied to duration.
            ordered = args.wav[cycle % len(args.wav) :] + args.wav[: cycle % len(args.wav)]
            for source in ordered:
                streamed = selected_variant(source, args.variants_root, cycle)
                record = run_once(
                    client,
                    source,
                    streamed,
                    args,
                    cycle=cycle,
                    measured=measured,
                )
                records.append(record)
                raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "gpu": args.gpu_limit,
                            "cycle": cycle + 1,
                            "measured": measured,
                            "sample": source.name,
                            "duration_s": record["source_duration_s"],
                            "tool_ok": record["valid_tool_path"],
                            "speech_end_to_first_token_ms": record[
                                "speech_end_to_first_slm_token_ms"
                            ],
                            "speech_end_to_last_token_ms": record[
                                "speech_end_to_last_slm_token_ms"
                            ],
                            "speech_end_to_first_audio_ms": record[
                                "speech_end_to_first_audio_byte_ms"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = summarize(records, args)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["tool_path_success_rate"] < 1.0:
        print("WARNING: not every measured run completed the intended tool path", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
