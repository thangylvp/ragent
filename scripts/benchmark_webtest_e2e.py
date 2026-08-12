#!/usr/bin/env python3
"""Benchmark speech-end -> last SLM token through the real WebSocket pipeline.

The source waveform is streamed at real-time pace.  ``--speech-end-ms`` is a
reference annotation within that waveform, so the reported endpoint delay is
not confused with the utterance duration or client-side playback time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import wave
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="mono 16 kHz PCM16 WAV")
    parser.add_argument(
        "--speech-end-ms",
        type=float,
        required=True,
        help="last speech point measured from the start of the source WAV",
    )
    parser.add_argument("--backend", default="omnivad")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--prefix-ms", type=int, default=500)
    parser.add_argument("--suffix-ms", type=int, default=1000)
    parser.add_argument("--packet-ms", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        properties = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.getcomptype(),
        )
        if properties != (1, 2, 16_000, "NONE"):
            raise ValueError(
                f"expected mono 16 kHz PCM16 WAV, got channels={properties[0]}, "
                f"width={properties[1]}, rate={properties[2]}, codec={properties[3]}"
            )
        return reader.readframes(reader.getnframes())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def aggregate(runs: list[dict]) -> dict:
    keys = [
        "speech_end_to_vad_ms",
        "speech_end_to_last_token_ms",
        "speech_end_to_result_ms",
    ]
    return {
        key: {
            "mean": round(statistics.mean(item[key] for item in runs), 3),
            "median": round(statistics.median(item[key] for item in runs), 3),
            "p95": round(percentile([item[key] for item in runs], 0.95), 3),
            "min": round(min(item[key] for item in runs), 3),
            "max": round(max(item[key] for item in runs), 3),
        }
        for key in keys
    }


def mean_nested(runs: list[dict], section: str, key: str) -> float:
    return round(statistics.mean(item[section][key] for item in runs), 3)


def one_run(client, pcm: bytes, args: argparse.Namespace, run_number: int) -> dict:
    packet_samples = 16_000 * args.packet_ms // 1000
    packet_bytes = packet_samples * 2
    prefix = bytes(16_000 * args.prefix_ms // 1000 * 2)
    suffix = bytes(16_000 * args.suffix_ms // 1000 * 2)
    stream = prefix + pcm + suffix
    stop_sending = threading.Event()
    sender_error: list[str] = []
    audio_started: list[float] = []

    with client.websocket_connect("/api/audio/stream") as websocket:
        websocket.send_json(
            {
                "event": "start_stream",
                "backend": args.backend,
                "sample_rate": 16_000,
                "channels": 1,
                "encoding": "pcm16le",
            }
        )
        started_event = websocket.receive_json()
        if started_event.get("event") != "stream_started":
            raise RuntimeError(f"expected stream_started, got {started_event}")

        def send_realtime() -> None:
            begin = time.perf_counter()
            audio_started.append(time.time() * 1000)
            try:
                for offset in range(0, len(stream), packet_bytes):
                    if stop_sending.is_set():
                        return
                    # A microphone can emit a packet only after collecting its final
                    # sample, so packet 0 arrives one packet interval after audio starts.
                    deadline = begin + (
                        (offset // packet_bytes) + 1
                    ) * args.packet_ms / 1000
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    websocket.send_bytes(stream[offset : offset + packet_bytes])
            except Exception as exc:  # server normally closes after first utterance
                if not stop_sending.is_set():
                    sender_error.append(f"{type(exc).__name__}: {exc}")

        sender = threading.Thread(target=send_realtime, daemon=True)
        sender.start()
        events: dict[str, dict] = {"stream_started": started_event}
        while "model_result" not in events:
            event = websocket.receive_json()
            name = event.get("event")
            if name == "error":
                raise RuntimeError(event.get("message", "unknown server error"))
            if name in {"utterance_finalized", "model_started", "model_result"}:
                events[name] = event
            if name == "utterance_finalized":
                stop_sending.set()
        stop_sending.set()
        sender.join(timeout=2)

    if sender_error:
        raise RuntimeError(sender_error[0])
    if not audio_started:
        raise RuntimeError("audio sender did not start")

    finalized = events["utterance_finalized"]
    model_result = events["model_result"]
    result = model_result["result"]
    model_timing = result.get("timings") or {}
    speech_end_wall_ms = (
        audio_started[0] + args.prefix_ms + args.speech_end_ms
    )
    endpoint_delay_ms = (
        finalized["vad_finalized_timestamp_ms"] - speech_end_wall_ms
    )
    to_last_token_ms = endpoint_delay_ms + model_result["last_token_from_vad_ms"]
    to_result_ms = endpoint_delay_ms + model_result["end_to_end_from_vad_ms"]
    return {
        "run": run_number,
        "cold_start": run_number == 1,
        "route": result["route"],
        "calls": result["calls"],
        "output_tokens": result["output_tokens"],
        "captured_audio_ms": finalized["duration_ms"],
        "vad_endpoint_audio_ms": finalized["endpoint_audio_ms"],
        "speech_end_to_vad_ms": round(endpoint_delay_ms, 3),
        "speech_end_to_last_token_ms": round(to_last_token_ms, 3),
        "speech_end_to_result_ms": round(to_result_ms, 3),
        "vad": {
            "frames": finalized["vad_frames_processed"],
            "compute_total_ms": finalized["vad_process_total_ms"],
            "compute_mean_frame_ms": finalized["vad_process_mean_ms"],
            "compute_max_frame_ms": finalized["vad_process_max_ms"],
        },
        "post_vad": {
            **model_result["component_timings"],
            **model_timing,
        },
    }


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise ValueError("use at least two runs to distinguish cold and warm latency")
    os.environ.setdefault("WEBTEST_MODEL_MODE", "local")
    os.environ.setdefault("WEBTEST_DEVICE", "cuda")

    from fastapi.testclient import TestClient
    from demo.backend.app import app

    pcm = read_pcm(args.wav)
    duration_ms = len(pcm) / 2 / 16_000 * 1000
    if not 0 <= args.speech_end_ms <= duration_ms:
        raise ValueError("--speech-end-ms must fall inside the source WAV")

    client = TestClient(app)
    runs = [one_run(client, pcm, args, index + 1) for index in range(args.runs)]
    warm_runs = runs[1:]
    warm_generation_ms = mean_nested(
        warm_runs,
        "post_vad",
        "generation_to_last_token_ms",
    )
    warm_first_to_last_ms = mean_nested(
        warm_runs,
        "post_vad",
        "first_to_last_token_ms",
    )
    output_tokens = statistics.mean(item["output_tokens"] for item in warm_runs)
    report = {
        "definition": "wall time from annotated source speech end through last generated token",
        "source_wav": str(args.wav.resolve()),
        "source_duration_ms": round(duration_ms, 3),
        "speech_end_ms": args.speech_end_ms,
        "backend": args.backend,
        "packet_ms": args.packet_ms,
        "runs": runs,
        "cold": aggregate(runs[:1]),
        "warm": aggregate(warm_runs),
        "warm_component_mean_ms": {
            "capture_write": mean_nested(warm_runs, "post_vad", "capture_write_ms"),
            "model_dispatch": mean_nested(warm_runs, "post_vad", "model_dispatch_ms"),
            "audio_decode": mean_nested(warm_runs, "post_vad", "audio_decode_ms"),
            "prompt_render": mean_nested(warm_runs, "post_vad", "prompt_render_ms"),
            "feature_extraction": mean_nested(
                warm_runs,
                "post_vad",
                "feature_extraction_ms",
            ),
            "host_to_gpu": mean_nested(
                warm_runs,
                "post_vad",
                "host_to_device_ms",
            ),
            "model_to_first_token": mean_nested(
                warm_runs,
                "post_vad",
                "generation_to_first_token_ms",
            ),
            "first_to_last_token": warm_first_to_last_ms,
            "model_to_last_token": warm_generation_ms,
            "decode_and_parse_after_last_token": mean_nested(
                warm_runs,
                "post_vad",
                "decode_parse_ms",
            ),
        },
        "warm_vad_mean": {
            "online_compute_total_ms": mean_nested(
                warm_runs,
                "vad",
                "compute_total_ms",
            ),
            "compute_per_frame_ms": mean_nested(
                warm_runs,
                "vad",
                "compute_mean_frame_ms",
            ),
        },
        "warm_rates": {
            "effective_output_tokens_per_second": round(
                output_tokens / warm_generation_ms * 1000,
                3,
            ),
            "tokens_per_second_after_first_token": round(
                max(0.0, output_tokens - 1) / warm_first_to_last_ms * 1000,
                3,
            ),
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
