#!/usr/bin/env python3
"""Aggregate Jetson full-system JSONL and tegrastats logs into report data."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable


LATENCY_METRICS = (
    "speech_end_to_first_slm_token_ms",
    "speech_end_to_last_slm_token_ms",
    "speech_end_to_first_audio_byte_ms",
    "audio_ready_to_first_slm_token_ms",
    "audio_ready_to_last_slm_token_ms",
    "audio_ready_to_first_audio_byte_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--conditions", nargs="+", type=int, default=[100, 70, 50, 30])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "stddev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def values(records: list[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]) -> list[float]:
    output = []
    for record in records:
        value = getter(record)
        if value is not None:
            output.append(float(value))
    return output


def component_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, Callable[[dict[str, Any]], Any]] = {
        "vad_endpoint_ms": lambda row: row["speech_end_to_vad_finalized_ms"],
        "vad_compute_total_ms": lambda row: row["vad"]["compute_total_ms"],
        "enhancement_compute_total_ms": lambda row: row["enhancement"]["compute_total_ms"],
        "enhancement_algorithmic_delay_ms": lambda row: row["enhancement"][
            "algorithmic_delay_ms"
        ],
        "capture_write_ms": lambda row: row["slm"]["pipeline_timings"][
            "capture_write_ms"
        ],
        "slm_request_to_first_token_ms": lambda row: row["slm"]["timings"][
            "request_to_first_token_ms"
        ],
        "slm_first_to_last_token_ms": lambda row: row["slm"]["timings"][
            "first_to_last_token_ms"
        ],
        "slm_request_to_last_token_ms": lambda row: row["slm"]["timings"][
            "request_to_last_token_ms"
        ],
        "model_adapter_after_last_token_ms": lambda row: row["slm"]["timings"][
            "adapter_total_ms"
        ]
        - row["slm"]["timings"]["request_to_last_token_ms"],
        "execute_ms": lambda row: row["harness"]["timings"]["execute_ms"],
        "static_audio_lookup_ms": lambda row: row["harness"]["timings"]["tts_ms"],
        "last_token_to_response_ready_ms": lambda row: row[
            "audio_ready_to_response_ready_ms"
        ]
        - row["audio_ready_to_last_slm_token_ms"],
        "response_ready_to_first_audio_byte_ms": lambda row: row[
            "audio_ready_to_first_audio_byte_ms"
        ]
        - row["audio_ready_to_response_ready_ms"],
    }
    result = {
        name: distribution(values(records, getter))
        for name, getter in components.items()
    }
    decode_tps = []
    effective_tps = []
    for row in records:
        tokens = row["slm"].get("output_tokens")
        decode_ms = row["slm"]["timings"].get("first_to_last_token_ms")
        request_ms = row["slm"]["timings"].get("request_to_last_token_ms")
        if tokens is not None and decode_ms and decode_ms > 0:
            decode_tps.append(
                max(0.0, float(tokens) - 1) / float(decode_ms) * 1000
            )
        if tokens is not None and request_ms and request_ms > 0:
            effective_tps.append(float(tokens) / float(request_ms) * 1000)
    result["decode_tokens_per_second_after_first"] = distribution(decode_tps)
    result["effective_output_tokens_per_second"] = distribution(effective_tps)
    return result


def parse_tegrastats(path: Path) -> dict[str, Any]:
    patterns = {
        "ram_used_mb": re.compile(r"RAM (\d+)/(\d+)MB"),
        "gpu_load_percent": re.compile(r"GR3D_FREQ (\d+)%"),
        "gpu_temp_c": re.compile(r"gpu@([\d.]+)C"),
        "gpu_soc_power_mw": re.compile(r"VDD_GPU_SOC (\d+)mW"),
        "cpu": re.compile(r"CPU \[([^]]+)\]"),
    }
    collected: dict[str, list[float]] = {
        "ram_used_mb": [],
        "gpu_load_percent": [],
        "gpu_temp_c": [],
        "gpu_soc_power_mw": [],
        "mean_cpu_load_percent": [],
    }
    active_collected: dict[str, list[float]] = {
        name: [] for name in collected
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line_values: dict[str, float] = {}
        for name in ("ram_used_mb", "gpu_load_percent", "gpu_temp_c", "gpu_soc_power_mw"):
            match = patterns[name].search(line)
            if match:
                line_values[name] = float(match.group(1))
        cpu = patterns["cpu"].search(line)
        if cpu:
            loads = [float(item) for item in re.findall(r"(\d+)%@", cpu.group(1))]
            if loads:
                line_values["mean_cpu_load_percent"] = statistics.mean(loads)
        for name, value in line_values.items():
            collected[name].append(value)
        if line_values.get("gpu_load_percent", 0) > 0:
            for name, value in line_values.items():
                active_collected[name].append(value)
    gpu_active = [item for item in collected["gpu_load_percent"] if item > 0]
    return {
        "samples": len(collected["gpu_load_percent"]),
        "sample_interval_ms": 100,
        "overall": {name: distribution(items) for name, items in collected.items()},
        "gpu_active": distribution(gpu_active),
        "active_period": {
            name: distribution(items) for name, items in active_collected.items()
        },
        "gpu_active_fraction": round(
            len(gpu_active) / len(collected["gpu_load_percent"]), 6
        )
        if collected["gpu_load_percent"]
        else None,
    }


def quality(records: list[dict[str, Any]], expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    exact_total = 0
    name_total = 0
    for sample, target in expected.items():
        sample_rows = [row for row in records if row["sample"] == sample]
        expected_call = target["expected_call"]
        exact = sum(row["slm"]["calls"] == [expected_call] for row in sample_rows)
        name = sum(
            len(row["slm"]["calls"]) == 1
            and row["slm"]["calls"][0].get("name") == expected_call["name"]
            for row in sample_rows
        )
        exact_total += exact
        name_total += name
        outputs = Counter(
            json.dumps(row["slm"]["calls"], sort_keys=True, ensure_ascii=False)
            for row in sample_rows
        )
        rows.append(
            {
                "sample": sample,
                "runs": len(sample_rows),
                "tool_name_accuracy": round(name / len(sample_rows), 6) if sample_rows else None,
                "exact_call_accuracy": round(exact / len(sample_rows), 6) if sample_rows else None,
                "unique_predictions": [
                    {"calls": json.loads(item), "count": count}
                    for item, count in outputs.items()
                ],
            }
        )
    total = sum(row["runs"] for row in rows)
    return {
        "runs": total,
        "tool_name_accuracy": round(name_total / total, 6) if total else None,
        "exact_call_accuracy": round(exact_total / total, 6) if total else None,
        "samples": rows,
    }


def main() -> int:
    args = parse_args()
    corpus_payload = json.loads(args.corpus.read_text(encoding="utf-8"))
    expected = {item["file"]: item for item in corpus_payload["samples"]}
    output: dict[str, Any] = {
        "schema_version": 1,
        "result_root": str(args.result_root.resolve()),
        "corpus": corpus_payload,
        "conditions": {},
    }
    all_condition_records: dict[int, list[dict[str, Any]]] = {}
    for condition in args.conditions:
        directory = args.result_root / f"p{condition}"
        records = [
            json.loads(line)
            for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = [row for row in records if row["measured"]]
        valid = [row for row in measured if row["valid_tool_path"]]
        all_condition_records[condition] = measured
        by_sample = {}
        for sample in expected:
            rows = [row for row in valid if row["sample"] == sample]
            by_sample[sample] = {
                "duration_seconds": expected[sample]["duration_seconds"],
                "valid_runs": len(rows),
                "latency_ms": {
                    metric: distribution(values(rows, lambda row, key=metric: row[key]))
                    for metric in LATENCY_METRICS
                },
            }
        output["conditions"][str(condition)] = {
            "measured_runs": len(measured),
            "valid_tool_path_runs": len(valid),
            "valid_tool_path_rate": round(len(valid) / len(measured), 6),
            "latency_ms": {
                metric: distribution(values(valid, lambda row, key=metric: row[key]))
                for metric in LATENCY_METRICS
            },
            "by_sample": by_sample,
            "components_ms": component_metrics(valid),
            "quality": quality(measured, expected),
            "resources": parse_tegrastats(directory / "tegrastats.log"),
        }

    baseline = output["conditions"].get("100", {}).get("by_sample", {})
    for condition, body in output["conditions"].items():
        for sample, row in body["by_sample"].items():
            base_row = baseline.get(sample, {}).get("latency_ms", {})
            row["median_degradation_vs_p100_percent"] = {}
            for metric in LATENCY_METRICS:
                current = row["latency_ms"][metric]
                base = base_row.get(metric)
                row["median_degradation_vs_p100_percent"][metric] = (
                    round((current["median"] / base["median"] - 1) * 100, 3)
                    if current and base and base["median"]
                    else None
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
