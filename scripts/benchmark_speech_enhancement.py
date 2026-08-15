#!/usr/bin/env python3
"""Benchmark streaming denoisers for the speech-to-tool-call pipeline.

The benchmark deliberately treats the downstream SLM result as the primary
metric.  Signal metrics are still reported, but they do not prove that a
denoiser preserved command words or argument values.

This script expects model artifacts under ``outputs/denoise/models``.  They are
kept out of Git because they are downloaded third-party weights.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import onnxruntime as ort
import sherpa_onnx
import soundfile as sf
import torch
from scipy.signal import correlate, correlation_lags, resample_poly


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = (
    REPO_ROOT.parent
    / "stc/outputs/models/route_v1_best_step1250_hf/samples"
)
DEFAULT_TOOLS = DEFAULT_SAMPLES.parent / "tools_openai.json"
DEFAULT_MODELS = REPO_ROOT / "outputs/denoise/models"
DEFAULT_CAPTURES = REPO_ROOT / "outputs/demo/captures"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/denoise/benchmark"


@dataclass
class EnhancementResult:
    samples: np.ndarray
    compute_ms: float
    frame_ms_mean: float
    frame_ms_p99: float
    frame_ms_max: float
    algorithmic_delay_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 5.0])
    parser.add_argument(
        "--sample",
        action="append",
        help="WAV basename to include; repeat this option. Defaults to all tool samples.",
    )
    parser.add_argument(
        "--denoiser",
        action="append",
        help="Denoiser name to include; repeat this option. Defaults to every available model.",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument(
        "--skip-slm",
        dest="skip_slm",
        action="store_true",
        help="Skip the downstream vLLM tool-call check.",
    )
    return parser.parse_args()


def read_mono_16k(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != 16_000:
        divisor = math.gcd(sample_rate, 16_000)
        samples = resample_poly(samples, 16_000 // divisor, sample_rate // divisor)
    return np.asarray(samples, dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(samples, -1.0, 1.0), 16_000, subtype="PCM_16")


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)) + 1e-12))


def build_quiet_frame_noise(captures_dir: Path, seed: int = 7) -> np.ndarray:
    """Build a repeatable ambient track from low-energy captured frames.

    The web captures are utterance-level rather than continuous room recordings,
    so selecting their quietest 20 ms frames reduces the chance of copying speech
    into the test mixture.  The resulting benchmark is useful but is not a
    replacement for a dedicated noise-only room recording.
    """

    frames: list[np.ndarray] = []
    frame_size = 320
    paths = sorted(captures_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)[-80:]
    for path in paths:
        try:
            samples = read_mono_16k(path)
        except Exception:
            continue
        for offset in range(0, len(samples) - frame_size + 1, frame_size):
            frame = samples[offset : offset + frame_size]
            if np.max(np.abs(frame)) > 1e-5:
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"no usable 16 kHz capture frames found under {captures_dir}")
    energies = np.asarray([rms(frame) for frame in frames])
    limit = float(np.quantile(energies, 0.35))
    quiet = [frame for frame, energy in zip(frames, energies) if energy <= limit]
    rng = np.random.default_rng(seed)
    rng.shuffle(quiet)
    return np.concatenate(quiet).astype(np.float32)


def repeat_to_length(noise: np.ndarray, length: int, offset: int) -> np.ndarray:
    if len(noise) == 0:
        raise ValueError("noise track is empty")
    start = offset % len(noise)
    required = length + start
    tiled = np.tile(noise, math.ceil(required / len(noise)))
    return tiled[start : start + length].copy()


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_level = rms(clean)
    noise_level = rms(noise)
    target_noise_level = clean_level / (10 ** (snr_db / 20))
    mixed = clean + noise * (target_noise_level / max(noise_level, 1e-12))
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak
    return mixed.astype(np.float32)


def timing(times: list[float], samples: np.ndarray, delay_ms: float) -> EnhancementResult:
    millis = np.asarray(times, dtype=np.float64) * 1000
    return EnhancementResult(
        samples=samples.astype(np.float32),
        compute_ms=float(millis.sum()),
        frame_ms_mean=float(millis.mean()),
        frame_ms_p99=float(np.quantile(millis, 0.99)),
        frame_ms_max=float(millis.max()),
        algorithmic_delay_ms=delay_ms,
    )


def create_sherpa_runner(kind: str, path: Path) -> Callable[[np.ndarray], EnhancementResult]:
    config_class = (
        sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig
        if kind == "gtcrn"
        else sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig
    )
    model_config = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        **{kind: config_class(model=str(path))},
        num_threads=2,
    )
    denoiser = sherpa_onnx.OnlineSpeechDenoiser(
        sherpa_onnx.OnlineSpeechDenoiserConfig(model=model_config)
    )
    hop = denoiser.frame_shift_in_samples

    def run(samples: np.ndarray) -> EnhancementResult:
        denoiser.reset()
        output: list[float] = []
        times: list[float] = []
        for offset in range(0, len(samples), hop):
            frame = samples[offset : offset + hop]
            if len(frame) < hop:
                frame = np.pad(frame, (0, hop - len(frame)))
            started = time.perf_counter()
            result = denoiser.run(frame, 16_000)
            times.append(time.perf_counter() - started)
            output.extend(result.samples)
        started = time.perf_counter()
        result = denoiser.flush()
        times.append(time.perf_counter() - started)
        output.extend(result.samples)
        enhanced = np.asarray(output, dtype=np.float32)
        if len(enhanced) < len(samples):
            enhanced = np.pad(enhanced, (0, len(samples) - len(enhanced)))
        # Both families are causal. Their frame shift is the conservative
        # algorithmic-delay figure used for integration planning.
        return timing(times, enhanced[: len(samples)], hop / 16_000 * 1000)

    return run


def make_ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def create_fastenhancer_runner(path: Path) -> Callable[[np.ndarray], EnhancementResult]:
    session = make_ort_session(path)
    hop = 256
    n_fft = 512

    def run(samples: np.ndarray) -> EnhancementResult:
        inputs = {
            item.name: np.zeros(item.shape, dtype=np.float32)
            for item in session.get_inputs()
            if item.name.startswith("cache_in_")
        }
        padded = np.pad(samples, (0, n_fft))
        output: list[np.ndarray] = []
        times: list[float] = []
        for offset in range(0, len(samples) + n_fft - hop, hop):
            inputs["wav_in"] = padded[None, offset : offset + hop]
            started = time.perf_counter()
            result = session.run(None, inputs)
            times.append(time.perf_counter() - started)
            output.append(result[0][0])
            for index, cache in enumerate(result[1:]):
                inputs[f"cache_in_{index}"] = cache
        enhanced = np.concatenate(output)
        delay = n_fft - hop
        enhanced = enhanced[delay : delay + len(samples)]
        return timing(times, enhanced, delay / 16_000 * 1000)

    return run


def create_ulunas_runner(path: Path) -> Callable[[np.ndarray], EnhancementResult]:
    session = make_ort_session(path)
    window = torch.hann_window(512)

    def run(samples: np.ndarray) -> EnhancementResult:
        source = torch.from_numpy(samples)[None]
        spectrum = torch.view_as_real(
            torch.stft(
                source,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=window,
                return_complex=True,
            )
        ).numpy()
        caches = {
            "conv_cache": np.zeros((1, 5358), dtype=np.float32),
            "tfa_cache": np.zeros((1, 402), dtype=np.float32),
            "inter_cache": np.zeros((1, 1056), dtype=np.float32),
        }
        output: list[np.ndarray] = []
        times: list[float] = []
        for index in range(spectrum.shape[2]):
            started = time.perf_counter()
            result = session.run(
                None,
                {"mix": spectrum[:, :, index : index + 1, :], **caches},
            )
            times.append(time.perf_counter() - started)
            output.append(result[0])
            caches = {
                "conv_cache": result[1],
                "tfa_cache": result[2],
                "inter_cache": result[3],
            }
        enhanced_spectrum = torch.from_numpy(np.concatenate(output, axis=2))
        complex_spectrum = torch.complex(
            enhanced_spectrum[..., 0], enhanced_spectrum[..., 1]
        )
        enhanced = torch.istft(
            complex_spectrum[0],
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=window,
            onesided=True,
            length=len(samples),
        ).numpy()
        return timing(times, enhanced, 16.0)

    return run


def create_runners(models_dir: Path) -> dict[str, Callable[[np.ndarray], EnhancementResult]]:
    definitions = {
        "gtcrn": (create_sherpa_runner, "gtcrn", models_dir / "gtcrn_simple.onnx"),
        "dpdfnet2": (
            create_sherpa_runner,
            "dpdfnet",
            models_dir / "dpdfnet2.onnx",
        ),
        "dpdfnet4": (
            create_sherpa_runner,
            "dpdfnet",
            models_dir / "dpdfnet4.onnx",
        ),
        "dpdfnet8": (
            create_sherpa_runner,
            "dpdfnet",
            models_dir / "dpdfnet8.onnx",
        ),
        "fastenhancer_b": (
            create_fastenhancer_runner,
            None,
            models_dir / "fastenhancer_b_dns.onnx",
        ),
        "fastenhancer_s": (
            create_fastenhancer_runner,
            None,
            models_dir / "fastenhancer_s_dns.onnx",
        ),
        "ulunas": (
            create_ulunas_runner,
            None,
            REPO_ROOT
            / "outputs/denoise/repos/ul-unas/ulunas_onnx/onnx_models/ulunas_stream_simple.onnx",
        ),
    }
    runners = {}
    for name, (factory, kind, path) in definitions.items():
        if not path.is_file():
            continue
        runners[name] = factory(kind, path) if kind else factory(path)
    return runners


def align_to_reference(reference: np.ndarray, estimate: np.ndarray, max_lag: int = 800) -> np.ndarray:
    probe = min(len(reference), len(estimate), 16_000 * 10)
    correlation = correlate(estimate[:probe], reference[:probe], mode="full", method="fft")
    lags = correlation_lags(probe, probe, mode="full")
    allowed = np.abs(lags) <= max_lag
    lag = int(lags[allowed][np.argmax(np.abs(correlation[allowed]))])
    if lag > 0:
        estimate = estimate[lag:]
    elif lag < 0:
        estimate = np.pad(estimate, (-lag, 0))
    if len(estimate) < len(reference):
        estimate = np.pad(estimate, (0, len(reference) - len(estimate)))
    return estimate[: len(reference)]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = align_to_reference(reference, estimate).astype(np.float64)
    reference -= reference.mean()
    estimate -= estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    error = estimate - target
    return float(10 * np.log10(np.dot(target, target) / (np.dot(error, error) + 1e-12)))


class SlmClient:
    def __init__(self, base_url: str, tools_path: Path):
        self.base_url = base_url.rstrip("/")
        self.tools = json.loads(tools_path.read_text(encoding="utf-8"))
        self.client = httpx.Client(timeout=180, trust_env=False)
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        self.model = response.json()["data"][0]["id"]

    def infer(self, wav_path: Path) -> tuple[list[dict], float]:
        encoded = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
                        }
                    ],
                }
            ],
            "tools": self.tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = []
        for item in message.get("tool_calls") or []:
            function = item["function"]
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append({"name": function["name"], "arguments": arguments})
        return calls, elapsed_ms


def exact_match(actual: list[dict], expected: list[dict]) -> bool:
    if len(actual) != len(expected):
        return False
    for got, wanted in zip(actual, expected):
        if got.get("name") != wanted.get("name"):
            return False
        if "arguments" in wanted and got.get("arguments") != wanted["arguments"]:
            return False
    return True


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads((args.samples_dir / "expected_outputs.json").read_text())
    selected_names = args.sample or [
        name
        for name, item in expected.items()
        if item["expected_tool_calls"][0]["name"] != "non_tool"
    ]
    runners = create_runners(args.models_dir)
    if args.denoiser:
        missing = sorted(set(args.denoiser) - set(runners))
        if missing:
            raise ValueError(f"requested denoisers are unavailable: {missing}")
        runners = {name: runners[name] for name in args.denoiser}
    if not runners:
        raise RuntimeError("no denoiser artifacts are available")

    noise = build_quiet_frame_noise(args.captures_dir)
    slm = None if args.skip_slm else SlmClient(args.vllm_url, args.tools)
    rows: list[dict] = []
    case_index = 0
    for sample_name in selected_names:
        clean = read_mono_16k(args.samples_dir / sample_name)
        wanted = expected[sample_name]["expected_tool_calls"]
        for snr_db in args.snr_db:
            ambient = repeat_to_length(noise, len(clean), case_index * 997)
            case_index += 1
            noisy = mix_at_snr(clean, ambient, snr_db)
            case_dir = args.output_dir / f"{Path(sample_name).stem}__snr_{snr_db:g}db"
            write_wav(case_dir / "clean.wav", clean)
            noisy_path = case_dir / "noisy.wav"
            write_wav(noisy_path, noisy)

            noisy_calls: list[dict] = []
            noisy_slm_ms = None
            if slm:
                noisy_calls, noisy_slm_ms = slm.infer(noisy_path)
            rows.append(
                {
                    "sample": sample_name,
                    "snr_db": snr_db,
                    "denoiser": "none",
                    "si_sdr_db": round(si_sdr(clean, noisy), 3),
                    "compute_ms": 0.0,
                    "rtf": 0.0,
                    "frame_ms_mean": 0.0,
                    "frame_ms_p99": 0.0,
                    "frame_ms_max": 0.0,
                    "algorithmic_delay_ms": 0.0,
                    "slm_ms": None if noisy_slm_ms is None else round(noisy_slm_ms, 3),
                    "calls": noisy_calls,
                    "exact_tool_match": None if slm is None else exact_match(noisy_calls, wanted),
                }
            )
            for denoiser_name, runner in runners.items():
                result = runner(noisy)
                output_path = case_dir / f"{denoiser_name}.wav"
                write_wav(output_path, result.samples)
                calls: list[dict] = []
                slm_ms = None
                if slm:
                    calls, slm_ms = slm.infer(output_path)
                duration_seconds = len(noisy) / 16_000
                rows.append(
                    {
                        "sample": sample_name,
                        "snr_db": snr_db,
                        "denoiser": denoiser_name,
                        "si_sdr_db": round(si_sdr(clean, result.samples), 3),
                        "compute_ms": round(result.compute_ms, 3),
                        "rtf": round(result.compute_ms / 1000 / duration_seconds, 5),
                        "frame_ms_mean": round(result.frame_ms_mean, 4),
                        "frame_ms_p99": round(result.frame_ms_p99, 4),
                        "frame_ms_max": round(result.frame_ms_max, 4),
                        "algorithmic_delay_ms": result.algorithmic_delay_ms,
                        "slm_ms": None if slm_ms is None else round(slm_ms, 3),
                        "calls": calls,
                        "exact_tool_match": None if slm is None else exact_match(calls, wanted),
                    }
                )
            print(f"finished {sample_name} at {snr_db:g} dB")

    aggregates = {}
    for name in ["none", *runners]:
        items = [row for row in rows if row["denoiser"] == name]
        aggregates[name] = {
            "cases": len(items),
            "exact_tool_accuracy": (
                None
                if slm is None
                else round(statistics.mean(row["exact_tool_match"] for row in items), 4)
            ),
            "si_sdr_db_mean": round(statistics.mean(row["si_sdr_db"] for row in items), 3),
            "rtf_mean": round(statistics.mean(row["rtf"] for row in items), 5),
            "frame_ms_p99_max": round(max(row["frame_ms_p99"] for row in items), 4),
            "algorithmic_delay_ms": max(row["algorithmic_delay_ms"] for row in items),
        }
    report = {
        "notes": [
            "Primary metric is exact downstream tool name and arguments.",
            "Ambient noise is synthesized from the quietest 35% of 20 ms frames in recent web captures.",
            "Record a dedicated noise-only room track before making a production decision.",
            "SI-SDR is time-aligned within +/-50 ms and is only a supporting signal metric.",
        ],
        "snr_db": args.snr_db,
        "samples": selected_names,
        "aggregates": aggregates,
        "rows": rows,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(aggregates, indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
