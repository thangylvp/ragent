"""Streaming PCM16 speech enhancement adapters.

Enhancement is intentionally a separate component from VAD. It transforms the
microphone signal; VAD then decides which transformed frames form an utterance.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class EnhancementUpdate:
    pcm16le: bytes
    frames_processed: int
    compute_ms: float


class LiveSpeechEnhancer(Protocol):
    name: str
    sample_rate: int
    frame_samples: int
    algorithmic_delay_ms: float
    frames_processed: int
    compute_total_ms: float
    compute_max_ms: float

    def process(self, pcm16le: bytes) -> EnhancementUpdate: ...

    def reset(self) -> None: ...


class PassthroughEnhancer:
    name = "none"
    sample_rate = 16_000
    frame_samples = 1
    algorithmic_delay_ms = 0.0

    def __init__(self):
        self.frames_processed = 0
        self.compute_total_ms = 0.0
        self.compute_max_ms = 0.0

    def process(self, pcm16le: bytes) -> EnhancementUpdate:
        return EnhancementUpdate(bytes(pcm16le), 0, 0.0)

    def reset(self) -> None:
        self.frames_processed = 0
        self.compute_total_ms = 0.0
        self.compute_max_ms = 0.0


class FastEnhancerOnnx:
    """Official FastEnhancer waveform-to-waveform streaming ONNX runtime."""

    sample_rate = 16_000
    frame_samples = 256
    algorithmic_delay_ms = 16.0

    def __init__(self, name: str, model_path: str | Path):
        import onnxruntime as ort

        self.name = name
        self.model_path = str(model_path)
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._cache_shapes = {
            item.name: tuple(int(value) for value in item.shape)
            for item in self._session.get_inputs()
            if item.name.startswith("cache_in_")
        }
        self._pending = bytearray()
        self._caches: dict[str, np.ndarray] = {}
        self.frames_processed = 0
        self.compute_total_ms = 0.0
        self.compute_max_ms = 0.0
        self.reset()

    def process(self, pcm16le: bytes) -> EnhancementUpdate:
        if len(pcm16le) % 2:
            raise ValueError("PCM16LE input has an odd byte count")
        self._pending.extend(pcm16le)
        frame_bytes = self.frame_samples * 2
        output = bytearray()
        processed = 0
        compute_ms = 0.0
        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
            inputs = {
                "wav_in": (samples / 32768.0)[None],
                **self._caches,
            }
            started = time.perf_counter()
            result = self._session.run(None, inputs)
            elapsed_ms = (time.perf_counter() - started) * 1000
            compute_ms += elapsed_ms
            self.compute_total_ms += elapsed_ms
            self.compute_max_ms = max(self.compute_max_ms, elapsed_ms)
            self.frames_processed += 1
            processed += 1
            enhanced = np.clip(result[0][0], -1.0, 1.0)
            output.extend(np.rint(enhanced * 32767.0).astype("<i2").tobytes())
            for index, cache in enumerate(result[1:]):
                self._caches[f"cache_in_{index}"] = cache
        return EnhancementUpdate(bytes(output), processed, compute_ms)

    def reset(self) -> None:
        self._pending.clear()
        self._caches = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self._cache_shapes.items()
        }
        self.frames_processed = 0
        self.compute_total_ms = 0.0
        self.compute_max_ms = 0.0


def live_enhancer_catalog(fastenhancer_s_model: str | Path) -> list[dict]:
    model_path = Path(fastenhancer_s_model)
    ort_available = importlib.util.find_spec("onnxruntime") is not None
    available = ort_available and model_path.is_file()
    if not ort_available:
        reason = "onnxruntime is not installed"
    elif not model_path.is_file():
        reason = f"model is missing: {model_path}"
    else:
        reason = None
    return [
        {
            "name": "fastenhancer_s",
            "available": available,
            "description": "FastEnhancer-S, causal 16 kHz DNS model (ICASSP 2026)",
            "algorithmic_delay_ms": 16.0,
            "reason": reason,
        },
        {
            "name": "none",
            "available": True,
            "description": "No server-side enhancement (A/B baseline)",
            "algorithmic_delay_ms": 0.0,
            "reason": None,
        },
    ]


def create_live_enhancer(
    name: str,
    *,
    fastenhancer_s_model: str | Path,
) -> LiveSpeechEnhancer:
    normalized = name.strip().lower()
    if normalized == "none":
        return PassthroughEnhancer()
    if normalized == "fastenhancer_s":
        catalog = {item["name"]: item for item in live_enhancer_catalog(fastenhancer_s_model)}
        item = catalog[normalized]
        if not item["available"]:
            raise RuntimeError(item["reason"])
        return FastEnhancerOnnx(normalized, fastenhancer_s_model)
    raise ValueError(f"unknown speech enhancer: {name}")
