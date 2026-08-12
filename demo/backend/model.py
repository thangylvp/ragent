"""Car CommandASR inference adapters used after VAD finalizes an utterance."""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from typing import Protocol

from slm.modeling.qwen3_tool_calls import parse, render_tool_calls

_AUDIO_BLOCK = "<|audio_start|><|audio_pad|><|audio_end|>"


def load_tools(model_dir: str | Path) -> list[dict]:
    path = Path(model_dir) / "tools_openai.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"car tool catalog is missing: {path}; point WEBTEST_MODEL_DIR at the STC export"
        )
    tools = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"invalid tool catalog: {path}")
    return tools


def summarize_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
        }
        for item in tools
    ]


def _result(
    raw: str,
    calls: list[dict],
    latency_ms: float,
    output_tokens=None,
    timings: dict | None = None,
) -> dict:
    names = [call.get("name") for call in calls]
    if not calls:
        route = "abstain"
    elif "non_tool" in names:
        route = "non_tool" if len(calls) == 1 else "invalid_mixed"
    else:
        route = "tool"
    transcript = None
    if route == "non_tool":
        value = (calls[0].get("arguments") or {}).get("text")
        transcript = value if isinstance(value, str) else None
    return {
        "route": route,
        "calls": calls,
        "transcript": transcript,
        "raw": raw,
        "latency_ms": round(latency_ms, 1),
        "output_tokens": output_tokens,
        "timings": timings or {},
    }


class SpeechModel(Protocol):
    def infer(self, wav_path: str | Path, tools: list[dict]) -> dict: ...

    @property
    def info(self) -> dict: ...


class DisabledModel:
    @property
    def info(self) -> dict:
        return {"kind": "disabled", "ready": True}

    def infer(self, wav_path: str | Path, tools: list[dict]) -> dict:
        return _result("", [], 0.0, timings={"to_last_token_ms": 0.0})


class VllmModel:
    def __init__(self, settings):
        import httpx

        self.settings = settings
        headers = (
            {"Authorization": f"Bearer {settings.vllm_api_key}"}
            if settings.vllm_api_key
            else {}
        )
        self.client = httpx.Client(
            timeout=settings.vllm_timeout_s,
            trust_env=False,
            headers=headers,
        )
        self._ready = False
        self._error: str | None = None
        self._last_latency_ms: float | None = None
        self.probe()

    def probe(self) -> None:
        try:
            response = self.client.get(f"{self.settings.vllm_base_url}/models", timeout=3)
            response.raise_for_status()
            self._ready = True
            self._error = None
        except Exception as exc:
            self._ready = False
            self._error = f"{type(exc).__name__}: {exc}"

    def infer(self, wav_path: str | Path, tools: list[dict]) -> dict:
        encoded = base64.b64encode(Path(wav_path).read_bytes()).decode("ascii")
        payload = {
            "model": self.settings.vllm_model,
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
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.settings.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        try:
            response = self.client.post(
                f"{self.settings.vllm_base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            calls = []
            for item in message.get("tool_calls") or []:
                function = item.get("function") or {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({"name": function["name"], "arguments": arguments})
            raw = render_tool_calls(calls) if calls else (message.get("content") or "")
            latency_ms = (time.perf_counter() - started) * 1000
            self._last_latency_ms = latency_ms
            self._ready = True
            self._error = None
            usage = body.get("usage") or {}
            return _result(
                raw,
                calls,
                latency_ms,
                usage.get("completion_tokens"),
                timings={
                    "remote_request_to_last_token_ms": round(latency_ms, 3),
                    "to_last_token_ms": round(latency_ms, 3),
                },
            )
        except Exception as exc:
            self._ready = False
            self._error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def info(self) -> dict:
        return {
            "kind": "vllm",
            "ready": self._ready,
            "model": self.settings.vllm_model,
            "base_url": self.settings.vllm_base_url,
            "latency_ms": self._last_latency_ms,
            "error": self._error,
        }


class LocalCommandAsrModel:
    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._torch = None
        self._device: str | None = None
        self._eos_token_id: int | None = None
        self._error: str | None = None
        self._last_latency_ms: float | None = None
        if settings.eager_model:
            self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        from slm.modeling.command_asr import CommandASR
        from slm.modeling.qwen3_tool_calls import assistant_eos_token_id

        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not self.settings.allow_cpu_model:
            raise RuntimeError(
                "CUDA is unavailable. Set WEBTEST_ALLOW_CPU_MODEL=1 to permit very slow CPU "
                "generation, or use WEBTEST_MODEL_MODE=vllm."
            )
        dtype = getattr(torch, self.settings.dtype)
        model, tokenizer, processor = CommandASR._load_dir(
            self.settings.model_dir, dtype
        )
        model.eval().to(device)
        self._model = model
        self._tokenizer = tokenizer
        self._processor = processor
        self._torch = torch
        self._device = device
        self._eos_token_id = assistant_eos_token_id(tokenizer)

    def infer(self, wav_path: str | Path, tools: list[dict]) -> dict:
        with self._lock:
            adapter_started = time.perf_counter()
            try:
                load_started = time.perf_counter()
                self._load()
                load_ms = (time.perf_counter() - load_started) * 1000
                result = self._infer_loaded(wav_path, tools)
                timings = result["timings"]
                timings["load_ms"] = round(load_ms, 3)
                timings["adapter_total_ms"] = round(
                    (time.perf_counter() - adapter_started) * 1000,
                    3,
                )
                timings["to_last_token_ms"] = round(
                    load_ms + timings["loaded_to_last_token_ms"],
                    3,
                )
                return result
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                raise

    def _infer_loaded(self, wav_path: str | Path, tools: list[dict]) -> dict:
        import soundfile as sf

        assert self._model is not None
        assert self._eos_token_id is not None
        torch = self._torch
        tokenizer = self._tokenizer
        loaded_started = time.perf_counter()
        audio_started = time.perf_counter()
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        audio_decode_ms = (time.perf_counter() - audio_started) * 1000
        if sample_rate != 16_000 or audio.ndim != 1:
            raise ValueError("captured model audio must be mono 16 kHz")
        prompt_started = time.perf_counter()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": _AUDIO_BLOCK}],
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompt_ms = (time.perf_counter() - prompt_started) * 1000
        feature_started = time.perf_counter()
        encoded = self._processor(
            text=prompt,
            audio=audio,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        feature_extraction_ms = (time.perf_counter() - feature_started) * 1000
        device = self._device
        transfer_started = time.perf_counter()
        input_ids = encoded["input_ids"].to(device)
        features = encoded["input_features"].to(device).to(
            self._model.thinker.audio_tower.proj2.weight.dtype
        )
        attention_mask = encoded["attention_mask"].to(device)
        feature_attention_mask = encoded["feature_attention_mask"].to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        transfer_ms = (time.perf_counter() - transfer_started) * 1000
        eos = self._eos_token_id
        generation_started = time.perf_counter()

        class _FirstTokenTimer:
            def __init__(self):
                self.elapsed_ms: float | None = None

            def __call__(self, current_input_ids, scores):
                if self.elapsed_ms is None:
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    self.elapsed_ms = (time.perf_counter() - generation_started) * 1000
                return scores

        first_token_timer = _FirstTokenTimer()
        with torch.no_grad():
            output = self._model.thinker.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=(tokenizer.pad_token_id or eos),
                logits_processor=[first_token_timer],
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000
        first_token_ms = first_token_timer.elapsed_ms
        loaded_to_last_token_ms = (time.perf_counter() - loaded_started) * 1000
        generated = output[0][input_ids.shape[1] :]
        decode_started = time.perf_counter()
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        calls = parse(raw)
        decode_parse_ms = (time.perf_counter() - decode_started) * 1000
        loaded_total_ms = (time.perf_counter() - loaded_started) * 1000
        self._last_latency_ms = generation_ms
        self._error = None
        return _result(
            raw,
            calls,
            generation_ms,
            int(generated.numel()),
            timings={
                "audio_decode_ms": round(audio_decode_ms, 3),
                "prompt_render_ms": round(prompt_ms, 3),
                "feature_extraction_ms": round(feature_extraction_ms, 3),
                "host_to_device_ms": round(transfer_ms, 3),
                "generation_to_first_token_ms": round(first_token_ms, 3)
                if first_token_ms is not None
                else None,
                "first_to_last_token_ms": round(generation_ms - first_token_ms, 3)
                if first_token_ms is not None
                else None,
                "generation_to_last_token_ms": round(generation_ms, 3),
                "decode_parse_ms": round(decode_parse_ms, 3),
                "loaded_to_last_token_ms": round(loaded_to_last_token_ms, 3),
                "loaded_total_ms": round(loaded_total_ms, 3),
            },
        )

    @property
    def info(self) -> dict:
        return {
            "kind": "local",
            "ready": self._model is not None,
            "model_dir": self.settings.model_dir,
            "device": self._device or self.settings.device,
            "latency_ms": self._last_latency_ms,
            "error": self._error,
        }


def build_model(settings) -> SpeechModel:
    mode = settings.model_mode.strip().lower()
    if mode == "local":
        return LocalCommandAsrModel(settings)
    if mode == "vllm":
        return VllmModel(settings)
    if mode == "disabled":
        return DisabledModel()
    raise ValueError("WEBTEST_MODEL_MODE must be local, vllm, or disabled")
