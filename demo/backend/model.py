"""Standalone STCC vLLM adapter used after VAD finalizes an utterance."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Protocol

from harness.tool_calls import render_tool_calls


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
        return _result(
            "",
            [],
            0.0,
            timings={
                "request_to_first_token_ms": 0.0,
                "request_to_last_token_ms": 0.0,
                "to_last_token_ms": 0.0,
            },
        )


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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        try:
            content_parts: list[str] = []
            streamed_calls: dict[int, dict[str, str]] = {}
            first_token_ms: float | None = None
            last_token_ms: float | None = None
            output_tokens = None
            with self.client.stream(
                "POST",
                f"{self.settings.vllm_base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    body = json.loads(data)
                    usage = body.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_tokens = usage["completion_tokens"]
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    meaningful = False
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        meaningful = True
                    for item in delta.get("tool_calls") or []:
                        index = int(item.get("index", 0))
                        target = streamed_calls.setdefault(
                            index,
                            {"name": "", "arguments": ""},
                        )
                        function = item.get("function") or {}
                        name_fragment = function.get("name") or ""
                        argument_fragment = function.get("arguments") or ""
                        target["name"] += name_fragment
                        target["arguments"] += argument_fragment
                        meaningful = meaningful or bool(name_fragment or argument_fragment)
                    if meaningful:
                        elapsed = (time.perf_counter() - started) * 1000
                        if first_token_ms is None:
                            first_token_ms = elapsed
                        last_token_ms = elapsed

            calls = []
            for index in sorted(streamed_calls):
                item = streamed_calls[index]
                arguments_text = item["arguments"] or "{}"
                try:
                    arguments = json.loads(arguments_text)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments_text}
                calls.append({"name": item["name"], "arguments": arguments})
            raw = render_tool_calls(calls) if calls else "".join(content_parts)
            latency_ms = (time.perf_counter() - started) * 1000
            first_token_ms = first_token_ms if first_token_ms is not None else latency_ms
            last_token_ms = last_token_ms if last_token_ms is not None else latency_ms
            self._last_latency_ms = latency_ms
            self._ready = True
            self._error = None
            return _result(
                raw,
                calls,
                latency_ms,
                output_tokens,
                timings={
                    "request_to_first_token_ms": round(first_token_ms, 3),
                    "request_to_last_token_ms": round(last_token_ms, 3),
                    "first_to_last_token_ms": round(last_token_ms - first_token_ms, 3),
                    "adapter_total_ms": round(latency_ms, 3),
                    "to_last_token_ms": round(last_token_ms, 3),
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


def build_model(settings) -> SpeechModel:
    mode = settings.model_mode.strip().lower()
    if mode == "vllm":
        return VllmModel(settings)
    if mode == "disabled":
        return DisabledModel()
    raise ValueError("WEBTEST_MODEL_MODE must be vllm or disabled")
