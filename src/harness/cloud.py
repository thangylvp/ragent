"""Small cloud-agent adapter for non-tool conversation turns."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý giọng nói thân thiện trên một robot. Luôn trả lời "
    "bằng tiếng Việt tự nhiên, ngắn gọn và dễ nghe. Không dùng markdown. "
    "Câu trả lời thường chỉ nên dài một hoặc hai câu."
)


@dataclass(frozen=True, slots=True)
class CloudReply:
    text: str
    latency_ms: float
    model: str


class DisabledCloudAgent:
    """Useful offline fallback: the demo still exposes routing and execution."""

    enabled = False

    def reply(self, text: str) -> CloudReply:
        raise RuntimeError("cloud agent is not configured")

    def reset(self) -> None:
        return None

    @property
    def info(self) -> dict[str, Any]:
        return {"enabled": False, "ready": False, "model": "disabled"}


class OpenAICloudAgent:
    """Short Vietnamese conversational fallback using the Responses API."""

    enabled = True

    def __init__(self, settings):
        import httpx

        self.model = settings.cloud_model
        self._url = f"{settings.openai_base_url}/responses"
        self._timeout = settings.cloud_timeout_s
        self._max_turns = settings.cloud_history_turns
        self._history: list[dict[str, str]] = []
        self._lock = threading.RLock()
        self._ready: bool | None = None
        self._error: str | None = None
        self._terminal_error = False
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=self._timeout,
            trust_env=True,
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        if isinstance(body.get("output_text"), str) and body["output_text"].strip():
            return body["output_text"].strip()
        parts: list[str] = []
        for item in body.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    value = content.get("text")
                    if isinstance(value, str):
                        parts.append(value)
        return "".join(parts).strip()

    def reply(self, text: str) -> CloudReply:
        with self._lock:
            if self._terminal_error:
                raise RuntimeError(self._error or "cloud authentication is unavailable")
            started = time.perf_counter()
            messages = [*self._history, {"role": "user", "content": text}]
            try:
                response = self._client.post(
                    self._url,
                    json={
                        "model": self.model,
                        "instructions": SYSTEM_INSTRUCTION,
                        "input": messages,
                        "reasoning": {"effort": "none"},
                        "max_output_tokens": 220,
                    },
                )
                response.raise_for_status()
                self._ready = True
                self._error = None
            except Exception as exc:
                self._ready = False
                self._error = f"{type(exc).__name__}: {exc}"
                response = getattr(exc, "response", None)
                self._terminal_error = getattr(response, "status_code", None) in {401, 403}
                raise
            answer = self._extract_text(response.json())
            if not answer:
                raise RuntimeError("cloud agent returned no text")
            self._history.extend(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._history = self._history[-self._max_turns * 2 :]
            return CloudReply(
                text=answer,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                model=self.model,
            )

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "ready": self._ready,
            "provider": "openai",
            "model": self.model,
            "error": self._error,
            "terminal_error": self._terminal_error,
        }


class GeminiCloudAgent:
    """Short Vietnamese conversational fallback using Gemini GenerateContent."""

    enabled = True
    _THINKING_LEVELS = {"minimal", "low", "medium", "high"}

    def __init__(self, settings):
        import httpx

        self.model = settings.cloud_model
        self.thinking_level = settings.gemini_thinking_level
        if self.thinking_level not in self._THINKING_LEVELS:
            raise ValueError(
                "DEMO_GEMINI_THINKING_LEVEL must be minimal, low, medium, or high"
            )
        model_path = quote(self.model, safe="-._")
        self._url = f"{settings.gemini_base_url}/models/{model_path}:generateContent"
        self._max_turns = settings.cloud_history_turns
        self._history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._ready: bool | None = None
        self._error: str | None = None
        self._terminal_error = False
        self._client = httpx.Client(
            headers={"x-goog-api-key": settings.gemini_api_key},
            timeout=settings.cloud_timeout_s,
            trust_env=True,
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        parts: list[str] = []
        for candidate in body.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                value = part.get("text")
                if isinstance(value, str) and not part.get("thought"):
                    parts.append(value)
            if parts:
                break
        return "".join(parts).strip()

    def reply(self, text: str) -> CloudReply:
        with self._lock:
            if self._terminal_error:
                raise RuntimeError(self._error or "Gemini authentication is unavailable")
            started = time.perf_counter()
            contents = [
                *self._history,
                {"role": "user", "parts": [{"text": text}]},
            ]
            try:
                response = self._client.post(
                    self._url,
                    json={
                        "systemInstruction": {
                            "parts": [{"text": SYSTEM_INSTRUCTION}],
                        },
                        "contents": contents,
                        "generationConfig": {
                            # Gemini 3 thinking tokens share this ceiling with
                            # the visible answer. The system prompt, not a tiny
                            # token cap, keeps spoken replies to 1–2 sentences.
                            "maxOutputTokens": 1024,
                            "thinkingConfig": {
                                "thinkingLevel": self.thinking_level,
                            },
                        },
                    },
                )
                response.raise_for_status()
                body = response.json()
                self._ready = True
                self._error = None
            except Exception as exc:
                self._ready = False
                self._error = f"{type(exc).__name__}: {exc}"
                failed_response = getattr(exc, "response", None)
                self._terminal_error = getattr(failed_response, "status_code", None) in {
                    400,
                    401,
                    403,
                }
                raise
            answer = self._extract_text(body)
            if not answer:
                feedback = body.get("promptFeedback") or {}
                block_reason = feedback.get("blockReason")
                finish_reasons = [
                    item.get("finishReason")
                    for item in body.get("candidates") or []
                    if item.get("finishReason")
                ]
                detail = block_reason or ",".join(finish_reasons) or "empty response"
                raise RuntimeError(f"Gemini returned no answer: {detail}")
            self._history.extend(
                [
                    {"role": "user", "parts": [{"text": text}]},
                    {"role": "model", "parts": [{"text": answer}]},
                ]
            )
            self._history = self._history[-self._max_turns * 2 :]
            return CloudReply(
                text=answer,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                model=self.model,
            )

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "ready": self._ready,
            "provider": "gemini",
            "model": self.model,
            "thinking_level": self.thinking_level,
            "error": self._error,
            "terminal_error": self._terminal_error,
        }


def build_cloud_agent(settings):
    if not settings.cloud_enabled:
        return DisabledCloudAgent()
    provider = settings.cloud_provider.strip().lower()
    if provider == "gemini":
        if not settings.gemini_api_key:
            return DisabledCloudAgent()
        return GeminiCloudAgent(settings)
    if provider == "openai":
        if not settings.openai_api_key:
            return DisabledCloudAgent()
        return OpenAICloudAgent(settings)
    raise ValueError("DEMO_CLOUD_PROVIDER must be gemini or openai")
