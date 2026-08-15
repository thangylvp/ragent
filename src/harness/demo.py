"""One-turn orchestration for the microphone-driven RAGENT demo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from execute import ExecutionStatus, SimulatedHardware

from .cloud import build_cloud_agent
from .responses import ResponseLibrary, ResponseTemplate
from .tts import SynthesizingAudioStore, build_static_audio, build_tts


@dataclass(slots=True)
class HarnessResult:
    route: str
    assistant_text: str
    transcript: str | None
    calls: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    audio: list[dict[str, Any]]
    hardware: dict[str, Any]
    timings: dict[str, Any]
    cloud_model: str | None = None
    errors: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "assistant_text": self.assistant_text,
            "transcript": self.transcript,
            "calls": self.calls,
            "executions": self.executions,
            "audio": self.audio,
            "hardware": self.hardware,
            "timings": self.timings,
            "cloud_model": self.cloud_model,
            "errors": self.errors or [],
        }


class DemoHarness:
    def __init__(
        self,
        executor,
        cloud,
        dynamic_tts,
        static_audio=None,
        responses=None,
    ):
        self.executor = executor
        self.cloud = cloud
        self.dynamic_tts = dynamic_tts
        # Kept only for lightweight unit-test construction. Production always
        # passes ManifestAudioStore explicitly in build_demo_harness().
        self.static_audio = static_audio or SynthesizingAudioStore(dynamic_tts)
        self.responses = responses or ResponseLibrary()

    def process(self, slm_result: dict[str, Any]) -> HarnessResult:
        started = time.perf_counter()
        route = slm_result.get("route") or "abstain"
        transcript = slm_result.get("transcript")
        raw_calls = slm_result.get("calls") or []
        calls = list(raw_calls) if isinstance(raw_calls, list) else []
        executions: list[dict[str, Any]] = []
        cloud_ms = 0.0
        execute_ms = 0.0
        cloud_model = None
        errors: list[str] = []
        static_response: ResponseTemplate | None = None

        if route == "non_tool" and transcript:
            try:
                reply = self.cloud.reply(transcript)
                assistant_text = reply.text
                cloud_ms = reply.latency_ms
                cloud_model = reply.model
                effective_route = "cloud"
            except Exception as exc:
                errors.append(f"cloud: {type(exc).__name__}: {exc}")
                static_response = self.responses.cloud_unavailable
                assistant_text = static_response.text
                effective_route = "cloud_error"
        elif route == "tool" and calls:
            if len(calls) != 1:
                static_response = self.responses.multiple_calls
                assistant_text = static_response.text
                effective_route = "multiple_tool_calls"
            else:
                execute_started = time.perf_counter()
                result = self.executor.execute(calls[0])
                execute_ms = (time.perf_counter() - execute_started) * 1000
                executions = [result.as_dict()]
                if result.status is ExecutionStatus.BUSY:
                    static_response = self.responses.busy(result.name)
                    effective_route = "busy"
                elif result.status is ExecutionStatus.MISSING_REQUIRED:
                    static_response = self.responses.missing(result.name, result.missing)
                    effective_route = "missing_required"
                elif result.status is ExecutionStatus.REJECTED:
                    static_response = self.responses.reject(result.name, result.error)
                    effective_route = "rejected"
                else:
                    static_response = self.responses.success(result.name)
                    effective_route = "executed"
                assistant_text = static_response.text
        else:
            static_response = (
                self.responses.repeat if route in {"abstain", None} else self.responses.invalid_call
            )
            assistant_text = static_response.text
            effective_route = "repeat" if static_response is self.responses.repeat else "invalid_call"

        audio_started = time.perf_counter()
        synthesis_ms = 0.0
        audio_mode = "static_cache" if static_response else "dynamic_cloud_tts"
        try:
            clip = (
                self.static_audio.resolve(static_response)
                if static_response
                else self.dynamic_tts.synthesize(assistant_text)
            )
            audio = [clip.as_dict()]
            synthesis_ms = clip.synthesis_ms
        except Exception as exc:
            error_prefix = "static_audio" if static_response else "tts"
            errors.append(f"{error_prefix}: {type(exc).__name__}: {exc}")
            audio = []
            if static_response is None:
                # A free-form cloud answer cannot be pre-rendered. If cloud TTS
                # fails, play one fixed local explanation instead.
                try:
                    fallback = self.static_audio.resolve(self.responses.tts_unavailable)
                    audio = [fallback.as_dict()]
                    audio_mode = "static_tts_error_fallback"
                except Exception as fallback_exc:
                    errors.append(
                        "static_audio: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )
        tts_ms = (time.perf_counter() - audio_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        return HarnessResult(
            route=effective_route,
            assistant_text=assistant_text,
            transcript=transcript,
            calls=calls,
            executions=executions,
            audio=audio,
            hardware=self.executor.snapshot(),
            timings={
                "cloud_ms": round(cloud_ms, 3),
                "execute_ms": round(max(0.0, execute_ms), 3),
                "tts_ms": round(tts_ms, 3),
                "tts_synthesis_ms": round(synthesis_ms, 3),
                "audio_mode": audio_mode,
                "harness_total_ms": round(total_ms, 3),
            },
            cloud_model=cloud_model,
            errors=errors,
        )

    def reset(self) -> dict[str, Any]:
        self.cloud.reset()
        return self.executor.reset()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "cloud": self.cloud.info,
            "tts": self.dynamic_tts.info,
            "static_audio": self.static_audio.info,
            "hardware": self.executor.snapshot(),
        }


def build_demo_harness(settings, tools: list[dict[str, Any]]) -> DemoHarness:
    return DemoHarness(
        executor=SimulatedHardware(tools),
        cloud=build_cloud_agent(settings),
        dynamic_tts=build_tts(settings),
        static_audio=build_static_audio(settings),
    )
