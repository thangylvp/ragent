from __future__ import annotations

import json
from pathlib import Path

from execute import SimulatedHardware
from harness import CloudReply, DemoHarness, ResponseLibrary, ToneTTS


class StubCloud:
    enabled = True

    def reply(self, text: str) -> CloudReply:
        return CloudReply(f"Tôi đã nghe: {text}", 12.0, "stub-cloud")

    def reset(self) -> None:
        return None

    @property
    def info(self):
        return {"enabled": True, "model": "stub-cloud"}


class FailingCloud(StubCloud):
    def reply(self, text: str) -> CloudReply:
        raise RuntimeError("invalid demo credential")


class TrackingTTS:
    enabled = True

    def __init__(self, root):
        self.inner = ToneTTS(root)
        self.calls: list[str] = []

    def synthesize(self, text):
        self.calls.append(text)
        return self.inner.synthesize(text)

    @property
    def info(self):
        return {"enabled": True, "model": "tracking-tts"}


class FailingTTS(TrackingTTS):
    def synthesize(self, text):
        self.calls.append(text)
        raise RuntimeError("cloud TTS unavailable")


class TrackingStaticAudio:
    def __init__(self, root):
        self.inner = ToneTTS(root)
        self.keys: list[str] = []

    def resolve(self, template):
        self.keys.append(template.key)
        return self.inner.synthesize(template.text)

    @property
    def info(self):
        return {"ready": True, "mode": "test-static"}


def _harness(tmp_path, *, cloud=None, dynamic_tts=None):
    dynamic = dynamic_tts or TrackingTTS(tmp_path / "dynamic")
    static = TrackingStaticAudio(tmp_path / "static")
    return (
        DemoHarness(
            SimulatedHardware(_tools()),
            cloud or StubCloud(),
            dynamic,
            static,
        ),
        dynamic,
        static,
    )


def _tools() -> list[dict]:
    path = (
        Path(__file__).resolve().parents[3]
        / "stc/outputs/models/route_v1_best_step1250_hf/tools_openai.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_non_tool_routes_to_cloud_and_tts(tmp_path):
    harness, dynamic, static = _harness(tmp_path)
    result = harness.process(
        {
            "route": "non_tool",
            "transcript": "Hôm nay là thứ mấy?",
            "calls": [{"name": "non_tool", "arguments": {"text": "Hôm nay là thứ mấy?"}}],
        }
    )
    assert result.route == "cloud"
    assert result.cloud_model == "stub-cloud"
    assert result.audio[0]["url"].endswith(".wav")
    assert result.hardware["action_count"] == 0
    assert dynamic.calls == [result.assistant_text]
    assert static.keys == []


def test_tool_success_missing_and_busy_responses(tmp_path):
    harness, dynamic, static = _harness(tmp_path)
    success = harness.process(
        {
            "route": "tool",
            "calls": [{"name": "control_trunk", "arguments": {"action": "open"}}],
        }
    )
    assert success.route == "executed"
    assert success.hardware["access"]["trunk"] == "open"
    assert "cốp" in success.assistant_text
    assert static.keys[-1] == "success.control_trunk"

    missing = harness.process(
        {"route": "tool", "calls": [{"name": "play_media", "arguments": {}}]}
    )
    assert missing.route == "missing_required"
    assert "nội dung nào" in missing.assistant_text
    assert static.keys[-1] == "missing.play_media"

    harness.executor.set_busy(True)
    busy = harness.process(
        {
            "route": "tool",
            "calls": [{"name": "control_trunk", "arguments": {"action": "close"}}],
        }
    )
    assert busy.route == "busy"
    assert "đang bận" in busy.assistant_text
    assert static.keys[-1] == "busy.control_trunk"
    assert dynamic.calls == []


def test_cloud_failure_uses_cached_server_audio(tmp_path):
    harness, dynamic, static = _harness(tmp_path, cloud=FailingCloud())
    result = harness.process(
        {
            "route": "non_tool",
            "transcript": "Xin chào",
            "calls": [{"name": "non_tool", "arguments": {"text": "Xin chào"}}],
        }
    )
    assert result.route == "cloud_error"
    assert result.audio and result.audio[0]["url"].endswith(".wav")
    assert result.errors and result.errors[0].startswith("cloud:")
    assert static.keys == ["system.cloud_unavailable"]
    assert dynamic.calls == []


def test_reject_uses_detailed_cached_tool_response(tmp_path):
    harness, dynamic, static = _harness(tmp_path)
    result = harness.process(
        {
            "route": "tool",
            "calls": [{"name": "set_temperature", "arguments": {"value": 12}}],
        }
    )
    assert result.route == "rejected"
    assert "16 đến 30 độ C" in result.assistant_text
    assert static.keys == ["reject.set_temperature"]
    assert dynamic.calls == []
    assert result.hardware["action_count"] == 0


def test_multiple_calls_are_rejected_before_any_hardware_mutation(tmp_path):
    harness, dynamic, static = _harness(tmp_path)
    result = harness.process(
        {
            "route": "tool",
            "calls": [
                {"name": "set_fog_lights", "arguments": {"state": "on"}},
                {"name": "play_media", "arguments": {}},
            ],
        }
    )
    assert result.route == "multiple_tool_calls"
    assert result.executions == []
    assert result.hardware["action_count"] == 0
    assert result.hardware["lighting"]["fog"] == "off"
    assert static.keys == ["system.multiple_tool_calls"]
    assert dynamic.calls == []


def test_dynamic_tts_failure_plays_fixed_local_fallback(tmp_path):
    failing = FailingTTS(tmp_path / "dynamic")
    harness, _, static = _harness(tmp_path, dynamic_tts=failing)
    result = harness.process(
        {
            "route": "non_tool",
            "transcript": "Hôm nay là thứ mấy?",
            "calls": [],
        }
    )
    assert result.route == "cloud"
    assert result.audio
    assert static.keys == ["system.cloud_tts_unavailable"]
    assert result.timings["audio_mode"] == "static_tts_error_fallback"
    assert result.errors[0].startswith("tts:")


def test_all_execution_templates_have_stable_unique_keys():
    names = [item["function"]["name"] for item in _tools()]
    templates = ResponseLibrary().all_templates(names)
    keys = [template.key for template in templates]
    assert len(keys) == 135
    assert len(keys) == len(set(keys))
    for name in names:
        if name == "non_tool":
            continue
        assert f"success.{name}" in keys
        assert f"missing.{name}" in keys
        assert f"busy.{name}" in keys
        assert f"reject.{name}" in keys
    assert "reject.control_trunk.vehicle_not_stopped" in keys
    assert "missing.set_temperature" in keys
