"""Deterministic WebSocket smoke test; no microphone, GPU, or model server."""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from demo.backend import app as app_module
from demo.backend.model import DisabledModel
from execute import SimulatedHardware
from harness import DemoHarness, DisabledCloudAgent, ToneTTS


def _audio_frames() -> list[bytes]:
    frames = []
    sample_index = 0
    for kind, count in (("silence", 25), ("tone", 60), ("silence", 40)):
        for _ in range(count):
            values = []
            for _ in range(320):
                value = (
                    0
                    if kind == "silence"
                    else int(10_000 * math.sin(2 * math.pi * 220 * sample_index / 16_000))
                )
                values.append(value)
                sample_index += 1
            frames.append(struct.pack("<320h", *values))
    return frames


class WebtestSmokeTest(unittest.TestCase):
    def test_frontend_keeps_observability_inside_each_ai_turn(self):
        frontend = Path(app_module.SETTINGS.frontend_dir)
        html = (frontend / "index.html").read_text(encoding="utf-8")
        javascript = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertNotIn('class="latency-strip"', html)
        self.assertIn("SLM RAW OUTPUT", javascript)
        self.assertIn('role === "slm"', javascript)
        self.assertIn('role === "cloud"', javascript)
        self.assertIn('addTurn("cloud", cloudText, cloudMeta)', javascript)
        self.assertIn("PARSED TOOL CALL", javascript)
        self.assertIn("data-turn-first-audio", javascript)
        self.assertIn("Audio → cloud LLM", javascript)
        self.assertIn("audio_to_cloud_llm_ms", javascript)
        self.assertIn("turnDiagnostics(message.turn_id, response)", javascript)
        self.assertIn("attachInputAudio(turn, message.capture_url", javascript)
        self.assertIn("Phát lại audio đầu vào của lượt này", javascript)
        self.assertIn('id="vehicleToggle"', html)
        self.assertNotIn("speechSynthesis", javascript)

    def test_energy_vad_finalizes_before_disabled_model_boundary(self):
        with tempfile.TemporaryDirectory(prefix="vad_webtest_") as directory:
            root = Path(directory)
            harness = DemoHarness(
                SimulatedHardware(app_module.TOOLS),
                DisabledCloudAgent(),
                ToneTTS(root / "voice"),
            )
            with (
                patch.object(app_module, "MODEL", DisabledModel()),
                patch.object(app_module, "HARNESS", harness),
                patch.object(app_module, "CAPTURE_DIR", root),
                patch.object(app_module, "VOICE_DIR", root / "voice"),
            ):
                client = TestClient(app_module.app)
                response = client.get("/api/config")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["tools"]), 33)

                running = client.post(
                    "/api/hardware/vehicle?running=true&speed_kph=5"
                )
                self.assertEqual(running.status_code, 200)
                self.assertTrue(running.json()["vehicle"]["running"])
                blocked = harness.process(
                    {
                        "route": "tool",
                        "calls": [
                            {
                                "name": "control_trunk",
                                "arguments": {"action": "open"},
                            }
                        ],
                    }
                )
                self.assertEqual(blocked.route, "rejected")
                self.assertEqual(blocked.hardware["access"]["trunk"], "closed")
                client.post("/api/hardware/reset")

                with client.websocket_connect("/api/audio/stream") as websocket:
                    websocket.send_json(
                        {
                            "event": "start_stream",
                            "backend": "energy",
                            "enhancer": "none",
                            "sample_rate": 16_000,
                        }
                    )
                    for frame in _audio_frames():
                        websocket.send_bytes(frame)

                    events = []
                    while not any(item["event"] == "assistant_response" for item in events):
                        events.append(websocket.receive_json())
                    response_event = next(
                        item for item in events if item["event"] == "assistant_response"
                    )
                    websocket.send_json(
                        {
                            "event": "playback_finished",
                            "turn_id": response_event["turn_id"],
                        }
                    )
                    while not any(
                        item["event"] == "input_gate" and item.get("state") == "open"
                        for item in events
                    ):
                        events.append(websocket.receive_json())

                event_names = [item["event"] for item in events]
                self.assertIn("utterance_started", event_names)
                self.assertIn("utterance_finalized", event_names)
                self.assertIn("model_started", event_names)
                self.assertLess(
                    event_names.index("utterance_finalized"),
                    event_names.index("model_started"),
                )
                finalized = next(
                    item for item in events if item["event"] == "utterance_finalized"
                )
                result = next(item for item in events if item["event"] == "model_result")
                self.assertGreater(finalized["endpoint_audio_ms"], 0)
                self.assertGreaterEqual(finalized["utterance_start_audio_ms"], 0)
                self.assertGreater(
                    finalized["utterance_end_audio_ms"],
                    finalized["utterance_start_audio_ms"],
                )
                self.assertGreater(finalized["vad_frames_processed"], 0)
                self.assertEqual(finalized["enhancement_frames_processed"], 0)
                self.assertIn("before_enhancement_capture_url", finalized)
                before_response = client.get(finalized["before_enhancement_capture_url"])
                model_input_response = client.get(finalized["capture_url"])
                self.assertEqual(before_response.status_code, 200)
                self.assertEqual(before_response.content, model_input_response.content)
                self.assertIn("vad_finalized_timestamp_ms", finalized)
                self.assertGreaterEqual(
                    result["timings"]["audio_to_last_llm_token_ms"], 0
                )
                self.assertIn("model_dispatch_ms", result["timings"])
                assistant = next(
                    item for item in events if item["event"] == "assistant_response"
                )
                self.assertEqual(assistant["response"]["route"], "repeat")
                self.assertTrue(assistant["response"]["audio"][0]["url"])
                audio_response = client.get(assistant["response"]["audio"][0]["url"])
                self.assertEqual(audio_response.status_code, 200)
                self.assertTrue(audio_response.content.startswith(b"RIFF"))
                self.assertTrue(
                    any(
                        item["event"] == "input_gate" and item.get("state") == "open"
                        for item in events
                    )
                )

    def test_aligned_pcm_window_shifts_and_zero_pads(self):
        source = struct.pack("<5h", 10, 20, 30, 40, 50)
        shifted = app_module._aligned_pcm_window(
            source,
            start_sample=-2,
            sample_count=5,
        )
        self.assertEqual(struct.unpack("<5h", shifted), (0, 0, 10, 20, 30))

        overrun = app_module._aligned_pcm_window(
            source,
            start_sample=3,
            sample_count=4,
        )
        self.assertEqual(struct.unpack("<4h", overrun), (40, 50, 0, 0))

    def test_voice_endpoint_serves_dynamic_and_static_cache_roots(self):
        with tempfile.TemporaryDirectory(prefix="voice_roots_") as directory:
            root = Path(directory)
            dynamic = root / "dynamic"
            static = root / "static"
            dynamic.mkdir()
            static.mkdir()
            dynamic_id = "1" * 32
            static_id = "2" * 32
            dynamic_payload = b"RIFFdynamic"
            static_payload = b"RIFFstatic"
            (dynamic / f"{dynamic_id}.wav").write_bytes(dynamic_payload)
            (static / f"{static_id}.wav").write_bytes(static_payload)
            with (
                patch.object(app_module, "VOICE_DIR", dynamic),
                patch.object(app_module, "STATIC_VOICE_DIR", static),
            ):
                client = TestClient(app_module.app)
                self.assertEqual(
                    client.get(f"/api/voice/{dynamic_id}.wav").content,
                    dynamic_payload,
                )
                self.assertEqual(
                    client.get(f"/api/voice/{static_id}.wav").content,
                    static_payload,
                )
                self.assertEqual(client.get(f"/api/voice/{'3' * 32}.wav").status_code, 404)


if __name__ == "__main__":
    unittest.main()
