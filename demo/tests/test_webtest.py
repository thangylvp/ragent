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
    def test_energy_vad_finalizes_before_disabled_model_boundary(self):
        with tempfile.TemporaryDirectory(prefix="vad_webtest_") as directory:
            with (
                patch.object(app_module, "MODEL", DisabledModel()),
                patch.object(app_module, "CAPTURE_DIR", Path(directory)),
            ):
                client = TestClient(app_module.app)
                response = client.get("/api/config")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()["tools"]), 33)

                with client.websocket_connect("/api/audio/stream") as websocket:
                    websocket.send_json(
                        {
                            "event": "start_stream",
                            "backend": "energy",
                            "sample_rate": 16_000,
                        }
                    )
                    for frame in _audio_frames():
                        websocket.send_bytes(frame)

                    events = []
                    while not any(item["event"] == "model_result" for item in events):
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
                self.assertGreater(finalized["vad_frames_processed"], 0)
                self.assertIn("vad_finalized_timestamp_ms", finalized)
                self.assertGreaterEqual(result["last_token_from_vad_ms"], 0)
                self.assertIn("model_dispatch_ms", result["component_timings"])


if __name__ == "__main__":
    unittest.main()
