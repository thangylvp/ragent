"""Fast deterministic tests for the dependency-free VAD baseline."""

from __future__ import annotations

import struct
import unittest

from vad import EnergyVad, EnergyVadConfig, VadEventKind, VadState


def frame(config: EnergyVadConfig, amplitude: int) -> bytes:
    return struct.pack(f"<{config.frame_samples}h", *([amplitude] * config.frame_samples))


class EnergyVadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnergyVadConfig()
        self.silence = frame(self.config, 0)
        self.speech = frame(self.config, 8_000)

    def feed(self, vad: EnergyVad, pcm: bytes, count: int):
        return [event for _ in range(count) for event in vad.process_frame(pcm)]

    def test_segments_one_utterance_with_preroll_and_trimmed_tail(self):
        vad = EnergyVad(self.config)
        self.assertEqual(self.feed(vad, self.silence, 20), [])

        start_events = self.feed(vad, self.speech, 10)
        self.assertEqual(
            [event.kind for event in start_events], [VadEventKind.SPEECH_STARTED]
        )
        self.assertEqual(vad.state, VadState.SPEECH)

        end_events = self.feed(vad, self.silence, 30)
        self.assertEqual(
            [event.kind for event in end_events], [VadEventKind.SPEECH_ENDED]
        )
        ended = end_events[0]
        self.assertEqual(ended.reason, "end_silence")
        self.assertEqual(ended.speech_ms, 200)
        self.assertIsNotNone(ended.audio_pcm16le)
        # The output retains pre-roll, speech, and the configured trailing tail.
        self.assertGreaterEqual(ended.buffered_ms, 600)
        self.assertEqual(vad.state, VadState.IDLE)

    def test_rejects_a_short_noise_burst(self):
        vad = EnergyVad(self.config)
        self.feed(vad, self.silence, 15)
        start_events = self.feed(vad, self.speech, 6)
        self.assertEqual(start_events[0].kind, VadEventKind.SPEECH_STARTED)

        events = self.feed(vad, self.silence, 30)
        self.assertEqual(
            [event.kind for event in events], [VadEventKind.SEGMENT_REJECTED]
        )
        self.assertEqual(events[0].reason, "too_short")

    def test_listening_gate_rejects_partial_speech_and_ignores_tts(self):
        vad = EnergyVad(self.config)
        self.feed(vad, self.speech, 5)
        events = vad.set_listening(False)
        self.assertEqual(
            [event.kind for event in events], [VadEventKind.SEGMENT_REJECTED]
        )
        self.assertEqual(events[0].reason, "listening_gate_closed")
        self.assertEqual(self.feed(vad, self.speech, 100), [])
        self.assertEqual(vad.state, VadState.IDLE)

        self.assertEqual(vad.set_listening(True), [])
        events = self.feed(vad, self.speech, 5)
        self.assertEqual([event.kind for event in events], [VadEventKind.SPEECH_STARTED])

    def test_max_duration_emits_once_then_waits_for_silence(self):
        config = EnergyVadConfig(
            pre_roll_ms=40,
            start_confirm_ms=40,
            min_speech_ms=80,
            end_silence_ms=100,
            trailing_silence_ms=40,
            max_utterance_ms=200,
        )
        vad = EnergyVad(config)
        speech = frame(config, 8_000)
        silence = frame(config, 0)

        events = self.feed(vad, speech, 20)
        self.assertEqual(
            [event.kind for event in events],
            [VadEventKind.SPEECH_STARTED, VadEventKind.MAX_DURATION_REACHED],
        )
        self.assertEqual(vad.state, VadState.COOLDOWN)
        self.assertEqual(self.feed(vad, speech, 20), [])
        self.feed(vad, silence, 5)
        self.assertEqual(vad.state, VadState.IDLE)

    def test_requires_exact_frame_size(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            EnergyVad(self.config).process_frame(b"too short")

    def test_config_cannot_drop_start_confirmation_audio(self):
        with self.assertRaisesRegex(ValueError, "pre_roll_ms"):
            EnergyVadConfig(pre_roll_ms=40, start_confirm_ms=100)


if __name__ == "__main__":
    unittest.main()
