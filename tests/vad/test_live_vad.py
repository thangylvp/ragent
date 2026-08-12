"""Tests for the detector-neutral live PCM event adapter."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vad import VadEventKind
from vad.live import EventModelLiveVad, pcm_dbfs


@dataclass
class _Result:
    frame_idx: int
    is_speech: bool
    smoothed_prob: float
    is_speech_start: bool = False
    is_speech_end: bool = False
    speech_start_frame: int = -1
    speech_end_frame: int = -1


class _FakeEventModel:
    def __init__(self):
        self.frame = 0

    def process(self, pcm16le: bytes):
        self.frame += 1
        return _Result(
            frame_idx=self.frame,
            is_speech=2 <= self.frame <= 3,
            smoothed_prob=0.9 if 2 <= self.frame <= 3 else 0.1,
            is_speech_start=self.frame == 3,
            speech_start_frame=1 if self.frame == 3 else -1,
            is_speech_end=self.frame == 5,
            speech_end_frame=4 if self.frame == 5 else -1,
        )

    def reset(self):
        self.frame = 0


class LiveVadTest(unittest.TestCase):
    def test_event_model_retains_retroactive_onset_and_emits_pcm(self):
        session = EventModelLiveVad("fake", _FakeEventModel())
        frames = [bytes([index, 0]) * 160 for index in range(1, 6)]
        events = []
        for frame in frames:
            events.extend(session.process_frame(frame).events)

        self.assertEqual(
            [event.kind for event in events],
            [VadEventKind.SPEECH_STARTED, VadEventKind.SPEECH_ENDED],
        )
        self.assertEqual(events[1].audio_pcm16le, b"".join(frames[:4]))
        self.assertEqual(events[1].utterance_start_sample, 0)
        self.assertEqual(events[1].utterance_end_sample, 640)

    def test_pcm_dbfs_handles_silence_and_full_scale(self):
        self.assertEqual(pcm_dbfs(b"\x00\x00" * 160), -120.0)
        self.assertAlmostEqual(pcm_dbfs(b"\xff\x7f" * 160), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
