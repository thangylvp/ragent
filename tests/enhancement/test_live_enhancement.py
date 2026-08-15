"""Tests for the streaming speech-enhancement contract."""

from __future__ import annotations

import struct
import unittest

import numpy as np

from enhancement.live import FastEnhancerOnnx, PassthroughEnhancer


class _IdentitySession:
    def run(self, _outputs, inputs):
        return [inputs["wav_in"], inputs["cache_in_0"]]


class LiveEnhancementTest(unittest.TestCase):
    def test_passthrough_returns_input_without_buffering(self):
        enhancer = PassthroughEnhancer()
        pcm = struct.pack("<4h", -100, 0, 100, 200)
        update = enhancer.process(pcm)
        self.assertEqual(update.pcm16le, pcm)
        self.assertEqual(update.frames_processed, 0)

    def test_fastenhancer_buffers_until_a_complete_256_sample_frame(self):
        enhancer = FastEnhancerOnnx.__new__(FastEnhancerOnnx)
        enhancer.name = "fake"
        enhancer._session = _IdentitySession()
        enhancer._cache_shapes = {"cache_in_0": (1, 2)}
        enhancer._pending = bytearray()
        enhancer._caches = {}
        enhancer.frames_processed = 0
        enhancer.compute_total_ms = 0.0
        enhancer.compute_max_ms = 0.0
        enhancer.reset()

        source = np.arange(-128, 128, dtype="<i2").tobytes()
        first = enhancer.process(source[:200])
        second = enhancer.process(source[200:])

        self.assertEqual(first.pcm16le, b"")
        self.assertEqual(second.frames_processed, 1)
        self.assertEqual(len(second.pcm16le), len(source))
        actual = np.frombuffer(second.pcm16le, dtype="<i2")
        expected = np.frombuffer(source, dtype="<i2")
        self.assertLessEqual(int(np.max(np.abs(actual - expected))), 1)


if __name__ == "__main__":
    unittest.main()
