# Voice activity detection

VAD is a first-class component because it converts an open microphone stream
into bounded model turns and needs independent data, evaluation, versioning
and edge tuning. The harness, web demo and future robot microphone adapter all
consume the same `VadEngine` interface and endpointing configuration.

## Package

```text
vad/
├── base.py          # Implemented: VadEngine protocol and event types
├── energy.py        # Implemented: adaptive energy baseline + segmenter
├── stream.py        # Planned: transport validation/resampling boundary
└── metrics.py       # Planned: corpus-level segmentation metrics
```

## Stream contract

- Input: mono PCM signed 16-bit little-endian, 16 kHz, fixed 20 ms frames.
- Output events: `speech_started`, `speech_ended`, `segment_rejected` and
  `max_duration_reached`.
- A finalized event carries a complete WAV/PCM utterance plus onset, offset,
  speech duration, total buffered duration and the endpoint reason.
- Pre-roll retains speech that precedes the confirmed start trigger.
- Timestamps use a monotonic clock and sample indices, not wall-clock time.
- Thresholds and durations come from versioned harness configuration.

The first baseline adapts to the ambient noise floor and uses hysteresis:
speech must cross a higher threshold to start than it needs to remain active.
The starting configuration will use approximately 300 ms pre-roll, 100 ms
start confirmation, 600 ms end silence and a 15 s maximum utterance, but these
are tuning seeds rather than product constants.

## Robot-specific behavior

Version 1 is half-duplex. The harness closes the VAD speech gate while its own TTS
is playing, then performs a short noise-floor recalibration before listening
again. This prevents the robot from interpreting itself as a user. Supporting
barge-in later requires acoustic echo cancellation plus a separate evaluation
set; browser-provided echo cancellation is not accepted as production proof.

## Evaluation

Evaluate on continuously recorded sessions, not only isolated command WAVs.
Report false activations per hour, missed-speech rate, clipped onset/offset,
endpoint delay, rejected-short-segment rate, maximum-duration cuts and segment
purity. Test silence, far-field speech, accents, fans, music/TV, robot motors,
motion, TTS playback and speech immediately after TTS.
