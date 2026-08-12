# Test plan

The automated suite is split by ownership:

```text
tests/
├── vad/       # Streaming classification, endpointing and corpus metrics
├── slm/       # Model contracts, metrics, data and serving compatibility
├── harness/   # Turn state, slot completion and cloud routing
├── execute/   # Robot validation, state transitions, safety and dispatch fakes
└── e2e/       # Full recorded/streamed scenarios through all three components
```

It will cover:

- contract parsing and version compatibility;
- VAD segmentation, pre-roll, timeout and robot-playback gating;
- partial-call acceptance and required-field detection;
- route, tool-name and argument scoring;
- deterministic robot simulator state transitions and rejected calls;
- manifest validation and split leakage checks;
- prompt/template parity between training and vLLM serving;
- standalone checkpoint export audits;
- sample audio inference on supported vLLM versions;
- end-to-end recorded scenarios and browser/API smoke tests.

Unit tests must use small fixtures and fake model clients. GPU integration and
latency tests will be marked separately. VAD corpus evaluation additionally
reports false activations per hour, speech miss rate, clipped-onset duration,
endpoint delay and segment purity for silence, far-field speech, motor noise
and robot TTS playback.
