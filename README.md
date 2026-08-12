# Robot Speech-to-Action

Train and evaluate a small speech-language model that maps Vietnamese robot
audio directly to one of two outputs:

1. `non_tool(text=...)` for queries that must be forwarded to a cloud agent;
2. a structured robot tool call containing only arguments stated or reliably
   resolved from the conversation context.

The model starts from `Qwen/Qwen3-ASR-0.6B` and follows the same direct
audio-to-tool architecture as the existing car `stc` project. It is a parser,
not an agent and not an execution runtime.

## Responsibility boundary

| Component | Owns |
| --- | --- |
| Speech-to-action SLM (`src/slm`) | Speech recognition, route selection, tool selection, argument extraction |
| VAD (`src/vad`) | Streaming speech detection, utterance buffering, endpointing and playback gating |
| Agent harness (`src/harness`, placeholder) | Turn orchestration, conversation history, missing-field detection, follow-up questions, cloud forwarding, response/TTS loop |
| Execution layer (`src/execute`, placeholder) | Schema validation, robot state, preconditions, safety policy, action execution and result reporting |
| Component webtest (`demo`, implemented) | Streams microphone audio through selectable VADs, then displays the existing car STC model output without execution |
| Future robot demo (`demo/docs/DESIGN.md`, design only) | Will compose the robot SLM, harness and execution contracts after they are defined |

The SLM must not invent required values merely to make a call executable. If a
required field is absent, it emits the selected tool with that field omitted;
the harness decides what to ask next.

See [docs/architecture.md](docs/architecture.md) for the data flow and
[docs/migration-map.md](docs/migration-map.md) for the planned reuse from
`stc`. The runnable component page is documented in
[demo/README.md](demo/README.md); the future robot end-to-end surface remains
a design in [demo/docs/DESIGN.md](demo/docs/DESIGN.md).

## Current status

The reusable Qwen3-ASR model wrapper, vendored audio backend and Qwen3 tool-call
codec have been migrated from `stc` into `src/slm/modeling`. Robot training,
data generation and the production tool catalog remain deferred until that
catalog is agreed.

VAD can now be compared independently through adapters for FireRedVAD,
OmniVAD-Kit, Silero VAD, WebRTC VAD and the dependency-free energy baseline.
The harness and execution layer remain design placeholders. A narrow webtest
now exercises live VAD cuts against the existing car checkpoint, but it does
not assume or implement robot conversation-loop behavior.

## Repository layout

```text
robot-speech-to-action/
├── configs/                     # Data, training, evaluation and serving configs
├── contracts/                   # Versioned boundary contracts and robot tool catalog
├── data/                        # Manifest conventions; raw audio is external/ignored
├── demo/                        # Runnable VAD→car-STC webtest; future robot demo design
├── docs/                        # Architecture, decisions and migration notes
├── scripts/                     # Thin CLI entry points (planned)
├── src/
│   ├── slm/                     # Speech model, data, training, evaluation and serving
│   │   ├── checkpoint/          # Standalone checkpoint export/audit
│   │   ├── contracts/           # Output and tool-schema validation
│   │   ├── data/                # Manifest preparation and audio transforms
│   │   ├── data_gen/            # Text and speech data generation
│   │   ├── engine/              # Training loop and hooks
│   │   ├── eval/                # Routing, tool-call, ASR and acoustic evaluation
│   │   ├── modeling/            # Qwen3-ASR direct audio-to-tool integration
│   │   ├── robot/               # Robot catalog, entities and normalization
│   │   └── serving/             # vLLM request/response helpers and benchmarks
│   ├── vad/                     # Streaming VAD, optional backend adapters and endpointing
│   │   └── eval/                # Independent VAD evaluation
│   ├── harness/                 # Conversation loop
│   │   └── eval/                # Independent harness evaluation
│   └── execute/                 # Robot execution layer
│       └── eval/                # Independent execution evaluation
└── tests/                       # Deterministic unit and contract tests
```

Run the current CPU-only VAD tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests/vad -p 'test_*.py' -v
```

See [configs/vad/README.md](configs/vad/README.md) for reproducible component
benchmark commands, [the initial VAD result](docs/vad-benchmark-2026-08-12.md),
and [src/slm/modeling/ORIGIN.md](src/slm/modeling/ORIGIN.md) for the exact SLM
migration boundary.

## References

- Existing local car implementation: `../stc`
- Existing robot gateway and action names: `../robot-agent-gateway`
- Existing robot test recordings: `../data_test_robot`
- VoiceTeller reference: <https://github.com/distil-labs/distil-voice-assistant-banking>
