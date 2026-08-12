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
| Speech-to-action SLM (this repo) | Speech recognition, route selection, tool selection, argument extraction |
| Agent harness (future repo/service) | Conversation history, missing-field detection, follow-up questions, cloud forwarding, response/TTS loop |
| Execution layer (future repo/service) | Schema validation, robot state, preconditions, safety policy, action execution and result reporting |

The SLM must not invent required values merely to make a call executable. If a
required field is absent, it emits the selected tool with that field omitted;
the harness decides what to ask next.

See [docs/architecture.md](docs/architecture.md) for the data flow and
[docs/migration-map.md](docs/migration-map.md) for the planned reuse from
`stc`.

## Current status

This first commit is an architecture scaffold. It deliberately does not define
the production robot tool catalog yet. The catalog should be agreed with the
agent harness and execution layer before data generation begins.

## Repository layout

```text
robot-speech-to-action/
├── configs/                     # Data, training, evaluation and serving configs
├── contracts/                   # Versioned boundary contracts and robot tool catalog
├── data/                        # Manifest conventions; raw audio is external/ignored
├── docs/                        # Architecture, decisions and migration notes
├── scripts/                     # Thin CLI entry points (planned)
├── src/robot_speech_to_action/
│   ├── checkpoint/              # Standalone checkpoint export/audit
│   ├── contracts/               # Output and tool-schema validation
│   ├── data/                    # Manifest preparation and audio transforms
│   ├── data_gen/                # Text and speech data generation
│   ├── engine/                  # Training loop and hooks
│   ├── eval/                    # Routing, tool-call, ASR and acoustic evaluation
│   ├── modeling/                # Qwen3-ASR direct audio-to-tool model integration
│   ├── robot/                   # Robot domain catalog, entities and normalization
│   └── serving/                 # vLLM request/response helpers and benchmarks
└── tests/                       # Deterministic unit and contract tests
```

## References

- Existing local car implementation: `../stc`
- Existing robot gateway and action names: `../robot-agent-gateway`
- Existing robot test recordings: `../data_test_robot`
- VoiceTeller reference: <https://github.com/distil-labs/distil-voice-assistant-banking>
