# Migration map from the car project

The new repository is based on the architecture and training machinery in
`../stc`, not on its vehicle ontology. Migration should happen module by
module with tests, rather than by copying the entire working tree.

| Source in `stc` | Destination | Plan |
| --- | --- | --- |
| `src/modeling/command_asr.py` | `src/robot_speech_to_action/modeling/` | Reuse Qwen3-ASR audio-to-tool integration; remove car naming |
| `src/engine/` | `src/robot_speech_to_action/engine/` | Reuse trainer and hooks after isolating generic dependencies |
| `src/data/` | `src/robot_speech_to_action/data/` | Reuse manifest loading, audio transforms and samplers |
| `src/data_gen/text/` | `src/robot_speech_to_action/data_gen/text/` | Reuse balanced/diverse generation with robot prompts and schemas |
| `src/data_gen/speech/` | `src/robot_speech_to_action/data_gen/speech/` | Reuse TTS and augmentation pipelines |
| `src/checkpoint/` and `tools/export_checkpoint.py` | `checkpoint/` and `scripts/` | Reuse standalone Hugging Face export and compatibility audit |
| `src/eval/command/` | `src/robot_speech_to_action/eval/` | Generalize scoring to route/tool/argument/missing-slot metrics |
| `src/car/` | none | Do not copy; replace with a versioned robot contract |
| car-generated data and checkpoints | none | Do not copy or mix into robot training |

## External inputs

- `../robot-agent-gateway` is a source for candidate action and mission names,
  but its current regex/action configuration is not yet a formal tool schema.
- `../data_test_robot` is an external evaluation-data source. Audio remains
  outside Git; only reviewed manifests and provenance may be committed.
- `../distil-voice-assistant-banking` is a conceptual reference for partial
  calls, deterministic slot elicitation and conversation history. Its model
  architecture is different: separate ASR plus text SLM, whereas this project
  continues training Qwen3-ASR for direct speech-to-tool output.

## Migration order

1. Freeze and version the robot tool catalog with the execution team.
2. Implement output-contract validation and exact-match scoring.
3. Migrate generic data preparation and audit existing robot recordings.
4. Generate balanced text data, including missing-slot and non-tool cases.
5. Generate/augment speech and create leakage-safe train/validation/test sets.
6. Migrate the Qwen3-ASR training and checkpoint-export path.
7. Train in stages: ASR adaptation, routing/tool alignment, then speech SFT.
8. Evaluate offline, export standalone checkpoint and benchmark edge serving.
