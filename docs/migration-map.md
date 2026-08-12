# Migration map from the car project

The new repository is based on the architecture and training machinery in
`../stc`, not on its vehicle ontology. Migration should happen module by
module with tests, rather than by copying the entire working tree.

| Source in `stc` | Destination | Status |
| --- | --- | --- |
| `src/modeling/command_asr.py` | `src/slm/modeling/` | Migrated: generic Qwen3-ASR audio-to-tool model, codec and vendor backend |
| `src/engine/` | `src/slm/engine/` | Deferred: reuse trainer and hooks after isolating generic dependencies |
| `src/data/` | `src/slm/data/` | Deferred: reuse manifest loading, audio transforms and samplers |
| `src/data_gen/text/` | `src/slm/data_gen/text/` | Deferred: reuse generation only after robot prompts and schemas exist |
| `src/data_gen/speech/` | `src/slm/data_gen/speech/` | Deferred: reuse TTS and augmentation pipelines |
| `src/checkpoint/` and `tools/export_checkpoint.py` | `checkpoint/` and `scripts/` | Deferred: reuse standalone Hugging Face export and compatibility audit |
| `src/eval/command/` | `src/slm/eval/` | Deferred: generalize scoring to route/tool/argument/missing-slot metrics |
| `src/car/` | none | Do not copy; replace with a versioned robot contract |
| `demo/` | `demo/` | First component webtest implemented: streaming VAD into the existing car model with no execution; robot integration remains deferred until the harness is designed |
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

1. Migrate and independently verify the reusable Qwen3-ASR model component.
2. Compare candidate VAD components and collect a continuously annotated VAD
   evaluation corpus.
3. Freeze and version the robot tool catalog with the execution team.
4. Implement output-contract validation and exact-match scoring.
5. Migrate generic data preparation and audit existing robot recordings.
6. Generate balanced text data, including missing-slot and non-tool cases.
7. Generate/augment speech and create leakage-safe train/validation/test sets.
8. Migrate the Qwen3-ASR training and checkpoint-export path.
9. Train in stages: ASR adaptation, routing/tool alignment, then speech SFT.
10. Design the harness contract, then connect the real components to the
    end-to-end demo and perform edge benchmarking.
