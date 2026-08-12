# SLM model source origin

The `CommandASR` wrapper, Qwen3 tool-call codec, and vendored Qwen3-ASR
backend in this directory were migrated from the current local `../stc`
implementation on 2026-08-12. Imports were namespaced under `slm`; the car
catalog and car-specific training data were not copied.

This migration intentionally includes only the reusable model component:

- `CommandASR`, checkpoint save/load helpers and sparse supervised loss;
- the Qwen3 tool-call render/parse codec;
- the registry/build hook;
- the vendored Qwen3-ASR configuration, model and processor.

The `stc` trainer, car datasets, prompts, schemas and checkpoint are not a
robot model and remain outside this repository. They will be generalized only
after the robot tool contract is frozen.

The vendored Qwen3-ASR files retain their upstream Apache-2.0 headers and came
originally from `qwen-asr==0.0.6`, as documented in
`audio_encoder/qwen3asr_vendor/__init__.py`.
