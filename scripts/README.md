# Planned command-line entry points

Scripts will remain thin wrappers around importable package code:

```text
scripts/
├── audit_robot_contract.py
├── audit_recordings.py
├── prepare_manifests.py
├── generate_text.py
├── generate_speech.py
├── train.py
├── evaluate.py
├── export_checkpoint.py
├── test_vllm_audio.py
└── benchmark_vllm.py
```

The first implementation milestone is contract audit plus dataset inventory;
training commands come only after the robot schema is frozen.
