# Planned command-line entry points

Scripts will remain thin wrappers around importable package code:

```text
scripts/
├── audit_robot_contract.py
├── audit_recordings.py
├── prepare_manifests.py
├── generate_text.py
├── generate_speech.py
├── evaluate_vad.py
├── train.py
├── evaluate.py
├── export_checkpoint.py
├── test_vllm_audio.py
├── benchmark_vllm.py
└── benchmark_e2e.py
```

The first implementation milestone is contract audit plus dataset inventory;
training commands come only after the robot schema is frozen. The web server
will have its own `demo/run.sh`; it is an integration surface, not a training
entry point.
