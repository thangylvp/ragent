# Command-line entry points

Scripts will remain thin wrappers around importable package code:

```text
scripts/
├── audit_robot_contract.py
├── audit_recordings.py
├── prepare_manifests.py
├── generate_text.py
├── generate_speech.py
├── evaluate_vad.py            # Implemented: normalized per-backend JSON report
├── compare_vad.py             # Implemented: Markdown comparison + port agreement
├── train.py
├── evaluate.py
├── export_checkpoint.py
├── test_vllm_audio.py
├── benchmark_vllm.py
├── benchmark_e2e.py
└── benchmark_webtest_e2e.py   # Implemented: real-time speech-end→last-token timing
```

The remaining training/data commands come only after the robot schema is
frozen. `demo/run.sh` now launches the narrow VAD→car-STC component webtest.
The full robot integration surface remains deferred until the harness contract
is designed.
