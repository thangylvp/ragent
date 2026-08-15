# Command-line tools

This directory contains only demo, serving and performance tooling. Training
and dataset-generation entry points are intentionally out of scope.

```text
scripts/
├── smoke_test_vllm_audio.py        # One standalone-checkpoint audio request
├── benchmark_webtest_e2e.py        # Laptop/local component timing
├── benchmark_jetson_full_system.py # Repeated Jetson tool-call path
├── benchmark_jetson_cloud_path.py  # Repeated Jetson non-tool/cloud path
├── analyze_jetson_full_system.py   # Tool-call aggregates and resource summary
├── analyze_jetson_cloud_path.py    # Non-tool aggregates
├── benchmark_speech_enhancement.py # Enhancement + downstream call comparison
├── evaluate_vad.py                 # Normalized per-backend VAD report
├── compare_vad.py                  # VAD comparison and agreement report
├── create_pcm16_variants.py        # Byte-unique benchmark WAV variants
├── prewarm_demo_audio.py           # Fixed Vietnamese response cache
├── create_jetson_benchmark_slides.py
└── jetson/                          # vLLM/MPS/demo/measurement launchers
```

Use the exact command order and environment variables in
[`docs/jetson-reproduction.md`](../docs/jetson-reproduction.md). Runtime output
is written below the ignored `outputs/` tree; raw or private audio is never
committed.
