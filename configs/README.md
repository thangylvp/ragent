# Configuration layout

Planned configuration groups:

```text
configs/
├── data_gen/
│   ├── text_v1.yaml
│   └── speech_v1.yaml
├── vad/
│   └── energy_v1.yaml
├── harness/
│   └── conversation_v1.yaml
├── execute/
│   └── simulator_v1.yaml
├── training/
│   ├── asr_adaptation.yaml
│   ├── text_alignment.yaml
│   └── speech_sft.yaml
├── evaluation/
│   ├── offline.yaml
│   └── edge_benchmark.yaml
├── serving/
│   ├── vllm_019.yaml
│   └── vllm_latest.yaml
└── demo/
    ├── mock.yaml
    ├── local_model.yaml
    └── vllm.yaml
```

Configs will reference data and model roots through explicit project-specific
variables. Large datasets and checkpoints must not be stored in this Git
repository.

VAD configuration owns frame size, adaptive thresholds, pre-roll,
minimum-speech duration, end-silence duration and maximum utterance length.
The demo reads that same VAD configuration; it must not maintain a second
set of browser-only endpointing thresholds.
