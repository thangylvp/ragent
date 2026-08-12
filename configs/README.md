# Configuration layout

Planned configuration groups:

```text
configs/
├── data_gen/
│   ├── text_v1.yaml
│   └── speech_v1.yaml
├── training/
│   ├── asr_adaptation.yaml
│   ├── text_alignment.yaml
│   └── speech_sft.yaml
├── evaluation/
│   ├── offline.yaml
│   └── edge_benchmark.yaml
└── serving/
    ├── vllm_019.yaml
    └── vllm_latest.yaml
```

Configs will reference data and model roots through explicit project-specific
variables. Large datasets and checkpoints must not be stored in this Git
repository.
