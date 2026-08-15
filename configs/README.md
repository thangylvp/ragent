# Configuration and benchmark metadata

The repository keeps only configuration needed to reproduce the demo and
performance evaluation:

```text
configs/
├── benchmarks/
│   ├── jetson_tool_latency_corpus.json
│   └── jetson_cloud_latency_corpus.json
└── vad/
    └── README.md
```

The benchmark JSON files describe expected routes/calls and audio durations;
the private/raw WAV files remain outside Git. Machine-specific paths and
service addresses are supplied through the environment variables documented in
`docs/jetson-reproduction.md` and `demo/README.md`.
