# Data layout

Only small reviewed manifests, schemas and evaluation metadata belong in Git.
Raw recordings, generated audio and caches remain on external storage.

Planned layout:

```text
data/
├── manifests/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── eval/
│   ├── routing.jsonl
│   ├── tool_calls.jsonl
│   ├── missing_fields.jsonl
│   ├── multi_turn.jsonl
│   └── acoustic.jsonl
├── samples/                     # Small redistributable smoke-test audio only
├── raw/                         # Ignored; external recordings
└── generated/                   # Ignored; generated text and speech
```

Each example must retain provenance, speaker/group identity, transcript,
audio path, route, expected tool call, and generation/augmentation metadata.
Splits must be speaker- and semantic-template-safe to prevent leakage.
