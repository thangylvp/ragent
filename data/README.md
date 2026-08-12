# Data layout

Only small reviewed manifests, schemas and evaluation metadata belong in Git.
Raw recordings, generated audio and caches remain on external storage.

Planned layout:

```text
data/
├── slm/
│   ├── manifests/{train,validation,test}.jsonl
│   └── eval/{routing,tool_calls,missing_fields,multi_turn,acoustic}.jsonl
├── vad/
│   ├── manifests/{validation,test}.jsonl
│   └── sessions/                # Ignored continuous audio + speech boundaries
├── harness/
│   └── eval/                    # Reviewed multi-turn scenario manifests
├── execute/
│   └── eval/                    # State/safety scenario fixtures
├── e2e/
│   └── eval/                    # Full continuous-session expectations
├── samples/                     # Small redistributable smoke-test audio only
├── raw/                         # Ignored; external recordings
└── generated/                   # Ignored; generated text and speech
```

SLM examples retain provenance, speaker/group identity, transcript, audio
path, route, expected tool call and generation/augmentation metadata. VAD
examples additionally retain continuous-session speech boundaries, acoustic
condition, robot playback intervals and endpoint annotations. Splits must be
speaker-, session- and semantic-template-safe to prevent leakage.
