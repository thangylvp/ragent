# Robot end-to-end web demo

This top-level application will test the complete user path:

```text
browser microphone
  -> first-class streaming VAD
  -> speech-to-action SLM
  -> harness route / required-field decision
  -> cloud agent or robot execution layer
  -> visible response, state change and timing trace
```

It follows the lightweight FastAPI plus buildless HTML/CSS/JavaScript pattern
from `../stc/demo`, but it is not a copy of the car dashboard. See
[`docs/DESIGN.md`](docs/DESIGN.md) for the locked component boundaries,
streaming protocol, robot UI and test plan.

## Current status

The directory and interface design are scaffolded now, and the first VAD
baseline is implemented under `src/vad`. The runnable mock page comes after
the versioned robot tool contract and the first harness/execution interfaces
are implemented, so the demo imports those interfaces instead of creating
incompatible demo-only behavior.

## Planned layout

```text
demo/
├── README.md
├── requirements.txt
├── run.sh
├── docs/DESIGN.md
├── backend/
│   ├── app.py                   # HTTP, WebSocket and static files only
│   ├── schemas.py               # Demo transport/event models
│   ├── pipeline.py              # Thin composition of public component APIs
│   └── adapters/                # Mock, local-model and vLLM wiring
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/
│       ├── app.js
│       ├── audio-stream.js      # PCM capture; no authoritative VAD logic
│       ├── api.js
│       ├── conversation.js
│       ├── robot-state.js
│       └── trace.js
└── tests/
    ├── test_stream.py
    ├── test_pipeline.py
    └── test_browser_smoke.py
```
