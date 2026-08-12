# Robot end-to-end demo design

## Purpose

The demo is a human-operated integration test for the VAD, SLM, agent harness
and execution layer. Component evaluations remain authoritative for accuracy;
the demo exposes boundary mistakes, turn-taking failures, state drift and
latency that isolated tests cannot show.

It must remain thin. Business logic belongs in `src/vad`, `src/slm`,
`src/harness` or `src/execute`; the demo only composes those public APIs and
renders their events.

## Differences from the car demo

| Car `stc` demo | Robot demo |
| --- | --- |
| Click once to start and once to stop recording | Continuous PCM stream with server-side VAD; push-to-talk remains a diagnostic fallback |
| Car HMI controls dominate the page | Conversation, robot mission/motion/safety state and action timeline dominate |
| Car simulator is called directly by demo pipeline | Harness calls the execution-layer interface; the demo never bypasses it |
| Browser produces a complete WAV before upload | Browser sends fixed PCM frames; VAD emits a finalized utterance |
| Cabin-specific reducers | Schema-driven robot state snapshot plus generic events until the catalog is frozen |

## End-to-end flow

```text
getUserMedia
  -> AudioWorklet (mono PCM frames)
  -> WebSocket /api/audio/stream
  -> vad.VadEngine + segmenter
  -> completed utterance
  -> SLM adapter (mock | local checkpoint | vLLM)
  -> harness conversation loop
       -> non_tool -> cloud adapter -> reply/TTS
       -> incomplete tool -> deterministic follow-up/TTS
       -> complete tool -> execute gateway -> result/TTS
  -> structured stage events over the WebSocket
  -> conversation, robot-state and latency panels
```

Audio classification and endpointing happen on the server. The browser may
show a live RMS meter but must not decide the canonical start/end boundaries.
This ensures the page tests the first-class VAD component that will also
receive robot microphone audio.

## Transport contract

The microphone stream uses one WebSocket per session:

- client sends a JSON `start_stream` message with PCM format metadata;
- client then sends binary 16 kHz mono PCM16LE frames, normally 20 ms each;
- server emits `vad_state`, `utterance_started`, `utterance_finalized`,
  `stage_started`, `stage_finished`, `follow_up`, `tool_result`,
  `cloud_reply` and `error` events;
- `stop_stream` flushes or rejects the current segment deterministically;
- reconnect creates no implicit model turn and does not execute a partial call.

Typed text and uploaded WAV endpoints remain available to isolate microphone
or VAD failures during debugging.

Every finalized turn receives one `turn_id`. Stage events carry monotonic
timestamps so the UI can display:

- captured and retained audio duration;
- VAD onset confirmation and endpoint delay;
- SLM queue time, TTFT, decode time and output tokens/second when available;
- harness validation/follow-up time;
- execution or cloud latency;
- speech-end-to-result and total end-to-end latency.

## Page layout

The page has four primary areas:

1. **Listen and VAD** — microphone permission, stream state, calibrated noise
   floor, live level, `idle/speech/ending` state, utterance duration, manual
   push-to-talk and WAV upload.
2. **Conversation** — user transcript/tool intent, deterministic follow-up,
   cloud response and execution response in one chronological timeline.
3. **Decision trace** — raw SLM output, route, extracted arguments, missing
   fields, safety/validation outcome and per-stage timings.
4. **Robot state** — connection, battery, pose, motion, mission, safety gate,
   current/last action and a generic raw state/event view. Fields are rendered
   only when supplied by the execution contract.

The UI never fabricates a successful state change. Rejected, timed-out and
cancelled calls stay visible with the previous robot snapshot.

## Modes

- `mock`: CPU-only deterministic SLM, cloud and robot simulator; used by CI
  and frontend development.
- `local`: standalone speech-to-action checkpoint in the demo backend.
- `vllm`: OpenAI-compatible remote model server; harness and execute stay
  local unless explicitly configured otherwise.
- `robot`: real execution gateway; opt-in and visibly distinguished from the
  simulator. Actions still pass through execution validation and safety.

Real-robot mode must never be the default. The page shows an armed/disarmed
state and requires the execution layer to authorize actions; UI controls do
not constitute a safety mechanism.

## VAD test cases

The demo smoke suite replays continuous PCM streams with known boundaries:

- silence followed by one command and trailing silence;
- two commands separated by sufficient silence;
- short cough/tap that must be rejected;
- far-field command with fan or motor noise;
- a command immediately after robot TTS finishes;
- robot TTS while the half-duplex gate is closed;
- a long utterance reaching the configured maximum;
- disconnect during speech, which must not execute a partial call.

CI uses fake clocks and recorded fixtures. Hardware runs additionally report
VAD and end-to-end metrics from real-time streaming.
