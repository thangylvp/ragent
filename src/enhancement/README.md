# Speech enhancement component

This component runs before VAD:

```text
microphone PCM -> echo cancellation -> speech enhancement -> VAD -> SLM
```

`fastenhancer_s` is the current webtest default. It is the official causal
16 kHz DNS checkpoint from FastEnhancer (ICASSP 2026), executed on CPU with
ONNX Runtime. Its waveform model consumes and emits 256 samples per step and
has 16 ms algorithmic delay. `none` remains available for direct A/B tests.

Browser noise suppression is disabled when using this component. Cascading a
browser denoiser and a neural denoiser can remove command words and makes the
input device/browser dependent. Browser echo cancellation stays enabled; the
robot deployment should replace it with an explicit AEC tied to the robot's
speaker reference.
