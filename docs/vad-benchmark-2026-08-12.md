# Initial VAD component benchmark — 2026-08-12

This is a component smoke benchmark, not a final VAD selection. It covers 247
isolated 16 kHz robot/user recordings (1,940.119 seconds) on an Intel Core
i9-14900HX CPU. Every current clip contains user speech; the `vmo_2305_16k`
subdirectories name the background condition rather than negative-only audio.

Consequently, activation rate is a clip-level speech hit rate. Detected-audio
ratio is useful for comparing segmentation behavior but is **not accuracy**
without manual onset/offset labels: a larger value may retain speech or may
retain extra noise. This corpus cannot measure false activations.

## Runtime

| Backend | Runtime | Load ms | Realtime × | Process peak RSS MB |
| --- | --- | ---: | ---: | ---: |
| Energy baseline | Python stdlib | 0.9 | 1,360.5 | 119.5 |
| WebRTC VAD | native CPU | 11.5 | 10,719.1 | 119.5 |
| FireRedVAD | PyTorch CPU | 770.9 | 245.7 | 554.0 |
| OmniVAD-Kit | ncnn CPU | 131.3 | 35.6 | 120.8 |
| Silero VAD v6 | PyTorch CPU | 825.0 | 244.9 | 531.5 |

The process memory figure includes Python, the evaluator and any loaded
framework; it is not model-weight size. Throughput is sequential offline WAV
processing, not endpoint delay. OmniVAD uses frame-oriented streaming calls,
whereas the PyTorch reference processes a complete file, so throughput alone
does not compare kernel efficiency. All backends are comfortably faster than
real time on this machine.

Tested package versions: FireRedVAD 0.0.2, OmniVAD 0.2.13, Silero VAD 6.2.1,
WebRTC VAD 2.0.14 and PyTorch 2.10.0+cu128. All detection ran on CPU.

## Speech hit rate

| Acoustic condition | Energy | WebRTC | FireRed | OmniVAD | Silero |
| --- | ---: | ---: | ---: | ---: | ---: |
| crowd | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| park | 96.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| piano | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| quiet | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| washing dishes | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| robot moving | 100.0% | 100.0% | 100.0% | 100.0% | 96.5% |
| robot stationary | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| VMO speech | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

FireRedVAD and OmniVAD-Kit are the same FireRed streaming acoustic model behind
different deployment runtimes. On the 247 common files they had 100% activation
agreement, 77.73% exact segment-count agreement, 96.09% mean speech-mask IoU on
a 10 ms grid, and 129.2 ms mean absolute detected-duration difference per file.
That is strong but not bit-identical agreement in this test.

## Current conclusion

Keep FireRedVAD as the PyTorch reference and OmniVAD-Kit as the primary edge
candidate. OmniVAD has much lower observed process memory and startup cost and
still runs 35.6 times faster than real time. Keep WebRTC and Silero as useful
baselines; neither should be selected or rejected from activation coverage
alone.

Before choosing the production VAD, record continuous sessions with precise
speech boundaries and real negative intervals: robot motors, TTS playback,
fans, music, crowd speech not addressed to the robot, handling noise and true
silence. Then measure false activations/hour, missed-speech duration, onset
clipping, endpoint delay, segment purity, CPU use and streaming latency on the
target robot.

Sources: [FireRedVAD official repository](https://github.com/FireRedTeam/FireRedVAD),
[OmniVAD-Kit official repository](https://github.com/lifeiteng/OmniVAD-Kit),
[Silero VAD official repository](https://github.com/snakers4/silero-vad).
