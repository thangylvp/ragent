# Recent speech enhancement review and local decision (2026-08-12)

## Decision

Use the official 16 kHz `FastEnhancer-S` streaming ONNX checkpoint as the
current webtest default, before VAD. Keep `none` selectable for A/B testing.
This is a provisional component decision, not a claim that it is universally
best.

The webtest retains synchronized before/after WAVs for every turn. If live
Vietnamese recognition is worse after enhancement, prefer `none`; the small
synthetic sweep below is insufficient to override evidence from the actual
room, microphone and browser capture chain.

Why this model:

- It preserved all 7/7 clean STCC tool calls in the local regression sweep.
- It tied the best tested systems at 5/14 exact calls on severe 0 and 5 dB
  mixtures made from recent room captures.
- It is causal, open, CPU-friendly, and has an official waveform-to-waveform
  ONNX checkpoint with 16 ms algorithmic delay.
- Isolated local streaming RTF was about 0.021. In the full Python web path,
  CPU contention and thread dispatch raised the measured per-frame cost, but
  it remained comfortably real-time.

Do not make `DPDFNet2` always-on despite its stronger signal metrics and 4/7
exact calls at 5 dB: it changed 3 of 7 clean commands in this small Vietnamese
regression set. `DPDFNet4` preserved 6/7 clean commands and remains a useful
quality challenger. A future degradation gate can revisit these choices after
we collect true clean/noisy robot recordings.

## What recent top-tier work says

### General ML venues

| Work | Venue | Main idea | Fit for this robot parser |
| --- | --- | --- | --- |
| [FIRING-Net](https://proceedings.iclr.cc/paper_files/paper/2025/hash/51adbf4e5b0162824800f08a4486ba6b-Abstract-Conference.html) | ICLR 2025 | Recycles filtered-out features and lets speech/noise representations refine each other | Interesting discriminative model; no production-ready edge runtime comparable to the tested ONNX systems |
| [GenSE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/acde98fb254b8021d194ccdb80a1241e-Abstract-Conference.html) | ICLR 2025 | Treats enhancement as hierarchical language modeling over semantic and acoustic tokens | High perceptual potential, but generative reconstruction is too large and risky for exact low-latency command arguments |
| [FINALLY](https://papers.nips.cc/paper_files/paper/2024/hash/01b3dea1871f7cea1e0e6be1f2f085bc-Abstract-Conference.html) | NeurIPS 2024 | Universal 48 kHz GAN enhancement using WavLM perceptual loss | Good studio-quality/offline reference; not the cleanest 16 kHz causal Jetson path |
| [Speech Robust Bench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/605e02ae04cba1ebf6a08206299e76b9-Abstract-Conference.html) | ICLR 2025 | Evaluates ASR under diverse real corruptions | Directly relevant: evaluate/train the SLM for corruptions instead of assuming enhancement solves robustness |
| [Large Language Models are Efficient Learners of Noise-Robust Speech Recognition](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13f80f05afe56daf8b65fc4384ab6d09-Abstract-Conference.html) | ICLR 2024 | Represents noise conditions in language space for robust ASR | Supports noise-aware model training as a complementary path |
| [Are Deep Speech Denoising Models Robust to Adversarial Noise?](https://openreview.net/pdf?id=WtH2JxKJKf) | ICLR 2026 | Shows several DNS models can be made unintelligible by hidden adversarial noise | Important safety warning: enhancement output cannot be trusted as a safety boundary |

The ICML 2024 and 2025 proceedings search did not surface a directly relevant,
accepted, open, causal monaural speech-enhancement system. Papers that use the
word “denoising” there are often about diffusion, representation learning, or
theoretical in-context denoising rather than acoustic speech enhancement. Venue
prestige is therefore not a useful deployment filter by itself.

### Speech and audio venues

| Work | Venue/status | Deployment facts | Local conclusion |
| --- | --- | --- | --- |
| [FastEnhancer](https://github.com/aask1357/fastenhancer) | ICASSP 2026 | Causal; official 16/48 kHz ONNX; T/B/S/M/L sizes; 16 ms delay for the tested S model | Current default |
| [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas) | IEEE/ACM TASLP 2026 | Ultra-light model; official checkpoint and streaming ONNX | Deployable, but 6/7 clean and 0/14 noisy exact calls locally |
| [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) | ICASSP 2024 | Tiny causal model; 48.2K parameters and streaming implementation | Excellent tiny baseline; weaker local noisy command preservation |
| [aTENNuate](https://www.isca-archive.org/interspeech_2025/pei25_interspeech.html) | Interspeech 2025 | Raw-waveform state-space denoising with low parameter/MAC/latency claims | Promising research; no stable tested edge artifact in this checkout |
| [DeepFilterGAN](https://www.isca-archive.org/interspeech_2025/serbest25_interspeech.html) | Interspeech 2025 | Low-latency full-band predictive enhancement plus stochastic regeneration | Perceptually attractive, but stochastic regeneration is unnecessary risk for command parsing |
| [FATE](https://www.isca-archive.org/interspeech_2025/dang25_interspeech.html) | Interspeech 2025 | Classify the degradation first; only run the matching enhancement path | Correct production direction; prevents unnecessary clean-speech processing |
| [Diffusion Buffer](https://www.isca-archive.org/interspeech_2025/lay25_interspeech.html) | Interspeech 2025 | Online diffusion with roughly 0.3–1 s latency | Too slow for the primary robot turn path |
| [URGENT 2025 challenge](https://www.isca-archive.org/interspeech_2025/saijo25_interspeech.html) | Interspeech 2025 | Tests multilingual, multi-degradation universal enhancement | Useful future benchmark design; winning universal systems are not necessarily small/causal |
| [CAGCRN](https://www.isca-archive.org/interspeech_2025/wang25d_interspeech.html) | Interspeech 2025 | Joint acoustic echo cancellation and noise suppression in about 0.07M parameters | Especially relevant when robot TTS/barge-in is enabled |

DeepFilterNet3 and RNNoise remain mature engineering baselines. They are not
the newest papers, but maturity, licensing, and target-runtime tooling often
matter more than a small benchmark gain. DeepFilterNet is full-band 48 kHz,
whereas this parser consumes 16 kHz audio, so it adds resampling and compute
without a clear downstream benefit.

## Local benchmark

The test used the seven bundled Vietnamese tool-call WAVs. Every clean source
produced the expected call without enhancement. Severe noisy cases were made at
0 and 5 dB by rescaling the lowest-energy 20 ms frames from recent web
captures. This is deliberately difficult and may amplify residual speech or
browser artifacts; it is not a substitute for a dedicated room-noise capture.

| Frontend | Clean exact | Noisy exact, 14 cases | 5 dB exact | Mean SI-SDR on two official UL-UNAS pairs | Isolated streaming RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 7/7 | 4/14 | 2/7 | 1.57 dB | 0 |
| GTCRN | 7/7 | 1/14 | 1/7 | 4.73 dB | 0.032 |
| DPDFNet2 | 4/7 | 5/14 | **4/7** | 9.91 dB | 0.080 |
| DPDFNet4 | 6/7 | 5/14 | 3/7 | 9.85 dB | 0.12–0.14 |
| DPDFNet8 | 6/7 | 4/14 | 3/7 | **10.20 dB** | 0.22 |
| FastEnhancer-B | 7/7 | 2/14 | 2/7 | 6.39 dB | **0.013** |
| **FastEnhancer-S** | **7/7** | **5/14** | 3/7 | 8.78 dB | **0.021** |
| UL-UNAS | 6/7 | 0/14 | 0/7 | 6.99 dB | 0.088 including STFT/iSTFT |

The full web path was also checked with the 2 s fog-light sample. One warm run
with FastEnhancer-S returned the correct call in 500.5 ms from annotated speech
end to the last vLLM token; the no-enhancement comparison was 456.1 ms. The
observed difference (44.4 ms) includes 16 ms model delay, frame buffering,
endpoint movement, CPU contention, and run-to-run model variance. It is not a
pure denoiser latency measurement.

Reproduce the full sweep with the isolated benchmark environment:

```bash
.venvs/ulunas/bin/python scripts/benchmark_speech_enhancement.py \
  --snr-db 0 5 \
  --output-dir outputs/denoise/benchmark-2026-08-12
```

The JSON result and listenable A/B WAVs are under the ignored output directory.

## Production best practice

1. **Fix capture geometry first.** Put microphones away from fans/motors and
   speakers. A small microphone array plus beamforming usually gives a
   far-field robot more information than any single-channel denoiser.
2. **AEC before denoising.** For robot TTS or barge-in, feed the exact speaker
   playback reference into an acoustic echo canceller, then run noise
   suppression. A denoiser is not an echo canceller.
3. **Do not stack uncontrolled denoisers.** With server enhancement enabled,
   browser `noiseSuppression` is off and AGC is off. Echo cancellation remains
   on only for the browser demo.
4. **Train for the deployment channel.** Mix Vietnamese training utterances
   with measured fan, wheel, servo, office, music, TV, competing speech, and
   robot playback at multiple SNRs/distances/reverberation levels. Include
   clean audio and outputs from the deployed enhancer. Exact tool/argument loss
   matters more than perceptual MOS.
5. **Gate or bypass when clean.** FATE's analyze-then-enhance idea is sound.
   Before adding a gate, collect enough clean and degraded robot data to prove
   the gate itself does not make worse decisions.
6. **Evaluate continuous operation.** Report false starts/hour on noise-only
   recordings, command miss rate, clipped onset/tail, exact tool calls, p95/p99
   frame time, speech-end latency, double-talk behavior, and temperature/power
   on Jetson.
7. **Keep a safe fallback.** Low-confidence or contradictory outputs should
   cause a clarification or cloud route, never direct unsafe execution. The
   ICLR 2026 robustness result is a reminder that a neural denoiser is not a
   security control.

## Next data to record

Before promoting the frontend to the robot, record synchronized 16 kHz PCM in
these conditions:

- 10 minutes of silence/ambient noise per room;
- each robot motor/fan/action separately and in combination;
- robot TTS playback with and without a human speaking over it;
- near (0.5 m), normal (1–2 m), and far-field commands;
- a fixed command list with clean close-mic reference and robot-mic recording.

That data enables meaningful DNS improvement, exact-call accuracy, VAD false
activation/hour, AEC, and microphone-array comparisons.
