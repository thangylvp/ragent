# Component webtest frontend

The buildless HTML/CSS/JavaScript page captures microphone audio with an
`AudioWorklet`, resamples it to 16 kHz PCM16LE and sends fixed packets over a
WebSocket. It displays levels and structured backend events, but canonical VAD
decisions remain server-side in `src/vad`.
