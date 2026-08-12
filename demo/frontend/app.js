"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    backend: $("backend"), start: $("start"), stop: $("stop"), status: $("status"),
    vadState: $("vadState"), levelBar: $("levelBar"), levelText: $("levelText"),
    audioTime: $("audioTime"), confidence: $("confidence"), frameCost: $("frameCost"),
    utteranceTime: $("utteranceTime"), capture: $("capture"), modelBadge: $("modelBadge"),
    route: $("route"), emptyResult: $("emptyResult"), result: $("result"),
    transcriptBox: $("transcriptBox"), transcript: $("transcript"), calls: $("calls"),
    e2eLatency: $("e2eLatency"), modelLatency: $("modelLatency"),
    outputTokens: $("outputTokens"), timings: $("timings"), raw: $("raw"),
    trace: $("trace"), tools: $("tools"), toolCount: $("toolCount"),
  };
  let socket = null;
  let audio = null;
  let listening = false;
  let config = null;

  const escapeHtml = (value) => String(value == null ? "" : value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

  function addTrace(label, detail = "") {
    const item = document.createElement("li");
    const now = new Date().toLocaleTimeString([], { hour12: false });
    item.innerHTML = `<time>${now}</time><strong>${escapeHtml(label)}</strong><span>${escapeHtml(detail)}</span>`;
    ui.trace.prepend(item);
  }

  function setState(value) {
    const normalized = (value || "idle").toLowerCase();
    ui.vadState.textContent = normalized.toUpperCase();
    ui.vadState.className = `state ${normalized}`;
  }

  function setLevel(dbfs) {
    const value = Number.isFinite(dbfs) ? dbfs : -120;
    const width = Math.max(0, Math.min(100, (value + 70) / 60 * 100));
    ui.levelBar.style.width = `${width}%`;
    ui.levelText.textContent = `${value.toFixed(1)} dBFS`;
  }

  function modelBadge(info) {
    const ready = info && info.ready;
    ui.modelBadge.classList.toggle("ready", !!ready);
    const target = info ? (info.model || info.kind || "model") : "unknown";
    let state = "offline";
    if (ready) state = "ready";
    else if (info && info.error) state = "error";
    else if (info && info.kind === "local") state = "loads on first turn";
    ui.modelBadge.querySelector("span").textContent = `${target} · ${state}`;
    ui.modelBadge.title = info && info.error ? info.error : "";
  }

  function resetResult() {
    ui.emptyResult.hidden = false;
    ui.result.hidden = true;
    ui.route.textContent = "WAITING";
    ui.route.className = "state neutral";
    ui.calls.innerHTML = "";
    ui.raw.textContent = "";
    ui.transcriptBox.hidden = true;
  }

  function renderResult(payload) {
    const result = payload.result;
    ui.emptyResult.hidden = true;
    ui.result.hidden = false;
    ui.route.textContent = result.route.toUpperCase();
    ui.route.className = `state ${result.route === "tool" ? "tool" : result.route}`;
    ui.e2eLatency.textContent = `${payload.end_to_end_from_vad_ms.toFixed(1)} ms`;
    ui.modelLatency.textContent = `${result.latency_ms.toFixed(1)} ms`;
    ui.outputTokens.textContent = result.output_tokens == null ? "—" : result.output_tokens;
    const timingRows = [
      ["Capture WAV", payload.component_timings && payload.component_timings.capture_write_ms],
      ["Lazy model load", result.timings && result.timings.load_ms],
      ["Audio decode", result.timings && result.timings.audio_decode_ms],
      ["Prompt render", result.timings && result.timings.prompt_render_ms],
      ["Feature extraction", result.timings && result.timings.feature_extraction_ms],
      ["Host → GPU", result.timings && result.timings.host_to_device_ms],
      ["Generate to first token", result.timings && result.timings.generation_to_first_token_ms],
      ["First → last token", result.timings && result.timings.first_to_last_token_ms],
      ["Generate through last token", result.timings && result.timings.generation_to_last_token_ms],
      ["Decode + parse", result.timings && result.timings.decode_parse_ms],
    ].filter((row) => row[1] != null);
    ui.timings.innerHTML = timingRows.map((row) =>
      `<div><dt>${escapeHtml(row[0])}</dt><dd>${Number(row[1]).toFixed(2)} ms</dd></div>`
    ).join("");
    ui.raw.textContent = result.raw || "(empty output)";
    ui.transcriptBox.hidden = !result.transcript;
    ui.transcript.textContent = result.transcript || "";
    ui.calls.innerHTML = (result.calls || []).length
      ? result.calls.map((call) => `
          <article class="call">
            <h3>${escapeHtml(call.name)}</h3>
            <pre>${escapeHtml(JSON.stringify(call.arguments || {}, null, 2))}</pre>
          </article>`).join("")
      : '<p class="empty small">No structured call parsed.</p>';
    modelBadge(payload.model);
  }

  async function stopAudio() {
    if (!audio) return;
    const current = audio;
    audio = null;
    try { current.node.disconnect(); } catch (_) { /* no-op */ }
    try { current.source.disconnect(); } catch (_) { /* no-op */ }
    current.stream.getTracks().forEach((track) => track.stop());
    try { await current.context.close(); } catch (_) { /* no-op */ }
  }

  async function openMicrophone(onPcm) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    await context.audioWorklet.addModule("/static/audio-worklet.js");
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "pcm16-worklet");
    const silent = context.createGain();
    silent.gain.value = 0;
    node.port.onmessage = (event) => {
      if (event.data.type === "pcm") onPcm(event.data.buffer);
      if (event.data.type === "level") setLevel(event.data.dbfs);
    };
    source.connect(node);
    node.connect(silent).connect(context.destination);
    audio = { stream, context, source, node, silent };
    addTrace("Microphone opened", `${context.sampleRate} Hz browser input → 16 kHz PCM`);
  }

  function connectSocket() {
    return new Promise((resolve, reject) => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/api/audio/stream`);
      socket.binaryType = "arraybuffer";
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error("WebSocket connection failed"));
      socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
      socket.onclose = () => {
        if (listening) finishListening();
      };
    });
  }

  async function startListening() {
    if (listening) return;
    resetResult();
    ui.capture.hidden = true;
    ui.utteranceTime.textContent = "—";
    ui.status.textContent = "Opening microphone…";
    try {
      await connectSocket();
      socket.send(JSON.stringify({
        event: "start_stream", backend: ui.backend.value, sample_rate: 16000,
        channels: 1, encoding: "pcm16le",
      }));
      await openMicrophone((buffer) => {
        if (socket && socket.readyState === WebSocket.OPEN) socket.send(buffer);
      });
      listening = true;
      ui.start.disabled = true;
      ui.backend.disabled = true;
      ui.stop.disabled = false;
      ui.status.textContent = "Listening continuously. Speak once, then stay quiet for endpointing.";
      addTrace("Listening started", ui.backend.value);
    } catch (error) {
      if (socket) {
        try { socket.close(); } catch (_) { /* no-op */ }
        socket = null;
      }
      await finishListening();
      ui.status.textContent = error.message;
      addTrace("Start failed", error.message);
    }
  }

  async function finishListening() {
    listening = false;
    await stopAudio();
    ui.start.disabled = false;
    ui.backend.disabled = false;
    ui.stop.disabled = true;
    setState("idle");
    setLevel(-120);
  }

  async function stopListening() {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ event: "stop_stream" }));
    }
    await finishListening();
    ui.status.textContent = "Stopped. Partial speech was discarded and not sent to the model.";
    addTrace("Manual stop", "partial audio discarded");
  }

  async function handleEvent(message) {
    switch (message.event) {
      case "stream_started":
        addTrace("VAD initialized", `${message.backend}, ${message.frame_ms} ms frames`);
        break;
      case "vad_frame":
        setState(message.state);
        setLevel(message.level_dbfs);
        ui.audioTime.textContent = `${(message.audio_ms / 1000).toFixed(2)} s`;
        ui.confidence.textContent = message.confidence == null ? "—" : message.confidence.toFixed(3);
        ui.frameCost.textContent = `${message.process_ms.toFixed(2)} ms`;
        break;
      case "utterance_started":
        setState("speech");
        ui.status.textContent = "Speech confirmed. Keep talking; silence will finalize the turn.";
        addTrace("Speech started", `retroactive onset ${message.start_ms} ms`);
        break;
      case "utterance_rejected":
        addTrace("Segment rejected", message.reason || "too short");
        break;
      case "utterance_finalized":
        await finishListening();
        ui.utteranceTime.textContent = `${(message.duration_ms / 1000).toFixed(2)} s`;
        ui.capture.src = `${message.capture_url}?v=${Date.now()}`;
        ui.capture.hidden = false;
        ui.status.textContent = "Utterance finalized. Running the car model…";
        addTrace("Utterance finalized", `${message.duration_ms} ms · ${message.reason}`);
        break;
      case "model_started":
        ui.route.textContent = "RUNNING";
        ui.route.className = "state running";
        addTrace("Model started", message.model.kind);
        break;
      case "model_result":
        renderResult(message);
        ui.status.textContent = "Complete. Review the captured WAV and structured output.";
        addTrace("Model complete", `${message.result.route} · ${message.result.latency_ms} ms`);
        break;
      case "stream_stopped":
        addTrace("Stream stopped", message.reason);
        break;
      case "error":
        await finishListening();
        ui.status.textContent = message.message;
        ui.route.textContent = "ERROR";
        ui.route.className = "state error";
        addTrace("Error", message.message);
        break;
      default:
        addTrace(message.event || "Unknown event", "");
    }
  }

  async function boot() {
    if (!navigator.mediaDevices || !window.AudioWorkletNode) {
      ui.status.textContent = "This browser does not support the required microphone AudioWorklet API.";
      ui.start.disabled = true;
      return;
    }
    try {
      const response = await fetch("/api/config");
      if (!response.ok) throw new Error(await response.text());
      config = await response.json();
      modelBadge(config.model);
      ui.backend.innerHTML = config.vad_backends.map((item) => `
        <option value="${escapeHtml(item.name)}" ${item.available ? "" : "disabled"}>
          ${escapeHtml(item.name)} — ${escapeHtml(item.description)}${item.available ? "" : " (unavailable)"}
        </option>`).join("");
      ui.backend.value = config.default_vad;
      ui.toolCount.textContent = config.tools.length;
      ui.tools.innerHTML = config.tools.map((tool) =>
        `<span title="${escapeHtml(tool.description)}">${escapeHtml(tool.name)}</span>`).join("");
      addTrace("Webtest ready", `${config.tools.length} original STC tools loaded`);
    } catch (error) {
      ui.status.textContent = `Cannot load server config: ${error.message}`;
      ui.start.disabled = true;
    }
  }

  ui.start.addEventListener("click", startListening);
  ui.stop.addEventListener("click", stopListening);
  $("clearTrace").addEventListener("click", () => { ui.trace.innerHTML = ""; });
  window.addEventListener("beforeunload", () => { if (socket) socket.close(); });
  boot();
})();
