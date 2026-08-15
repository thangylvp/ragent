"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    backend: $("backend"), enhancer: $("enhancer"), start: $("start"), stop: $("stop"),
    status: $("status"), listenTitle: $("listenTitle"), listenOrb: $("listenOrb"),
    levelBar: $("levelBar"), levelText: $("levelText"), conversation: $("conversation"),
    modelBadge: $("modelBadge"), cloudBadge: $("cloudBadge"),
    trace: $("trace"), beforeCapture: $("beforeCapture"), capture: $("capture"),
    busyToggle: $("busyToggle"), vehicleToggle: $("vehicleToggle"), hardwareMode: $("hardwareMode"), battery: $("battery"),
    batteryBar: $("batteryBar"), climatePower: $("climatePower"), temperature: $("temperature"),
    fan: $("fan"), playback: $("playback"), mediaSource: $("mediaSource"), volume: $("volume"),
    trunk: $("trunk"), driverWindow: $("driverWindow"), sunroof: $("sunroof"),
    driveMode: $("driveMode"), vehicleState: $("vehicleState"), ambient: $("ambient"), connectivity: $("connectivity"),
    actionCount: $("actionCount"), hardwareEvents: $("hardwareEvents"),
  };
  let socket = null;
  let audioInput = null;
  let listening = false;
  let inputGate = false;
  let serverInputGate = false;
  let inputPreviewPlayer = null;
  const turns = new Map();

  const escapeHtml = (value) => String(value == null ? "" : value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const ms = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)} ms` : "—";
  const viState = (value) => ({ on: "Bật", off: "Tắt", playing: "Đang phát", paused: "Tạm dừng", stopped: "Đã dừng", open: "Mở", closed: "Đóng" }[value] || value || "—");

  function trace(label, detail = "") {
    const item = document.createElement("li");
    item.innerHTML = `<time>${new Date().toLocaleTimeString("vi-VN", { hour12: false })}</time>${escapeHtml(label)} ${escapeHtml(detail)}`;
    ui.trace.prepend(item);
  }

  function setLevel(dbfs) {
    const value = Number.isFinite(dbfs) ? dbfs : -120;
    ui.levelBar.style.width = `${Math.max(0, Math.min(100, (value + 70) / 60 * 100))}%`;
    ui.levelText.textContent = `${value.toFixed(1)} dBFS`;
  }

  function setStage(stage, title, detail) {
    ui.listenOrb.className = `listen-orb ${stage}`;
    ui.listenTitle.textContent = title;
    ui.status.textContent = detail;
  }

  function badge(element, ready, label, detail) {
    element.className = `pill ${ready === true ? "ready" : ready === null ? "configured" : "error"}`;
    element.innerHTML = `<i></i><b>${escapeHtml(label)}</b> ${escapeHtml(detail)}`;
  }

  function renderHardware(state) {
    if (!state) return;
    const busy = !!state.busy;
    ui.busyToggle.checked = busy;
    ui.vehicleToggle.checked = !!state.vehicle.running;
    ui.hardwareMode.className = `mode ${busy ? "busy" : "ready"}`;
    ui.hardwareMode.innerHTML = `<i></i>${busy ? "ĐANG BẬN" : "SẴN SÀNG"}`;
    ui.battery.textContent = `${state.battery}%`;
    ui.batteryBar.style.width = `${state.battery}%`;
    ui.climatePower.textContent = state.climate.power === "on" ? "Đang bật" : "Đang tắt";
    ui.temperature.textContent = `${state.climate.temperature.driver}°C`;
    ui.fan.textContent = `${state.climate.fan} / 7`;
    ui.playback.textContent = viState(state.media.playback);
    ui.mediaSource.textContent = state.media.source || "—";
    ui.volume.textContent = state.media.muted ? "Tắt tiếng" : `${state.media.volume}%`;
    ui.trunk.textContent = `Cốp ${viState(state.access.trunk).toLowerCase()}`;
    ui.driverWindow.textContent = state.access.windows.driver ? `${state.access.windows.driver}%` : "Đóng";
    ui.sunroof.textContent = state.access.sunroof ? `${state.access.sunroof}%` : "Đóng";
    ui.driveMode.textContent = state.drive.mode;
    ui.vehicleState.textContent = state.vehicle.running
      ? `Đang chạy · ${Number(state.vehicle.speed_kph || 0).toFixed(0)} km/h`
      : "Đã dừng";
    ui.ambient.textContent = viState(state.lighting.ambient.state);
    const activeConnections = Object.entries(state.connectivity)
      .filter(([key, value]) => key !== "device" && value === "on").map(([key]) => key);
    ui.connectivity.textContent = activeConnections.join(" + ") || "Tắt";
    ui.actionCount.textContent = `${state.action_count} lệnh`;
    const events = [...(state.events || [])].reverse();
    ui.hardwareEvents.innerHTML = events.length
      ? events.map((event) => `<li><i></i><span>${escapeHtml(event)}</span></li>`).join("")
      : "<li><i></i><span>Hệ thống đã sẵn sàng</span></li>";
  }

  function clearWelcome() {
    const welcome = ui.conversation.querySelector(".welcome");
    if (welcome) welcome.remove();
  }

  function addTurn(role, text, meta = "", code = "") {
    clearWelcome();
    const turn = document.createElement("article");
    turn.className = `turn ${role}`;
    const avatar = role === "user"
      ? "BẠN"
      : role === "slm"
        ? "SLM"
        : role === "cloud"
          ? "CLOUD"
          : "AI";
    turn.innerHTML = `<span class="avatar">${avatar}</span><div class="bubble"><p>${escapeHtml(text)}</p>${code ? `<code class="tool-code">${escapeHtml(code)}</code>` : ""}<div class="meta">${meta}</div></div>`;
    ui.conversation.append(turn);
    ui.conversation.scrollTop = ui.conversation.scrollHeight;
    return turn;
  }

  function refreshInputGate() {
    inputGate = serverInputGate && inputPreviewPlayer === null;
  }

  function attachInputAudio(turn, captureUrl, durationMs) {
    const container = document.createElement("section");
    container.className = "input-audio";
    container.innerHTML = `<div><span>INPUT AUDIO</span><small>${(durationMs / 1000).toFixed(1)} giây · bấm để nghe lại</small></div>`;
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    player.src = `${captureUrl}?v=${Date.now()}`;
    player.setAttribute("aria-label", "Phát lại audio đầu vào của lượt này");
    player.addEventListener("play", () => {
      const previous = inputPreviewPlayer;
      inputPreviewPlayer = player;
      if (previous && previous !== player) previous.pause();
      refreshInputGate();
      if (listening) setStage("processing", "Đang nghe lại audio đầu vào", "Micro tạm dừng để không thu lại âm thanh đang phát");
    });
    const restoreCapture = () => {
      if (inputPreviewPlayer !== player) return;
      inputPreviewPlayer = null;
      refreshInputGate();
      if (listening && inputGate) setStage("live", "Đang lắng nghe", "Bạn có thể nói yêu cầu tiếp theo");
    };
    player.addEventListener("pause", restoreCapture);
    player.addEventListener("ended", restoreCapture);
    container.append(player);
    turn.querySelector(".bubble").append(container);
  }

  function slmDetails(result) {
    const calls = Array.isArray(result.calls) ? result.calls : [];
    const parsed = result.route === "tool"
      ? JSON.stringify(calls, null, 2)
      : result.route === "non_tool"
        ? (result.transcript || "(empty transcript)")
        : "(no parsed action)";
    return `
      <section class="slm-output">
        <div>
          <span>SLM RAW OUTPUT</span>
          <pre>${escapeHtml(result.raw || "(empty output)")}</pre>
        </div>
        <div>
          <span>${result.route === "tool" ? "PARSED TOOL CALL" : "PARSED OUTPUT"}</span>
          <pre>${escapeHtml(parsed)}</pre>
        </div>
      </section>`;
  }

  function turnDiagnostics(turnId, response) {
    const turn = turns.get(turnId);
    const message = turn && turn.modelMessage;
    if (!message) return "";
    const result = message.result;
    const timing = { ...(result.timings || {}), ...(message.timings || {}) };
    const componentRows = [
      ["Capture WAV", timing.capture_write_ms],
      ["Model dispatch", timing.model_dispatch_ms],
      ["SLM request total", timing.model_thread_ms || result.latency_ms],
      ["SLM request → first token", timing.request_to_first_token_ms],
      ["First → last token", timing.first_to_last_token_ms],
      ["Harness", response.timings && response.timings.harness_total_ms],
      ["Cloud", response.timings && response.timings.cloud_ms],
      ["Execute", response.timings && response.timings.execute_ms],
      ["TTS / cache", response.timings && response.timings.tts_ms],
    ].filter((row) => row[1] != null);
    const cloudMilestone = response.route === "cloud"
      ? `<article><span>Audio → cloud LLM</span><strong>${ms(response.timings && response.timings.audio_to_cloud_llm_ms)}</strong></article>`
      : "";
    return `
      <section class="turn-observability">
        <div class="turn-times" aria-label="Độ trễ của lượt này">
          <article><span>Audio → first SLM token</span><strong>${ms(timing.audio_to_first_llm_token_ms)}</strong></article>
          <article><span>Audio → last SLM token</span><strong>${ms(timing.audio_to_last_llm_token_ms)}</strong></article>
          ${cloudMilestone}
          <article class="accent"><span>Audio → first audio</span><strong data-turn-first-audio>—</strong></article>
        </div>
        <details class="turn-breakdown">
          <summary>Component timing</summary>
          <dl>${componentRows.map((row) => `<div><dt>${escapeHtml(row[0])}</dt><dd>${ms(row[1])}</dd></div>`).join("")}</dl>
        </details>
      </section>`;
  }

  function markFirstAudio(turnId, elapsed) {
    const turn = turns.get(turnId);
    turn && turn.assistantElement
      ?.querySelector("[data-turn-first-audio]")
      ?.replaceChildren(document.createTextNode(ms(elapsed)));
  }

  async function playResponse(turnId, response, bubble) {
    const clips = (response.audio || []).filter((clip) => clip.url);
    if (!clips.length) {
      trace("Audio unavailable", "Không có clip cục bộ để phát");
      send({ event: "playback_finished", turn_id: turnId });
      setStage("live", "Không có âm thanh phản hồi", "Kiểm tra static-audio manifest hoặc dịch vụ TTS đám mây");
      return;
    }
    let firstStarted = false;
    for (const clip of clips) {
      const player = document.createElement("audio");
      player.controls = true;
      player.preload = "auto";
      player.src = `${clip.url}?v=${Date.now()}`;
      bubble.querySelector(".bubble").append(player);
      const played = new Promise((resolve) => player.addEventListener("ended", resolve, { once: true }));
      player.addEventListener("playing", () => {
        if (firstStarted) return;
        firstStarted = true;
        const turn = turns.get(turnId);
        const elapsed = turn ? performance.now() - turn.baselinePerf : NaN;
        markFirstAudio(turnId, elapsed);
        send({ event: "audio_playback_started", turn_id: turnId, audio_from_vad_ms: elapsed });
        trace("Audio playback", ms(elapsed));
      }, { once: true });
      try {
        await player.play();
      } catch (_) {
        setStage("processing", "Chờ phát âm thanh", "Bấm nút Play trong câu trả lời để tiếp tục");
      }
      await played;
    }
    send({ event: "playback_finished", turn_id: turnId });
    setStage("live", "Đang lắng nghe", "Bạn có thể nói yêu cầu tiếp theo");
  }

  function send(value) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(value));
  }

  async function stopMicrophone() {
    if (!audioInput) return;
    const current = audioInput;
    audioInput = null;
    try { current.node.disconnect(); } catch (_) { /* no-op */ }
    try { current.source.disconnect(); } catch (_) { /* no-op */ }
    current.stream.getTracks().forEach((track) => track.stop());
    try { await current.context.close(); } catch (_) { /* no-op */ }
  }

  async function openMicrophone(onPcm) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
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
    audioInput = { stream, context, source, node, silent };
  }

  function connectSocket() {
    return new Promise((resolve, reject) => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/api/audio/stream`);
      socket.binaryType = "arraybuffer";
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error("Không thể kết nối WebSocket"));
      socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
      socket.onclose = () => { if (listening) finishListening(); };
    });
  }

  async function startListening() {
    if (listening) return;
    try {
      setStage("processing", "Đang khởi tạo", "Đang mở micro và tải VAD…");
      await connectSocket();
      send({ event: "start_stream", backend: ui.backend.value, enhancer: ui.enhancer.value, sample_rate: 16000, channels: 1, encoding: "pcm16le" });
      serverInputGate = true;
      refreshInputGate();
      await openMicrophone((buffer) => {
        if (inputGate && socket && socket.readyState === WebSocket.OPEN) socket.send(buffer);
      });
      listening = true;
      ui.start.disabled = true; ui.stop.disabled = false; ui.backend.disabled = true; ui.enhancer.disabled = true;
      setStage("live", "Đang lắng nghe", "Hãy nói tự nhiên; hệ thống sẽ tự nhận biết khi bạn dừng");
      trace("Microphone started", `${ui.enhancer.value} → ${ui.backend.value}`);
    } catch (error) {
      await finishListening();
      setStage("", "Không thể bắt đầu", error.message);
    }
  }

  async function finishListening() {
    listening = false; serverInputGate = false; inputGate = false;
    await stopMicrophone();
    ui.start.disabled = false; ui.stop.disabled = true; ui.backend.disabled = false; ui.enhancer.disabled = false;
    setLevel(-120);
  }

  async function stopListening() {
    send({ event: "stop_stream" });
    await finishListening();
    setStage("", "Đã dừng", "Bấm Bắt đầu khi bạn muốn tiếp tục");
  }

  async function handleEvent(message) {
    switch (message.event) {
      case "stream_started":
        serverInputGate = true;
        refreshInputGate();
        renderHardware(message.hardware);
        trace("Pipeline ready", `${message.enhancer} → ${message.backend}`);
        break;
      case "vad_frame":
        setLevel(message.level_dbfs);
        if (inputGate && message.is_speech) setStage("live", "Đang nghe bạn nói", `Độ tin cậy giọng nói ${Number(message.confidence || 0).toFixed(2)}`);
        break;
      case "utterance_started":
        setStage("live", "Đã phát hiện giọng nói", "Hãy tiếp tục nói, khoảng lặng sẽ kết thúc lượt");
        break;
      case "utterance_rejected":
        trace("VAD rejected", message.reason || "too short");
        break;
      case "input_gate":
        serverInputGate = message.state === "open";
        refreshInputGate();
        if (inputGate) { renderHardware(message.hardware); setStage("live", "Đang lắng nghe", "Bạn có thể nói yêu cầu tiếp theo"); }
        break;
      case "utterance_finalized": {
        serverInputGate = false;
        refreshInputGate();
        turns.set(message.turn_id, { baselinePerf: performance.now() });
        ui.beforeCapture.src = `${message.before_enhancement_capture_url}?v=${Date.now()}`;
        ui.capture.src = `${message.capture_url}?v=${Date.now()}`;
        const turn = addTurn("user", "Audio đầu vào", `<span>${(message.duration_ms / 1000).toFixed(1)} giây</span><span>đầu vào SLM</span>`);
        attachInputAudio(turn, message.capture_url, message.duration_ms);
        turns.get(message.turn_id).userElement = turn;
        setStage("processing", "Đang suy nghĩ", "SLM đang phân tích lời nói");
        trace("Utterance finalized", `${message.duration_ms} ms`);
        break;
      }
      case "model_started":
        trace("SLM started", message.model.kind);
        break;
      case "model_result": {
        const result = message.result;
        const turn = turns.get(message.turn_id);
        if (turn) turn.modelMessage = message;
        const description = result.route === "non_tool"
          ? `Chuyển tiếp hội thoại: ${result.transcript || "không có bản ghi"}`
          : result.route === "tool"
            ? `Tool call: ${(result.calls || []).map((call) => call.name).join(" + ") || "không hợp lệ"}`
            : "Không xác định được route";
        if (turn) {
          const timing = { ...(result.timings || {}), ...(message.timings || {}) };
          const meta = `<span class="route-chip">${escapeHtml(result.route)}</span><span>TTFT ${ms(timing.audio_to_first_llm_token_ms)}</span><span>hoàn tất ${ms(timing.audio_to_last_llm_token_ms)}</span>`;
          const slmTurn = addTurn("slm", description, meta);
          slmTurn.querySelector(".meta").insertAdjacentHTML("beforebegin", slmDetails(result));
          turn.slmElement = slmTurn;
        }
        trace("SLM complete", `${result.route} · ${ms(result.latency_ms)}`);
        break;
      }
      case "assistant_response": {
        const response = message.response;
        const turn = turns.get(message.turn_id);
        if (response.route === "cloud" || response.route === "cloud_error") {
          const cloudText = response.route === "cloud"
            ? response.assistant_text
            : "Cloud Agent không trả về được câu trả lời cho lượt này.";
          const cloudLatency = response.route === "cloud"
            ? ms(response.timings && response.timings.cloud_ms)
            : "request failed";
          const cloudMeta = `<span class="route-chip">${escapeHtml(response.route)}</span><span>${escapeHtml(response.cloud_model || "cloud unavailable")}</span><span>${cloudLatency}</span>`;
          const cloudTurn = addTurn("cloud", cloudText, cloudMeta);
          if (turn) turn.cloudElement = cloudTurn;
        }
        const audioMode = response.timings && response.timings.audio_mode;
        const meta = `<span class="route-chip">${escapeHtml(response.route)}</span><span>final voice output</span><span>${escapeHtml(audioMode || "audio")}</span>`;
        const bubble = addTurn("assistant", response.assistant_text, meta);
        if (turn) turn.assistantElement = bubble;
        bubble.querySelector(".bubble").insertAdjacentHTML(
          "beforeend",
          turnDiagnostics(message.turn_id, response),
        );
        renderHardware(response.hardware);
        setStage("processing", "Đang phản hồi", "Đang phát câu trả lời tiếng Việt");
        trace("Harness complete", `${response.route} · ${ms(response.timings.harness_total_ms)}`);
        if ((response.errors || []).some((error) => error.startsWith("cloud:"))) {
          badge(ui.cloudBadge, false, "Cloud", "authentication error");
        }
        playResponse(message.turn_id, response, bubble);
        break;
      }
      case "playback_acknowledged":
        break;
      case "stream_stopped":
        trace("Stream stopped");
        break;
      case "error":
        await finishListening();
        setStage("", "Có lỗi xảy ra", message.message);
        trace("Error", message.message);
        break;
      default:
        trace(message.event || "Unknown event");
    }
  }

  async function setBusy() {
    const response = await fetch(`/api/hardware/busy?busy=${ui.busyToggle.checked}`, { method: "POST" });
    renderHardware(await response.json());
  }

  async function setVehicleRunning() {
    const running = ui.vehicleToggle.checked;
    const response = await fetch(`/api/hardware/vehicle?running=${running}&speed_kph=${running ? 5 : 0}`, { method: "POST" });
    renderHardware(await response.json());
  }

  async function resetDemo() {
    const response = await fetch("/api/hardware/reset", { method: "POST" });
    renderHardware(await response.json());
    ui.conversation.innerHTML = `<div class="welcome"><span class="welcome-icon">✦</span><h2>Đã bắt đầu cuộc trò chuyện mới</h2><p>Bạn có thể nói một yêu cầu khác.</p></div>`;
    turns.clear();
    trace("Demo reset");
  }

  async function boot() {
    try {
      const response = await fetch("/api/config");
      if (!response.ok) throw new Error(await response.text());
      const config = await response.json();
      const modelReady = config.model.ready || config.model.kind === "local";
      badge(ui.modelBadge, modelReady, "SLM", config.model.model || config.model.kind);
      badge(
        ui.cloudBadge,
        config.harness.cloud.enabled ? config.harness.cloud.ready : false,
        "Cloud",
        config.harness.cloud.enabled
          ? `${config.harness.cloud.model} · ${
              config.harness.cloud.ready == null
                ? "configured"
                : config.harness.cloud.ready
                  ? "ready"
                  : "error"
            }`
          : "disabled",
      );
      ui.backend.innerHTML = config.vad_backends.map((item) => `<option value="${escapeHtml(item.name)}" ${item.available ? "" : "disabled"}>${escapeHtml(item.name)}${item.available ? "" : " (unavailable)"}</option>`).join("");
      ui.backend.value = config.default_vad;
      ui.enhancer.innerHTML = config.enhancers.map((item) => `<option value="${escapeHtml(item.name)}" ${item.available ? "" : "disabled"}>${escapeHtml(item.name)}${item.available ? "" : " (unavailable)"}</option>`).join("");
      ui.enhancer.value = config.default_enhancer;
      $("voiceDisclosure").textContent = config.voice_disclosure;
      renderHardware(config.harness.hardware);
      trace("Demo ready", `${config.tools.length} tools`);
    } catch (error) {
      ui.start.disabled = true;
      setStage("", "Không thể tải demo", error.message);
    }
  }

  ui.start.addEventListener("click", startListening);
  ui.stop.addEventListener("click", stopListening);
  $("settingsToggle").addEventListener("click", () => { $("audioSettings").hidden = !$("audioSettings").hidden; });
  ui.busyToggle.addEventListener("change", setBusy);
  ui.vehicleToggle.addEventListener("change", setVehicleRunning);
  $("resetHardware").addEventListener("click", resetDemo);
  $("newConversation").addEventListener("click", resetDemo);
  window.addEventListener("beforeunload", () => { if (socket) socket.close(); });
  boot();
})();
