(function () {
  const cfg = window.TRUSTMARK_PLAYER_CONFIG || {};
  const playlistUrl = cfg.playlistUrl;
  const statusUrl = cfg.statusUrl;
  const playUrl = cfg.playUrl;
  const pauseUrl = cfg.pauseUrl;
  const heartbeatUrl = cfg.heartbeatUrl;
  const heartbeatIntervalMs = cfg.heartbeatIntervalMs || 10000;

  const overlay = document.getElementById("overlay");
  const spinner = document.getElementById("spinner");
  const statusText = document.getElementById("statusText");
  const video = document.getElementById("video");
  const playPauseBtn = document.getElementById("playPauseBtn");
  const fullscreenBtn = document.getElementById("fullscreenBtn");
  const timeline = document.getElementById("timeline");
  const timeLabel = document.getElementById("timeLabel");
  const sessionLabel = document.getElementById("sessionLabel");

  let hls = null;
  let currentStatus = null;
  let expectedGeneration = 0;
  let readinessHandle = null;
  let heartbeatHandle = null;
  let timelineDragging = false;
  let transitionPending = false;
  let startOnReady = false;
  let playerAttached = false;
  let playbackStarted = false;
  let pendingStartPositionSeconds = 0;
  let displayAnchorPositionSeconds = 0;
  let invalidated = false;

  function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) {
      seconds = 0;
    }
    const whole = Math.floor(seconds);
    const hrs = Math.floor(whole / 3600);
    const mins = Math.floor((whole % 3600) / 60);
    const secs = whole % 60;
    if (hrs > 0) {
      return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }
    return `${mins}:${String(secs).padStart(2, "0")}`;
  }

  function showOverlay(message, spinning) {
    overlay.classList.remove("hidden");
    statusText.textContent = message || "Preparing stream...";
    statusText.classList.remove("error-text");
    spinner.classList.toggle("hidden", !spinning);
  }

  function showError(message) {
    overlay.classList.remove("hidden");
    spinner.classList.add("hidden");
    statusText.textContent = message || "Stream failed.";
    statusText.classList.add("error-text");
  }

  function hideOverlay() {
    overlay.classList.add("hidden");
  }

  function setControlsDisabled(disabled) {
    playPauseBtn.disabled = disabled;
    fullscreenBtn.disabled = disabled;
    timeline.disabled = disabled;
  }

  function applyInvalidatedState() {
    invalidated = true;
    transitionPending = false;
    startOnReady = false;
    stopHeartbeat();
    if (readinessHandle !== null) {
      window.clearInterval(readinessHandle);
      readinessHandle = null;
    }
    destroyHls();
    setControlsDisabled(true);
    showError("The current session has been invalidated.");
  }

  function loadingMessageForStatus(status) {
    if (!status || !status.spool_bytes_ever_arrived) {
      return "Connecting to source and buffering stream...";
    }
    if (!status.spool_media_ready) {
      return "Preparing playable startup buffer...";
    }
    return "Buffering stream...";
  }

  function destroyHls() {
    if (hls) {
      hls.destroy();
      hls = null;
    }
    playerAttached = false;
    playbackStarted = false;
    video.removeAttribute("src");
    try {
      video.load();
    } catch (_err) {}
  }

  async function postJson(url, payload) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    if (!resp.ok) {
      let detail = "Request failed.";
      try {
        const data = await resp.json();
        detail = data.detail || detail;
      } catch (_err) {}
      throw new Error(detail);
    }
    return resp.json();
  }

  async function fetchStatus() {
    const resp = await fetch(statusUrl, { cache: "no-store" });
    if (!resp.ok) {
      throw new Error("Failed to fetch stream status.");
    }
    return resp.json();
  }

  function syncPlayPauseButton() {
    const localPlaying = playerAttached && !video.paused && !video.ended;
    playPauseBtn.textContent = localPlaying ? "Pause" : "Play";
  }

  function visibleTimelinePosition() {
    if (timelineDragging) {
      return Number(timeline.value || 0);
    }
    if (!currentStatus) {
      return pendingStartPositionSeconds || 0;
    }
    if (playbackStarted && playerAttached) {
      return Math.min(
        currentStatus.duration_seconds || 0,
        displayAnchorPositionSeconds + (video.currentTime || 0),
      );
    }
    if (transitionPending) {
      return pendingStartPositionSeconds;
    }
    if (playerAttached && currentStatus.state === "streaming") {
      return displayAnchorPositionSeconds;
    }
    return currentStatus.saved_position_seconds || currentStatus.logical_position_seconds || 0;
  }

  function currentLogicalPosition() {
    if (!currentStatus) {
      return pendingStartPositionSeconds || 0;
    }
    if (playbackStarted && playerAttached && currentStatus.state === "streaming") {
      return Math.min(
        currentStatus.duration_seconds || 0,
        displayAnchorPositionSeconds + (video.currentTime || 0),
      );
    }
    if (transitionPending) {
      return pendingStartPositionSeconds;
    }
    if (playerAttached && currentStatus.state === "streaming") {
      return displayAnchorPositionSeconds;
    }
    return currentStatus.saved_position_seconds || currentStatus.logical_position_seconds || 0;
  }

  function updateTimeline(positionSeconds, durationSeconds) {
    timeline.max = String(Math.max(1, durationSeconds || 1));
    if (!timelineDragging) {
      timeline.value = String(Math.min(durationSeconds || 0, Math.max(0, positionSeconds || 0)));
    }
    timeLabel.textContent = `${formatTime(positionSeconds || 0)} / ${formatTime(durationSeconds || 0)}`;
  }

  function markPlaybackStarted() {
    playbackStarted = true;
    if (currentStatus) {
      displayAnchorPositionSeconds = currentStatus.anchor_time_seconds || pendingStartPositionSeconds || 0;
      updateTimeline(visibleTimelinePosition(), currentStatus.duration_seconds);
    }
  }

  function applyStatus(status) {
    currentStatus = status;
    expectedGeneration = status.generation || expectedGeneration;
    if (status.state === "invalidated") {
      applyInvalidatedState();
      sessionLabel.textContent = "";
      return;
    }
    invalidated = false;
    setControlsDisabled(false);
    if (!playbackStarted && status.state === "streaming") {
      displayAnchorPositionSeconds = status.anchor_time_seconds || pendingStartPositionSeconds || 0;
    }
    updateTimeline(visibleTimelinePosition(), status.duration_seconds);
    sessionLabel.textContent = "";
    syncPlayPauseButton();
  }

  function startHeartbeat() {
    if (heartbeatHandle !== null) {
      window.clearInterval(heartbeatHandle);
    }
    heartbeatHandle = window.setInterval(async function () {
      if (!currentStatus) {
        return;
      }
      if (!(transitionPending || currentStatus.state === "buffering" || currentStatus.state === "streaming")) {
        return;
      }
      try {
        await postJson(heartbeatUrl, {
          position_seconds: currentLogicalPosition(),
        });
      } catch (_err) {}
    }, heartbeatIntervalMs);
  }

  function stopHeartbeat() {
    if (heartbeatHandle !== null) {
      window.clearInterval(heartbeatHandle);
      heartbeatHandle = null;
    }
  }

  function attachPlayer(autoPlay) {
    destroyHls();
    const src = `${playlistUrl}?generation=${expectedGeneration || 0}&t=${Date.now()}`;
    displayAnchorPositionSeconds = currentStatus
      ? (currentStatus.anchor_time_seconds || pendingStartPositionSeconds || 0)
      : pendingStartPositionSeconds;
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
      playerAttached = true;
      if (autoPlay) {
        hideOverlay();
        video.play().catch(() => {});
        startHeartbeat();
      } else {
        showOverlay("Ready. Press Play to start.", false);
      }
      syncPlayPauseButton();
      return;
    }
    if (!(window.Hls && Hls.isSupported())) {
      showError("This browser cannot play HLS.");
      return;
    }
    hls = new Hls({
      lowLatencyMode: false,
      backBufferLength: 90,
      maxBufferLength: 60,
      liveSyncDurationCount: 3,
      liveMaxLatencyDurationCount: 6,
    });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(Hls.Events.MANIFEST_PARSED, function () {
      video.currentTime = 0;
      playerAttached = true;
      if (autoPlay) {
        hideOverlay();
        video.play().catch(() => {});
        startHeartbeat();
      } else {
        showOverlay("Ready. Press Play to start.", false);
      }
      syncPlayPauseButton();
    });
    hls.on(Hls.Events.ERROR, function (_event, data) {
      if (data && data.fatal) {
        showError("Playback pipeline error.");
      }
    });
  }

  async function waitForGeneration(generation) {
    if (readinessHandle !== null) {
      window.clearInterval(readinessHandle);
    }
    showOverlay("Connecting to source and buffering stream...", true);
    readinessHandle = window.setInterval(async function () {
      try {
        const status = await fetchStatus();
        applyStatus(status);
        if (status.state === "invalidated") {
          return;
        }
        if (status.state === "error") {
          window.clearInterval(readinessHandle);
          readinessHandle = null;
          transitionPending = false;
          startOnReady = false;
          stopHeartbeat();
          showError(status.error || "Stream failed.");
          return;
        }
        if (status.generation === generation && !status.ready_for_playback) {
          showOverlay(loadingMessageForStatus(status), true);
          return;
        }
        if (status.generation === generation && status.ready_for_playback) {
          window.clearInterval(readinessHandle);
          readinessHandle = null;
          transitionPending = false;
          const shouldAutoPlay = startOnReady;
          startOnReady = false;
          attachPlayer(shouldAutoPlay);
        }
      } catch (_err) {
        transitionPending = false;
        startOnReady = false;
        stopHeartbeat();
        showError("Failed to check stream status.");
      }
    }, 1000);
  }

  async function requestStartAt(positionSeconds) {
    if (invalidated) {
      return;
    }
    transitionPending = true;
    startOnReady = true;
    playbackStarted = false;
    pendingStartPositionSeconds = positionSeconds;
    displayAnchorPositionSeconds = positionSeconds;
    destroyHls();
    updateTimeline(positionSeconds, currentStatus ? currentStatus.duration_seconds : 0);
    showOverlay("Connecting to source and buffering stream...", true);
    const data = await postJson(playUrl, {
      position_seconds: positionSeconds,
    });
    expectedGeneration = data.generation;
    const status = await fetchStatus();
    applyStatus(status);
    if (status.state === "invalidated") {
      return;
    }
    await waitForGeneration(expectedGeneration);
  }

  async function requestPause() {
    if (!currentStatus || transitionPending || invalidated) {
      return;
    }
    const pos = currentLogicalPosition();
    transitionPending = true;
    startOnReady = false;
    playbackStarted = false;
    pendingStartPositionSeconds = pos;
    displayAnchorPositionSeconds = pos;
    stopHeartbeat();
    try {
      try {
        video.pause();
      } catch (_err) {}
      destroyHls();
      showOverlay("Playback paused.", false);
      syncPlayPauseButton();
      await postJson(pauseUrl, {
        position_seconds: pos,
      });
      const status = await fetchStatus();
      applyStatus(status);
    } catch (err) {
      showError(err.message || "Pause failed.");
    } finally {
      transitionPending = false;
    }
  }

  async function requestFullscreen() {
    if (invalidated) {
      return;
    }
    const target = video;
    if (!target) {
      return;
    }
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
          return;
        }
        if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen();
          return;
        }
        return;
      }
      if (target.requestFullscreen) {
        await target.requestFullscreen();
        return;
      }
      if (target.webkitRequestFullscreen) {
        target.webkitRequestFullscreen();
        return;
      }
      if (target.webkitEnterFullscreen) {
        target.webkitEnterFullscreen();
        return;
      }
      if (target.webkitEnterFullScreen) {
        target.webkitEnterFullScreen();
      }
    } catch (_err) {}
  }

  playPauseBtn.addEventListener("click", async function () {
    if (transitionPending || invalidated) {
      return;
    }
    try {
      if (playerAttached && currentStatus && currentStatus.state === "streaming" && !video.paused && !video.ended) {
        await requestPause();
      } else if (playerAttached && currentStatus && currentStatus.state === "streaming") {
        hideOverlay();
        await video.play().catch(() => {});
        startHeartbeat();
        syncPlayPauseButton();
      } else {
        await requestStartAt(currentLogicalPosition());
      }
    } catch (err) {
      transitionPending = false;
      stopHeartbeat();
      showError(err.message || "Playback control failed.");
    }
  });

  fullscreenBtn.addEventListener("click", async function () {
    if (invalidated) {
      return;
    }
    await requestFullscreen();
  });

  timeline.addEventListener("input", function () {
    if (invalidated) {
      return;
    }
    timelineDragging = true;
    updateTimeline(Number(timeline.value), currentStatus ? currentStatus.duration_seconds : 0);
  });

  timeline.addEventListener("change", async function () {
    if (invalidated) {
      return;
    }
    timelineDragging = false;
    const target = Number(timeline.value);
    try {
      await requestStartAt(target);
    } catch (err) {
      transitionPending = false;
      stopHeartbeat();
      showError(err.message || "Seek failed.");
    }
  });

  video.addEventListener("timeupdate", function () {
    if (!currentStatus || timelineDragging) {
      return;
    }
    updateTimeline(visibleTimelinePosition(), currentStatus.duration_seconds);
  });

  video.addEventListener("play", function () {
    syncPlayPauseButton();
  });

  video.addEventListener("playing", function () {
    markPlaybackStarted();
    syncPlayPauseButton();
  });

  video.addEventListener("pause", function () {
    syncPlayPauseButton();
  });

  window.setInterval(async function () {
    try {
      const status = await fetchStatus();
      applyStatus(status);
        if (status.state === "invalidated") {
          return;
        }
      if (status.state === "error") {
        stopHeartbeat();
        startOnReady = false;
        showError(status.error || "Stream failed.");
      } else if (!transitionPending && status.state === "paused") {
        stopHeartbeat();
        showOverlay("Playback paused.", false);
      } else if (!transitionPending && status.state === "streaming" && !playerAttached) {
        showOverlay("Ready. Press Play to start.", false);
      } else if (transitionPending && (status.state === "buffering" || status.state === "streaming")) {
        showOverlay(loadingMessageForStatus(status), true);
      }
      syncPlayPauseButton();
    } catch (_err) {}
  }, 2000);

  window.addEventListener("pagehide", function () {
    if (!currentStatus || currentStatus.state !== "streaming") {
      return;
    }
    const payload = JSON.stringify({
      position_seconds: currentLogicalPosition(),
    });
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        pauseUrl,
        new Blob([payload], { type: "application/json" }),
      );
    }
  });

  (async function init() {
    try {
      showOverlay("Preparing stream...", true);
      const status = await fetchStatus();
      applyStatus(status);
      if (status.state === "invalidated") {
        return;
      }
      if (status.state === "error") {
        showError(status.error || "Stream failed.");
        return;
      }
      if (status.state === "streaming") {
        if (status.ready_for_playback) {
          pendingStartPositionSeconds = status.anchor_time_seconds || 0;
          displayAnchorPositionSeconds = pendingStartPositionSeconds;
          attachPlayer(false);
          showOverlay("Ready. Press Play to start.", false);
        } else {
          transitionPending = true;
          startOnReady = false;
          showOverlay(loadingMessageForStatus(status), true);
          await waitForGeneration(status.generation);
        }
        return;
      }
      if (status.state === "buffering") {
        transitionPending = true;
        startOnReady = false;
        showOverlay(loadingMessageForStatus(status), true);
        await waitForGeneration(status.generation);
        return;
      }
      if (status.state === "paused") {
        showOverlay("Playback paused. Press Play to resume.", false);
        syncPlayPauseButton();
        return;
      }
      showOverlay("Press Play to start.", false);
      syncPlayPauseButton();
    } catch (err) {
      transitionPending = false;
      startOnReady = false;
      stopHeartbeat();
      showError(err.message || "Failed to initialize player.");
    }
  })();
})();
