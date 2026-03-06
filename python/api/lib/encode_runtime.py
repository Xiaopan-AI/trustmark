import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib import request as urllib_request
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import Request
from loguru import logger

from python.api.lib.schemas import DeviceEnum, ModelTypeEnum, SessionStateEnum
from python.api.lib.trustmark_runtime import build_trustmark

WINDOW_SEGMENTS = 6
FRAME_LOG_EVERY = 24
HEARTBEAT_INTERVAL_SECONDS = 10.0
CLIENT_IDLE_TIMEOUT_SECONDS = 30.0
CLIENT_WATCHDOG_POLL_SECONDS = 1.0
SOURCE_CONNECT_TIMEOUT_SECONDS = 30.0
SPOOL_RETRY_BACKOFF_SECONDS = 1.0
SPOOL_RETRY_MAX_BACKOFF_SECONDS = 5.0


@dataclass
class SessionState:
    stream_id: str
    source_url: str
    wm_secret: int
    device: str
    model_type: ModelTypeEnum
    duration_seconds: float
    fps: float
    width: int
    height: int
    inference_scale: float
    prebuffer_seconds: int
    segment_seconds: int
    gpu_batch_target: int
    gpu_batch_max: int
    gpu_flush_ms: int
    use_nvenc: bool
    has_audio_track: bool = False
    encoder_backend: str = "pending"
    hls_root_dir: str = ""
    hls_dir: str = ""
    playlist_path: str = ""
    hls_segment_prefix: str = "seg_"
    hls_writer: Optional[Any] = None
    hls_writer_started: bool = False
    hls_writer_stderr_tail: str = ""
    state: SessionStateEnum = SessionStateEnum.PREPARED
    error: str = ""
    started: bool = False
    current_pts: float = 0.0
    frames_processed: int = 0
    spool_path: str = ""
    spool_started: bool = False
    spool_thread: Optional[threading.Thread] = None
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    spool_complete: bool = False
    spool_error: str = ""
    spool_last_error: str = ""
    spool_started_at: float = 0.0
    spool_connected_at: float = 0.0
    spool_attempt_count: int = 0
    spool_bytes_ever_arrived: bool = False
    spool_retry_deadline: float = 0.0
    spool_media_ready: bool = False
    spool_media_ready_at: float = 0.0
    spool_media_last_error: str = ""
    generation: int = 0
    active_generation: int = 0
    anchor_time_seconds: float = 0.0
    saved_position_seconds: float = 0.0
    logical_position_seconds: float = 0.0
    last_client_position_seconds: float = 0.0
    last_heartbeat_at: float = 0.0
    invalidated_at: float = 0.0
    controlled_stop_reason: str = ""
    end_of_stream_reached: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(init=False)
    worker: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self.cond = threading.Condition(self.lock)

    @property
    def secret_bits(self) -> str:
        return "".join(str((self.wm_secret >> i) & 1) for i in reversed(range(56)))


class StreamRegistry:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._ffmpeg_bin = self._resolve_binary("ffmpeg")
        self._ffprobe_bin = self._resolve_binary("ffprobe")
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._client_watchdog_loop,
            daemon=True,
            name="stream-client-watchdog",
        )
        self._watchdog_thread.start()
        logger.info(
            "StreamRegistry initialized with ffmpeg='{}' ffprobe='{}'",
            self._ffmpeg_bin,
            self._ffprobe_bin,
        )

    def _resolve_binary(self, binary_name: str) -> str:
        binary_path = shutil_which(binary_name)
        if not binary_path:
            logger.error("Required binary '{}' not found in PATH", binary_name)
            raise RuntimeError(f"{binary_name} not found in PATH.")
        logger.debug("Resolved binary '{}' to '{}'", binary_name, binary_path)
        return binary_path

    def create_session(
        self,
        source_url: str,
        wm_secret: int,
        device: str,
        model_type: ModelTypeEnum,
        inference_scale: float,
        prebuffer_seconds: int,
        segment_seconds: int,
        gpu_batch_target: int,
        gpu_batch_max: int,
        gpu_flush_ms: int,
        use_nvenc: bool,
    ) -> SessionState:
        logger.info(
            "Create session requested source_url='{}' device='{}' model='{}' inf_scale={} prebuffer={}s segment={}s batch_target={} batch_max={} flush_ms={} use_nvenc={}",
            source_url,
            device,
            model_type.value,
            inference_scale,
            prebuffer_seconds,
            segment_seconds,
            gpu_batch_target,
            gpu_batch_max,
            gpu_flush_ms,
            use_nvenc,
        )
        self._validate_source_url(source_url)
        duration, fps, width, height, has_audio_track, seeded_spool_path = resolve_source_metadata(
            self._ffprobe_bin,
            source_url,
        )
        stream_id = uuid.uuid4().hex[:12]
        effective_batch_target = max(1, min(64, int(gpu_batch_target)))
        effective_batch_max = max(effective_batch_target, min(64, int(gpu_batch_max)))
        initial_spool_path = seeded_spool_path or create_temp_spool_file(stream_id)
        seeded_spool_bytes = os.path.getsize(initial_spool_path) if seeded_spool_path and os.path.exists(initial_spool_path) else 0
        seeded_now = time.time() if seeded_spool_path else 0.0
        session = SessionState(
            stream_id=stream_id,
            source_url=source_url,
            wm_secret=wm_secret,
            device=device,
            model_type=model_type,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            inference_scale=float(inference_scale),
            prebuffer_seconds=int(prebuffer_seconds),
            segment_seconds=int(segment_seconds),
            gpu_batch_target=effective_batch_target,
            gpu_batch_max=effective_batch_max,
            gpu_flush_ms=int(gpu_flush_ms),
            use_nvenc=bool(use_nvenc),
            has_audio_track=has_audio_track,
            spool_path=initial_spool_path,
            hls_root_dir=create_temp_hls_root_dir(stream_id),
            spool_started=bool(seeded_spool_path),
            bytes_downloaded=seeded_spool_bytes,
            total_bytes=seeded_spool_bytes or None,
            spool_complete=bool(seeded_spool_path),
            spool_bytes_ever_arrived=bool(seeded_spool_path),
            spool_started_at=seeded_now,
            spool_connected_at=seeded_now,
            spool_media_ready=bool(seeded_spool_path),
            spool_media_ready_at=seeded_now,
        )
        with self._lock:
            self._sessions[stream_id] = session
        logger.info(
            "[{}] Session created duration={:.3f}s fps={:.3f} size={}x{} device='{}' model='{}' inf_scale={} prebuffer={}s segment={}s batch_target={} batch_max={} flush_ms={} use_nvenc={} has_audio={} spool='{}' hls_root='{}'",
            stream_id,
            duration,
            fps,
            width,
            height,
            device,
            model_type.value,
            session.inference_scale,
            session.prebuffer_seconds,
            session.segment_seconds,
            session.gpu_batch_target,
            session.gpu_batch_max,
            session.gpu_flush_ms,
            session.use_nvenc,
            session.has_audio_track,
            session.spool_path,
            session.hls_root_dir,
        )
        return session

    def get(self, stream_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(stream_id)
        if not session:
            logger.warning("[{}] Session lookup failed", stream_id)
            raise KeyError(stream_id)
        return session

    def ensure_started(self, session: SessionState) -> None:
        with session.lock:
            if session.state == SessionStateEnum.INVALIDATED:
                return
            self._launch_spooler_locked(session)

    def play(self, session: SessionState, position_seconds: Optional[float] = None) -> int:
        with session.lock:
            if session.state == SessionStateEnum.INVALIDATED:
                raise RuntimeError("The current session has been invalidated.")
            session.error = ""
            if not session.spool_complete:
                session.spool_error = ""
            session.spool_last_error = ""
            self._launch_spooler_locked(session)
            target = self._clamp_time_unlocked(
                session,
                position_seconds if position_seconds is not None else session.saved_position_seconds,
            )
            session.saved_position_seconds = target
            session.logical_position_seconds = target
            session.last_client_position_seconds = target
            session.last_heartbeat_at = time.time()
            session.controlled_stop_reason = "play_restart" if session.worker is not None else ""
        self._replace_run(session, target)
        with session.lock:
            return session.active_generation

    def pause(self, session: SessionState, position_seconds: Optional[float]) -> int:
        with session.lock:
            if session.state == SessionStateEnum.INVALIDATED:
                raise RuntimeError("The current session has been invalidated.")
            pos = self._clamp_time_unlocked(
                session,
                position_seconds if position_seconds is not None else self._best_known_position_unlocked(session),
            )
            session.saved_position_seconds = pos
            session.logical_position_seconds = pos
            session.last_client_position_seconds = pos
            session.controlled_stop_reason = "pause"
            generation = session.active_generation
            session.state = SessionStateEnum.PAUSED
            session.cond.notify_all()
        self._stop_active_run(session, cleanup_hls=True)
        with session.lock:
            session.state = SessionStateEnum.PAUSED
            session.current_pts = pos
            session.cond.notify_all()
            return generation

    def heartbeat(self, session: SessionState, position_seconds: float) -> None:
        with session.lock:
            if session.state == SessionStateEnum.INVALIDATED:
                raise RuntimeError("The current session has been invalidated.")
            pos = self._clamp_time_unlocked(session, position_seconds)
            session.last_heartbeat_at = time.time()
            session.last_client_position_seconds = pos
            if session.state in (SessionStateEnum.BUFFERING, SessionStateEnum.STREAMING):
                session.logical_position_seconds = pos
            session.cond.notify_all()

    def invalidate_by_secret(self, wm_secret: int) -> list[SessionState]:
        with self._lock:
            sessions = [session for session in self._sessions.values() if session.wm_secret == wm_secret]
        invalidated_sessions: list[SessionState] = []
        now = time.time()
        for session in sessions:
            with session.lock:
                if session.state == SessionStateEnum.INVALIDATED:
                    invalidated_sessions.append(session)
                    continue
                session.invalidated_at = now
                session.state = SessionStateEnum.INVALIDATED
                session.error = "The current session has been invalidated."
                session.controlled_stop_reason = "invalidated"
                session.stop_event.set()
                session.cond.notify_all()
            self._stop_active_run(session, cleanup_hls=True)
            with session.lock:
                session.state = SessionStateEnum.INVALIDATED
                session.error = "The current session has been invalidated."
                session.cond.notify_all()
            invalidated_sessions.append(session)
        return invalidated_sessions

    def _validate_source_url(self, source_url: str) -> None:
        parsed = urlparse(source_url)
        logger.debug("Validating source URL '{}'", source_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS URLs are supported.")
        if not parsed.path.lower().endswith(".mp4"):
            raise ValueError("MVP supports MP4 URLs only.")
        logger.debug("Source URL validated successfully '{}'", source_url)

    def _launch_spooler_locked(self, session: SessionState) -> None:
        if session.spool_thread and session.spool_thread.is_alive():
            return
        if session.spool_complete:
            return
        now = time.time()
        if session.spool_started_at <= 0 or session.spool_error:
            session.spool_started_at = now
            session.spool_retry_deadline = now + SOURCE_CONNECT_TIMEOUT_SECONDS
            session.spool_error = ""
            session.spool_last_error = ""
            session.spool_connected_at = 0.0
            session.spool_attempt_count = 0
        session.spool_started = True
        spool_thread = threading.Thread(
            target=self._download_to_spool,
            args=(session,),
            daemon=True,
            name=f"spool-{session.stream_id}",
        )
        session.spool_thread = spool_thread
        logger.info("[{}] Launching downloader thread", session.stream_id)
        spool_thread.start()

    def _download_to_spool(self, session: SessionState) -> None:
        logger.info("[{}] Progressive spool download starting '{}'", session.stream_id, session.spool_path)
        backoff_seconds = SPOOL_RETRY_BACKOFF_SECONDS
        try:
            while True:
                with session.lock:
                    if session.stop_event.is_set() or session.spool_complete:
                        session.cond.notify_all()
                        return
                    session.spool_attempt_count += 1
                    attempt = session.spool_attempt_count
                    deadline = session.spool_retry_deadline
                    session.bytes_downloaded = 0
                    session.total_bytes = None
                    session.spool_last_error = ""
                    session.spool_media_ready = False
                    session.spool_media_ready_at = 0.0
                    session.spool_media_last_error = ""
                    session.cond.notify_all()
                try:
                    with open(session.spool_path, "wb"):
                        pass
                    req = urllib_request.Request(
                        session.source_url,
                        headers={
                            "User-Agent": "Mozilla/5.0",
                            "Accept": "*/*",
                        },
                    )
                    with urllib_request.urlopen(req, timeout=60) as response, open(session.spool_path, "wb") as spool_file:
                        total_bytes = response.headers.get("Content-Length")
                        with session.lock:
                            session.total_bytes = int(total_bytes) if total_bytes else None
                            if session.spool_connected_at <= 0:
                                session.spool_connected_at = time.time()
                            session.spool_last_error = ""
                            session.cond.notify_all()
                        logger.info(
                            "[{}] Spool response opened attempt={} total_bytes={}",
                            session.stream_id,
                            attempt,
                            session.total_bytes if session.total_bytes is not None else "unknown",
                        )
                        chunk_size = 1024 * 1024
                        chunk_idx = 0
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            spool_file.write(chunk)
                            spool_file.flush()
                            chunk_idx += 1
                            with session.lock:
                                session.bytes_downloaded += len(chunk)
                                if not session.spool_bytes_ever_arrived:
                                    session.spool_bytes_ever_arrived = True
                                session.cond.notify_all()
                                bytes_downloaded = session.bytes_downloaded
                            if chunk_idx == 1 or chunk_idx % 8 == 0:
                                refresh_spool_media_ready(session, self._ffprobe_bin)
                            if chunk_idx == 1 or chunk_idx % 8 == 0:
                                logger.debug(
                                    "[{}] Spool progress bytes_downloaded={} available_seconds={:.3f}",
                                    session.stream_id,
                                    bytes_downloaded,
                                    estimate_available_seconds(session),
                                )
                        with session.lock:
                            session.spool_complete = True
                            session.spool_last_error = ""
                            session.spool_error = ""
                            session.cond.notify_all()
                        if not refresh_spool_media_ready(session, self._ffprobe_bin):
                            with session.lock:
                                session.spool_error = session.spool_media_last_error or "Local spool file is not playable."
                                session.cond.notify_all()
                            logger.error(
                                "[{}] Spool completed but local media probe still failed: {}",
                                session.stream_id,
                                session.spool_error,
                            )
                            return
                        logger.info(
                            "[{}] Spool download complete bytes_downloaded={} available_seconds={:.3f}",
                            session.stream_id,
                            session.bytes_downloaded,
                            estimate_available_seconds(session),
                        )
                        return
                except Exception as exc:
                    with session.lock:
                        session.spool_last_error = str(exc)
                        session.cond.notify_all()
                        now = time.time()
                        deadline_exhausted = deadline > 0 and now >= deadline
                        if deadline_exhausted:
                            session.spool_error = str(exc)
                            session.cond.notify_all()
                    if deadline_exhausted:
                        logger.exception("[{}] Spool download failed after retry window: {}", session.stream_id, exc)
                        return
                    logger.warning(
                        "[{}] Spool attempt {} failed transiently, retrying in {:.1f}s: {}",
                        session.stream_id,
                        attempt,
                        backoff_seconds,
                        exc,
                    )
                    if session.stop_event.wait(backoff_seconds):
                        return
                    backoff_seconds = min(SPOOL_RETRY_MAX_BACKOFF_SECONDS, backoff_seconds * 2.0)
        finally:
            with session.lock:
                if session.spool_thread is threading.current_thread():
                    session.spool_thread = None
                session.cond.notify_all()

    def _replace_run(self, session: SessionState, anchor_time_seconds: float) -> None:
        self._stop_active_run(session, cleanup_hls=True)
        with session.lock:
            self._start_run_locked(session, anchor_time_seconds)

    def _start_run_locked(self, session: SessionState, anchor_time_seconds: float) -> None:
        self._launch_spooler_locked(session)
        generation = session.generation + 1
        hls_dir = create_generation_hls_dir(session, generation)
        target = self._clamp_time_unlocked(session, anchor_time_seconds)
        session.generation = generation
        session.active_generation = generation
        session.hls_dir = hls_dir
        session.playlist_path = os.path.join(hls_dir, "stream.m3u8")
        session.hls_segment_prefix = "seg_"
        session.anchor_time_seconds = target
        session.saved_position_seconds = target
        session.logical_position_seconds = target
        session.current_pts = target
        session.frames_processed = 0
        session.encoder_backend = "pending"
        session.hls_writer = None
        session.hls_writer_started = False
        session.hls_writer_stderr_tail = ""
        session.controlled_stop_reason = ""
        session.end_of_stream_reached = False
        session.error = ""
        session.started = True
        session.state = SessionStateEnum.BUFFERING
        session.stop_event = threading.Event()
        worker = threading.Thread(
            target=self._worker_loop,
            args=(session, generation, target, session.stop_event),
            daemon=True,
            name=f"stream-{session.stream_id}-g{generation}",
        )
        session.worker = worker
        logger.info(
            "[{}] Launching worker generation={} state='{}' anchor={:.3f}s segment_seconds={} prebuffer_seconds={}",
            session.stream_id,
            generation,
            session.state.value,
            target,
            session.segment_seconds,
            session.prebuffer_seconds,
        )
        worker.start()

    def _stop_active_run(self, session: SessionState, cleanup_hls: bool) -> None:
        with session.lock:
            worker = session.worker
            stop_event = session.stop_event
            hls_dir = session.hls_dir
            stop_event.set()
            session.cond.notify_all()
        stop_hls_writer(session)
        if worker is not None and worker.is_alive():
            worker.join(timeout=10)
        if cleanup_hls and hls_dir:
            cleanup_generation_hls_dir(hls_dir)
        with session.lock:
            if session.worker is worker:
                session.worker = None
            if cleanup_hls and session.hls_dir == hls_dir:
                session.hls_dir = ""
                session.playlist_path = ""
            session.hls_writer = None
            session.hls_writer_started = False
            session.cond.notify_all()

    def _client_watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(CLIENT_WATCHDOG_POLL_SECONDS):
            with self._lock:
                sessions = list(self._sessions.values())
            now = time.time()
            for session in sessions:
                try:
                    self._handle_stale_client(session, now)
                except Exception:
                    logger.exception("[{}] Client watchdog failed", session.stream_id)

    def _handle_stale_client(self, session: SessionState, now: float) -> None:
        with session.lock:
            if session.state not in (SessionStateEnum.BUFFERING, SessionStateEnum.STREAMING):
                return
            if session.last_heartbeat_at <= 0:
                return
            if now - session.last_heartbeat_at < CLIENT_IDLE_TIMEOUT_SECONDS:
                return
            pause_position = self._clamp_time_unlocked(session, self._best_known_position_unlocked(session))
            session.saved_position_seconds = pause_position
            session.logical_position_seconds = pause_position
            session.last_client_position_seconds = pause_position
            session.current_pts = pause_position
            session.controlled_stop_reason = "timeout"
            session.state = SessionStateEnum.PAUSED
            logger.info(
                "[{}] Client heartbeat timeout after {:.1f}s, auto-pausing at {:.3f}s",
                session.stream_id,
                now - session.last_heartbeat_at,
                pause_position,
            )
        self._stop_active_run(session, cleanup_hls=True)
        with session.lock:
            if session.state != SessionStateEnum.INVALIDATED:
                session.state = SessionStateEnum.PAUSED
            session.cond.notify_all()

    def _best_known_position_unlocked(self, session: SessionState) -> float:
        if session.last_client_position_seconds > 0:
            return session.last_client_position_seconds
        if session.logical_position_seconds > 0:
            return session.logical_position_seconds
        return session.saved_position_seconds

    def _clamp_time_unlocked(self, session: SessionState, time_seconds: Optional[float]) -> float:
        raw = 0.0 if time_seconds is None else float(time_seconds)
        if session.duration_seconds > 0:
            return max(0.0, min(float(session.duration_seconds), raw))
        return max(0.0, raw)

    def _worker_loop(
        self,
        session: SessionState,
        generation: int,
        anchor_time_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        reader = None
        hls_writer = None
        try:
            logger.info(
                "[{}] Worker starting generation={} device='{}' model='{}' anchor={:.3f}s fps={:.3f}",
                session.stream_id,
                generation,
                session.device,
                session.model_type.value,
                anchor_time_seconds,
                session.fps or 25.0,
            )
            tm = build_trustmark(
                model_type=session.model_type,
                device=_enum_device_from_runtime(session.device),
            )
            target_spool_seconds = required_spool_target_seconds(session, anchor_time_seconds)
            wait_until_spooled(session, target_spool_seconds)
            if stop_event.is_set():
                logger.info("[{}] Worker generation={} cancelled before start", session.stream_id, generation)
                return

            writer_cmd, encoder_backend = build_hls_writer_command(
                self._ffmpeg_bin,
                session,
                anchor_time_seconds,
            )
            if session.use_nvenc and encoder_backend != "h264_nvenc":
                logger.warning("[{}] NVENC requested but unavailable, falling back to libx264", session.stream_id)
            logger.info(
                "[{}] Starting ffmpeg HLS writer generation={} encoder='{}' playlist='{}'",
                session.stream_id,
                generation,
                encoder_backend,
                session.playlist_path,
            )
            logger.debug("[{}] ffmpeg HLS writer command: {}", session.stream_id, writer_cmd)
            hls_writer = subprocess.Popen(
                writer_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with session.lock:
                if generation != session.active_generation:
                    stop_event.set()
                session.encoder_backend = encoder_backend
                session.hls_writer = hls_writer
                session.hls_writer_started = True
                session.cond.notify_all()
            if stop_event.is_set():
                return

            reader = FFmpegPipeReader(
                ffmpeg_bin=self._ffmpeg_bin,
                source_url=session.spool_path,
                width=session.width,
                height=session.height,
                seek_seconds=anchor_time_seconds,
                stream_id=f"{session.stream_id}-g{generation}",
            )

            batch_frames = []
            total_encoded = 0
            last_flush = time.time()
            first_playlist_logged = False

            def flush_batch() -> None:
                nonlocal batch_frames, total_encoded, last_flush, first_playlist_logged
                if not batch_frames:
                    return
                scaled_batch = []
                upscale_sizes = []
                if session.inference_scale < 0.999:
                    for frame in batch_frames:
                        orig_h, orig_w = frame.shape[:2]
                        scaled_w = max(2, int(round(orig_w * session.inference_scale)))
                        scaled_h = max(2, int(round(orig_h * session.inference_scale)))
                        scaled_batch.append(cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA))
                        upscale_sizes.append((orig_w, orig_h))
                else:
                    scaled_batch = batch_frames
                    upscale_sizes = [None] * len(batch_frames)

                encoded_batch = tm.encode_batch_numpy(scaled_batch, session.secret_bits, "binary")
                if hls_writer is None or hls_writer.stdin is None:
                    raise RuntimeError("ffmpeg HLS writer stdin is unavailable.")

                for idx, out_frame in enumerate(encoded_batch):
                    if upscale_sizes[idx] is not None:
                        orig_w, orig_h = upscale_sizes[idx]
                        out_frame = cv2.resize(out_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    hls_writer.stdin.write(out_frame.tobytes())
                hls_writer.stdin.flush()

                total_encoded += len(encoded_batch)
                media_seconds = float(total_encoded) / float(session.fps or 25.0)
                with session.lock:
                    if generation != session.active_generation:
                        stop_event.set()
                    session.frames_processed = total_encoded
                    session.current_pts = anchor_time_seconds + media_seconds
                    session.logical_position_seconds = session.current_pts
                    if session.state not in (SessionStateEnum.ERROR, SessionStateEnum.PAUSED, SessionStateEnum.INVALIDATED):
                        session.state = (
                            SessionStateEnum.STREAMING
                            if _is_session_ready_for_playback_unlocked(session)
                            else SessionStateEnum.BUFFERING
                        )
                    session.cond.notify_all()

                if total_encoded % FRAME_LOG_EVERY == 0:
                    logger.debug(
                        "[{}] Generation={} encoded={} logical_position={:.3f}s segments={}",
                        session.stream_id,
                        generation,
                        total_encoded,
                        anchor_time_seconds + media_seconds,
                        hls_segment_count(session),
                    )

                if not first_playlist_logged and playlist_exists(session) and hls_segment_count(session) > 0:
                    logger.info(
                        "[{}] ffmpeg-managed HLS output ready generation={} playlist='{}' segments={}",
                        session.stream_id,
                        generation,
                        session.playlist_path,
                        hls_segment_count(session),
                    )
                    first_playlist_logged = True

                batch_frames = []
                last_flush = time.time()

            while not stop_event.is_set():
                ret, frame = reader.read_frame()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_age_ms = (time.time() - last_flush) * 1000.0
                if len(batch_frames) >= session.gpu_batch_max:
                    flush_batch()
                elif len(batch_frames) >= session.gpu_batch_target and batch_age_ms >= session.gpu_flush_ms:
                    flush_batch()
                elif batch_frames and batch_age_ms >= session.gpu_flush_ms and len(batch_frames) >= max(1, min(8, session.gpu_batch_target)):
                    flush_batch()

            flush_batch()

            if hls_writer and hls_writer.stdin:
                try:
                    hls_writer.stdin.close()
                except Exception:
                    pass
            writer_err = b""
            if hls_writer and hls_writer.stderr:
                writer_err = hls_writer.stderr.read()
                hls_writer.stderr.close()
            if hls_writer:
                hls_writer.wait(timeout=60)
            if writer_err:
                session.hls_writer_stderr_tail = writer_err.decode("utf-8", errors="ignore")[-500:]
            if stop_event.is_set():
                logger.info(
                    "[{}] Worker generation={} stopped cleanly reason='{}'",
                    session.stream_id,
                    generation,
                    session.controlled_stop_reason or "unknown",
                )
                return
            if hls_writer and hls_writer.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg HLS writer failed: {writer_err.decode('utf-8', errors='ignore') or 'unknown error'}"
                )

            with session.lock:
                if generation == session.active_generation and session.state != SessionStateEnum.INVALIDATED:
                    session.current_pts = min(session.duration_seconds, session.current_pts) if session.duration_seconds > 0 else session.current_pts
                    session.logical_position_seconds = session.current_pts
                    session.saved_position_seconds = session.current_pts
                    session.end_of_stream_reached = True
                    session.state = SessionStateEnum.DONE
                    session.cond.notify_all()
            logger.info("[{}] Worker generation={} reached end of stream", session.stream_id, generation)
        except Exception as exc:
            if stop_event.is_set():
                logger.info(
                    "[{}] Worker generation={} exited during controlled stop reason='{}'",
                    session.stream_id,
                    generation,
                    session.controlled_stop_reason or "unknown",
                )
            else:
                with session.lock:
                    if generation == session.active_generation:
                        detail = str(exc)
                        if reader is not None and reader.stderr_tail():
                            detail = f"{detail} | ffmpeg: {reader.stderr_tail()}"
                        if session.hls_writer_stderr_tail:
                            detail = f"{detail} | hls: {session.hls_writer_stderr_tail}"
                        if session.state != SessionStateEnum.INVALIDATED:
                            session.state = SessionStateEnum.ERROR
                            session.error = detail
                        session.cond.notify_all()
                    else:
                        detail = str(exc)
                logger.exception("[{}] Worker generation={} failed: {}", session.stream_id, generation, detail)
        finally:
            if reader is not None:
                reader.close()
            stop_hls_writer(session)
            with session.lock:
                if session.worker is threading.current_thread():
                    session.worker = None
                session.cond.notify_all()
            logger.info("[{}] Worker generation={} exiting state='{}'", session.stream_id, generation, session.state.value)

    def cleanup_all(self) -> None:
        self._watchdog_stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                with session.lock:
                    session.controlled_stop_reason = "cleanup"
                    session.stop_event.set()
                self._stop_active_run(session, cleanup_hls=True)
                cleanup_hls_root_dir(session)
                cleanup_spool_file(session)
            except Exception:
                logger.exception("[{}] Failed during registry cleanup", session.stream_id)


def shutil_which(binary_name: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for base in paths:
        candidate = os.path.join(base, binary_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def create_temp_spool_file(stream_id: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f"trustmark_{stream_id}_", suffix=".mp4")
    os.close(fd)
    logger.info("[{}] Created temp spool file '{}'", stream_id, path)
    return path


def create_temp_hls_root_dir(stream_id: str) -> str:
    path = tempfile.mkdtemp(prefix=f"trustmark_hls_{stream_id}_")
    logger.info("[{}] Created temp HLS root dir '{}'", stream_id, path)
    return path


def create_generation_hls_dir(session: SessionState, generation: int) -> str:
    path = os.path.join(session.hls_root_dir, f"gen_{generation:06d}")
    os.makedirs(path, exist_ok=True)
    logger.info("[{}] Created generation HLS dir '{}' generation={}", session.stream_id, path, generation)
    return path


def cleanup_generation_hls_dir(path: str) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_spool_file(session: SessionState) -> None:
    if session.spool_path and os.path.exists(session.spool_path):
        os.remove(session.spool_path)
        logger.info("[{}] Removed temp spool file '{}'", session.stream_id, session.spool_path)


def cleanup_hls_root_dir(session: SessionState) -> None:
    if session.hls_root_dir and os.path.isdir(session.hls_root_dir):
        shutil.rmtree(session.hls_root_dir, ignore_errors=True)
        logger.info("[{}] Removed temp HLS root dir '{}'", session.stream_id, session.hls_root_dir)


def estimate_available_seconds(session: SessionState) -> float:
    with session.lock:
        return estimate_available_seconds_unlocked(session)


def estimate_available_seconds_unlocked(session: SessionState) -> float:
    if session.spool_complete:
        return float(session.duration_seconds or 0.0)
    if session.total_bytes and session.total_bytes > 0 and session.duration_seconds > 0:
        ratio = min(1.0, float(session.bytes_downloaded) / float(session.total_bytes))
        return float(session.duration_seconds) * ratio
    return 0.0


def probe_local_spool_media_ready(ffprobe_bin: str, spool_path: str) -> tuple[bool, str]:
    if not spool_path or not os.path.exists(spool_path):
        return False, "Local spool file does not exist yet."
    if os.path.getsize(spool_path) <= 0:
        return False, "Local spool file is still empty."
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        spool_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffprobe error"
        return False, detail
    return True, ""


def refresh_spool_media_ready(session: SessionState, ffprobe_bin: str) -> bool:
    ready, detail = probe_local_spool_media_ready(ffprobe_bin, session.spool_path)
    with session.lock:
        if ready:
            session.spool_media_ready = True
            if session.spool_media_ready_at <= 0:
                session.spool_media_ready_at = time.time()
            session.spool_media_last_error = ""
        elif not session.spool_media_ready:
            session.spool_media_ready = False
            session.spool_media_last_error = detail
        session.cond.notify_all()
    return ready


def spool_retry_deadline_remaining_unlocked(session: SessionState) -> float:
    if session.spool_retry_deadline <= 0:
        return 0.0
    return max(0.0, session.spool_retry_deadline - time.time())


def spool_startup_ready_unlocked(session: SessionState, target_seconds: float) -> bool:
    available_seconds = estimate_available_seconds_unlocked(session)
    return available_seconds + 0.05 >= target_seconds and session.spool_media_ready


def startup_segment_count(session: SessionState) -> int:
    return max(1, int(math.ceil(float(session.prebuffer_seconds) / float(session.segment_seconds))))


def prebuffer_frame_target(session: SessionState) -> int:
    return max(1, int(round(float(session.prebuffer_seconds) * float(session.fps or 25.0))))


def required_spool_target_seconds(session: SessionState, anchor_time_seconds: float) -> float:
    target = float(anchor_time_seconds) + float(session.prebuffer_seconds) + float(session.segment_seconds)
    if session.duration_seconds > 0:
        return min(float(session.duration_seconds), target)
    return max(0.0, target)


def wait_until_spooled(session: SessionState, target_seconds: float) -> None:
    logger.debug(
        "[{}] Waiting for spool target={:.3f}s current_available={:.3f}s",
        session.stream_id,
        target_seconds,
        estimate_available_seconds(session),
    )
    with session.lock:
        while True:
            available_seconds = estimate_available_seconds_unlocked(session)
            if spool_startup_ready_unlocked(session, target_seconds):
                logger.debug(
                    "[{}] Spool ready target={:.3f}s available={:.3f}s complete={} media_ready={}",
                    session.stream_id,
                    target_seconds,
                    available_seconds,
                    session.spool_complete,
                    session.spool_media_ready,
                )
                return
            if session.stop_event.is_set():
                raise RuntimeError("Session stop requested while waiting for spool.")
            remaining = spool_retry_deadline_remaining_unlocked(session)
            if remaining <= 0:
                unmet = []
                if available_seconds + 0.05 < target_seconds:
                    unmet.append("enough data was buffered")
                if not session.spool_media_ready:
                    unmet.append("local spool became playable")
                if unmet:
                    detail = session.spool_media_last_error or session.spool_error
                    message = "Source warmup timed out before " + " and ".join(unmet)
                    if detail:
                        message = f"{message}: {detail}"
                    raise RuntimeError(message)
                if session.spool_error:
                    raise RuntimeError(f"Spool download failed: {session.spool_error}")
            session.cond.wait(timeout=0.5)


def parse_fraction(raw_value: str) -> float:
    if not raw_value or raw_value in {"0/0", "N/A"}:
        return 0.0
    if "/" in raw_value:
        num, den = raw_value.split("/", 1)
        den_val = float(den)
        if den_val == 0:
            return 0.0
        return float(num) / den_val
    return float(raw_value)


def extract_kv(output: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", output, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def probe_source_metadata(ffprobe_bin: str, source_url: str):
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        source_url,
    ]
    logger.info("ffprobe metadata probe started source='{}'", source_url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffprobe error"
        raise ValueError(f"Cannot probe source_url metadata: {detail}")
    output = proc.stdout
    width = int(extract_kv(output, "width") or 0)
    height = int(extract_kv(output, "height") or 0)
    fps = parse_fraction(extract_kv(output, "avg_frame_rate")) or 25.0
    duration = float(extract_kv(output, "duration") or 0.0)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source dimensions from ffprobe.")
    return duration, fps, width, height


def probe_source_has_audio(ffprobe_bin: str, source_url: str) -> bool:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def download_probe_copy(source_url: str) -> str:
    fd, path = tempfile.mkstemp(prefix="trustmark_probe_", suffix=".mp4")
    os.close(fd)
    req = urllib_request.Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urllib_request.urlopen(req, timeout=60) as response, open(path, "wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    return path


def resolve_source_metadata(ffprobe_bin: str, source_url: str) -> tuple[float, float, int, int, bool, Optional[str]]:
    logger.info("Remote-first metadata probe started source='{}'", source_url)
    try:
        duration, fps, width, height = probe_source_metadata(ffprobe_bin, source_url)
        has_audio = probe_source_has_audio(ffprobe_bin, source_url)
        return duration, fps, width, height, has_audio, None
    except Exception:
        logger.warning("Remote-first metadata probe failed source='{}'", source_url)
    probe_path = download_probe_copy(source_url)
    duration, fps, width, height = probe_source_metadata(ffprobe_bin, probe_path)
    has_audio = probe_source_has_audio(ffprobe_bin, probe_path)
    return duration, fps, width, height, has_audio, probe_path


class FFmpegPipeReader:
    def __init__(
        self,
        ffmpeg_bin: str,
        source_url: str,
        width: int,
        height: int,
        seek_seconds: float = 0.0,
        stream_id: str = "-",
    ):
        self.stream_id = stream_id
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self._stderr_tail = ""
        self._frames_read = 0
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if seek_seconds > 0:
            cmd += ["-ss", str(seek_seconds)]
        cmd += [
            "-i",
            source_url,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        logger.info(
            "[{}] Starting ffmpeg reader seek={:.3f}s size={}x{}",
            self.stream_id,
            seek_seconds,
            width,
            height,
        )
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _read_exact(self, size: int) -> bytes:
        if not self.proc.stdout:
            return b""
        chunks = []
        got = 0
        while got < size:
            chunk = self.proc.stdout.read(size - got)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def read_frame(self):
        raw = self._read_exact(self.frame_bytes)
        if len(raw) < self.frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._frames_read += 1
        return True, frame

    def close(self):
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            if self.proc.stderr:
                err = self.proc.stderr.read()
                if err:
                    self._stderr_tail = err.decode("utf-8", errors="ignore")[-500:]
                self.proc.stderr.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            pass

    def stderr_tail(self) -> str:
        return self._stderr_tail


@lru_cache(maxsize=4)
def ffmpeg_supports_nvenc(ffmpeg_bin: str) -> bool:
    cmd = [ffmpeg_bin, "-hide_banner", "-encoders"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return False
    return proc.returncode == 0 and "h264_nvenc" in proc.stdout


def build_hls_writer_command(
    ffmpeg_bin: str,
    session: SessionState,
    anchor_time_seconds: float,
) -> tuple[list[str], str]:
    gop = max(1, int(round((session.fps or 25.0) * session.segment_seconds)))
    encoder_backend = "libx264"
    if session.use_nvenc and ffmpeg_supports_nvenc(ffmpeg_bin):
        encoder_backend = "h264_nvenc"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-re",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{session.width}x{session.height}",
        "-r",
        str(session.fps or 25.0),
        "-i",
        "-",
    ]
    if anchor_time_seconds > 0:
        cmd += [
            "-ss",
            str(anchor_time_seconds),
        ]
    cmd += [
        "-i",
        session.spool_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        encoder_backend,
    ]
    if encoder_backend == "h264_nvenc":
        cmd += [
            "-preset",
            "p1",
            "-tune",
            "ll",
            "-rc",
            "cbr",
            "-b:v",
            "6M",
            "-maxrate",
            "8M",
            "-bufsize",
            "12M",
        ]
    else:
        cmd += [
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
        ]
    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-af",
        "aresample=async=1000:first_pts=0",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "cfr",
        "-g",
        str(gop),
        "-sc_threshold",
        "0",
        "-muxpreload",
        "0",
        "-muxdelay",
        "0",
        "-f",
        "hls",
        "-hls_time",
        str(session.segment_seconds),
        "-hls_list_size",
        str(WINDOW_SEGMENTS),
        "-hls_flags",
        "delete_segments+append_list+independent_segments",
        "-hls_segment_filename",
        os.path.join(session.hls_dir, f"{session.hls_segment_prefix}%06d.ts"),
        session.playlist_path,
    ]
    return cmd, encoder_backend


def stop_hls_writer(session: SessionState) -> None:
    proc = session.hls_writer
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        if proc.stderr:
            err = proc.stderr.read()
            if err:
                session.hls_writer_stderr_tail = err.decode("utf-8", errors="ignore")[-500:]
            proc.stderr.close()
    except Exception:
        pass
    session.hls_writer = None
    session.hls_writer_started = False


def hls_segment_filename(session: SessionState, seq: int) -> str:
    return os.path.join(session.hls_dir, f"{session.hls_segment_prefix}{seq:06d}.ts")


def list_hls_segment_seqs(session: SessionState) -> list[int]:
    if not session.hls_dir or not os.path.isdir(session.hls_dir):
        return []
    seqs = []
    prefix = session.hls_segment_prefix
    for name in os.listdir(session.hls_dir):
        if not (name.startswith(prefix) and name.endswith(".ts")):
            continue
        raw = name[len(prefix):-3]
        if raw.isdigit():
            seqs.append(int(raw))
    seqs.sort()
    return seqs


def hls_segment_count(session: SessionState) -> int:
    return len(list_hls_segment_seqs(session))


def playlist_exists(session: SessionState) -> bool:
    return bool(session.playlist_path) and os.path.exists(session.playlist_path) and os.path.getsize(session.playlist_path) > 0


def rewrite_playlist_for_fastapi(session: SessionState, playlist_text: str) -> str:
    lines = []
    prefix = session.hls_segment_prefix
    for line in playlist_text.splitlines():
        if line.startswith(prefix) and line.endswith(".ts"):
            raw = line[len(prefix):-3]
            if raw.isdigit():
                line = f"/encode/{session.stream_id}/segments/{int(raw)}.ts"
        lines.append(line)
    return "\n".join(lines) + "\n"


def get_window_bounds(session: SessionState):
    seqs = list_hls_segment_seqs(session)
    if not seqs:
        return None, None
    start = session.anchor_time_seconds + float(seqs[0]) * float(session.segment_seconds)
    end = session.anchor_time_seconds + (float(seqs[-1]) + 1.0) * float(session.segment_seconds)
    return start, end


def _is_session_ready_for_playback_unlocked(session: SessionState) -> bool:
    if session.state in (SessionStateEnum.ERROR, SessionStateEnum.INVALIDATED):
        return False
    if session.state == SessionStateEnum.DONE:
        return playlist_exists(session) and hls_segment_count(session) > 0
    if session.state in (SessionStateEnum.PREPARED, SessionStateEnum.PAUSED):
        return False
    return (
        session.frames_processed >= prebuffer_frame_target(session)
        and playlist_exists(session)
        and hls_segment_count(session) >= startup_segment_count(session)
    )


def absolute_url(request: Request, route_name: str, **params) -> str:
    return str(request.url_for(route_name, **params))


@lru_cache(maxsize=1)
def load_player_script_template() -> str:
    script_path = os.path.join(os.path.dirname(__file__), "player_page.js")
    with open(script_path, "r", encoding="utf-8") as fh:
        return fh.read().replace("</", "<\\/")


def build_player_page(
    playlist_url: str,
    status_url: str,
    play_url: str,
    pause_url: str,
    heartbeat_url: str,
) -> str:
    config_json = json.dumps(
        {
            "playlistUrl": playlist_url,
            "statusUrl": status_url,
            "playUrl": play_url,
            "pauseUrl": pause_url,
            "heartbeatUrl": heartbeat_url,
            "heartbeatIntervalMs": int(HEARTBEAT_INTERVAL_SECONDS * 1000),
        }
    ).replace("</", "<\\/")
    player_script = load_player_script_template()
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Watermark Stream Player</title>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        background: #000;
        color: #fff;
        font-family: Arial, sans-serif;
      }}
      .player-shell {{
        width: min(1100px, 100vw);
        margin: 0 auto;
        padding: 16px;
        box-sizing: border-box;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }}
      .video-wrap {{
        position: relative;
        width: 100%;
        background: #000;
      }}
      video {{
        width: 100%;
        max-height: 72vh;
        background: #000;
        display: block;
      }}
      .overlay {{
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        background: rgba(0, 0, 0, 0.65);
      }}
      .controls {{
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}
      .controls-row {{
        display: flex;
        gap: 12px;
        align-items: center;
      }}
      button {{
        padding: 8px 14px;
        background: #1f1f1f;
        color: #fff;
        border: 1px solid #444;
        border-radius: 6px;
        cursor: pointer;
      }}
      button:disabled,
      input[type="range"]:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
      }}
      input[type="range"] {{
        width: 100%;
      }}
      .spinner {{
        width: 42px;
        height: 42px;
        border: 4px solid #333;
        border-top-color: #fff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }}
      .status-text {{
        color: #ddd;
        font-size: 15px;
      }}
      .meta-text {{
        color: #aaa;
        font-size: 13px;
      }}
      .error-text {{
        color: #ff8a8a;
      }}
      .hidden {{
        display: none;
      }}
      @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
      }}
    </style>
  </head>
  <body>
    <div class="player-shell">
      <div class="video-wrap">
        <video id="video" playsinline></video>
        <div id="overlay" class="overlay">
          <div id="spinner" class="spinner"></div>
          <div id="statusText" class="status-text">Preparing stream...</div>
        </div>
      </div>
      <div class="controls">
        <div class="controls-row">
          <button id="playPauseBtn" type="button">Play</button>
          <button id="fullscreenBtn" type="button">Fullscreen</button>
          <div id="timeLabel" class="status-text">0:00 / 0:00</div>
        </div>
        <input id="timeline" type="range" min="0" max="1" step="0.1" value="0" />
        <div id="sessionLabel" class="meta-text">Connecting...</div>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      window.TRUSTMARK_PLAYER_CONFIG = {config_json};
    </script>
    <script>
{player_script}
    </script>
  </body>
</html>
"""


def _enum_device_from_runtime(device: str) -> DeviceEnum:
    if device == "cpu":
        return DeviceEnum.CPU
    return DeviceEnum.CUDA_0


registry = StreamRegistry()
