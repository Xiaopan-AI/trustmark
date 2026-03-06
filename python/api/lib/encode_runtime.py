import math
import os
import re
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

from python.api.lib.schemas import ModelTypeEnum, SessionStateEnum
from python.api.lib.trustmark_runtime import build_trustmark

WINDOW_SEGMENTS = 6
FRAME_LOG_EVERY = 24
DEFAULT_PREBUFFER_SECONDS = 4
DEFAULT_SEGMENT_SECONDS = 2
DEFAULT_GPU_BATCH_TARGET = 60
DEFAULT_GPU_BATCH_MAX = 64
DEFAULT_GPU_FLUSH_MS = 12
DEFAULT_USE_NVENC = True
DEFAULT_INFERENCE_SCALE = 0.5


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
        duration, fps, width, height, has_audio_track = resolve_source_metadata(
            self._ffprobe_bin,
            source_url,
        )
        stream_id = uuid.uuid4().hex[:12]
        effective_batch_target = max(1, min(64, int(gpu_batch_target)))
        effective_batch_max = max(effective_batch_target, min(64, int(gpu_batch_max)))
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
            spool_path=create_temp_spool_file(stream_id),
            hls_dir=create_temp_hls_dir(stream_id),
        )
        session.playlist_path = os.path.join(session.hls_dir, "stream.m3u8")
        with self._lock:
            self._sessions[stream_id] = session
        logger.info(
            "[{}] Session created duration={:.3f}s fps={:.3f} size={}x{} device='{}' model='{}' inf_scale={} prebuffer={}s segment={}s batch_target={} batch_max={} flush_ms={} use_nvenc={} has_audio={} spool='{}' hls_dir='{}'",
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
            session.hls_dir,
        )
        return session

    def get(self, stream_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(stream_id)
        if not session:
            logger.warning("[{}] Session lookup failed", stream_id)
            raise KeyError(stream_id)
        return session

    def _validate_source_url(self, source_url: str) -> None:
        parsed = urlparse(source_url)
        logger.debug("Validating source URL '{}'", source_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS URLs are supported.")
        if not parsed.path.lower().endswith(".mp4"):
            raise ValueError("MVP supports MP4 URLs only.")
        logger.debug("Source URL validated successfully '{}'", source_url)

    def ensure_started(self, session: SessionState) -> None:
        with session.lock:
            self._launch_spooler_locked(session)
            self._launch_worker_locked(session)

    def _launch_spooler_locked(self, session: SessionState) -> None:
        if session.spool_started and session.spool_thread and session.spool_thread.is_alive():
            logger.debug("[{}] Spooler already running", session.stream_id)
            return
        if session.spool_complete:
            logger.debug("[{}] Spool already complete", session.stream_id)
            return
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

    def _launch_worker_locked(self, session: SessionState) -> None:
        if session.worker is not None and session.worker.is_alive():
            logger.debug("[{}] Worker already running", session.stream_id)
            return
        session.started = True
        session.state = SessionStateEnum.BUFFERING
        logger.info(
            "[{}] Launching worker state='{}' segment_seconds={} prebuffer_seconds={} startup_segments={} inf_scale={} batch_target={} batch_max={} flush_ms={} use_nvenc={}",
            session.stream_id,
            session.state.value,
            session.segment_seconds,
            session.prebuffer_seconds,
            startup_segment_count(session),
            session.inference_scale,
            session.gpu_batch_target,
            session.gpu_batch_max,
            session.gpu_flush_ms,
            session.use_nvenc,
        )
        worker = threading.Thread(
            target=self._worker_loop,
            args=(session,),
            daemon=True,
            name=f"stream-{session.stream_id}",
        )
        session.worker = worker
        worker.start()

    def _download_to_spool(self, session: SessionState) -> None:
        logger.info("[{}] Progressive spool download starting '{}'", session.stream_id, session.spool_path)
        try:
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
                    session.cond.notify_all()
                logger.info(
                    "[{}] Spool response opened total_bytes={}",
                    session.stream_id,
                    session.total_bytes if session.total_bytes is not None else "unknown",
                )
                chunk_size = 1024 * 1024
                chunk_idx = 0
                while not session.stop_event.is_set():
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    spool_file.write(chunk)
                    spool_file.flush()
                    chunk_idx += 1
                    with session.lock:
                        session.bytes_downloaded += len(chunk)
                        session.cond.notify_all()
                        bytes_downloaded = session.bytes_downloaded
                    if chunk_idx == 1 or chunk_idx % 8 == 0:
                        logger.debug(
                            "[{}] Spool progress bytes_downloaded={} available_seconds={:.3f}",
                            session.stream_id,
                            bytes_downloaded,
                            estimate_available_seconds(session),
                        )
                with session.lock:
                    session.spool_complete = True
                    session.cond.notify_all()
                logger.info(
                    "[{}] Spool download complete bytes_downloaded={} available_seconds={:.3f}",
                    session.stream_id,
                    session.bytes_downloaded,
                    estimate_available_seconds(session),
                )
        except Exception as exc:
            with session.lock:
                session.spool_error = str(exc)
                session.cond.notify_all()
            logger.exception("[{}] Spool download failed: {}", session.stream_id, exc)

    def _worker_loop(self, session: SessionState) -> None:
        reader = None
        tm = None
        logger.info(
            "[{}] Worker starting device='{}' model='{}' fps={:.3f} inf_scale={} prebuffer_seconds={} segment_seconds={} batch_target={} batch_max={} flush_ms={} use_nvenc={}",
            session.stream_id,
            session.device,
            session.model_type.value,
            session.fps or 25.0,
            session.inference_scale,
            session.prebuffer_seconds,
            session.segment_seconds,
            session.gpu_batch_target,
            session.gpu_batch_max,
            session.gpu_flush_ms,
            session.use_nvenc,
        )
        try:
            logger.info("[{}] Initializing TrustMark on '{}'", session.stream_id, session.device)
            tm = build_trustmark(
                model_type=session.model_type,
                device=_enum_device_from_runtime(session.device),
            )
            logger.info("[{}] TrustMark initialized successfully", session.stream_id)

            if session.duration_seconds > 0:
                wait_until_spooled(session, session.duration_seconds)

            writer_cmd, encoder_backend = build_hls_writer_command(self._ffmpeg_bin, session)
            if session.use_nvenc and encoder_backend != "h264_nvenc":
                logger.warning("[{}] NVENC requested but unavailable, falling back to libx264", session.stream_id)
            logger.info("[{}] Starting ffmpeg HLS writer encoder='{}' playlist='{}'", session.stream_id, encoder_backend, session.playlist_path)
            logger.debug("[{}] ffmpeg HLS writer command: {}", session.stream_id, writer_cmd)
            hls_writer = subprocess.Popen(
                writer_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with session.lock:
                session.encoder_backend = encoder_backend
                session.hls_writer = hls_writer
                session.hls_writer_started = True

            logger.info("[{}] Opening ffmpeg reader seek=0.000s source='{}'", session.stream_id, session.spool_path)
            reader = FFmpegPipeReader(
                ffmpeg_bin=self._ffmpeg_bin,
                source_url=session.spool_path,
                width=session.width,
                height=session.height,
                seek_seconds=0.0,
                stream_id=session.stream_id,
            )

            batch_frames = []
            total_encoded = 0
            last_flush = time.time()
            first_batch_logged = False
            first_playlist_logged = False

            def flush_batch() -> None:
                nonlocal batch_frames, total_encoded, last_flush, first_batch_logged, first_playlist_logged
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

                infer_started = time.time()
                encoded_batch = tm.encode_batch_numpy(scaled_batch, session.secret_bits, "binary")
                infer_ms = (time.time() - infer_started) * 1000.0
                if not first_batch_logged:
                    logger.info(
                        "[{}] First batch encoded size={} inf_scale={} infer_ms={:.2f}",
                        session.stream_id,
                        len(batch_frames),
                        session.inference_scale,
                        infer_ms,
                    )
                    logger.info("[{}] First frame encoded successfully", session.stream_id)
                    first_batch_logged = True

                if hls_writer.stdin is None:
                    raise RuntimeError("ffmpeg HLS writer stdin is unavailable.")

                for idx, out_frame in enumerate(encoded_batch):
                    if upscale_sizes[idx] is not None:
                        orig_w, orig_h = upscale_sizes[idx]
                        out_frame = cv2.resize(out_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    hls_writer.stdin.write(out_frame.tobytes())
                hls_writer.stdin.flush()

                total_encoded += len(encoded_batch)
                with session.lock:
                    session.frames_processed = total_encoded
                    session.current_pts = float(total_encoded) / float(session.fps or 25.0)
                    if session.state != SessionStateEnum.ERROR:
                        session.state = (
                            SessionStateEnum.STREAMING
                            if _is_session_ready_for_playback_unlocked(session)
                            else SessionStateEnum.BUFFERING
                        )
                    session.cond.notify_all()

                if total_encoded % FRAME_LOG_EVERY == 0:
                    logger.debug(
                        "[{}] Encoded {} frames total pts={:.3f}s segments={} playlist_exists={}",
                        session.stream_id,
                        total_encoded,
                        float(total_encoded) / float(session.fps or 25.0),
                        hls_segment_count(session),
                        playlist_exists(session),
                    )

                if not first_playlist_logged and playlist_exists(session) and hls_segment_count(session) > 0:
                    logger.info(
                        "[{}] ffmpeg-managed HLS output ready playlist='{}' segments={}",
                        session.stream_id,
                        session.playlist_path,
                        hls_segment_count(session),
                    )
                    if session.has_audio_track:
                        logger.info("[{}] First generated HLS segment includes muxed original audio", session.stream_id)
                    first_playlist_logged = True

                batch_frames = []
                last_flush = time.time()

            while not session.stop_event.is_set():
                ret, frame = reader.read_frame()
                if not ret:
                    break
                if total_encoded == 0 and not batch_frames:
                    logger.info("[{}] First frame read from ingest", session.stream_id)
                batch_frames.append(frame)
                batch_age_ms = (time.time() - last_flush) * 1000.0
                if len(batch_frames) >= session.gpu_batch_max:
                    flush_batch()
                elif len(batch_frames) >= session.gpu_batch_target and batch_age_ms >= session.gpu_flush_ms:
                    flush_batch()
                elif batch_frames and batch_age_ms >= session.gpu_flush_ms and len(batch_frames) >= max(1, min(8, session.gpu_batch_target)):
                    flush_batch()

            flush_batch()

            if hls_writer.stdin:
                hls_writer.stdin.close()
            writer_err = b""
            if hls_writer.stderr:
                writer_err = hls_writer.stderr.read()
                hls_writer.stderr.close()
            hls_writer.wait(timeout=60)
            if writer_err:
                session.hls_writer_stderr_tail = writer_err.decode("utf-8", errors="ignore")[-500:]
            if hls_writer.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg HLS writer failed: {writer_err.decode('utf-8', errors='ignore') or 'unknown error'}"
                )
            with session.lock:
                session.hls_writer = None
                session.hls_writer_started = False
                if session.state != SessionStateEnum.ERROR:
                    session.state = SessionStateEnum.DONE
                session.cond.notify_all()
            logger.info("[{}] ffmpeg HLS writer finished state='{}'", session.stream_id, session.state.value)

        except Exception as exc:
            with session.lock:
                session.state = SessionStateEnum.ERROR
                detail = str(exc)
                if reader is not None and reader.stderr_tail():
                    detail = f"{detail} | ffmpeg: {reader.stderr_tail()}"
                if session.hls_writer_stderr_tail:
                    detail = f"{detail} | hls: {session.hls_writer_stderr_tail}"
                session.error = detail
                session.cond.notify_all()
            logger.exception("[{}] Worker failed: {}", session.stream_id, detail)
        finally:
            if reader is not None:
                reader.close()
            stop_hls_writer(session)
            logger.info("[{}] Worker exiting state='{}'", session.stream_id, session.state.value)

    def cleanup_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                session.stop_event.set()
                stop_hls_writer(session)
                cleanup_hls_dir(session)
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


def create_temp_hls_dir(stream_id: str) -> str:
    path = tempfile.mkdtemp(prefix=f"trustmark_hls_{stream_id}_")
    logger.info("[{}] Created temp HLS dir '{}'", stream_id, path)
    return path


def cleanup_spool_file(session: SessionState) -> None:
    if session.spool_path and os.path.exists(session.spool_path):
        os.remove(session.spool_path)
        logger.info("[{}] Removed temp spool file '{}'", session.stream_id, session.spool_path)


def cleanup_hls_dir(session: SessionState) -> None:
    if session.hls_dir and os.path.isdir(session.hls_dir):
        for name in os.listdir(session.hls_dir):
            try:
                os.remove(os.path.join(session.hls_dir, name))
            except IsADirectoryError:
                pass
            except FileNotFoundError:
                pass
        os.rmdir(session.hls_dir)
        logger.info("[{}] Removed temp HLS dir '{}'", session.stream_id, session.hls_dir)


def estimate_available_seconds(session: SessionState) -> float:
    with session.lock:
        if session.spool_complete:
            return float(session.duration_seconds or 0.0)
        if session.total_bytes and session.total_bytes > 0 and session.duration_seconds > 0:
            ratio = min(1.0, float(session.bytes_downloaded) / float(session.total_bytes))
            return float(session.duration_seconds) * ratio
        return 0.0


def startup_segment_count(session: SessionState) -> int:
    return max(1, int(math.ceil(float(session.prebuffer_seconds) / float(session.segment_seconds))))


def prebuffer_frame_target(session: SessionState) -> int:
    return max(1, int(round(float(session.prebuffer_seconds) * float(session.fps or 25.0))))


def wait_until_spooled(session: SessionState, target_seconds: float) -> None:
    logger.debug(
        "[{}] Waiting for spool target={:.3f}s current_available={:.3f}s",
        session.stream_id,
        target_seconds,
        estimate_available_seconds(session),
    )
    with session.lock:
        while not session.stop_event.is_set():
            if session.spool_error:
                raise RuntimeError(f"Spool download failed: {session.spool_error}")
            available_seconds = estimate_available_seconds_unlocked(session)
            if available_seconds + 0.05 >= target_seconds:
                logger.debug(
                    "[{}] Spool ready target={:.3f}s available={:.3f}s complete={}",
                    session.stream_id,
                    target_seconds,
                    available_seconds,
                    session.spool_complete,
                )
                return
            session.state = SessionStateEnum.BUFFERING
            session.cond.wait(timeout=0.5)
        raise RuntimeError("Session stop requested while waiting for spool.")


def estimate_available_seconds_unlocked(session: SessionState) -> float:
    if session.spool_complete:
        return float(session.duration_seconds or 0.0)
    if session.total_bytes and session.total_bytes > 0 and session.duration_seconds > 0:
        ratio = min(1.0, float(session.bytes_downloaded) / float(session.total_bytes))
        return float(session.duration_seconds) * ratio
    return 0.0


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
    logger.debug("ffprobe command: {}", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffprobe error"
        logger.error("ffprobe probe failed source='{}' detail='{}'", source_url, detail)
        raise ValueError(f"Cannot probe source_url metadata: {detail}")

    output = proc.stdout
    width = int(extract_kv(output, "width") or 0)
    height = int(extract_kv(output, "height") or 0)
    fps = parse_fraction(extract_kv(output, "avg_frame_rate")) or 25.0
    duration = float(extract_kv(output, "duration") or 0.0)
    if width <= 0 or height <= 0:
        logger.error("ffprobe returned invalid dimensions source='{}' size={}x{}", source_url, width, height)
        raise ValueError("Invalid source dimensions from ffprobe.")
    logger.info(
        "ffprobe metadata success source='{}' duration={:.3f}s fps={:.3f} size={}x{}",
        source_url,
        duration,
        fps,
        width,
        height,
    )
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
    logger.info("ffprobe audio probe started source='{}'", source_url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown ffprobe error"
        logger.warning("ffprobe audio probe failed source='{}' detail='{}'", source_url, detail)
        return False
    has_audio = bool(proc.stdout.strip())
    logger.info("ffprobe audio probe result source='{}' has_audio={}", source_url, has_audio)
    return has_audio


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


def resolve_source_metadata(ffprobe_bin: str, source_url: str) -> tuple[float, float, int, int, bool]:
    logger.info("Remote-first metadata probe started source='{}'", source_url)
    try:
        duration, fps, width, height = probe_source_metadata(ffprobe_bin, source_url)
        has_audio = probe_source_has_audio(ffprobe_bin, source_url)
        logger.info(
            "Remote-first metadata probe succeeded source='{}' duration={:.3f}s fps={:.3f} size={}x{} has_audio={}",
            source_url,
            duration,
            fps,
            width,
            height,
            has_audio,
        )
        return duration, fps, width, height, has_audio
    except Exception as exc:
        logger.warning("Remote-first metadata probe failed source='{}' detail='{}'", source_url, exc)

    logger.info("Falling back to local probe copy for source='{}'", source_url)
    probe_path = download_probe_copy(source_url)
    try:
        duration, fps, width, height = probe_source_metadata(ffprobe_bin, probe_path)
        has_audio = probe_source_has_audio(ffprobe_bin, probe_path)
        logger.info(
            "Fallback local metadata probe succeeded source='{}' duration={:.3f}s fps={:.3f} size={}x{} has_audio={}",
            source_url,
            duration,
            fps,
            width,
            height,
            has_audio,
        )
        return duration, fps, width, height, has_audio
    finally:
        try:
            os.remove(probe_path)
        except FileNotFoundError:
            pass


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
        logger.debug("[{}] ffmpeg reader command: {}", self.stream_id, cmd)
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
            logger.debug(
                "[{}] ffmpeg reader short read bytes={} expected={}",
                self.stream_id,
                len(raw),
                self.frame_bytes,
            )
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._frames_read += 1
        if self._frames_read == 1:
            logger.info("[{}] ffmpeg reader produced first frame", self.stream_id)
        elif self._frames_read % FRAME_LOG_EVERY == 0:
            logger.debug("[{}] ffmpeg reader frames_read={}", self.stream_id, self._frames_read)
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
        if self._stderr_tail:
            logger.warning("[{}] ffmpeg reader stderr tail: {}", self.stream_id, self._stderr_tail)
        logger.info("[{}] ffmpeg reader closed after {} frames", self.stream_id, self._frames_read)

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


def build_hls_writer_command(ffmpeg_bin: str, session: SessionState) -> tuple[list[str], str]:
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
                line = f"/streams/{session.stream_id}/segments/{int(raw)}.ts"
        lines.append(line)
    return "\n".join(lines) + "\n"


def get_window_bounds(session: SessionState):
    seqs = list_hls_segment_seqs(session)
    if not seqs:
        return None, None
    start = float(seqs[0]) * float(session.segment_seconds)
    end = (float(seqs[-1]) + 1.0) * float(session.segment_seconds)
    return start, end


def _is_session_ready_for_playback_unlocked(session: SessionState) -> bool:
    if session.state == SessionStateEnum.ERROR:
        return False
    if session.state == SessionStateEnum.DONE:
        return playlist_exists(session) and hls_segment_count(session) > 0
    return (
        session.frames_processed >= prebuffer_frame_target(session)
        and playlist_exists(session)
        and hls_segment_count(session) >= startup_segment_count(session)
    )


def absolute_url(request: Request, route_name: str, **params) -> str:
    return str(request.url_for(route_name, **params))


def build_player_page(playlist_url: str, status_url: str) -> str:
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
        align-items: center;
        justify-content: center;
      }}
      video {{
        width: 100%;
        max-height: 72vh;
        background: #000;
        display: block;
      }}
      .panel {{
        width: 100%;
      }}
      .loading {{
        min-height: 72vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
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
        font-size: 16px;
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
      <div class="panel">
        <div id="loading" class="loading">
          <div id="spinner" class="spinner"></div>
          <div id="statusText" class="status-text">Preparing stream...</div>
        </div>
        <video id="video" class="hidden" controls autoplay playsinline></video>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      (function() {{
        const playlistUrl = {playlist_url!r};
        const statusUrl = {status_url!r};
        const loading = document.getElementById("loading");
        const spinner = document.getElementById("spinner");
        const statusText = document.getElementById("statusText");
        const video = document.getElementById("video");
        let started = false;
        let pollHandle = null;

        function attachPlayer() {{
          if (started) {{
            return;
          }}
          started = true;
          if (pollHandle !== null) {{
            window.clearInterval(pollHandle);
            pollHandle = null;
          }}
          loading.classList.add("hidden");
          video.classList.remove("hidden");
          if (video.canPlayType("application/vnd.apple.mpegurl")) {{
            video.src = playlistUrl;
            video.play().catch(() => {{}});
            return;
          }}
          if (window.Hls && Hls.isSupported()) {{
            const hls = new Hls({{
              lowLatencyMode: false,
              backBufferLength: 90,
              maxBufferLength: 60,
              liveSyncDurationCount: 3,
              liveMaxLatencyDurationCount: 6,
            }});
            hls.loadSource(playlistUrl);
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED, function() {{
              video.play().catch(() => {{}});
            }});
            return;
          }}
          spinner.classList.add("hidden");
          statusText.textContent = "This browser cannot play HLS.";
          statusText.classList.add("error-text");
        }}

        function showError(message) {{
          if (pollHandle !== null) {{
            window.clearInterval(pollHandle);
            pollHandle = null;
          }}
          spinner.classList.add("hidden");
          statusText.textContent = message || "Stream failed to start.";
          statusText.classList.add("error-text");
        }}

        async function checkReadiness() {{
          if (started) {{
            return;
          }}
          try {{
            const resp = await fetch(statusUrl, {{ cache: "no-store" }});
            if (!resp.ok) {{
              showError("Failed to check stream status.");
              return;
            }}
            const status = await resp.json();
            if (status.state === "error") {{
              showError(status.error || "Stream failed to start.");
              return;
            }}
            if (status.ready_for_playback) {{
              attachPlayer();
            }}
          }} catch (_err) {{
            showError("Failed to check stream status.");
          }}
        }}

        checkReadiness();
        pollHandle = window.setInterval(checkReadiness, 1000);
      }})();
    </script>
  </body>
</html>
"""


def _enum_device_from_runtime(device: str):
    from python.api.lib.schemas import DeviceEnum

    if device == "cpu":
        return DeviceEnum.CPU
    return DeviceEnum.CUDA_0


registry = StreamRegistry()
