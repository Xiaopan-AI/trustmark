import glob
from collections import deque
import math
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from trustmark import TrustMark


@dataclass
class LiveStreamConfig:
    prebuffer_seconds: int = 4
    segment_seconds: int = 2
    hls_list_size: int = 6
    ffmpeg_preset: str = "ultrafast"
    max_concurrent_jobs: int = 1
    max_batch_size: int = 64
    min_batch_size: int = 1
    decode_queue_size: int = 256
    encoded_queue_size: int = 256
    ffmpeg_queue_size: int = 256
    use_nvenc: bool = True
    default_decode_workers: int = 2
    max_decode_workers: int = 8


class LiveJobState:
    INIT = "INIT"
    BUFFERING = "BUFFERING"
    STREAMING = "STREAMING"
    DONE = "DONE"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class HardwareVideoReader:
    def __init__(self, video_path: str, use_hardware: bool = True):
        self.video_path = video_path
        self.use_hardware = use_hardware
        self.reader = None
        self.cap = None
        self.hw_available = False
        if use_hardware:
            try:
                self.hw_available = self._init_hardware_reader()
            except Exception:
                self.hw_available = False
        if not self.hw_available:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

    def _init_hardware_reader(self) -> bool:
        import av

        container = av.open(self.video_path)
        stream = container.streams.video[0]
        hw_codecs = ["h264_cuvid", "hevc_cuvid", "h264_nvdec", "hevc_nvdec"]
        for codec_name in hw_codecs:
            try:
                stream.codec_context.codec = codec_name
                self.reader = container
                return True
            except Exception:
                continue
        self.reader = av.open(self.video_path)
        return True

    def get_properties(self):
        if self.hw_available and self.reader:
            stream = self.reader.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 25.0
            return fps, int(stream.width), int(stream.height)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height

    def get_total_frames(self) -> int:
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return 0

    def read_frames(self):
        if self.hw_available and self.reader:
            for frame in self.reader.decode(video=0):
                frame_rgb = frame.to_ndarray(format="rgb24")
                frame_bgr = frame_rgb[..., ::-1].copy()
                yield True, frame_bgr
            yield False, None
            return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                yield False, None
                return
            yield True, frame

    def release(self):
        if self.reader:
            self.reader.close()
        if self.cap:
            self.cap.release()


class FFmpegPipeReader:
    def __init__(self, video_path: str, width: int, height: int):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found in PATH")
        self.proc = subprocess.Popen(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_tail = ""

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

    def read_frames(self):
        while True:
            raw = self._read_exact(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                yield False, None
                return
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            yield True, frame

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
                    tail = err.decode("utf-8", errors="ignore")
                    self._stderr_tail = tail[-500:]
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


class _StaticDirHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, fmt, *args):
        return


class LiveStreamManager:
    def __init__(self):
        self.config = LiveStreamConfig()
        self._lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}
        self._base_dir = Path(__file__).resolve().parent / "live_stream_output"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._server_port = self._start_static_server(str(self._base_dir))

    def _start_static_server(self, directory: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = ThreadingHTTPServer(
            ("127.0.0.1", port),
            lambda *args, **kwargs: _StaticDirHandler(*args, directory=directory, **kwargs),
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._httpd = server
        return port

    def _supports_nvenc(self, ffmpeg_bin: str) -> bool:
        try:
            probe = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            return "h264_nvenc" in probe.stdout
        except Exception:
            return False

    def _build_ffmpeg_cmd(
        self,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps: float,
        segment_seconds: int,
        output_dir: Path,
        playlist_path: str,
        use_nvenc: bool,
    ):
        gop = max(1, int(round(fps * segment_seconds)))
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
        ]
        if use_nvenc:
            cmd += [
                "-c:v",
                "h264_nvenc",
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
                "-c:v",
                "libx264",
                "-preset",
                self.config.ffmpeg_preset,
                "-tune",
                "zerolatency",
                "-profile:v",
                "baseline",
                "-level",
                "3.1",
            ]
        cmd += [
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-sc_threshold",
            "0",
            "-f",
            "hls",
            "-hls_time",
            str(segment_seconds),
            "-hls_list_size",
            str(self.config.hls_list_size),
            "-hls_flags",
            "delete_segments+append_list+independent_segments",
            "-hls_segment_filename",
            str(output_dir / "seg_%06d.ts"),
            playlist_path,
        ]
        return cmd

    def start_job(
        self,
        video_path: str,
        wm_secret: str,
        device: str,
        model_type: str,
        prebuffer_seconds: int = 4,
        segment_seconds: int = 2,
        gpu_batch_target: Optional[int] = None,
        gpu_batch_max: Optional[int] = None,
        gpu_flush_ms: Optional[int] = None,
        use_nvenc: Optional[bool] = None,
        decode_mode: Optional[str] = None,
        decode_workers: Optional[int] = None,
        inference_scale: Optional[float] = None,
    ) -> str:
        with self._lock:
            active = [j for j in self._jobs.values() if j["state"] in (LiveJobState.INIT, LiveJobState.BUFFERING, LiveJobState.STREAMING)]
            if len(active) >= self.config.max_concurrent_jobs:
                raise RuntimeError("Another live stream job is already running.")

            job_id = uuid.uuid4().hex[:12]
            output_dir = self._base_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            secret_int = int(wm_secret)
            secret_bits = "".join([str((secret_int >> i) & 1) for i in reversed(range(56))])
            batch_target = int(gpu_batch_target if gpu_batch_target is not None else 12)
            batch_target = max(self.config.min_batch_size, min(self.config.max_batch_size, batch_target))
            batch_max = int(gpu_batch_max if gpu_batch_max is not None else max(batch_target, 20))
            batch_max = max(batch_target, min(self.config.max_batch_size, batch_max))
            flush_ms = int(gpu_flush_ms if gpu_flush_ms is not None else 12)
            flush_ms = max(1, min(1000, flush_ms))
            resolved_nvenc = self.config.use_nvenc if use_nvenc is None else bool(use_nvenc)
            resolved_decode_mode = (decode_mode or "auto").lower()
            if resolved_decode_mode not in ("auto", "single_opencv", "ffmpeg_pipe"):
                resolved_decode_mode = "auto"
            resolved_decode_workers = int(decode_workers if decode_workers is not None else self.config.default_decode_workers)
            resolved_decode_workers = max(1, min(self.config.max_decode_workers, resolved_decode_workers))
            resolved_inference_scale = float(inference_scale if inference_scale is not None else 1.0)
            if resolved_inference_scale not in (1.0, 0.75, 0.5):
                resolved_inference_scale = 1.0

            stop_event = threading.Event()
            job = {
                "job_id": job_id,
                "state": LiveJobState.INIT,
                "error": "",
                "warnings": "",
                "input_video_path": video_path,
                "output_dir": str(output_dir),
                "playlist_path": str(output_dir / "stream.m3u8"),
                "playlist_url": f"http://127.0.0.1:{self._server_port}/{job_id}/stream.m3u8",
                "player_url": f"http://127.0.0.1:{self._server_port}/{job_id}/player.html",
                "fps": 0.0,
                "width": 0,
                "height": 0,
                "total_frames": 0,
                "frames_processed": 0,
                "segments_written": 0,
                "start_time": time.time(),
                "ready_time": None,
                "device": device,
                "model_type": model_type,
                "secret_bits": secret_bits,
                "prebuffer_seconds": int(prebuffer_seconds),
                "segment_seconds": int(segment_seconds),
                "stop_event": stop_event,
                "gpu_batch_target": batch_target,
                "gpu_batch_max": batch_max,
                "gpu_flush_ms": flush_ms,
                "use_nvenc": resolved_nvenc,
                "decode_mode": resolved_decode_mode,
                "decode_workers": resolved_decode_workers,
                "inference_scale": resolved_inference_scale,
                "encoder_backend": "pending",
                "decode_q_depth": 0,
                "encoded_q_depth": 0,
                "ffmpeg_q_depth": 0,
                "reorder_buffer_size": 0,
                "decode_fps": 0.0,
                "gpu_encode_fps": 0.0,
                "writer_fps": 0.0,
                "avg_batch_size": 0.0,
                "gpu_starvation_pct": 0.0,
                "decoder_fps_list": "",
                "dropped_frames": 0,
                "decode_backend": "unknown",
                "decode_read_errors": 0,
                "avg_frame_read_ms": 0.0,
                "ffmpeg_decode_stderr_tail": "",
                "avg_decode_enqueue_ms": 0.0,
                "avg_gpu_batch_infer_ms": 0.0,
                "avg_gpu_batch_emit_ms": 0.0,
                "avg_reorder_enqueue_ms": 0.0,
                "avg_writer_write_ms": 0.0,
                "probe_latest": "Probe idle",
                "probe_log": "",
            }
            self._jobs[job_id] = job
            self._write_player_html(output_dir / "player.html", job["playlist_url"])

            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
            job["thread"] = thread
            thread.start()
            return job_id

    def get_status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {"exists": False, "state": "NOT_FOUND"}
            return {
                "exists": True,
                "job_id": job_id,
                "state": job["state"],
                "error": job["error"],
                "warnings": job.get("warnings", ""),
                "frames_processed": job["frames_processed"],
                "total_frames": job["total_frames"],
                "segments_written": job["segments_written"],
                "playlist_url": job["playlist_url"] if job["state"] in (LiveJobState.STREAMING, LiveJobState.DONE) else "",
                "player_url": job["player_url"] if job["state"] in (LiveJobState.STREAMING, LiveJobState.DONE) else "",
                "encoder_backend": job.get("encoder_backend", "unknown"),
                "decode_mode": job.get("decode_mode", "auto"),
                "decode_workers": job.get("decode_workers", 1),
                "decode_q_depth": job.get("decode_q_depth", 0),
                "encoded_q_depth": job.get("encoded_q_depth", 0),
                "ffmpeg_q_depth": job.get("ffmpeg_q_depth", 0),
                "reorder_buffer_size": job.get("reorder_buffer_size", 0),
                "decode_fps": job.get("decode_fps", 0.0),
                "gpu_encode_fps": job.get("gpu_encode_fps", 0.0),
                "writer_fps": job.get("writer_fps", 0.0),
                "avg_batch_size": job.get("avg_batch_size", 0.0),
                "gpu_starvation_pct": job.get("gpu_starvation_pct", 0.0),
                "decoder_fps_list": job.get("decoder_fps_list", ""),
                "dropped_frames": job.get("dropped_frames", 0),
                "decode_backend": job.get("decode_backend", "unknown"),
                "decode_read_errors": job.get("decode_read_errors", 0),
                "avg_frame_read_ms": job.get("avg_frame_read_ms", 0.0),
                "ffmpeg_decode_stderr_tail": job.get("ffmpeg_decode_stderr_tail", ""),
                "avg_decode_enqueue_ms": job.get("avg_decode_enqueue_ms", 0.0),
                "avg_gpu_batch_infer_ms": job.get("avg_gpu_batch_infer_ms", 0.0),
                "avg_gpu_batch_emit_ms": job.get("avg_gpu_batch_emit_ms", 0.0),
                "avg_reorder_enqueue_ms": job.get("avg_reorder_enqueue_ms", 0.0),
                "avg_writer_write_ms": job.get("avg_writer_write_ms", 0.0),
                "probe_latest": job.get("probe_latest", "Probe idle"),
                "probe_log": job.get("probe_log", ""),
                "gpu_batch_target": job.get("gpu_batch_target", 0),
                "gpu_batch_max": job.get("gpu_batch_max", 0),
                "gpu_flush_ms": job.get("gpu_flush_ms", 0),
                "inference_scale": job.get("inference_scale", 1.0),
            }

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job["stop_event"].set()
            return True

    def _set_state(self, job_id: str, state: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["state"] = state
                if state == LiveJobState.STREAMING and self._jobs[job_id]["ready_time"] is None:
                    self._jobs[job_id]["ready_time"] = time.time()

    def _set_error(self, job_id: str, err: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["error"] = err
                self._jobs[job_id]["state"] = LiveJobState.ERROR

    def _run_job(self, job_id: str):
        try:
            with self._lock:
                job = self._jobs[job_id]
                stop_event = job["stop_event"]
                video_path = job["input_video_path"]
                output_dir = Path(job["output_dir"])
                playlist_path = job["playlist_path"]
                device = job["device"]
                model_type = job["model_type"]
                bits_str = job["secret_bits"]
                prebuffer_seconds = job["prebuffer_seconds"]
                segment_seconds = max(1, job["segment_seconds"])
                batch_target = job["gpu_batch_target"]
                batch_max = job["gpu_batch_max"]
                flush_ms = job["gpu_flush_ms"]
                use_nvenc = job["use_nvenc"]
                decode_mode = job["decode_mode"]
                decode_workers = job["decode_workers"]
                inference_scale = job.get("inference_scale", 1.0)

            self._set_state(job_id, LiveJobState.BUFFERING)
            errors = []
            metrics = {
                "decoded_count": 0,
                "encoded_count": 0,
                "written_count": 0,
                "batch_frames_total": 0,
                "batch_count": 0,
                "gpu_poll_events": 0,
                "gpu_starve_events": 0,
                "dropped_frames": 0,
                "decode_enqueue_ms_total": 0.0,
                "gpu_batch_infer_ms_total": 0.0,
                "gpu_batch_emit_ms_total": 0.0,
                "reorder_enqueue_ms_total": 0.0,
                "writer_write_ms_total": 0.0,
            }
            metrics_lock = threading.Lock()
            t0 = time.time()

            video_reader = HardwareVideoReader(video_path, use_hardware=True)
            fps, width, height = video_reader.get_properties()
            total_frames = video_reader.get_total_frames()
            prebuffer_frames = max(1, int(round(prebuffer_seconds * fps)))
            prebuffer_segments = max(1, int(math.ceil(float(prebuffer_seconds) / float(segment_seconds))))
            decode_mode_final = decode_mode
            if decode_mode_final == "auto":
                decode_mode_final = "ffmpeg_pipe"
            if decode_mode_final not in ("single_opencv", "ffmpeg_pipe"):
                decode_mode_final = "single_opencv"

            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["fps"] = fps
                    self._jobs[job_id]["width"] = width
                    self._jobs[job_id]["height"] = height
                    self._jobs[job_id]["total_frames"] = total_frames
                    self._jobs[job_id]["decode_mode"] = decode_mode_final

            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

            use_nvenc_final = use_nvenc and self._supports_nvenc(ffmpeg_bin)
            if use_nvenc and not use_nvenc_final:
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["warnings"] = "NVENC unavailable, using libx264."

            ffmpeg_cmd = self._build_ffmpeg_cmd(
                ffmpeg_bin=ffmpeg_bin,
                width=width,
                height=height,
                fps=fps,
                segment_seconds=segment_seconds,
                output_dir=output_dir,
                playlist_path=playlist_path,
                use_nvenc=use_nvenc_final,
            )
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["encoder_backend"] = "h264_nvenc" if use_nvenc_final else "libx264"

            decode_q: "queue.Queue" = queue.Queue(maxsize=self.config.decode_queue_size)
            encoded_q: "queue.Queue" = queue.Queue(maxsize=self.config.encoded_queue_size)
            ffmpeg_q: "queue.Queue" = queue.Queue(maxsize=self.config.ffmpeg_queue_size)
            probe_stop_event = threading.Event()

            reader_done = threading.Event()
            gpu_done = threading.Event()
            writer_done = threading.Event()
            decode_sentinel = object()
            encoded_sentinel = object()
            writer_sentinel = object()

            def reader_loop():
                idx = 0
                local_count = 0
                read_ms_total = 0.0
                t_reader_start = time.time()
                ffmpeg_reader = None
                backend = "opencv"
                try:
                    if decode_mode_final == "ffmpeg_pipe":
                        ffmpeg_reader = FFmpegPipeReader(video_path, width, height)
                        frame_source = ffmpeg_reader.read_frames()
                        backend = "ffmpeg_pipe"
                    else:
                        frame_source = video_reader.read_frames()
                        backend = "opencv"

                    for ret, frame in frame_source:
                        if stop_event.is_set():
                            break
                        if not ret:
                            break
                        t_r = time.time()
                        decode_q.put((idx, frame))
                        enqueue_ms = (time.time() - t_r) * 1000.0
                        read_ms_total += enqueue_ms
                        idx += 1
                        local_count += 1
                        with metrics_lock:
                            metrics["decoded_count"] += 1
                            metrics["decode_enqueue_ms_total"] += enqueue_ms
                except Exception as e:
                    with metrics_lock:
                        metrics["dropped_frames"] += 1
                    if decode_mode_final == "ffmpeg_pipe":
                        # auto fallback to opencv when ffmpeg_pipe fails
                        if decode_mode == "auto":
                            try:
                                with self._lock:
                                    if job_id in self._jobs:
                                        self._jobs[job_id]["warnings"] = (
                                            (self._jobs[job_id]["warnings"] + " | " if self._jobs[job_id]["warnings"] else "")
                                            + f"ffmpeg_pipe failed ({e}), falling back to OpenCV."
                                        )
                                        self._jobs[job_id]["decode_mode"] = "single_opencv"
                                for ret, frame in video_reader.read_frames():
                                    if stop_event.is_set():
                                        break
                                    if not ret:
                                        break
                                    t_r = time.time()
                                    decode_q.put((idx, frame))
                                    enqueue_ms = (time.time() - t_r) * 1000.0
                                    read_ms_total += enqueue_ms
                                    idx += 1
                                    local_count += 1
                                    with metrics_lock:
                                        metrics["decoded_count"] += 1
                                        metrics["decode_enqueue_ms_total"] += enqueue_ms
                                backend = "opencv"
                            except Exception as e2:
                                errors.append(f"Reader fallback error: {e2}")
                        else:
                            errors.append(f"Reader error: {e}")
                    else:
                        errors.append(f"Reader error: {e}")
                finally:
                    elapsed_reader = max(0.001, time.time() - t_reader_start)
                    avg_read_ms = (read_ms_total / local_count) if local_count else 0.0
                    with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id]["decoder_fps_list"] = f"r0:{local_count/elapsed_reader:.1f}"
                            self._jobs[job_id]["decode_backend"] = backend
                            self._jobs[job_id]["decode_read_errors"] = len(errors)
                            self._jobs[job_id]["avg_frame_read_ms"] = avg_read_ms
                            if ffmpeg_reader is not None:
                                self._jobs[job_id]["ffmpeg_decode_stderr_tail"] = ffmpeg_reader.stderr_tail()
                    try:
                        decode_q.put(decode_sentinel, timeout=1)
                    except Exception:
                        pass
                    reader_done.set()
                    try:
                        video_reader.release()
                    except Exception:
                        pass
                    if ffmpeg_reader is not None:
                        ffmpeg_reader.close()

            def gpu_encode_loop():
                try:
                    tm = TrustMark(verbose=False, model_type=model_type, device=device)
                    batch_idx = []
                    batch_frames = []
                    last_flush = time.time()

                    def flush_batch():
                        nonlocal batch_idx, batch_frames, last_flush
                        if not batch_frames:
                            return
                        scaled_batch = []
                        upscale_sizes = []
                        if inference_scale < 0.999:
                            for f in batch_frames:
                                oh, ow = f.shape[:2]
                                nw = max(2, int(round(ow * inference_scale)))
                                nh = max(2, int(round(oh * inference_scale)))
                                scaled = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                                scaled_batch.append(scaled)
                                upscale_sizes.append((ow, oh))
                        else:
                            scaled_batch = batch_frames
                            upscale_sizes = [None] * len(batch_frames)

                        t_infer = time.time()
                        encoded = tm.encode_batch_numpy(scaled_batch, bits_str, "binary")
                        infer_ms = (time.time() - t_infer) * 1000.0
                        t_emit = time.time()
                        for i, (out_idx, out_frame) in enumerate(zip(batch_idx, encoded)):
                            if upscale_sizes[i] is not None:
                                ow, oh = upscale_sizes[i]
                                out_frame = cv2.resize(out_frame, (ow, oh), interpolation=cv2.INTER_LINEAR)
                            encoded_q.put((out_idx, out_frame))
                            metrics["encoded_count"] += 1
                        emit_ms = (time.time() - t_emit) * 1000.0
                        metrics["batch_count"] += 1
                        metrics["batch_frames_total"] += len(batch_frames)
                        metrics["gpu_batch_infer_ms_total"] += infer_ms
                        metrics["gpu_batch_emit_ms_total"] += emit_ms
                        batch_idx = []
                        batch_frames = []
                        last_flush = time.time()

                    while not stop_event.is_set():
                        metrics["gpu_poll_events"] += 1
                        timed_out = False
                        item = None
                        try:
                            item = decode_q.get(timeout=0.01)
                        except queue.Empty:
                            timed_out = True

                        if item is decode_sentinel:
                            flush_batch()
                            break

                        if item is not None:
                            idx, frame = item
                            batch_idx.append(idx)
                            batch_frames.append(frame)
                        elif timed_out and not batch_frames:
                            metrics["gpu_starve_events"] += 1

                        q_depth = decode_q.qsize()
                        dynamic_target = batch_target
                        if q_depth > batch_target * 2:
                            dynamic_target = min(batch_max, batch_target + q_depth // 4)
                        if len(batch_frames) >= dynamic_target:
                            flush_batch()
                        elif batch_frames and (timed_out and (time.time() - last_flush) * 1000.0 >= flush_ms):
                            flush_batch()

                    encoded_q.put(encoded_sentinel)
                except Exception as e:
                    errors.append(f"GPU encode error: {e}")
                    try:
                        encoded_q.put(encoded_sentinel, timeout=1)
                    except Exception:
                        pass
                finally:
                    gpu_done.set()

            def writer_loop():
                try:
                    while True:
                        try:
                            item = ffmpeg_q.get(timeout=0.2)
                        except queue.Empty:
                            if stop_event.is_set() and gpu_done.is_set():
                                break
                            continue
                        if item is writer_sentinel:
                            break
                        if ffmpeg_proc.stdin:
                            t_w = time.time()
                            ffmpeg_proc.stdin.write(item.tobytes())
                            metrics["writer_write_ms_total"] += (time.time() - t_w) * 1000.0
                            metrics["written_count"] += 1
                except Exception as e:
                    errors.append(f"Writer error: {e}")
                finally:
                    try:
                        if ffmpeg_proc.stdin:
                            ffmpeg_proc.stdin.flush()
                            ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    writer_done.set()

            def probe_loop():
                last_seg_path = ""
                try:
                    tm_probe = TrustMark(verbose=False, model_type=model_type, device=device)
                    while not probe_stop_event.is_set():
                        with self._lock:
                            if job_id not in self._jobs:
                                break
                            state_now = self._jobs[job_id]["state"]
                        if state_now != LiveJobState.STREAMING:
                            time.sleep(0.2)
                            continue

                        segs = sorted(glob.glob(str(output_dir / "seg_*.ts")))
                        if not segs:
                            time.sleep(0.5)
                            continue
                        seg_path = segs[-1]
                        if seg_path == last_seg_path:
                            time.sleep(0.5)
                            continue
                        last_seg_path = seg_path

                        cap = cv2.VideoCapture(seg_path)
                        try:
                            ret, frame_bgr = cap.read()
                            if not ret:
                                result = "No watermark"
                            else:
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                frame_pil = Image.fromarray(frame_rgb)
                                wm_secret, wm_present, _ = tm_probe.decode(frame_pil, "binary")
                                result = f"✅ watermark={wm_secret}" if wm_present else "No watermark"
                        except Exception as e:
                            result = f"Probe error: {e}"
                        finally:
                            cap.release()

                        ts = time.strftime("%H:%M:%S")
                        line = f"[{ts}] {result}"
                        with self._lock:
                            if job_id in self._jobs:
                                self._jobs[job_id]["probe_latest"] = result
                                old_log = self._jobs[job_id].get("probe_log", "")
                                lines = [x for x in old_log.splitlines() if x.strip()]
                                lines.append(line)
                                self._jobs[job_id]["probe_log"] = "\n".join(lines[-30:])
                        time.sleep(0.5)
                except Exception as e:
                    with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id]["probe_latest"] = f"Probe error: {e}"

            reader_thread = threading.Thread(target=reader_loop, daemon=True)
            gpu_thread = threading.Thread(target=gpu_encode_loop, daemon=True)
            writer_thread = threading.Thread(target=writer_loop, daemon=True)
            probe_thread = threading.Thread(target=probe_loop, daemon=True)
            reader_thread.start()
            gpu_thread.start()
            writer_thread.start()
            probe_thread.start()

            reorder_buffer = {}
            next_write_idx = 0
            stream_started = False
            last_prof_log_ts = 0.0
            fps_window = deque()  # (timestamp, decoded_count, encoded_count, written_count)

            while True:
                if stop_event.is_set() and gpu_done.is_set():
                    break

                try:
                    msg = encoded_q.get(timeout=0.05)
                    if msg is encoded_sentinel:
                        break
                    idx, frame = msg
                    reorder_buffer[idx] = frame
                except queue.Empty:
                    pass

                while next_write_idx in reorder_buffer:
                    t_rq = time.time()
                    ffmpeg_q.put(reorder_buffer.pop(next_write_idx))
                    metrics["reorder_enqueue_ms_total"] += (time.time() - t_rq) * 1000.0
                    next_write_idx += 1

                elapsed = max(0.001, time.time() - t0)
                now_ts = time.time()
                fps_window.append((now_ts, metrics["decoded_count"], metrics["encoded_count"], metrics["written_count"]))
                while len(fps_window) > 1 and (now_ts - fps_window[0][0]) > 10.0:
                    fps_window.popleft()
                if len(fps_window) > 1:
                    t_old, d_old, g_old, w_old = fps_window[0]
                    dt = max(0.001, now_ts - t_old)
                    dec_fps_win = (metrics["decoded_count"] - d_old) / dt
                    gpu_fps_win = (metrics["encoded_count"] - g_old) / dt
                    wr_fps_win = (metrics["written_count"] - w_old) / dt
                else:
                    dec_fps_win = 0.0
                    gpu_fps_win = 0.0
                    wr_fps_win = 0.0
                seg_count = len(glob.glob(str(output_dir / "seg_*.ts")))
                processed = metrics["encoded_count"]
                if not stream_started and processed >= prebuffer_frames and seg_count >= prebuffer_segments:
                    stream_started = True
                    self._set_state(job_id, LiveJobState.STREAMING)

                avg_batch = (
                    float(metrics["batch_frames_total"]) / float(metrics["batch_count"])
                    if metrics["batch_count"] > 0
                    else 0.0
                )
                avg_decode_enqueue_ms = (
                    float(metrics["decode_enqueue_ms_total"]) / float(metrics["decoded_count"])
                    if metrics["decoded_count"] > 0
                    else 0.0
                )
                avg_gpu_batch_infer_ms = (
                    float(metrics["gpu_batch_infer_ms_total"]) / float(metrics["batch_count"])
                    if metrics["batch_count"] > 0
                    else 0.0
                )
                avg_gpu_batch_emit_ms = (
                    float(metrics["gpu_batch_emit_ms_total"]) / float(metrics["batch_count"])
                    if metrics["batch_count"] > 0
                    else 0.0
                )
                avg_reorder_enqueue_ms = (
                    float(metrics["reorder_enqueue_ms_total"]) / float(metrics["written_count"])
                    if metrics["written_count"] > 0
                    else 0.0
                )
                avg_writer_write_ms = (
                    float(metrics["writer_write_ms_total"]) / float(metrics["written_count"])
                    if metrics["written_count"] > 0
                    else 0.0
                )
                starvation_pct = (
                    100.0 * float(metrics["gpu_starve_events"]) / float(metrics["gpu_poll_events"])
                    if metrics["gpu_poll_events"] > 0
                    else 0.0
                )
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["frames_processed"] = processed
                        self._jobs[job_id]["segments_written"] = seg_count
                        self._jobs[job_id]["decode_q_depth"] = decode_q.qsize()
                        self._jobs[job_id]["encoded_q_depth"] = encoded_q.qsize()
                        self._jobs[job_id]["ffmpeg_q_depth"] = ffmpeg_q.qsize()
                        self._jobs[job_id]["reorder_buffer_size"] = len(reorder_buffer)
                        self._jobs[job_id]["decode_fps"] = dec_fps_win
                        self._jobs[job_id]["gpu_encode_fps"] = gpu_fps_win
                        self._jobs[job_id]["writer_fps"] = wr_fps_win
                        self._jobs[job_id]["avg_batch_size"] = avg_batch
                        self._jobs[job_id]["gpu_starvation_pct"] = starvation_pct
                        self._jobs[job_id]["dropped_frames"] = metrics["dropped_frames"]
                        self._jobs[job_id]["avg_decode_enqueue_ms"] = avg_decode_enqueue_ms
                        self._jobs[job_id]["avg_gpu_batch_infer_ms"] = avg_gpu_batch_infer_ms
                        self._jobs[job_id]["avg_gpu_batch_emit_ms"] = avg_gpu_batch_emit_ms
                        self._jobs[job_id]["avg_reorder_enqueue_ms"] = avg_reorder_enqueue_ms
                        self._jobs[job_id]["avg_writer_write_ms"] = avg_writer_write_ms

                # Console profiler logging disabled; telemetry remains in UI.

            # Drain reorder buffer after GPU done.
            while next_write_idx in reorder_buffer:
                ffmpeg_q.put(reorder_buffer.pop(next_write_idx))
                next_write_idx += 1

            ffmpeg_q.put(writer_sentinel)
            reader_thread.join(timeout=5)
            gpu_thread.join(timeout=5)
            writer_thread.join(timeout=10)
            probe_stop_event.set()
            probe_thread.join(timeout=3)

            if stop_event.is_set():
                try:
                    ffmpeg_proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass
                self._set_state(job_id, LiveJobState.STOPPED)
            elif errors:
                try:
                    ffmpeg_proc.terminate()
                except Exception:
                    pass
                self._set_error(job_id, "; ".join(errors))
            else:
                _, stderr = ffmpeg_proc.communicate(timeout=60)
                if ffmpeg_proc.returncode != 0:
                    self._set_error(job_id, f"ffmpeg failed: {stderr.decode('utf-8', errors='ignore')}")
                else:
                    if Path(playlist_path).exists():
                        if self._jobs[job_id]["state"] == LiveJobState.BUFFERING:
                            self._set_state(job_id, LiveJobState.STREAMING)
                        self._set_state(job_id, LiveJobState.DONE)
                    else:
                        self._set_error(job_id, "HLS playlist was not generated.")

        except Exception as e:
            self._set_error(job_id, str(e))

    def _write_player_html(self, path: Path, playlist_url: str):
        path.write_text(
            f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <style>
      html, body {{ margin: 0; padding: 0; background: #000; }}
      #liveEncodedVideo {{ width: 100vw; height: 100vh; max-height: 520px; background: #000; }}
    </style>
  </head>
  <body>
    <video id="liveEncodedVideo" controls autoplay muted playsinline></video>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      (function() {{
        const video = document.getElementById("liveEncodedVideo");
        const src = "{playlist_url}";
        if (video.canPlayType("application/vnd.apple.mpegurl")) {{
          video.src = src;
        }} else if (window.Hls && Hls.isSupported()) {{
          const hls = new Hls({{ lowLatencyMode: true, backBufferLength: 30 }});
          hls.loadSource(src);
          hls.attachMedia(video);
        }} else {{
          document.body.innerHTML = "<div style='color:white;padding:12px'>HLS playback not supported.</div>";
        }}
      }})();
    </script>
  </body>
</html>
""",
            encoding="utf-8",
        )


def build_hls_player_html(player_url: str) -> str:
    if not player_url:
        return "<div>Waiting for stream...</div>"
    return f"""
<div style="width: 100%;">
  <iframe
    src="{player_url}"
    style="width:100%;height:560px;border:0;background:#000;"
    allow="autoplay; fullscreen">
  </iframe>
</div>
"""


stream_manager = LiveStreamManager()
