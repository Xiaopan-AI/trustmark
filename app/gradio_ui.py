import os
import cv2
import numpy as np
import time
import tempfile
import logging
import multiprocessing as mp
from queue import Empty, Full
from PIL import Image
import gradio as gr
import torch
from trustmark import TrustMark
from live_stream_backend import stream_manager, build_hls_player_html

# 环境清理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

logging.getLogger("PIL").setLevel(logging.WARNING)


# ============================
# Hardware-Accelerated Video Reading
# ============================

class HardwareVideoReader:
    """
    Wrapper for hardware-accelerated video decoding.
    Falls back to CPU decoding if hardware not available.
    """
    def __init__(self, video_path, use_hardware=True):
        self.video_path = video_path
        self.use_hardware = use_hardware
        self.reader = None
        self.cap = None
        self.hw_available = False
        
        # Try hardware-accelerated reading first
        if use_hardware:
            try:
                import av
                self.hw_available = self._init_hardware_reader()
            except ImportError:
                print("PyAV not installed, falling back to OpenCV CPU decoding")
                print("Install with: pip install av")
                self.hw_available = False
        
        # Fallback to OpenCV
        if not self.hw_available:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
    
    def _init_hardware_reader(self):
        """Initialize PyAV with hardware acceleration."""
        try:
            import av
            
            # Try to open with hardware decoding
            container = av.open(self.video_path)
            stream = container.streams.video[0]
            
            # Try hardware decoding codecs
            hw_codecs = ['h264_cuvid', 'hevc_cuvid', 'h264_nvdec', 'hevc_nvdec']
            
            for codec_name in hw_codecs:
                try:
                    stream.codec_context.codec = codec_name
                    self.reader = container
                    print(f"Hardware decoding enabled: {codec_name}")
                    return True
                except:
                    continue
            
            # Hardware failed, but PyAV still available for CPU decoding
            self.reader = av.open(self.video_path)
            print("Using PyAV with CPU decoding")
            return True
            
        except Exception as e:
            print(f"Hardware decode init failed: {e}")
            return False
    
    def get_properties(self):
        """Get video properties: fps, width, height."""
        if self.hw_available and self.reader:
            import av
            stream = self.reader.streams.video[0]
            fps = float(stream.average_rate)
            width = stream.width
            height = stream.height
            return fps, width, height
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return fps, width, height
    
    def read_frames(self):
        """Generator that yields frames as BGR numpy arrays."""
        if self.hw_available and self.reader:
            import av
            for frame in self.reader.decode(video=0):
                # Convert PyAV frame to numpy BGR
                frame_rgb = frame.to_ndarray(format='rgb24')
                frame_bgr = frame_rgb[..., ::-1].copy()
                yield True, frame_bgr
            yield False, None
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    yield False, None
                    break
                yield True, frame
    
    def release(self):
        """Release video resources."""
        if self.reader:
            self.reader.close()
        if self.cap:
            self.cap.release()


# MODEL_TYPE = "Q"  # Now dynamically selected by user
# ============================
# Device and Model Discovery
# ============================

def discover_devices():
    """
    Discover available compute devices (CPU and CUDA GPUs).
    Returns list of tuples: (display_label, device_string)
    """
    devices = []
    
    # CPU always available
    devices.append(("CPU", "cpu"))
    
    # Check for CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            compute_capability = f"{props.major}.{props.minor}"
            label = f"cuda:{i} - {name} ({memory_gb:.1f}GB, Compute {compute_capability})"
            devices.append((label, f"cuda:{i}"))
    
    return devices

def get_default_device():
    """
    Returns the default device string.
    Prefers first CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"

def get_model_types():
    """
    Returns list of available model types with descriptions.
    Format: (display_label, model_code)
    """
    return [
        ("P - High Quality (PSNR ~48) [Default]", "P"),
        ("Q - Balanced (PSNR ~43, ResNet50)", "Q"),
        ("C - Compact (PSNR ~39, ResNet18)", "C"),
        ("B - High Robustness (PSNR ~43)", "B")
    ]

# ============================
# 单进程编码
# ============================

def encode_video_sp(video_path, wm_secret, device, model_type):
    if not video_path:
        return None

    if not video_path:
        return None    
    tm = TrustMark(verbose=False, model_type=model_type, device=device)
    wm_secret = int(wm_secret)
    print(f"Encoding secret: {wm_secret}")
    bits_str = "".join([str((wm_secret >> i) & 1) for i in reversed(range(56))])
    print(f"Encoding bits: {bits_str}")
    try:
        video_reader = HardwareVideoReader(video_path, use_hardware=True)
        fps, width, height = video_reader.get_properties()
    except Exception as e:
        print(f"Video reader initialization failed: {e}")
        return None

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "encoded.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for ret, frame in video_reader.read_frames():
        if not ret:
            break

        wm_frame = tm.encode_numpy(frame, bits_str, "binary")
        out.write(wm_frame)

    video_reader.release()
    out.release()

    return output_path

# ============================
# 多进程编码 (Batch-capable)
# ============================
def watermark_worker_batch(in_q, out_q, bits_str, device, model_type, batch_size=8):
    """
    Batch-capable worker: collects multiple frames and processes them together.
    """
    try:
        from trustmark import TrustMark
        import sys
        
        # Initialize TrustMark with batch support
        tm_worker = TrustMark(verbose=False, model_type=model_type, device=device)
        
        # Check if batch method exists
        if not hasattr(tm_worker, 'encode_batch_numpy'):
            print("Warning: encode_batch_numpy not found, falling back to single-frame processing")
            # Fallback to original worker behavior
            while True:
                try:
                    item = in_q.get(timeout=5)
                    if item is None:
                        break
                    
                    frame_idx, frame_bgr = item
                    # Single frame processing
                    encoded_bgr = tm_worker.encode_numpy(frame_bgr, bits_str, "binary")
                    
                    while True:
                        try:
                            out_q.put((frame_idx, encoded_bgr), timeout=1)
                            break
                        except Full:
                            continue
                except Empty:
                    continue
            return
        
        # Batch processing mode
        while True:
            batch_frames = []
            batch_indices = []
            
            # Collect frames for batch
            timeout_counter = 0
            while len(batch_frames) < batch_size and timeout_counter < 3:
                try:
                    item = in_q.get(timeout=0.1)
                    if item is None:  # End signal
                        # Process remaining batch if any
                        if batch_frames:
                            encoded_batch = tm_worker.encode_batch_numpy(batch_frames, bits_str, "binary")
                            for idx, encoded_bgr in zip(batch_indices, encoded_batch):
                                while True:
                                    try:
                                        out_q.put((idx, encoded_bgr), timeout=1)
                                        break
                                    except Full:
                                        continue
                        return  # Exit worker
                    
                    frame_idx, frame_bgr = item
                    batch_frames.append(frame_bgr)
                    batch_indices.append(frame_idx)
                    
                except Empty:
                    timeout_counter += 1
                    # Process partial batch if we've been waiting
                    if len(batch_frames) > 0 and timeout_counter >= 2:
                        break
            
            # Process batch if we have frames
            if batch_frames:
                try:
                    encoded_batch = tm_worker.encode_batch_numpy(batch_frames, bits_str, "binary")
                    
                    # Put results in output queue
                    for idx, encoded_bgr in zip(batch_indices, encoded_batch):
                        while True:
                            try:
                                out_q.put((idx, encoded_bgr), timeout=1)
                                break
                            except Full:
                                continue
                except Exception as e:
                    print(f"Batch encoding error: {e}")
                    # Fall back to single-frame for this batch
                    for idx, frame_bgr in zip(batch_indices, batch_frames):
                        try:
                            encoded_bgr = tm_worker.encode_numpy(frame_bgr, bits_str, "binary")
                            while True:
                                try:
                                    out_q.put((idx, encoded_bgr), timeout=1)
                                    break
                                except Full:
                                    continue
                        except Exception as e2:
                            print(f"Frame {idx} encoding failed: {e2}")
                
    except Exception as e:
        print(f"Worker process error: {e}")
        import traceback
        traceback.print_exc()

# ============================
# Encode Logic (Multi-process)
# ============================
def encode_video(video_path, wm_secret_val, device, model_type):
    if not video_path:
        return None

    # 1. 准备秘密信息
    try:
        wm_secret_int = int(wm_secret_val)
        bits = [(wm_secret_int >> i) & 1 for i in reversed(range(56))]
        bits_str = "".join(map(str, bits))
    except ValueError:
        return None

    # 2. 打开视频获取信息 (with hardware acceleration)
    try:
        video_reader = HardwareVideoReader(video_path, use_hardware=True)
        fps, width, height = video_reader.get_properties()
    except Exception as e:
        print(f"Video reader initialization failed: {e}")
        return None
    
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "encoded.mp4")
    # 使用 mp4v 编码，如果追求体积小可改用 'avc1'
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 3. 设置多进程
    num_workers = 4  # 建议根据显存调整，若 8G 显存建议 1-2
    in_q = mp.Queue(maxsize=30)
    out_q = mp.Queue(maxsize=30)
    workers = []

    # Batch size per worker - adjust based on GPU memory
    if 'cuda:0' in device:
        batch_size = 8
    elif 'cuda:1' in device:
        batch_size = 4
    else:
        batch_size = 1

    for _ in range(num_workers):
        p = mp.Process(target=watermark_worker_batch, args=(in_q, out_q, bits_str, device, model_type, batch_size))
        p.start()
        workers.append(p)

    # 4. 主循环：读取、分发、回收、写入
    read_idx = 0
    write_idx = 0
    buffer = {}
    all_read = False

    print("Encoding started...")
    
    frame_iterator = None
    try:
        while write_idx < read_idx or not all_read:
            # 读取并放入输入队列 (hardware-accelerated)
            if not all_read and in_q.qsize() < 25:
                if frame_iterator is None:
                    frame_iterator = video_reader.read_frames()

                try:
                    ret, frame_bgr = next(frame_iterator)
                    if ret:
                        try:
                            in_q.put_nowait((read_idx, frame_bgr))
                            read_idx += 1
                        except Full:
                            pass
                    else:
                        all_read = True
                        for _ in range(num_workers):
                            in_q.put(None) # 发送停止信号
                except StopIteration:
                    all_read = True
                    for _ in range(num_workers):
                        in_q.put(None) # 发送停止信号

            # 从输出队列拿结果
            try:
                res = out_q.get(timeout=0.01)
                idx, frame_out = res
                buffer[idx] = frame_out
            except Empty:
                pass

            # 按顺序写入
            while write_idx in buffer:
                writer.write(buffer.pop(write_idx))
                write_idx += 1
    finally:
        video_reader.release()
        writer.release()
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    return output_path

# ============================
# Decode Logic (保持单进程即可，解码通常较快)
# ============================
# 注意：为了避免主进程预加载模型占用 worker 显存，建议在函数内部初始化 tm
def decode_video_watermark(video_path, device, model_type):
    from trustmark import TrustMark
    tm_decoder = TrustMark(verbose=False, model_type=model_type, device=device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "❌ Cannot open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(round(fps / 2)), 1)
    
    frame_idx = 0
    logs = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret: break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # frame_pil = Image.fromarray(frame_rgb).resize((512, 512), Image.BICUBIC)
            frame_pil = Image.fromarray(frame_rgb)
            wm_secret, wm_present, _ = tm_decoder.decode(frame_pil, "binary")
            timestamp = frame_idx / fps
            if wm_present:
                status = f"✅ secret={wm_secret}"
            else:
                status = "❌ No watermark detected"
            logs.append(f"[t={timestamp:.2f}s] {status}")

        frame_idx += 1

    cap.release()
    return "\n".join(logs) if logs else "No frames processed"

def secret_to_bits(secret_str):
    try:
        val = int(secret_str)
        return "".join([str((val >> i) & 1) for i in reversed(range(56))])
    except:
        return "Invalid integer"

# ============================
# Gradio UI
# ============================
def launch_app():
    with gr.Blocks(title="Video Watermark Tool") as demo:
        gr.Markdown("## 🎥 Video Watermark Encoder (Multi-Process)")
        
        # Settings Section - Universal Configuration
        gr.Markdown("### ⚙️ Configuration")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Discover devices and get default
                available_devices = discover_devices()
                device_labels = [d[0] for d in available_devices]
                device_values = [d[1] for d in available_devices]
                default_device_value = get_default_device()
                default_device_index = device_values.index(default_device_value) if default_device_value in device_values else 0
                
                device_selector = gr.Dropdown(
                    choices=device_labels,
                    value=device_labels[default_device_index],
                    label="Compute Device",
                    info="Select device for inference"
                )
                
                # Hidden state to store actual device string (cuda:0, cpu, etc.)
                device_state = gr.State(value=default_device_value)
                
            with gr.Column(scale=1):
                # Model type selection
                model_choices = get_model_types()
                model_labels = [m[0] for m in model_choices]
                model_values = [m[1] for m in model_choices]
                default_model_index = model_values.index("P")
                
                model_selector = gr.Dropdown(
                    choices=model_labels,
                    value=model_labels[default_model_index],
                    label="Model Type",
                    info="Select TrustMark model variant"
                )
                
                # Hidden state to store actual model code (Q, P, C, B)
                model_state = gr.State(value="P")
        
        # Device info display
        device_info = gr.Textbox(
            label="Current Configuration",
            value=f"Device: {device_labels[default_device_index]} | Model: P - High Quality (PSNR ~48) [Default]",
            interactive=False,
            lines=1
        )
        
        gr.Markdown("---")  # Visual separator
        
        # Update states when selections change
        def update_device_state(selected_label):
            # Find the device value corresponding to selected label
            for label, value in available_devices:
                if label == selected_label:
                    return value, f"Device: {selected_label} | Model: {model_state.value}"
            return device_state.value, device_info.value
        
        def update_model_state(selected_label):
            # Find the model value corresponding to selected label
            for label, value in model_choices:
                if label == selected_label:
                    return value, f"Device: {device_selector.value} | Model: {selected_label}"
            return model_state.value, device_info.value
        
        def update_device_info(device_label, model_label):
            return f"Device: {device_label} | Model: {model_label}"

        def _extract_video_path(video_value):
            if isinstance(video_value, str):
                return video_value
            if isinstance(video_value, dict):
                return video_value.get("path") or video_value.get("name")
            return None

        live_ui_cache = {
            "job_id": "",
            "status": "",
            "progress": "",
            "probe_log": "",
            "last_ui_push_ts": 0.0,
        }

        def start_live_encode(
            video_value,
            wm_secret,
            device,
            model_type,
            prebuffer_s,
            segment_s,
            gpu_batch_target,
            gpu_batch_max,
            gpu_flush_ms,
            use_nvenc,
            decode_mode,
            decode_workers,
            inference_scale,
        ):
            video_path = _extract_video_path(video_value)
            if not video_path:
                status_txt = "ERROR"
                progress_txt = "Please upload a source video."
                probe_txt = "Probe idle"
                live_ui_cache["job_id"] = ""
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                live_ui_cache["last_ui_push_ts"] = time.time()
                return "", status_txt, progress_txt, "<div>No stream.</div>", "", probe_txt
            try:
                job_id = stream_manager.start_job(
                    video_path=video_path,
                    wm_secret=wm_secret,
                    device=device,
                    model_type=model_type,
                    prebuffer_seconds=int(prebuffer_s),
                    segment_seconds=int(segment_s),
                    gpu_batch_target=int(gpu_batch_target),
                    gpu_batch_max=int(gpu_batch_max),
                    gpu_flush_ms=int(gpu_flush_ms),
                    use_nvenc=bool(use_nvenc),
                    decode_mode=str(decode_mode),
                    decode_workers=int(decode_workers),
                    inference_scale=float(inference_scale),
                )
                status_txt = "BUFFERING"
                progress_txt = "Job created. Preparing stream buffer..."
                probe_txt = "Probe idle"
                live_ui_cache["job_id"] = job_id
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                live_ui_cache["last_ui_push_ts"] = time.time()
                return job_id, status_txt, progress_txt, "<div>Buffering stream...</div>", "", probe_txt
            except Exception as e:
                status_txt = "ERROR"
                progress_txt = str(e)
                probe_txt = f"Probe error: {e}"
                live_ui_cache["job_id"] = ""
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                live_ui_cache["last_ui_push_ts"] = time.time()
                return "", status_txt, progress_txt, "<div>Failed to start stream.</div>", "", probe_txt

        def poll_live_encode(
            job_id,
            rendered_playlist_url=None,
        ):
            last_status = live_ui_cache["status"]
            last_progress = live_ui_cache["progress"]
            last_probe_log = live_ui_cache["probe_log"]
            last_ui_push_ts = live_ui_cache["last_ui_push_ts"]
            if not job_id:
                status_txt = "IDLE"
                progress_txt = "No active job."
                probe_txt = "Probe idle"
                status_out = status_txt if status_txt != last_status else gr.update()
                progress_out = progress_txt if progress_txt != last_progress else gr.update()
                probe_out = probe_txt if probe_txt != last_probe_log else gr.update()
                live_ui_cache["job_id"] = ""
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                return status_out, progress_out, "<div>Waiting for stream...</div>", "", probe_out
            status = stream_manager.get_status(job_id)
            if not status.get("exists"):
                status_txt = "NOT_FOUND"
                progress_txt = "Job not found."
                probe_txt = "Probe idle"
                status_out = status_txt if status_txt != last_status else gr.update()
                progress_out = progress_txt if progress_txt != last_progress else gr.update()
                probe_out = probe_txt if probe_txt != last_probe_log else gr.update()
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                return status_out, progress_out, "<div>Stream unavailable.</div>", "", probe_out

            state = status.get("state", "UNKNOWN")
            total = status.get("total_frames", 0)
            processed = status.get("frames_processed", 0)
            segs = status.get("segments_written", 0)
            batch_target = status.get("gpu_batch_target", 0)
            batch_max = status.get("gpu_batch_max", 0)
            batch_avg = status.get("avg_batch_size", 0.0)
            flush_ms = status.get("gpu_flush_ms", 0)
            backend = status.get("encoder_backend", "unknown")
            decode_mode = status.get("decode_mode", "auto")
            decode_workers = status.get("decode_workers", 1)
            decode_q = status.get("decode_q_depth", 0)
            encoded_q = status.get("encoded_q_depth", 0)
            ff_q = status.get("ffmpeg_q_depth", 0)
            reorder_sz = status.get("reorder_buffer_size", 0)
            dec_fps = status.get("decode_fps", 0.0)
            gpu_fps = status.get("gpu_encode_fps", 0.0)
            wr_fps = status.get("writer_fps", 0.0)
            starve_pct = status.get("gpu_starvation_pct", 0.0)
            decoder_fps_list = status.get("decoder_fps_list", "")
            dropped = status.get("dropped_frames", 0)
            decode_backend = status.get("decode_backend", "unknown")
            read_ms = status.get("avg_frame_read_ms", 0.0)
            inference_scale = status.get("inference_scale", 1.0)
            t_decode_q = status.get("avg_decode_enqueue_ms", 0.0)
            t_gpu_infer = status.get("avg_gpu_batch_infer_ms", 0.0)
            t_gpu_emit = status.get("avg_gpu_batch_emit_ms", 0.0)
            t_reorder_q = status.get("avg_reorder_enqueue_ms", 0.0)
            t_write = status.get("avg_writer_write_ms", 0.0)
            warn = status.get("warnings", "")
            probe_latest = status.get("probe_latest", "Probe idle")
            probe_log = status.get("probe_log", "Probe idle")
            if total and total > 0:
                progress = (
                    f"{processed}/{total} frames | seg={segs} | enc={backend} | "
                    f"fps(d/g/w)={dec_fps:.1f}/{gpu_fps:.1f}/{wr_fps:.1f} | "
                    f"decode={decode_mode}:{decode_workers}w ({decode_backend}, {read_ms:.2f}ms, {decoder_fps_list}) | "
                    f"inf_scale={inference_scale:.2f} | "
                    f"timers[ms] dq={t_decode_q:.3f} ginf={t_gpu_infer:.3f} gemit={t_gpu_emit:.3f} rq={t_reorder_q:.3f} wr={t_write:.3f} | "
                    f"batch(t/m/avg)={batch_target}/{batch_max}/{batch_avg:.1f} flush={flush_ms}ms | "
                    f"q(d/e/f)={decode_q}/{encoded_q}/{ff_q} reorder={reorder_sz} | "
                    f"gpu_starve={starve_pct:.1f}% dropped={dropped}"
                )
            else:
                progress = (
                    f"{processed} frames | seg={segs} | enc={backend} | "
                    f"fps(d/g/w)={dec_fps:.1f}/{gpu_fps:.1f}/{wr_fps:.1f} | "
                    f"decode={decode_mode}:{decode_workers}w ({decode_backend}, {read_ms:.2f}ms, {decoder_fps_list}) | "
                    f"inf_scale={inference_scale:.2f} | "
                    f"timers[ms] dq={t_decode_q:.3f} ginf={t_gpu_infer:.3f} gemit={t_gpu_emit:.3f} rq={t_reorder_q:.3f} wr={t_write:.3f} | "
                    f"batch(t/m/avg)={batch_target}/{batch_max}/{batch_avg:.1f} flush={flush_ms}ms | "
                    f"q(d/e/f)={decode_q}/{encoded_q}/{ff_q} reorder={reorder_sz} | "
                    f"gpu_starve={starve_pct:.1f}% dropped={dropped}"
                )
            if warn:
                progress = f"{progress} | note={warn}"

            # Lightweight bottleneck hint from telemetry
            hint = ""
            if state in ("BUFFERING", "STREAMING", "DONE"):
                if starve_pct > 30.0 and decode_q < max(4, batch_target // 2):
                    hint = "Hint: DECODE_BOUND -> try Decode Mode=ffmpeg_pipe, raise Decode Workers, lower GPU Batch Target."
                elif ff_q > 64 and wr_fps < gpu_fps * 0.8:
                    hint = "Hint: WRITER_BOUND -> enable NVENC, lower segment seconds, reduce GPU Batch Max slightly."
                elif decode_q > 64 and starve_pct < 10.0 and gpu_fps < dec_fps * 0.8:
                    hint = "Hint: GPU_BOUND -> raise GPU Batch Target/Max and reduce GPU Flush(ms)."
                elif reorder_sz > 64:
                    hint = "Hint: REORDER_BACKLOG -> reduce Decode Workers or lower batch settings."
                else:
                    hint = "Hint: BALANCED -> fine tune one knob at a time."
            if hint:
                progress = f"{progress} | {hint}"

            if state in ("STREAMING", "DONE"):
                player_url = status.get("player_url", "")
                if player_url and player_url != rendered_playlist_url:
                    html = build_hls_player_html(player_url)
                    status_txt = state
                    progress_txt = f"{progress} | probe={probe_latest}"
                    probe_txt = probe_log
                    status_out = status_txt if status_txt != last_status else gr.update()
                    progress_out = progress_txt if progress_txt != last_progress else gr.update()
                    probe_out = probe_txt if probe_txt != last_probe_log else gr.update()
                    live_ui_cache["status"] = status_txt
                    live_ui_cache["progress"] = progress_txt
                    live_ui_cache["probe_log"] = probe_txt
                    live_ui_cache["last_ui_push_ts"] = time.time()
                    return status_out, progress_out, html, player_url, probe_out
                html = gr.update()
            elif state == "ERROR":
                html = f"<div>Stream error: {status.get('error', 'Unknown error')}</div>"
                status_txt = state
                progress_txt = f"{progress} | probe={probe_latest}"
                probe_txt = probe_log
                status_out = status_txt if status_txt != last_status else gr.update()
                progress_out = progress_txt if progress_txt != last_progress else gr.update()
                probe_out = probe_txt if probe_txt != last_probe_log else gr.update()
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                live_ui_cache["last_ui_push_ts"] = time.time()
                return status_out, progress_out, html, rendered_playlist_url, probe_out
            elif state == "STOPPED":
                html = "<div>Stream stopped.</div>"
                status_txt = state
                progress_txt = f"{progress} | probe={probe_latest}"
                probe_txt = probe_log
                status_out = status_txt if status_txt != last_status else gr.update()
                progress_out = progress_txt if progress_txt != last_progress else gr.update()
                probe_out = probe_txt if probe_txt != last_probe_log else gr.update()
                live_ui_cache["status"] = status_txt
                live_ui_cache["progress"] = progress_txt
                live_ui_cache["probe_log"] = probe_txt
                live_ui_cache["last_ui_push_ts"] = time.time()
                return status_out, progress_out, html, "", probe_out
            else:
                html = "<div>Buffering stream...</div>"
            status_txt = state
            progress_txt = f"{progress} | probe={probe_latest}"
            probe_txt = probe_log
            now_ts = time.time()
            should_push = (state != last_status) or ((now_ts - float(last_ui_push_ts or 0.0)) >= 5.0)
            status_out = status_txt if (should_push and status_txt != last_status) else gr.update()
            progress_out = progress_txt if (should_push and progress_txt != last_progress) else gr.update()
            probe_out = probe_txt if (should_push and probe_txt != last_probe_log) else gr.update()
            live_ui_cache["status"] = status_txt
            live_ui_cache["progress"] = progress_txt
            live_ui_cache["probe_log"] = probe_txt
            if should_push:
                live_ui_cache["last_ui_push_ts"] = now_ts
            return status_out, progress_out, html, rendered_playlist_url, probe_out

        def stop_live_encode(job_id):
            if not job_id:
                return "IDLE", "No active job."
            ok = stream_manager.stop_job(job_id)
            if ok:
                return "STOPPED", "Stop signal sent."
            return "NOT_FOUND", "Job not found."
        
        device_selector.change(
            fn=update_device_state,
            inputs=[device_selector],
            outputs=[device_state, device_info]
        )
        
        model_selector.change(
            fn=update_model_state,
            inputs=[model_selector],
            outputs=[model_state, device_info]
        )

        with gr.Tabs():
            
            with gr.Tab("🔐 Encode_SP"):
                input_video = gr.Video(label="Input Video")
                wm_secret = gr.Textbox(label="Secret (Int)")
                bit_display = gr.Textbox(label="56-bit Binary", interactive=False)
                
                wm_secret.change(fn=secret_to_bits, inputs=wm_secret, outputs=bit_display)
                
                output_video = gr.Video(label="Output")
                encode_btn = gr.Button("Start Single-process Encoding")
                encode_btn.click(fn=encode_video_sp, inputs=[input_video, wm_secret, device_state, model_state], outputs=output_video)            
            
            
            
            with gr.Tab("🔐 Encode_MP"):
                input_video = gr.Video(label="Input Video")
                wm_secret = gr.Textbox(label="Secret (Int)")
                bit_display = gr.Textbox(label="56-bit Binary", interactive=False)
                
                wm_secret.change(fn=secret_to_bits, inputs=wm_secret, outputs=bit_display)
                
                output_video = gr.Video(label="Output")
                encode_btn = gr.Button("Start Multi-process Encoding")
                encode_btn.click(fn=encode_video, inputs=[input_video, wm_secret, device_state, model_state], outputs=output_video)
  
            with gr.Tab("🔓 Decode"):
                video_input = gr.Video()
                output_log = gr.Textbox(label="Log", lines=15)
                decode_btn = gr.Button("Decode")
                decode_btn.click(fn=decode_video_watermark, inputs=[video_input, device_state, model_state], outputs=output_log)

            with gr.Tab("⚡ Live Encode"):
                live_input_video = gr.Video(label="Input Video")
                live_secret = gr.Textbox(label="Secret (Int)")
                with gr.Accordion("Tuning Settings", open=False):
                    with gr.Row():
                        prebuffer_seconds = gr.Slider(minimum=2, maximum=12, value=4, step=1, label="Prebuffer Seconds")
                        segment_seconds = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="HLS Segment Seconds")
                    with gr.Row():
                        live_gpu_batch_target = gr.Slider(minimum=1, maximum=64, value=60, step=1, label="GPU Batch Target")
                        live_gpu_batch_max = gr.Slider(minimum=1, maximum=64, value=64, step=1, label="GPU Batch Max")
                        live_gpu_flush_ms = gr.Slider(minimum=1, maximum=100, value=12, step=1, label="GPU Flush (ms)")
                        live_use_nvenc = gr.Checkbox(value=True, label="Use NVENC Hardware Encoder")
                    with gr.Row():
                        live_decode_mode = gr.Dropdown(
                            choices=["auto", "single_opencv", "ffmpeg_pipe"],
                            value="auto",
                            label="Decode Mode",
                        )
                        live_decode_workers = gr.Slider(minimum=1, maximum=8, value=2, step=1, label="Decode Workers")
                        live_inference_scale = gr.Dropdown(
                            choices=["1.0", "0.75", "0.5"],
                            value="0.5",
                            label="Inference Scale",
                        )
                    gr.Markdown(
                        "### Tuning pointers\n"
                        "- `Decode Mode`: start with `auto`; force `ffmpeg_pipe` for higher decode throughput.\n"
                        "- `Decode Workers`: increase only if decode seems behind; too high may add overhead.\n"
                        "- `GPU Batch Target/Max`: increase for throughput, decrease if latency/reorder grows.\n"
                        "- `GPU Flush (ms)`: lower for responsiveness, higher for larger effective batches.\n"
                        "- `Use NVENC`: keep enabled to reduce writer bottleneck.\n"
                        "- Read telemetry: if `gpu_starve` is high and `decode_q` stays low, decode is the bottleneck."
                    )
                with gr.Row():
                    live_start_btn = gr.Button("Start Live Encode Stream")
                    live_stop_btn = gr.Button("Stop Stream")
                live_job_id = gr.Textbox(label="Job ID", interactive=False)
                live_status = gr.Textbox(label="Status", interactive=False)
                live_progress = gr.Textbox(label="Progress", interactive=False)
                with gr.Row():
                    with gr.Column(scale=3):
                        live_player = gr.HTML(label="Live Stream")
                    with gr.Column(scale=2):
                        live_probe_log = gr.Textbox(
                            label="Watermark Probe Log",
                            lines=20,
                            max_lines=20,
                            interactive=False,
                        )
                live_rendered_playlist_url = gr.State(value="")

                live_start_btn.click(
                    fn=start_live_encode,
                    inputs=[
                        live_input_video,
                        live_secret,
                        device_state,
                        model_state,
                        prebuffer_seconds,
                        segment_seconds,
                        live_gpu_batch_target,
                        live_gpu_batch_max,
                        live_gpu_flush_ms,
                        live_use_nvenc,
                        live_decode_mode,
                        live_decode_workers,
                        live_inference_scale,
                    ],
                    outputs=[
                        live_job_id,
                        live_status,
                        live_progress,
                        live_player,
                        live_rendered_playlist_url,
                        live_probe_log,
                    ],
                )
                live_stop_btn.click(
                    fn=stop_live_encode,
                    inputs=[live_job_id],
                    outputs=[live_status, live_progress],
                )

                live_timer = gr.Timer(value=1.0, active=True)
                live_timer.tick(
                    fn=poll_live_encode,
                    inputs=[live_job_id, live_rendered_playlist_url],
                    outputs=[
                        live_status,
                        live_progress,
                        live_player,
                        live_rendered_playlist_url,
                        live_probe_log,
                    ],
                )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # 关键：必须在 main 中设置 spawn 
    mp.set_start_method("spawn", force=True)
    launch_app()