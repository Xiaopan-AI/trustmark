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
        ("Q - Balanced (PSNR ~43, ResNet50) [Default]", "Q"),
        ("P - High Quality (PSNR ~48)", "P"),
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
                default_model_index = model_values.index("Q")
                
                model_selector = gr.Dropdown(
                    choices=model_labels,
                    value=model_labels[default_model_index],
                    label="Model Type",
                    info="Select TrustMark model variant"
                )
                
                # Hidden state to store actual model code (Q, P, C, B)
                model_state = gr.State(value="Q")
        
        # Device info display
        device_info = gr.Textbox(
            label="Current Configuration",
            value=f"Device: {device_labels[default_device_index]} | Model: Q - Balanced (PSNR ~43, ResNet50) [Default]",
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

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # 关键：必须在 main 中设置 spawn 
    mp.set_start_method("spawn", force=True)
    launch_app()