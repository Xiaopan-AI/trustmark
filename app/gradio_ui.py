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
from trustmark import TrustMark

# 环境清理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

logging.getLogger("PIL").setLevel(logging.WARNING)


MODEL_TYPE = "Q"
# ============================
# 单进程编码
# ============================

def encode_video_sp(video_path, wm_secret):
    if not video_path:
        return None

    if not video_path:
        return None    
    tm = TrustMark(verbose=False, model_type=MODEL_TYPE)
    wm_secret = int(wm_secret)
    print(f"Encoding secret: {wm_secret}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "encoded.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        bits = [(wm_secret >> i) & 1 for i in reversed(range(56))]
        # convert to string
        bits = "".join(map(str, bits))
        print(f"Encoding bits: {bits}")
        wm_img = tm.encode(pil_img, bits, "binary")
        wm_frame = cv2.cvtColor(np.array(wm_img), cv2.COLOR_RGB2BGR)
        out.write(wm_frame)

    cap.release()
    out.release()

    return output_path

# ============================
# 多进程编码
# ============================
def watermark_worker(in_q, out_q, bits_str):
    """
    独立进程：负责加载模型并进行编码
    """
    try:
        from trustmark import TrustMark
        # 在进程内初始化，避免 CUDA 句柄跨进程共享冲突
        tm_worker = TrustMark(verbose=False, model_type=MODEL_TYPE)
        
        while True:
            try:
                item = in_q.get(timeout=5)
                if item is None:  # 结束信号
                    break
                
                frame_idx, frame_pil = item
                # 编码
                encoded_pil = tm_worker.encode(frame_pil, bits_str, "binary")
                # 转换回 BGR 格式给 OpenCV
                encoded_bgr = cv2.cvtColor(np.array(encoded_pil), cv2.COLOR_RGB2BGR)
                
                # 放入输出队列
                while True:
                    try:
                        out_q.put((frame_idx, encoded_bgr), timeout=1)
                        break
                    except Full:
                        continue
            except Empty:
                continue
    except Exception as e:
        print(f"Worker process error: {e}")

# ============================
# Encode Logic (Multi-process)
# ============================
def encode_video(video_path, wm_secret_val):
    if not video_path:
        return None

    # 1. 准备秘密信息
    try:
        wm_secret_int = int(wm_secret_val)
        bits = [(wm_secret_int >> i) & 1 for i in reversed(range(56))]
        bits_str = "".join(map(str, bits))
    except ValueError:
        return None

    # 2. 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "encoded.mp4")
    # 使用 mp4v 编码，如果追求体积小可改用 'avc1'
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 3. 设置多进程
    num_workers = 4  # 建议根据显存调整，若 8G 显存建议 1-2
    in_q = mp.Queue(maxsize=30)
    out_q = mp.Queue(maxsize=30)
    workers = []

    for _ in range(num_workers):
        p = mp.Process(target=watermark_worker, args=(in_q, out_q, bits_str))
        p.start()
        workers.append(p)

    # 4. 主循环：读取、分发、回收、写入
    read_idx = 0
    write_idx = 0
    buffer = {}
    all_read = False

    print("Encoding started...")
    
    try:
        while write_idx < read_idx or not all_read:
            # 读取并放入输入队列
            if not all_read and in_q.qsize() < 25:
                ret, frame_bgr = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    try:
                        in_q.put_nowait((read_idx, frame_pil))
                        read_idx += 1
                    except Full:
                        pass
                else:
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
        cap.release()
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
def decode_video_watermark(video_path):
    from trustmark import TrustMark
    tm_decoder = TrustMark(verbose=False, model_type=MODEL_TYPE)
    
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

        with gr.Tabs():
            
            with gr.Tab("🔐 Encode_SP"):
                input_video = gr.Video(label="Input Video")
                wm_secret = gr.Textbox(label="Secret (Int)")
                bit_display = gr.Textbox(label="56-bit Binary", interactive=False)
                
                wm_secret.change(fn=secret_to_bits, inputs=wm_secret, outputs=bit_display)
                
                output_video = gr.Video(label="Output")
                encode_btn = gr.Button("Start Single-process Encoding")
                encode_btn.click(fn=encode_video_sp, inputs=[input_video, wm_secret], outputs=output_video)            
            
            
            
            with gr.Tab("🔐 Encode_MP"):
                input_video = gr.Video(label="Input Video")
                wm_secret = gr.Textbox(label="Secret (Int)")
                bit_display = gr.Textbox(label="56-bit Binary", interactive=False)
                
                wm_secret.change(fn=secret_to_bits, inputs=wm_secret, outputs=bit_display)
                
                output_video = gr.Video(label="Output")
                encode_btn = gr.Button("Start Multi-process Encoding")
                encode_btn.click(fn=encode_video, inputs=[input_video, wm_secret], outputs=output_video)
  
            with gr.Tab("🔓 Decode"):
                video_input = gr.Video()
                output_log = gr.Textbox(label="Log", lines=15)
                decode_btn = gr.Button("Decode")
                decode_btn.click(fn=decode_video_watermark, inputs=video_input, outputs=output_log)

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # 关键：必须在 main 中设置 spawn 
    mp.set_start_method("spawn", force=True)
    launch_app()