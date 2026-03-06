# Live Encode API (MVP)

Entrypoint: `python/api/app.py`

Run:

```bash
uvicorn python.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /streams`
- `POST /decode_frame`
- `GET /streams/{stream_id}/player`
- `GET /streams/{stream_id}/playlist.m3u8`
- `GET /streams/{stream_id}/segments/{seq}.ts`
- `POST /streams/{stream_id}/seek` (temporarily disabled in this debug pass)
- `GET /streams/{stream_id}/status`

Notes:

- `python/api/app.py` is now a thin FastAPI entrypoint that wires the encode and decode route modules.
- MVP supports `http(s)` MP4 source URLs only.
- `ffmpeg` and `ffprobe` must both be available in `PATH`.
- Backend debug instrumentation uses `loguru`.
- Source ingest uses eager remote metadata probing first, with a local temp-file probe fallback when needed.
- Source video is progressively spooled to a temp MP4 file before decode.
- HLS output is ffmpeg-managed in a session-local temp directory.
- The live HLS playlist and TS segments are served from session-local temp files, not rebuilt in Python memory.
- Startup readiness follows the live encode defaults: `prebuffer_seconds=4` and `segment_seconds=2` unless overridden per stream.
- The current debug posture uses a wider rolling HLS window and more conservative player buffering than the earlier low-latency setup.
- No TTL or GC cleanup in this version.

`POST /streams` settings:

- `device`: `CPU` or `CUDA_0`
- `model_type`: `P`, `Q`, `C`, or `B`
- `inference_scale`: `1.0`, `0.75`, or `0.5` (default `0.5`)
- `prebuffer_seconds`: default `4`
- `segment_seconds`: default `2`
- `gpu_batch_target`: default `60`
- `gpu_batch_max`: default `64`
- `gpu_flush_ms`: default `12`
- `use_nvenc`: default `true`

`POST /decode_frame` settings:

- `file`: required multipart local image upload
- `device`: `CPU` or `CUDA_0` (default `CPU`)
- `model_type`: `P`, `Q`, `C`, or `B` (default `P`)
- Success response returns decoded `wm_secret` as an integer
- Invalid or empty image uploads return `400`
- Valid images with no detectable watermark return `422`

`inference_scale` behavior:

- Frames are downscaled before watermark inference when `inference_scale < 1.0`.
- Watermark inference runs on the scaled frames.
- Encoded frames are upscaled back to the original source resolution before they are fed into the continuous ffmpeg HLS writer.
- This mirrors the proven Gradio live encode behavior; it is not watermark-strength control.

Recommended flow:

1. `POST /streams`
2. Open returned `player_url` in a browser
3. The player page shows a loading placeholder until startup buffering is ready, then begins playback from the beginning

Debugging:

- Watch the API server console logs while using `player_url`.
- The backend logs eager metadata probing, stream creation, worker startup, spool progress, ffmpeg HLS writer startup, batch watermark progress, timeline/segment continuity progress, playlist serving, and segment serving.
- Seek remains intentionally disabled during this debug phase.
- The player page performs startup-only status polling while the loading placeholder is shown, then stops polling after playback is attached.
- Output HLS segments retain original source audio when the source has an audio track; silent sources still stream as video-only.
- Temp spool files and temp HLS output directories are session-local and are cleaned up when the server process exits.
