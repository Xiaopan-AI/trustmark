import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from loguru import logger

from python.api.lib.encode_runtime import (
    _is_session_ready_for_playback_unlocked,
    absolute_url,
    build_player_page,
    estimate_available_seconds_unlocked,
    get_window_bounds,
    hls_segment_count,
    hls_segment_filename,
    list_hls_segment_seqs,
    playlist_exists,
    registry,
    rewrite_playlist_for_fastapi,
)
from python.api.lib.schemas import CreateStreamRequest, CreateStreamResponse, SessionStateEnum
from python.api.lib.trustmark_runtime import runtime_device

router = APIRouter()


@router.post("/streams", response_model=CreateStreamResponse)
def create_stream(req: CreateStreamRequest, request: Request):
    logger.info(
        "POST /streams received source_url='{}' device='{}' model='{}' inf_scale={} prebuffer={}s segment={}s batch_target={} batch_max={} flush_ms={} use_nvenc={}",
        req.source_url,
        req.device.value,
        req.model_type.value,
        req.inference_scale,
        req.prebuffer_seconds,
        req.segment_seconds,
        req.gpu_batch_target,
        req.gpu_batch_max,
        req.gpu_flush_ms,
        req.use_nvenc,
    )
    try:
        session = registry.create_session(
            source_url=str(req.source_url),
            wm_secret=req.wm_secret,
            device=runtime_device(req.device),
            model_type=req.model_type,
            inference_scale=req.inference_scale,
            prebuffer_seconds=req.prebuffer_seconds,
            segment_seconds=req.segment_seconds,
            gpu_batch_target=req.gpu_batch_target,
            gpu_batch_max=req.gpu_batch_max,
            gpu_flush_ms=req.gpu_flush_ms,
            use_nvenc=req.use_nvenc,
        )
    except ValueError as exc:
        logger.error("POST /streams validation failed: {}", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /streams failed unexpectedly: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stream_url = absolute_url(request, "get_playlist", stream_id=session.stream_id)
    player_url = absolute_url(request, "get_player", stream_id=session.stream_id)
    status_url = absolute_url(request, "stream_status", stream_id=session.stream_id)
    seek_url = f"{request.base_url}streams/{session.stream_id}/seek"
    return CreateStreamResponse(
        stream_id=session.stream_id,
        stream_url=stream_url,
        player_url=player_url,
        status_url=status_url,
        seek_url=seek_url,
        state=session.state,
        duration_seconds=session.duration_seconds,
        fps=session.fps,
        width=session.width,
        height=session.height,
        inference_scale=session.inference_scale,
        prebuffer_seconds=session.prebuffer_seconds,
        segment_seconds=session.segment_seconds,
        gpu_batch_target=session.gpu_batch_target,
        gpu_batch_max=session.gpu_batch_max,
        gpu_flush_ms=session.gpu_flush_ms,
        use_nvenc=session.use_nvenc,
        has_audio_track=session.has_audio_track,
        encoder_backend=session.encoder_backend,
    )


@router.get("/streams/{stream_id}/playlist.m3u8")
def get_playlist(stream_id: str):
    try:
        session = registry.get(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream_id not found.") from exc

    logger.info("[{}] Playlist requested", stream_id)
    registry.ensure_started(session)

    with session.lock:
        timeout_s = 30.0
        while (
            session.state not in (SessionStateEnum.ERROR, SessionStateEnum.DONE)
            and not _is_session_ready_for_playback_unlocked(session)
            and timeout_s > 0
        ):
            logger.debug(
                "[{}] Playlist waiting state='{}' hls_segments={} timeout_remaining={:.1f}s",
                stream_id,
                session.state.value,
                hls_segment_count(session),
                timeout_s,
            )
            tick = min(1.0, timeout_s)
            session.cond.wait(timeout=tick)
            timeout_s -= tick
        if session.state == SessionStateEnum.ERROR:
            logger.error("[{}] Playlist request failed due to session error '{}'", stream_id, session.error)
            raise HTTPException(status_code=500, detail=session.error or "stream worker failed.")
        if session.state != SessionStateEnum.DONE and not _is_session_ready_for_playback_unlocked(session):
            logger.warning(
                "[{}] Playlist startup wait timed out state='{}' hls_segments={}",
                stream_id,
                session.state.value,
                hls_segment_count(session),
            )
            raise HTTPException(status_code=503, detail="Stream is still buffering startup segments.")
    if not playlist_exists(session):
        raise HTTPException(status_code=503, detail="Playlist file is not ready yet.")

    with open(session.playlist_path, "r", encoding="utf-8") as playlist_file:
        text = rewrite_playlist_for_fastapi(session, playlist_file.read())
    logger.info("[{}] Playlist response served segments={}", stream_id, hls_segment_count(session))
    return PlainTextResponse(content=text, media_type="application/vnd.apple.mpegurl")


@router.get("/streams/{stream_id}/player")
def get_player(stream_id: str, request: Request):
    try:
        session = registry.get(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream_id not found.") from exc

    logger.info("[{}] Player page requested", stream_id)
    registry.ensure_started(session)
    html = build_player_page(
        playlist_url=absolute_url(request, "get_playlist", stream_id=stream_id),
        status_url=absolute_url(request, "stream_status", stream_id=stream_id),
    )
    return HTMLResponse(content=html)


@router.get("/streams/{stream_id}/segments/{seq}.ts")
def get_segment(stream_id: str, seq: int):
    try:
        session = registry.get(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream_id not found.") from exc

    seg_path = hls_segment_filename(session, seq)
    if not os.path.exists(seg_path):
        logger.warning("[{}] Segment miss seq={}", stream_id, seq)
        raise HTTPException(status_code=404, detail="segment not in active window.")
    logger.info("[{}] Segment served seq={} path='{}'", stream_id, seq, seg_path)
    return FileResponse(seg_path, media_type="video/mp2t", filename=os.path.basename(seg_path))


@router.get("/streams/{stream_id}/status")
def stream_status(stream_id: str):
    try:
        session = registry.get(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream_id not found.") from exc

    with session.lock:
        win_start, win_end = get_window_bounds(session)
        seqs = list_hls_segment_seqs(session)
        return {
            "stream_id": stream_id,
            "state": session.state,
            "error": session.error,
            "started": session.started,
            "source_url": session.source_url,
            "duration_seconds": session.duration_seconds,
            "fps": session.fps,
            "width": session.width,
            "height": session.height,
            "frames_processed": session.frames_processed,
            "current_anchor_seconds": session.current_pts,
            "window_start_seconds": win_start,
            "window_end_seconds": win_end,
            "active_segment_count": hls_segment_count(session),
            "next_seq": (seqs[-1] + 1) if seqs else 0,
            "inference_scale": session.inference_scale,
            "prebuffer_seconds": session.prebuffer_seconds,
            "segment_seconds": session.segment_seconds,
            "gpu_batch_target": session.gpu_batch_target,
            "gpu_batch_max": session.gpu_batch_max,
            "gpu_flush_ms": session.gpu_flush_ms,
            "use_nvenc": session.use_nvenc,
            "has_audio_track": session.has_audio_track,
            "encoder_backend": session.encoder_backend,
            "hls_writer_started": session.hls_writer_started,
            "playlist_path": session.playlist_path,
            "playlist_exists": playlist_exists(session),
            "hls_segment_count": hls_segment_count(session),
            "hls_writer_stderr_tail": session.hls_writer_stderr_tail,
            "spool_bytes_downloaded": session.bytes_downloaded,
            "spool_total_bytes": session.total_bytes,
            "spool_complete": session.spool_complete,
            "spool_error": session.spool_error,
            "spool_available_seconds": estimate_available_seconds_unlocked(session),
            "ready_for_playback": _is_session_ready_for_playback_unlocked(session),
        }
