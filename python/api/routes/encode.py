import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from python.api.lib.encode_runtime import (
    _is_session_ready_for_playback_unlocked,
    absolute_url,
    build_player_page,
    get_window_bounds,
    hls_segment_count,
    hls_segment_filename,
    playlist_exists,
    registry,
    rewrite_playlist_for_fastapi,
)
from python.api.lib.schemas import (
    CreateStreamRequest,
    CreateStreamResponse,
    HeartbeatRequest,
    PlaybackControlRequest,
    PlaybackControlResponse,
    SeekRequest,
    SessionStateEnum,
)

encode_router = APIRouter()
router = encode_router


def _session_or_404(stream_id: str):
    try:
        return registry.get(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown stream_id") from exc


def _reject_if_invalidated(session) -> None:
    with session.lock:
        if session.state == SessionStateEnum.INVALIDATED:
            raise HTTPException(status_code=409, detail="The current session has been invalidated.")


def _control_response(request: Request, session) -> PlaybackControlResponse:
    with session.lock:
        return PlaybackControlResponse(
            stream_id=session.stream_id,
            state=session.state,
            generation=session.active_generation,
            logical_position_seconds=session.logical_position_seconds,
            anchor_time_seconds=session.anchor_time_seconds,
            ready_for_playback=_is_session_ready_for_playback_unlocked(session),
            playlist_url=absolute_url(request, "get_playlist", stream_id=session.stream_id),
            status_url=absolute_url(request, "stream_status", stream_id=session.stream_id),
        )


@encode_router.post("/encode", response_model=CreateStreamResponse)
def create_stream(request: Request, req: CreateStreamRequest):
    try:
        session = registry.create_session(
            source_url=str(req.source_url),
            wm_secret=req.wm_secret,
            device=req.device.value.lower().replace("_0", ":0"),
            model_type=req.model_type,
            inference_scale=float(req.inference_scale),
            prebuffer_seconds=req.prebuffer_seconds,
            segment_seconds=req.segment_seconds,
            gpu_batch_target=req.gpu_batch_target,
            gpu_batch_max=req.gpu_batch_max,
            gpu_flush_ms=req.gpu_flush_ms,
            use_nvenc=req.use_nvenc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateStreamResponse(
        stream_id=session.stream_id,
        stream_url=absolute_url(request, "get_playlist", stream_id=session.stream_id),
        player_url=absolute_url(request, "get_player", stream_id=session.stream_id),
        status_url=absolute_url(request, "stream_status", stream_id=session.stream_id),
        play_url=absolute_url(request, "play_stream", stream_id=session.stream_id),
        pause_url=absolute_url(request, "pause_stream", stream_id=session.stream_id),
        resume_url=absolute_url(request, "resume_stream", stream_id=session.stream_id),
        seek_url=absolute_url(request, "seek_stream", stream_id=session.stream_id),
        heartbeat_url=absolute_url(request, "heartbeat_stream", stream_id=session.stream_id),
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


@encode_router.post("/encode/{stream_id}/play", response_model=PlaybackControlResponse, name="play_stream")
def play_stream(stream_id: str, request: Request, req: PlaybackControlRequest):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    registry.play(session, req.position_seconds)
    return _control_response(request, session)


@encode_router.post("/encode/{stream_id}/pause", response_model=PlaybackControlResponse, name="pause_stream")
def pause_stream(stream_id: str, request: Request, req: PlaybackControlRequest):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    generation = registry.pause(session, req.position_seconds)
    with session.lock:
        if session.active_generation == 0:
            session.active_generation = generation
    return _control_response(request, session)


@encode_router.post("/encode/{stream_id}/resume", response_model=PlaybackControlResponse, name="resume_stream")
def resume_stream(stream_id: str, request: Request):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    registry.play(session, None)
    return _control_response(request, session)


@encode_router.post("/encode/{stream_id}/seek", response_model=PlaybackControlResponse, name="seek_stream")
def seek_stream(stream_id: str, request: Request, req: SeekRequest):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    registry.play(session, req.time_seconds)
    return _control_response(request, session)


@encode_router.post("/encode/{stream_id}/heartbeat", response_model=PlaybackControlResponse, name="heartbeat_stream")
def heartbeat_stream(stream_id: str, request: Request, req: HeartbeatRequest):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    registry.heartbeat(session, req.position_seconds)
    return _control_response(request, session)


@encode_router.get("/encode/{stream_id}/playlist.m3u8", name="get_playlist")
def get_playlist(
    stream_id: str,
    generation: Optional[int] = Query(default=None, ge=0),
):
    session = _session_or_404(stream_id)
    timeout_seconds = 30.0
    started_at = time.time()
    while True:
        with session.lock:
            if generation is not None and session.active_generation and generation != session.active_generation:
                raise HTTPException(status_code=409, detail="Requested generation is no longer active.")
            if session.state == SessionStateEnum.INVALIDATED:
                raise HTTPException(status_code=409, detail="The current session has been invalidated.")
            if session.state in (SessionStateEnum.PREPARED, SessionStateEnum.PAUSED):
                raise HTTPException(status_code=409, detail="Stream is not currently playing.")
            if session.state == SessionStateEnum.ERROR:
                raise HTTPException(status_code=500, detail=session.error or "Stream failed.")
            if session.state == SessionStateEnum.DONE and playlist_exists(session):
                break
            if _is_session_ready_for_playback_unlocked(session):
                break
        if time.time() - started_at >= timeout_seconds:
            raise HTTPException(status_code=503, detail="Stream is still warming source connection or buffering.")
        time.sleep(0.25)

    with session.lock:
        playlist_path = session.playlist_path
    if not playlist_path or not os.path.exists(playlist_path):
        raise HTTPException(status_code=404, detail="Playlist not found.")
    with open(playlist_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return PlainTextResponse(
        rewrite_playlist_for_fastapi(session, content),
        media_type="application/vnd.apple.mpegurl",
    )


@encode_router.get("/encode/{stream_id}/segments/{seq}.ts", name="get_segment")
def get_segment(stream_id: str, seq: int):
    session = _session_or_404(stream_id)
    _reject_if_invalidated(session)
    segment_path = hls_segment_filename(session, seq)
    if not os.path.exists(segment_path):
        raise HTTPException(status_code=404, detail="Segment not found.")
    return FileResponse(segment_path, media_type="video/mp2t")


@encode_router.get("/encode/{stream_id}/player", response_class=HTMLResponse, name="get_player")
def get_player(stream_id: str, request: Request):
    _session_or_404(stream_id)
    return HTMLResponse(
        build_player_page(
            playlist_url=absolute_url(request, "get_playlist", stream_id=stream_id),
            status_url=absolute_url(request, "stream_status", stream_id=stream_id),
            play_url=absolute_url(request, "play_stream", stream_id=stream_id),
            pause_url=absolute_url(request, "pause_stream", stream_id=stream_id),
            heartbeat_url=absolute_url(request, "heartbeat_stream", stream_id=stream_id),
        )
    )


@encode_router.get("/encode/{stream_id}/status", name="stream_status")
def stream_status(stream_id: str):
    session = _session_or_404(stream_id)
    with session.lock:
        ready = _is_session_ready_for_playback_unlocked(session)
        window_start, window_end = get_window_bounds(session)
        return {
            "stream_id": session.stream_id,
            "state": session.state,
            "error": session.error,
            "duration_seconds": session.duration_seconds,
            "fps": session.fps,
            "width": session.width,
            "height": session.height,
            "frames_processed": session.frames_processed,
            "current_pts": session.current_pts,
            "saved_position_seconds": session.saved_position_seconds,
            "logical_position_seconds": session.logical_position_seconds,
            "anchor_time_seconds": session.anchor_time_seconds,
            "last_client_position_seconds": session.last_client_position_seconds,
            "last_heartbeat_at": session.last_heartbeat_at,
            "invalidated_at": session.invalidated_at,
            "generation": session.active_generation,
            "ready_for_playback": ready,
            "playlist_exists": playlist_exists(session),
            "hls_segment_count": hls_segment_count(session),
            "window_start_seconds": window_start,
            "window_end_seconds": window_end,
            "has_audio_track": session.has_audio_track,
            "encoder_backend": session.encoder_backend,
            "inference_scale": session.inference_scale,
            "prebuffer_seconds": session.prebuffer_seconds,
            "segment_seconds": session.segment_seconds,
            "gpu_batch_target": session.gpu_batch_target,
            "gpu_batch_max": session.gpu_batch_max,
            "gpu_flush_ms": session.gpu_flush_ms,
            "use_nvenc": session.use_nvenc,
            "spool_complete": session.spool_complete,
            "spool_error": session.spool_error,
            "spool_last_error": session.spool_last_error,
            "spool_attempt_count": session.spool_attempt_count,
            "spool_connected_at": session.spool_connected_at,
            "spool_started_at": session.spool_started_at,
            "spool_retry_deadline": session.spool_retry_deadline,
            "spool_retry_seconds_remaining": max(0.0, session.spool_retry_deadline - time.time()) if session.spool_retry_deadline > 0 else 0.0,
            "spool_bytes_ever_arrived": session.spool_bytes_ever_arrived,
            "spool_media_ready": session.spool_media_ready,
            "spool_media_ready_at": session.spool_media_ready_at,
            "spool_media_last_error": session.spool_media_last_error,
            "bytes_downloaded": session.bytes_downloaded,
            "total_bytes": session.total_bytes,
            "end_of_stream_reached": session.end_of_stream_reached,
        }
