from fastapi import APIRouter

from python.api.lib.encode_runtime import registry
from python.api.lib.schemas import InvalidateSessionRequest, InvalidateSessionResponse

action_router = APIRouter()
router = action_router


@action_router.post("/action/invalidate", response_model=InvalidateSessionResponse)
def invalidate_sessions(req: InvalidateSessionRequest):
    invalidated_sessions = registry.invalidate_by_secret(req.wm_secret)
    return InvalidateSessionResponse(
        wm_secret=req.wm_secret,
        invalidated_count=len(invalidated_sessions),
        invalidated_stream_ids=[session.stream_id for session in invalidated_sessions],
    )
