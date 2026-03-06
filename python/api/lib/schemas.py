from enum import Enum
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class DeviceEnum(str, Enum):
    CPU = "CPU"
    CUDA_0 = "CUDA_0"


class ModelTypeEnum(str, Enum):
    P = "P"
    Q = "Q"
    C = "C"
    B = "B"


class SessionStateEnum(str, Enum):
    PREPARED = "prepared"
    BUFFERING = "buffering"
    STREAMING = "streaming"
    DONE = "done"
    ERROR = "error"


class CreateStreamRequest(BaseModel):
    source_url: AnyHttpUrl = Field(..., description="HTTP/HTTPS MP4 URL")
    wm_secret: int = Field(..., ge=0, le=(1 << 56) - 1, description="Watermark secret integer")
    device: DeviceEnum = Field(default=DeviceEnum.CPU)
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.P)
    inference_scale: Literal[1.0, 0.75, 0.5] = Field(default=0.5)
    prebuffer_seconds: int = Field(default=4, ge=2, le=12)
    segment_seconds: int = Field(default=2, ge=1, le=4)
    gpu_batch_target: int = Field(default=60, ge=1, le=64)
    gpu_batch_max: int = Field(default=64, ge=1, le=64)
    gpu_flush_ms: int = Field(default=12, ge=1, le=100)
    use_nvenc: bool = Field(default=True)


class CreateStreamResponse(BaseModel):
    stream_id: str
    stream_url: str
    player_url: str
    status_url: str
    seek_url: str
    state: SessionStateEnum
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
    has_audio_track: bool
    encoder_backend: str


class SeekRequest(BaseModel):
    time_seconds: float = Field(..., ge=0.0)


class DecodeFrameResponse(BaseModel):
    wm_secret: int
    device: DeviceEnum
    model_type: ModelTypeEnum
