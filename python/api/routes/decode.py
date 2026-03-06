import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image

from python.api.lib.schemas import DecodeFrameResponse, DeviceEnum, ModelTypeEnum
from python.api.lib.trustmark_runtime import build_trustmark

router = APIRouter()
APP_SECRET_BITS = 56


@router.post("/decode_frame", response_model=DecodeFrameResponse)
async def decode_frame(
    file: UploadFile = File(...),
    device: DeviceEnum = Form(default=DeviceEnum.CPU),
    model_type: ModelTypeEnum = Form(default=ModelTypeEnum.P),
):
    logger.info(
        "POST /decode_frame received filename='{}' device='{}' model='{}'",
        file.filename or "<unnamed>",
        device.value,
        model_type.value,
    )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        logger.warning(
            "POST /decode_frame invalid image filename='{}' detail='{}'",
            file.filename or "<unnamed>",
            exc,
        )
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        tm = build_trustmark(model_type=model_type, device=device)
        wm_secret, wm_present, _ = tm.decode(image, "binary")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "POST /decode_frame failed filename='{}' device='{}' model='{}': {}",
            file.filename or "<unnamed>",
            device.value,
            model_type.value,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not wm_present:
        logger.warning(
            "POST /decode_frame no watermark detected filename='{}'",
            file.filename or "<unnamed>",
        )
        raise HTTPException(status_code=422, detail="No watermark detected in uploaded image.")

    decoded_bits = str(wm_secret)
    app_secret_bits = decoded_bits[:APP_SECRET_BITS]
    decoded_secret = int(app_secret_bits, 2)
    logger.info(
        "POST /decode_frame decoded filename='{}' wm_secret={}",
        file.filename or "<unnamed>",
        decoded_secret,
    )
    return DecodeFrameResponse(
        wm_secret=decoded_secret,
        device=device,
        model_type=model_type,
    )
