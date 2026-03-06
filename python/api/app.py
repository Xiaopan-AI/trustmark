import atexit
import os
import sys

from fastapi import FastAPI
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | <cyan>{thread.name}</cyan> | {message}",
)
logger.info("Initializing FastAPI watermark API module")

from python.api.lib.encode_runtime import registry
from python.api.routes import action_router, decode_router, encode_router

app = FastAPI(title="Watermarking Live Encode MVP")
app.include_router(encode_router)
app.include_router(decode_router)
app.include_router(action_router)
atexit.register(registry.cleanup_all)

logger.info("FastAPI app created title='{}'", app.title)
