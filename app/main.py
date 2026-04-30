from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.health import router as health_router
from app.api.stream import router as stream_router
from app.api.transcriptions import router as transcriptions_router
from app.core.recognizer import get_recognizer, reset_recognizer
from app.core.vad import get_vad, reset_vad

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "").lower() == "true" else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_is_internal_health_check: ContextVar[bool] = ContextVar("_is_internal_health_check", default=False)


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if _is_internal_health_check.get():
            return False
        return True


uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(HealthCheckFilter())


class InternalHealthCheckMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" and request.headers.get("X-Health-Check") == "internal":
            token = _is_internal_health_check.set(True)
            try:
                response = await call_next(request)
                return response
            finally:
                _is_internal_health_check.reset(token)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Loading Moonshine model...")
    try:
        get_recognizer()
        logger.info("Moonshine model loaded successfully")
    except Exception as e:
        logger.error("Failed to load Moonshine model: %s", e)
        raise

    try:
        vad = get_vad()
        if vad:
            logger.info("VAD model loaded successfully")
        else:
            logger.info("VAD model not found, WebSocket streaming disabled")
    except Exception as e:
        logger.warning("VAD model not loaded: %s", e)

    yield

    reset_recognizer()
    reset_vad()
    logger.info("Models unloaded")


app = FastAPI(
    title="Moonshine v2 STT Server",
    description="OpenAI-compatible Speech-to-Text API powered by Moonshine v2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(InternalHealthCheckMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(transcriptions_router)
app.include_router(stream_router)


@app.get("/")
async def root():
    return {
        "name": "Moonshine v2 STT Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": "/v1/audio/transcriptions",
        "websocket": "/v1/audio/stream",
    }
