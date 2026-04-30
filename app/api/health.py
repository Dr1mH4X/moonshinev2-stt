from __future__ import annotations

import os
import time

from fastapi import APIRouter

from app.schemas.responses import HealthResponse, ModelInfo, ModelListResponse

router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    model_name = os.environ.get("MODEL_NAME", "moonshine-v2")
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_name,
                created=int(_start_time),
            )
        ]
    )
