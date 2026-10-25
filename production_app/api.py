import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from production_app import model_service
from production_app.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_model()
    except Exception:
        logger.warning("Model not loaded at startup; will retry when requested.")
    yield


app = FastAPI(title="ML API", lifespan=lifespan)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"model": HealthResponse},
        503: {"model": HealthResponse},
    },
)
def health():
    ok = model_service.is_model_loaded()
    if not ok:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False},
        )
    return HealthResponse(status="healthy", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
    try:
        return model_service.predict(req.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")
