import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from production_app import model_service
from production_app.schemas import *

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_model()
    except Exception:
        logger.exception("Model failed to load")
    yield


app = FastAPI(title="ML API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    ok = model_service.is_model_loaded()
    return HealthResponse(status="healthy" if ok else "unhealthy", model_loaded=ok)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not model_service.is_model_loaded():
        raise HTTPException(503, "Model is currently unavailable")
    try:
        return model_service.predict(req.model_dump())
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Internal prediction error")
