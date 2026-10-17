import joblib
import pandas as pd
from production_app.config import settings

_model = None


def load_model():
    global _model
    _model = joblib.load(settings.model_path)


def is_model_loaded() -> bool:
    return _model is not None


def predict(features: dict) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded")
    X = pd.DataFrame([features])[FEATURE_ORDER]
    pred = int(_model.predict(X)[0])
    return {"predicted_class": pred, "label": CLASS_LABELS[pred]}
