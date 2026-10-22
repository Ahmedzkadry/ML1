from pathlib import Path
from typing import Any, Union
import joblib
import pandas as pd
from production_app.config import settings

_model = None


CLASS_LABELS = {0: "Malignant", 1: "Benign"}


def save_artifact(model: Any) -> None:
    model_path = Path(settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model() -> None:
    global _model
    _model = joblib.load(settings.model_path)


def is_model_loaded() -> bool:
    return _model is not None


def predict(features: Union[dict, pd.DataFrame]) -> dict:
    global _model

    if _model is None:
        load_model()

    if isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        df = pd.DataFrame([features])

    if hasattr(_model, "feature_names_in_"):
        feature_order = list(_model.feature_names_in_)
        df = df[feature_order]

    pred = int(_model.predict(df)[0])

    return {
        "predicted_class": pred,
        "readable_label": CLASS_LABELS[pred],
    }
