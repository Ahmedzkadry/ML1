import json
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

    if hasattr(model, "feature_names_in_"):
        features_path = Path(settings.selected_features_path)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features_path.write_text(
            json.dumps(list(model.feature_names_in_), indent=2),
            encoding="utf-8",
        )


def load_model() -> None:
    global _model
    model_path = Path(settings.model_path)
    if not model_path.exists():
        _model = None
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    _model = joblib.load(model_path)


def is_model_loaded() -> bool:
    model_path = Path(settings.model_path)
    if not model_path.exists():
        return False
    if _model is None:
        try:
            load_model()
        except Exception:
            return False
    return _model is not None


def get_selected_features() -> list[str]:
    global _model
    if _model is None and Path(settings.model_path).exists():
        load_model()
    if _model is not None and hasattr(_model, "feature_names_in_"):
        return list(_model.feature_names_in_)
    features_path = Path(settings.selected_features_path)
    if features_path.exists():
        return json.loads(features_path.read_text(encoding="utf-8"))
    return []


def predict(features: Union[dict, pd.DataFrame]) -> dict[str, Any]:
    global _model

    if not Path(settings.model_path).exists():
        _model = None
        raise FileNotFoundError(f"Model file not found at: {settings.model_path}")

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
