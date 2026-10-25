import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

from production_app.config import settings

DEFAULT_SELECTED_FEATURES: list[str] = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_concavity",
    "mean_concave_points",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_concave_points",
]


def load_selected_feature_names() -> list[str]:
    path = Path(settings.selected_features_path)
    if path.exists():
        try:
            features = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(features, list) and features:
                return [str(f) for f in features]
        except Exception:
            pass
    return DEFAULT_SELECTED_FEATURES


SELECTED_FEATURES: list[str] = load_selected_feature_names()

_field_definitions = {
    feature_name: (
        float,
        Field(
            ...,
            description=feature_name.replace("_", " ").capitalize(),
            ge=0.0,
        ),
    )
    for feature_name in SELECTED_FEATURES
}

PredictionRequest = create_model(
    "PredictionRequest",
    __config__=ConfigDict(extra="forbid"),
    **_field_definitions,
)


class PredictionResponse(BaseModel):
    predicted_class: int
    readable_label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
