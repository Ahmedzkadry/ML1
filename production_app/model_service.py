import os
from typing import Any
import joblib
import pandas as pd
from production_app.config import settings
from production_app.logging_config import logger


def save_artifact(model: Any, filepath: str = settings.model_path) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logger.info("Saving model artifact to %s", filepath)
    joblib.dump(model, filepath)


def load_artifact(filepath: str = settings.model_path) -> Any:
    if not os.path.exists(filepath):
        logger.error("Artifact not found at %s", filepath)
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    logger.info("Loading model artifact from %s", filepath)
    return joblib.load(filepath)


def predict(input_data: pd.DataFrame, filepath: str = settings.model_path) -> Any:
    model = load_artifact(filepath)
    logger.info("Running inference on %d record(s).", len(input_data))
    return model.predict(input_data)