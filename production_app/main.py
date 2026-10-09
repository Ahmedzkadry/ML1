from typing import Dict, Any
import pandas as pd
from production_app.data import get_train_test_data
from production_app.training import train_pipeline
from production_app.evaluation import evaluate_pipeline
from production_app.model_service import save_artifact, predict


def run_training_pipeline() -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = get_train_test_data()
    pipeline = train_pipeline(X_train, y_train)
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    save_artifact(pipeline)
    return metrics


def run_prediction_pipeline(sample_data: pd.DataFrame) -> Any:
    return predict(sample_data)