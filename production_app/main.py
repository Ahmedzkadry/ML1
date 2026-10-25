from typing import Any

import pandas as pd

from production_app.data import get_train_test_data
from production_app.evaluation import compare_models, evaluate_pipeline
from production_app.model_service import predict, save_artifact
from production_app.training import train_pipeline


def run_training_pipeline() -> dict[str, Any]:
    X_train, X_test, y_train, y_test = get_train_test_data()

    baseline_pipeline = train_pipeline(X_train, y_train, use_feature_selection=False)
    baseline_metrics = evaluate_pipeline(baseline_pipeline, X_test, y_test)

    selected_pipeline = train_pipeline(X_train, y_train, use_feature_selection=True)
    selected_metrics = evaluate_pipeline(selected_pipeline, X_test, y_test)

    comparison = compare_models(baseline_metrics, selected_metrics)
    save_artifact(selected_pipeline)

    selected_features = list(selected_pipeline.feature_names_in_)
    selected_metrics["selected_features"] = selected_features
    selected_metrics["comparison"] = comparison
    return selected_metrics


def run_prediction_pipeline(sample_data: pd.DataFrame) -> Any:
    return predict(sample_data)
