from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from production_app.logging_config import logger


def evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    logger.info("Evaluating model pipeline on test data.")
    if hasattr(pipeline, "feature_names_in_"):
        eval_df = X_test[list(pipeline.feature_names_in_)]
    else:
        eval_df = X_test

    predictions = pipeline.predict(eval_df)
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="weighted"))
    report = classification_report(y_test, predictions, output_dict=True)

    logger.info(
        "Evaluation metrics (%d features) -> Accuracy: %.4f | Weighted F1: %.4f",
        eval_df.shape[1],
        accuracy,
        f1,
    )
    return {
        "n_features": int(eval_df.shape[1]),
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report,
    }


def compare_models(
    baseline_metrics: dict[str, Any],
    selected_metrics: dict[str, Any],
) -> dict[str, Any]:
    comparison = {
        "baseline_n_features": baseline_metrics["n_features"],
        "selected_n_features": selected_metrics["n_features"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "selected_accuracy": selected_metrics["accuracy"],
        "baseline_f1_weighted": baseline_metrics["f1_weighted"],
        "selected_f1_weighted": selected_metrics["f1_weighted"],
    }
    logger.info(
        "Baseline (%d features) Accuracy: %.4f vs Selected (%d features) Accuracy: %.4f",
        comparison["baseline_n_features"],
        comparison["baseline_accuracy"],
        comparison["selected_n_features"],
        comparison["selected_accuracy"],
    )
    return comparison
