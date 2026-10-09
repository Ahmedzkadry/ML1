from typing import Dict, Any
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from production_app.logging_config import logger


def evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    logger.info("Evaluating model pipeline on test data.")
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    logger.info("Evaluation accuracy: %.4f", accuracy)
    return {"accuracy": accuracy, "classification_report": report}