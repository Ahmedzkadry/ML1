import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from production_app.preprocessing import create_preprocessor
from production_app.config import settings
from production_app.logging_config import logger


def build_pipeline() -> Pipeline:
    preprocessor = create_preprocessor()
    classifier = LogisticRegression(
        random_state=settings.random_state, max_iter=1000
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    logger.info("Starting pipeline training...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training completed.")
    return pipeline