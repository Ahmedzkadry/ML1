from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.feature_selection import select_features
from production_app.logging_config import logger
from production_app.preprocessing import create_preprocessor


def build_pipeline() -> Pipeline:
    preprocessor = create_preprocessor()
    classifier = LogisticRegression(random_state=settings.random_state, max_iter=1000)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: Optional[list[str]] = None,
    use_feature_selection: bool = True,
) -> Pipeline:
    logger.info("Starting pipeline training...")
    if selected_features is None and use_feature_selection:
        selected_features = select_features(X_train, y_train)

    train_df = X_train[selected_features] if selected_features is not None else X_train
    pipeline = build_pipeline()
    pipeline.fit(train_df, y_train)
    logger.info(
        "Pipeline training completed with %d features.",
        train_df.shape[1],
    )
    return pipeline
