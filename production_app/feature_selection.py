import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from production_app.config import settings
from production_app.logging_config import logger
from production_app.preprocessing import create_preprocessor


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = settings.n_features,
) -> list[str]:
    logger.info(
        X_train.shape[1],
        n_features,
    )
    preprocessor = create_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)

    estimator = LogisticRegression(
        random_state=settings.random_state,
        max_iter=1000,
    )
    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=1,
    )
    selector.fit(X_train_scaled, y_train)

    selected_features = list(X_train.columns[selector.support_])
    logger.info("Selected %d features: %s", len(selected_features), selected_features)
    return selected_features
