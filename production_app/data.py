import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from production_app.config import settings
from production_app.logging_config import logger


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df


def load_raw_data() -> pd.DataFrame:
    logger.info("Loading Breast Cancer Wisconsin dataset.")
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    return normalize_column_names(df)


def get_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_raw_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    logger.info(
        "Splitting data with test_size=%s, random_state=%s",
        settings.test_size,
        settings.random_state,
    )
    return train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )
