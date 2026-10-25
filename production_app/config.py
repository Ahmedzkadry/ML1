import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/model.joblib")
    selected_features_path: str = os.getenv(
        "SELECTED_FEATURES_PATH", "models/selected_features.json"
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    n_features: int = int(os.getenv("N_FEATURES", "10"))


settings = Settings()
