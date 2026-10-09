import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/model.joblib")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
