from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ISRO PS12 Dashboard"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # ML pipeline paths
    checkpoint_path: str = str(ROOT / "models/checkpoints/best_model.pth")
    data_root: str       = str(ROOT / "data/goes19/raw")
    output_dir: str      = str(ROOT / "outputs")

    # Dataset
    bt_min: float = 180.0
    bt_max: float = 320.0
    target_size: tuple = (512, 512)
    stride: int = 10

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"dev", "development"}:
                return True
        return value

settings = Settings()
