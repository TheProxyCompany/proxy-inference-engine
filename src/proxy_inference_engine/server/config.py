from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MODEL_PATH: str = "mlx-community/gemma-3-12b-it-qat-4bit"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


_settings: Settings | None = None


def load_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
