from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIModeEnum(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"


class LogLevelEnum(StrEnum):
    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class APIConfig(BaseSettings):
    MAX_FILE_SIZE_MB: int = 50
    ML_MODEL_API_KEY: str = Field("some-secret-api-key")
    BASE_URL: str = Field(..., alias="API_BASE_URL")
    MODE: APIModeEnum = Field(..., alias="API_MODE")
    LOG_LEVEL: LogLevelEnum = Field(..., alias="LOG_LEVEL")
    SESSION_SECRET_KEY: str = Field("your-session-secret-key", alias="SESSION_SECRET_KEY")
    LOCALHOST_CLIENT_ORIGIN: str = Field("", alias="LOCALHOST_CLIENT_ORIGIN")
    ALLOWED_ORIGINS_STR: str = Field("", alias="ALLOWED_ORIGINS")

    model_config = SettingsConfigDict(env_file="../env/ml_engine.env")

    @property
    def allowed_origins(self) -> list[str]:
        origins = {origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(",")}
        if self.MODE in (APIModeEnum.LOCAL, APIModeEnum.DEV, APIModeEnum.STAGE):
            origins.add(self.LOCALHOST_CLIENT_ORIGIN)
        return sorted(origins)
