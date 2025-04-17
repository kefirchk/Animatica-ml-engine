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
    ML_ENGINE_KEY: str = Field(..., alias="ML_ENGINE_KEY")
    MODE: APIModeEnum = Field(..., alias="API_MODE")
    LOG_LEVEL: LogLevelEnum = Field(..., alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file="../env/ml_engine.env")
