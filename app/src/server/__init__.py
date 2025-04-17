__version__ = "0.0.0"

from fastapi import Security
from fastapi.security import APIKeyHeader
from src.server.config import APIConfig
from src.server.exceptions import MLModelException

api_config = APIConfig()


async def validate_api_key(ml_engine_key: str = Security(APIKeyHeader(name="X-ML-Engine-Key", auto_error=False))):
    if ml_engine_key != api_config.ML_ENGINE_KEY:
        raise MLModelException("Invalid ML Engine Key")
    return ml_engine_key
