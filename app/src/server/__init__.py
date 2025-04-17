__version__ = "0.0.0"

from fastapi import Security
from fastapi.security import APIKeyHeader
from src.server.config import APIConfig
from src.server.exceptions import MLModelException

api_config = APIConfig()


async def validate_api_key(api_key: str = Security(APIKeyHeader(name="X-API-Key", auto_error=False))):
    if api_key != api_config.ML_MODEL_API_KEY:
        raise MLModelException("Invalid API Key")
    return api_key
