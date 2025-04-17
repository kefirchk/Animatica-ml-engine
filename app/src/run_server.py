import logging

import src.server as server
from fastapi import Depends, FastAPI
from src.server.config import APIConfig, LogLevelEnum
from src.server.exceptions import app_exception_handler, exception_handlers
from src.server.routers import router as fomm_router

logging.basicConfig(format="[PID:%(process)d] %(pathname)s:%(lineno)d %(message)s", level=logging.INFO)
config = APIConfig()

app = FastAPI(
    title=f"{config.MODE.capitalize()} Animatica ML API",
    description="This API is designed for the Animatica application.",
    swagger_ui_parameters={"displayRequestDuration": True},
    version=server.__version__,
    debug=(config.LOG_LEVEL == LogLevelEnum.DEBUG),
)

app.include_router(fomm_router, prefix="/api", dependencies=[Depends(server.validate_api_key)])

for exc_type in exception_handlers:
    app.add_exception_handler(exc_type, app_exception_handler)
