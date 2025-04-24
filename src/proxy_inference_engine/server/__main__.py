import uvicorn

from proxy_inference_engine.server.app import logger
from proxy_inference_engine.server.config import load_settings

if __name__ == "__main__":
    settings = load_settings()
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "proxy_inference_engine.server.app:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )
