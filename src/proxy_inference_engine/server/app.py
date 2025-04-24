import logging
import logging.config

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from proxy_inference_engine.engine import InferenceEngine
from proxy_inference_engine.server.config import load_settings
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.routes.chat import chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Application Setup ---
def create_app() -> FastAPI:
    logger.info("Starting application setup...")
    settings = load_settings()
    logger.info(f"Settings loaded. Model path (MVP): {settings.MODEL_PATH_MVP}")

    try:
        logger.info(f"Loading InferenceEngine from: {settings.MODEL_PATH_MVP}")
        inference_engine = InferenceEngine(settings.MODEL_PATH_MVP)
        logger.info("InferenceEngine loaded successfully.")
    except Exception as e:
        logger.exception(f"FATAL: Failed to load InferenceEngine: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize InferenceEngine: {e}") from e

    # --- Instantiate Services ---
    logger.info("Instantiating services...")

    # --- Create FastAPI App ---
    logger.info("Creating FastAPI application instance...")
    app = FastAPI(
        title="Proxy Inference Engine",
        description="A server for the Proxy Inference Engine.",
        version="0.1.0",
    )

    # --- Exception Handlers ---
    @app.exception_handler(InferenceError)
    async def inference_exception_handler(request: Request, exc: InferenceError):
        logger.error(f"Caught InferenceError: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Inference operation failed: {exc}"},
        )

    # Add other exception handlers if needed (e.g., for validation errors)
    logger.info("Exception handlers registered.")

    # --- Include Routers ---
    app.include_router(chat_router, prefix="/v1/chat", tags=["Chat"])
    logger.info("Routers included.")

    logger.info("Application setup complete.")
    return app


# Expose the app instance for uvicorn or other ASGI servers
app = create_app()
