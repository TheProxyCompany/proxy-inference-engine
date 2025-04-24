import logging
import logging.config
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

# from proxy_inference_engine.engine import InferenceEngine

from proxy_inference_engine.server.config import load_settings
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.routes.chat import (
    router as completions_router,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Application Setup ---
def create_app() -> FastAPI:
    logger.info("Starting application setup...")
    settings = load_settings()
    logger.info(f"Settings loaded. Model path (MVP): {settings.MODEL_PATH_MVP}")

    # --- MVP: Load Single Engine Directly ---
    # In the future, this will be replaced by EngineManager
    try:
        logger.info(f"Loading InferenceEngine from: {settings.MODEL_PATH_MVP}")
        # pie_engine = InferenceEngine(settings.MODEL_PATH_MVP)
        logger.info("InferenceEngine loaded successfully.")
    except Exception as e:
        logger.exception(f"FATAL: Failed to load InferenceEngine: {e}", exc_info=True)
        # Depending on desired behavior, you might exit or continue without the engine
        raise RuntimeError(f"Could not initialize InferenceEngine: {e}") from e

    # --- Instantiate Services ---
    logger.info("Instantiating services...")

    # --- Create FastAPI App ---
    logger.info("Creating FastAPI application instance...")
    app = FastAPI(
        title="Proxy Inference Engine",
        description="A proxy server for various inference engines.",
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
    app.include_router(completions_router, prefix="/v1", tags=["Legacy Completions"])
    logger.info("Routers included.")

    # --- Health Check Endpoint (Optional but Recommended) ---
    @app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
    async def health_check():
        # Basic check; could be expanded to check engine status, etc.
        return {"status": "ok"}

    logger.info("Application setup complete.")
    return app


# Expose the app instance for uvicorn or other ASGI servers
app = create_app()
