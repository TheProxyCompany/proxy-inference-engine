import logging
import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from proxy_inference_engine.engine import InferenceEngine

logger = logging.getLogger(__name__)

class AppState(BaseModel):
    """
    Pydantic model to hold application state.
    """
    inference_engine: InferenceEngine | None = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI, model_path: str):
    # --- Startup ---
    logger.info("Executing API startup sequence...")
    try:
        logger.info(f"Loading InferenceEngine from: {model_path}")
        engine_instance = InferenceEngine(model_path=model_path)
        app_state.inference_engine = engine_instance
        logger.info("InferenceEngine loaded successfully.")
    except Exception as e:
        logger.exception(f"CRITICAL: Failed to load InferenceEngine during startup: {e}")
        app_state.inference_engine = None

    yield

    # --- Shutdown ---
    logger.info("Executing API shutdown sequence...")
    if app_state.inference_engine:
        logger.info("Cleaning up InferenceEngine resources...")
        # Add any explicit cleanup needed for the engine
        app_state.inference_engine = None

    gc.collect()
    logger.info("Shutdown complete.")

def create_app() -> FastAPI:
    """Factory function to create the FastAPI application instance."""

    # Check if model path env var was set by cli.py
    if not os.environ.get(MODEL_PATH_ENV_VAR):
         # This should ideally not happen if cli.py runs first, but good to check
         logger.warning(f"{MODEL_PATH_ENV_VAR} not found in environment for factory.")
         # Depending on requirements, could raise RuntimeError here

    app = FastAPI(
        title="Proxy Inference Engine (PIE) API",
        description="API for optimized MLX inference with optional PSE structuring.",
        version="0.1.0", # Consider making this dynamic from pyproject.toml
        lifespan=lifespan # Use the lifespan context manager
    )

    # --- API Routes ---

    @app.get("/health")
    async def health_check():
        engine = app_state.get("inference_engine")
        model_loaded = engine is not None and hasattr(engine, 'model')
        if model_loaded:
            return {"status": "healthy", "message": "InferenceEngine loaded."}
        else:
            # Return 503 Service Unavailable if engine failed to load
             raise HTTPException(status_code=503, detail="Inference engine is not available.")

    # Include the router for /v1/responses
    app.include_router(responses_router.router)

    return app
