import asyncio
import logging
import logging.config
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from proxy_inference_engine.pie_core import (
    ResponseReader,
    init_request_writer,
    shutdown_request_writer,
)
from proxy_inference_engine.server.config import load_settings
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.ipc_dispatch import (
    SHUTDOWN_TASK_TIMEOUT_S,
    IPCState,
    run_response_dispatcher,
)
from proxy_inference_engine.server.routes.chat import chat_router
from proxy_inference_engine.server.routes.completions import completions_router
from proxy_inference_engine.server.routes.responses import responses_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown logic for IPC components."""
    logger.info("Application lifespan startup: Initializing IPC...")
    ipc_state: IPCState | None = None

    try:
        # Initialize C++ components first
        logger.info("Initializing C++ Request Writer...")
        init_request_writer()
        logger.info("C++ Request Writer Initialized.")

        # Initialize shared state object
        ipc_state = IPCState()

        # Initialize C++ Response Reader
        logger.info("Initializing C++ Response Reader...")
        ipc_state.response_reader = ResponseReader()
        logger.info("C++ Response Reader Initialized.")

        # Store state in app context *after* successful initialization
        app.state.ipc_state = ipc_state

        # Start the dispatcher task
        logger.info("Starting response dispatcher task...")
        ipc_state.dispatcher_task = asyncio.create_task(
            run_response_dispatcher(ipc_state)
        )
        await asyncio.sleep(0.1)  # Allow task scheduling
        if ipc_state.dispatcher_task.done():
            # Check if task exited immediately (e.g., due to reader init error logged inside)
            try:
                ipc_state.dispatcher_task.result()  # Raise exception if task failed
            except Exception as task_err:
                logger.critical(
                    "Dispatcher task failed immediately on startup: %s",
                    task_err,
                    exc_info=True,
                )
                raise RuntimeError("Dispatcher task failed to start") from task_err
        logger.info("Response dispatcher task started successfully.")
        logger.info("IPC Initialization complete.")

    except Exception as e:
        logger.critical("Failed during IPC initialization: %s", e, exc_info=True)
        # Ensure state is marked as None if setup fails
        app.state.ipc_state = None
        shutdown_request_writer()
        raise

    # --- Application runs here ---
    yield
    # --- End Application Run ---

    # --- Shutdown logic ---
    logger.info("Application lifespan shutdown: Cleaning up IPC...")
    ipc_state = getattr(app.state, "ipc_state", None)  # Retrieve state safely

    try:
        if (
            ipc_state
            and ipc_state.dispatcher_task
            and not ipc_state.dispatcher_task.done()
        ):
            logger.info("Stopping response dispatcher task...")
            ipc_state.dispatcher_task.cancel()
            try:
                await asyncio.wait_for(
                    ipc_state.dispatcher_task,
                    timeout=SHUTDOWN_TASK_TIMEOUT_S,
                )
                logger.info("Response dispatcher task stopped gracefully.")
            except asyncio.CancelledError:
                logger.info("Response dispatcher task was cancelled.")
            except TimeoutError:
                logger.warning(
                    "Response dispatcher task did not stop within timeout (%.1fs).",
                    SHUTDOWN_TASK_TIMEOUT_S,
                )
            except Exception as e:
                logger.error("Error stopping dispatcher task: %s", e, exc_info=True)
    finally:
        try:
            shutdown_request_writer()
            logger.info("C++ Request Writer shut down.")
        except Exception as e:
            logger.error("Error shutting down C++ Request Writer: %s", e, exc_info=True)
        logger.info("IPC cleanup complete.")

    logger.info("Application shutdown complete.")


# --- Application Setup ---
def create_app() -> FastAPI:
    logger.info("Starting application setup...")
    settings = load_settings()
    logger.info(
        f"Settings loaded. Model path (Note: Not directly used by app): {settings.MODEL_PATH}"
    )
    # Model is loaded by the separate C++ engine process

    logger.info("Creating FastAPI application instance...")
    app = FastAPI(
        title="Proxy Inference Engine API",
        description="API server interfacing with the PIE C++ Inference Engine.",
        version="0.1.0",
        lifespan=lifespan,  # Register the lifespan context manager
    )

    # --- Remove old engine dependency override ---
    # app.dependency_overrides[get_inference_engine] = _get_loaded_engine

    @app.exception_handler(InferenceError)
    async def inference_exception_handler(request: Request, exc: InferenceError):
        # This might need adjustment if errors are now reported differently via IPC
        logger.error(f"Caught InferenceError: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Inference operation failed: {exc}"},
        )

    logger.info("Exception handlers registered.")

    # --- Include Routers ---
    app.include_router(completions_router, prefix="/v1", tags=["Completions"])
    app.include_router(chat_router, prefix="/v1", tags=["Chat"])
    app.include_router(responses_router, prefix="/v1", tags=["Responses"])
    logger.info("Routers included.")

    logger.info("Application setup complete.")
    return app


# Expose the app instance for uvicorn or other ASGI servers
app = create_app()
