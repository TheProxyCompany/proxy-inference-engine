import logging
from fastapi import APIRouter, HTTPException, status

from proxy_inference_engine.server.models.chat import (
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from proxy_inference_engine.server.exceptions import InferenceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Create a legacy text completion",
    tags=["Legacy Completions"],
)
async def handle_completion_request(
    request: CompletionRequest,
    # service:
) -> CompletionResponse:
    """
    Handles requests to the `/v1/chat/completions` endpoint.

    This endpoint uses the OpenAI v1 chat completions API.
    """
    logger.info(f"Handling completion request for model: {request.model}")
    try:
        # response = await service.create_completion(request)
        # return response
        return CompletionResponse(
            id="cmpl-123",
            object="text_completion",
            created=123,
            model="gpt-3.5-turbo",
            choices=[],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            system_fingerprint="",
        )
    except InferenceError as e:
        logger.error(f"Inference error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        )
    except NotImplementedError as e:
        logger.error(f"Feature not implemented: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("An unexpected error occurred", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during completion.",
        ) from e
