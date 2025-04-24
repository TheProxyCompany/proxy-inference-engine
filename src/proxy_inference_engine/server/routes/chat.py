import logging

from fastapi import APIRouter, Depends, HTTPException, status

from proxy_inference_engine.engine.inference_engine import InferenceEngine
from proxy_inference_engine.interaction import Interaction
from proxy_inference_engine.server.app import get_inference_engine
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.models.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)

logger = logging.getLogger(__name__)

chat_router = APIRouter()


@chat_router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create a chat completion",
    tags=["Chat"],
)
async def handle_completion_request(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # noqa: B008
) -> ChatCompletionResponse:
    """
    Handles requests to the `/v1/chat/completions` endpoint.
    """
    logger.info(f"Handling chat completion request for model: {request.model}")

    input_interactions: list[Interaction] = [
        msg.to_interaction()
        for msg in request.messages
    ]

    inference_kwargs = {
        "max_completion_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": request.tool_choice,
        "tools": request.tools,
        "response_format": request.response_format,
    }

    inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}

    try:
        generated_text, metadata = await engine(
            input_interactions,
            **inference_kwargs,
        )
    except InferenceError as e:
        logger.error(f"Inference error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        ) from e
    except NotImplementedError as e:
        logger.error(f"Feature not implemented: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("An unexpected error occurred", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during completion.",
        ) from e

    finish_reason = metadata.get("finish_reason", "unknown")
    prompt_tokens = metadata.get("prompt_tokens", 0)
    completion_tokens = metadata.get("completion_tokens", 0)
    total_tokens = metadata.get("total_tokens", 0)

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content=generated_text,
        ),
        finish_reason=finish_reason,
    )

    usage = ChatCompletionUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    response = ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        usage=usage,
    )
    logger.info(f"Chat completion request successful. ID: {response.id}")
    return response
