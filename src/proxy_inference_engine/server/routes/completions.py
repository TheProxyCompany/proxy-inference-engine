import logging

from fastapi import APIRouter, Depends, HTTPException, status

from proxy_inference_engine.engine.inference_engine import InferenceEngine
from proxy_inference_engine.interaction import (
    Interaction,
    InteractionRole,
)
from proxy_inference_engine.server.dependencies import get_inference_engine
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.models.completions import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)

logger = logging.getLogger(__name__)

completions_router = APIRouter()


@completions_router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Create a text completion",
    tags=["Completions"],
)
async def handle_completion_request(
    request: CompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # noqa: B008
) -> CompletionResponse:
    """
    Handles requests to the `/completions` endpoint.

    This endpoint uses the OpenAI v1 chat completions API.
    """
    logger.info(f"Handling completion request for model: {request.model}")

    if request.stream:
        logger.warning("Streaming requested but not supported.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is not supported in this version.",
        )
    if request.n > 1:
        logger.warning("Request parameter 'n' > 1 is not supported, using n=1.")
        # If best_of is also set, log a warning or raise error based on validation
        if request.best_of is not None and request.best_of > 1:
            logger.warning(
                "Request parameters 'n' > 1 and 'best_of' > 1 are not supported, using n=1 and ignoring best_of."
            )

    input_interactions: list[Interaction]
    if isinstance(request.prompt, str):
        input_interactions = [
            Interaction.simple(
                role=InteractionRole.USER,
                content=request.prompt,
            )
        ]
    elif isinstance(request.prompt, list) and len(request.prompt) > 0:
        logger.warning(
            "Batch prompt input received, using only the first prompt in this version."
        )
        input_interactions = [
            Interaction.simple(
                role=InteractionRole.USER,
                content=request.prompt[0],
            )
        ]
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid prompt format. Expecting string or list of strings.",
        )

    inference_kwargs = {
        "max_completion_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
    }

    try:
        new_interaction = await engine(
            input_interactions,
            **inference_kwargs,
        )
        finish_reason = new_interaction.metadata.get("finish_reason", "unknown")
        prompt_tokens = new_interaction.metadata.get("prompt_tokens", 0)
        completion_tokens = new_interaction.metadata.get("completion_tokens", 0)
        total_tokens = new_interaction.metadata.get("total_tokens", 0)

        new_content = new_interaction.content[0].content if new_interaction.content else ""
        choice = CompletionChoice(
            index=0,
            text=new_content,
            logprobs=None,
            finish_reason=finish_reason,
        )

        usage = CompletionUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        response = CompletionResponse(
            model=request.model,
            choices=[choice],
            usage=usage,
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

    logger.info(f"Completion request successful. ID: {response.id}")
    return response
