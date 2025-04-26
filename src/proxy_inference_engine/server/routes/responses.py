import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from proxy_inference_engine.engine import InferenceEngine
from proxy_inference_engine.interaction import (
    Interaction,
    InteractionRole,
    InteractionType,
)
from proxy_inference_engine.server.dependencies import get_inference_engine
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.models.responses import (
    OutputFunctionCall,
    OutputMessage,
    OutputTextContent,
    ResponseObject,
    ResponseRequest,
    ResponseUsage,
)

logger = logging.getLogger(__name__)

responses_router = APIRouter()


@responses_router.post(
    "/responses",
    response_model=ResponseObject,
    summary="Create a model response",
    tags=["Responses"],
)
async def handle_response_request(
    request: ResponseRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # noqa: B008
) -> ResponseObject:
    """
    Handles requests to the `/v1/responses` endpoint (MVP: text-only).
    """
    logger.info(f"Handling response request for model: {request.model}")

    input_interactions: list[Interaction] = []
    if request.instructions:
        input_interactions.append(
            Interaction.simple(
                role=InteractionRole.SYSTEM, content=request.instructions
            )
        )
    input_interactions.append(
        Interaction.simple(role=InteractionRole.USER, content=request.input)
    )

    inference_kwargs = {
        "max_completion_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": request.tool_choice.to_dict() if request.tool_choice else None,
        "tools": [tool.to_dict() for tool in request.tools] if request.tools else None,
        "response_format": request.text.to_dict() if request.text else None,
    }
    inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}

    try:
        new_interaction = await engine(
            input_interactions,
            **inference_kwargs,
        )
        prompt_tokens = new_interaction.metadata.get("prompt_tokens", 0)
        completion_tokens = new_interaction.metadata.get("completion_tokens", 0)
        total_tokens = new_interaction.metadata.get("total_tokens", 0)

        response_content = []
        for item in new_interaction.content:
            match item.type:
                case InteractionType.TEXT:
                    response_content.append(OutputTextContent(text=item.content))
                case InteractionType.TOOL_CALL:
                    assert isinstance(item.content, dict)
                    name, arguments = item.content.get("name"), item.content.get("arguments")
                    assert isinstance(name, str)
                    assert isinstance(arguments, dict)
                    response_content.append(OutputFunctionCall(name=name, arguments=json.dumps(arguments)))
                case _:
                    logger.warning(f"Unknown content type: {item.type}")

        response_message = OutputMessage(content=response_content)
        usage = ResponseUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        response = ResponseObject(
            model=request.model,
            output=[response_message],
            usage=usage,
            min_p=request.min_p,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            parallel_tool_calls=request.parallel_tool_calls or False,
            tool_choice=request.tool_choice,
            tools=request.tools or [],
            text=request.text,
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

    logger.info(f"Response request successful. ID: {response.id}")
    return response
