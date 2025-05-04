import logging
from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from proxy_inference_engine.engine.inference_engine import InferenceEngine
from proxy_inference_engine.interaction import Interaction
from proxy_inference_engine.server.app import get_inference_engine
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from proxy_inference_engine.server.models.chat.output import (
    generate_chat_completion_id,
    get_current_timestamp,
)

logger = logging.getLogger(__name__)

chat_router = APIRouter()


@chat_router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
    tags=["Chat"],
)
async def handle_completion_request(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # noqa: B008
) -> dict[str, Any] | StreamingResponse:
    """
    Handles requests to the `/v1/chat/completions` endpoint.
    """
    logger.info(f"Chat completion request for model: {request.model}")

    input_interactions: list[Interaction] = [
        msg.to_interaction() for msg in request.messages
    ]

    tools = [tool.to_dict() for tool in request.tools] if request.tools else None
    tool_choice = request.tool_choice.to_dict() if request.tool_choice else None
    response_format = (
        request.response_format.to_dict() if request.response_format else None
    )
    stream_options = (
        request.stream_options.model_dump() if request.stream_options else None
    )

    inference_kwargs = {
        "max_completion_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": tool_choice,
        "tools": tools,
        "response_format": response_format,
        "logprobs": request.logprobs,
        "top_logprobs": request.top_logprobs,
        "stop": request.stop,
        "stream_options": stream_options,
    }

    # Remove None values
    inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}

    try:
        if request.stream:
            response = await handle_streaming_completion_request(
                request.model,
                engine,
                input_interactions,
                inference_kwargs,
            )
            return response
        else:
            response = await handle_non_streaming_completion_request(
                request.model,
                engine,
                input_interactions,
                inference_kwargs,
            )
            logger.info(f"Chat completion request successful. ID: {response.id}")
            return response.model_dump()
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


async def handle_non_streaming_completion_request(
    model: str,
    engine: InferenceEngine,
    input_interactions: list[Interaction],
    inference_kwargs: dict[str, Any],
) -> ChatCompletionResponse:
    """
    Handles non-streaming chat completion requests.
    """
    logger.info(f" Chat completion request for model: {model}")
    new_interaction = engine(input_interactions, **inference_kwargs)
    finish_reason = new_interaction.metadata.get("finish_reason", "unknown")
    prompt_tokens = new_interaction.metadata.get("prompt_tokens", 0)
    completion_tokens = new_interaction.metadata.get("completion_tokens", 0)
    total_tokens = new_interaction.metadata.get("total_tokens", 0)
    generated_tokens = new_interaction.metadata.get("generated_tokens", [])
    generated_logprobs = new_interaction.metadata.get("generated_logprobs", None)

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage.from_interaction(new_interaction),
        finish_reason=finish_reason,
        logprobs=ChatCompletionLogProbs.from_generation(
            generated_tokens,
            generated_logprobs,
            engine.tokenizer.decode,
        ),
    )

    usage = ChatCompletionUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    response = ChatCompletionResponse(
        model=model,
        choices=[choice],
        usage=usage,
    )
    return response


async def handle_streaming_completion_request(
    model: str,
    engine: InferenceEngine,
    input_interactions: list[Interaction],
    inference_kwargs: dict[str, Any],
) -> StreamingResponse:
    logger.info(f"Streaming chat completion request for model: {model}")

    chat_completion_id = generate_chat_completion_id()
    created_at = get_current_timestamp()

    stream_options: dict = inference_kwargs.get("stream_options", {})
    include_usage: bool = stream_options.get("include_usage", False)
    collect_logprobs: bool = inference_kwargs.get("logprobs", False)

    encoded_prompt = engine.prepare_engine(input_interactions, **inference_kwargs)

    async def stream() -> AsyncIterable[str]:
        generated_tokens = []
        generated_logprobs = []
        finish_reason = None
        try:
            for token_id, logprobs_map in engine.generate(
                encoded_prompt, **inference_kwargs
            ):
                generated_tokens.append(token_id)
                if collect_logprobs:
                    generated_logprobs.append(logprobs_map)

                delta = ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatMessage(
                        role="assistant" if not generated_tokens else None,
                        content=engine.tokenizer.decode(token_id),
                    ),
                    finish_reason=finish_reason,
                    logprobs=ChatCompletionLogProbs.from_generation(
                        generated_tokens,
                        generated_logprobs if collect_logprobs else None,
                        engine.tokenizer.decode,
                    ),
                )

                yield ChatCompletionChunk(
                    id=chat_completion_id,
                    created=created_at,
                    model=model,
                    choices=[delta],
                ).model_dump_json()
        except StopIteration as exc:
            finish_reason = exc.value
            assert isinstance(finish_reason, str)

        if not finish_reason:
            finish_reason = "stop"

        # send the final "done" chunk
        yield ChatCompletionChunk(
            id=chat_completion_id,
            created=created_at,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatMessage(role=None, content=None),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
        ).model_dump_json()

        if include_usage:
            # send the final "usage" chunk
            yield ChatCompletionChunk(
                id=chat_completion_id,
                created=created_at,
                model=model,
                choices=[],
                usage=ChatCompletionUsage(
                    input_tokens=encoded_prompt.size,
                    output_tokens=len(generated_tokens),
                    total_tokens=encoded_prompt.size + len(generated_tokens),
                ),
            ).model_dump_json()

    response = StreamingResponse(
        content=stream(),
        media_type="text/event-stream",
    )
    return response
