import asyncio
import json
import logging
import random
from collections.abc import AsyncIterable
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from proxy_inference_engine.interaction import Interaction
from proxy_inference_engine.pie_core import submit_request
from proxy_inference_engine.server.dependencies import IPCStateDep
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.ipc_dispatch import ResponseDeltaDict
from proxy_inference_engine.server.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
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
    summary="Create a chat completion",
    tags=["Chat"],
)
async def handle_completion_request(
    request: ChatCompletionRequest,
    ipc_state: IPCStateDep,
) -> ChatCompletionResponse | EventSourceResponse:
    """
    Handles requests to the `/v1/chat/completions` endpoint.
    """
    logger.info(f"Chat completion request for model: {request.model}")

    # 1. Generate request ID
    current_request_id = await ipc_state.get_next_request_id()
    logger.debug("Generated request ID: %d", current_request_id)

    # 2. Prepare arguments for C++ submit_request
    input_interactions: list[Interaction] = [
        msg.to_interaction() for msg in request.messages
    ]
    prompt_string = json.dumps(
        [interaction.to_dict() for interaction in input_interactions]
    )
    tools = [tool.to_dict() for tool in request.tools] if request.tools else None
    tool_schemas_str = json.dumps(tools) if tools else ""
    response_format_str = (
        json.dumps(request.response_format.to_dict()) if request.response_format else ""
    )

    # tool_choice = request.tool_choice.to_dict() if request.tool_choice else None
    # stream_options = (
    #     request.stream_options.model_dump() if request.stream_options else None
    # )

    submit_args = {
        "request_id": current_request_id,
        "prompt_string": prompt_string,
        "temperature": request.temperature,
        "top_p": request.top_p if request.top_p is not None else 1.0,
        "top_k": request.top_k if request.top_k is not None else -1,
        "min_p": request.min_p if request.min_p is not None else 0.0,
        "rng_seed": random.randint(0, 2**32 - 1),
        "logit_bias": {},
        "max_generated_tokens": request.max_completion_tokens or 1024,
        "stop_token_ids": [],
        "request_channel_id": 0,
        "response_channel_id": current_request_id,
        "tool_schemas_str": tool_schemas_str,
        "response_format_str": response_format_str,
    }

    # 3. Create and register the asyncio Queue for this request
    response_queue = asyncio.Queue[ResponseDeltaDict]()
    ipc_state.active_request_queues[current_request_id] = response_queue
    logger.debug("Registered queue for request ID: %d", current_request_id)

    # 4. Submit request to C++ engine (run sync C++ call in thread)
    try:
        logger.debug("Submitting request %d to C++ engine", current_request_id)
        await asyncio.to_thread(submit_request, **submit_args)
        logger.info("Submitted request %d successfully", current_request_id)
        # 5. Handle response (streaming or non-streaming)
        if request.stream:
            logger.debug("Handling streaming for request %d", current_request_id)
            return EventSourceResponse(
                content=stream_response_generator(
                    current_request_id,
                    response_queue,
                    request.model,
                ),
                media_type="text/event-stream",
            )
        else:
            logger.debug("Handling non-streaming for request %d", current_request_id)
            response_data = await gather_non_streaming_response(
                current_request_id, response_queue
            )
            usage = ChatCompletionUsage(
                input_tokens=0,
                output_tokens=response_data["completion_tokens"],
                total_tokens=response_data["completion_tokens"],
            )
            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content=response_data["full_content"]
                ),
                finish_reason=response_data["finish_reason"],
                logprobs=None,  # TODO: Accumulate logprobs if needed
            )
            final_response = ChatCompletionResponse(
                id=generate_chat_completion_id(),  # Generate final ID
                created=get_current_timestamp(),
                model=request.model,
                choices=[choice],
                usage=usage,
            )
            logger.info("Non-streaming request %d successful.", current_request_id)
            return final_response
    except Exception as e:
        logger.error(f"Error submitting request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during completion.",
        ) from e
    finally:
        if current_request_id in ipc_state.active_request_queues:
            _ = ipc_state.active_request_queues.pop(current_request_id, None)
            logger.debug("Cleaned up queue for request ID %d.", current_request_id)


async def gather_non_streaming_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
) -> dict[str, Any]:
    """Collects deltas from the queue until the final one for a non-streaming response."""
    full_content = ""
    finish_reason = "unknown"
    completion_tokens = 0

    while True:
        try:
            delta = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError as e:
            logger.error(
                "Timeout waiting for delta for non-streaming request %d", request_id
            )
            raise InferenceError("Timeout receiving response from engine.") from e

        full_content += delta.get("content", "")
        completion_tokens += len(delta.get("tokens", []))

        queue.task_done()  # Mark item as processed

        if delta.get("is_final_delta", False):
            finish_reason = delta.get("finish_reason", "stop")
            logger.debug(
                "Received final delta for request %d, reason: %s",
                request_id,
                finish_reason,
            )
            break

    return {
        "full_content": full_content.strip(),
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
    }


async def stream_response_generator(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    model_name: str,
) -> AsyncIterable[str]:
    """Generates Server-Sent Events (SSE) from the response queue."""
    completion_tokens = 0
    chat_completion_id = generate_chat_completion_id()  # ID for the whole stream
    created_at = get_current_timestamp()
    logger.info("Starting SSE stream for request %d", request_id)

    while True:
        try:
            delta_dict = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError:
            logger.error(
                "Timeout waiting for delta for streaming request %d", request_id
            )
            break

        chunk_choice = ChatCompletionChunkChoice(
            index=0,
            delta=ChatMessage(
                # only send assistant role for the first chunk
                role="assistant" if completion_tokens <= 0 else None,
                content=delta_dict.get("content", None),
            ),
            finish_reason=None,
            logprobs=None,
        )
        chunk = ChatCompletionChunk(
            id=chat_completion_id,
            created=created_at,
            model=model_name,
            choices=[chunk_choice],
        )

        token_list = delta_dict.get("tokens", [])
        completion_tokens += len(token_list)

        yield chunk.model_dump_json()

        queue.task_done()  # Mark item as processed

        if delta_dict.get("is_final_delta", False):
            finish_reason = delta_dict.get("finish_reason", "stop")

            logger.info(
                "Sending final delta for stream %d, reason: %s",
                request_id,
                finish_reason,
            )
            final_chunk_choice = ChatCompletionChunkChoice(
                index=0,
                delta=ChatMessage(role=None, content=None),
                finish_reason=finish_reason,
                logprobs=None,
            )
            final_chunk = ChatCompletionChunk(
                id=chat_completion_id,
                created=created_at,
                model=model_name,
                choices=[final_chunk_choice],
            )
            yield final_chunk.model_dump_json()
            break  # Exit loop after final delta

    yield "[DONE]"  # OpenAI compatible stream termination
    logger.info("Finished SSE stream for request %d", request_id)
