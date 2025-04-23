from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional, Literal

# --- Request Models ---


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"]
    image_url: str  # Assuming URL for now


# A union type for possible input items
InputItem = Union[InputText, InputImage]  # Add other types like file inputs later


class ReasoningOptions(BaseModel):
    # Placeholder for reasoning options for o-series models
    pass


class TextOptions(BaseModel):
    # Placeholder for text response configuration
    format: Dict[str, Any] = {"type": "text"}  # Default based on example


class ToolChoice(BaseModel):
    # Placeholder for tool choice configuration
    pass


class ToolDefinition(BaseModel):
    # Placeholder for tool definitions (built-in or function calls)
    pass


class CreateResponseRequest(BaseModel):
    input: Union[str, List[InputItem]]  # Allow simple string or detailed list
    model: str
    include: Optional[List[str]] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    metadata: Optional[Dict[str, str]] = Field(default=None, max_items=16)
    parallel_tool_calls: Optional[bool] = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[ReasoningOptions] = None
    service_tier: Optional[Literal["auto", "default", "flex"]] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    text: Optional[TextOptions] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None  # Can be string or object
    tools: Optional[List[ToolDefinition]] = None
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    # Add custom validation for metadata key/value lengths if needed


# --- Response Models ---


class OutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[Any] = []  # Define annotations structure if needed


class OutputMessage(BaseModel):
    type: Literal["message"]
    id: str
    status: str  # e.g., "completed"
    role: Literal["assistant"]
    content: List[OutputText]  # Assuming only text output for now


# Placeholder for other potential output types (e.g., tool calls)
OutputItem = OutputMessage  # Add other types later


class ResponseReasoning(BaseModel):
    effort: Optional[Any] = None
    summary: Optional[Any] = None


class UsageDetails(BaseModel):
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: Optional[UsageDetails] = None
    output_tokens: int
    output_tokens_details: Optional[UsageDetails] = None
    total_tokens: int


class CreateResponseResponse(BaseModel):
    id: str
    object: Literal["response"]
    created_at: int  # Unix timestamp
    status: str  # e.g., "completed"
    error: Optional[Any] = None  # Define error structure if needed
    incomplete_details: Optional[Any] = None  # Define details structure if needed
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[OutputItem]
    parallel_tool_calls: bool
    previous_response_id: Optional[str] = None
    reasoning: Optional[ResponseReasoning] = None
    store: bool
    temperature: Optional[float] = None  # Can be null in response
    text: Optional[TextOptions] = (
        None  # Re-use request model or define specific response one
    )
    tool_choice: Optional[Any] = None  # Use ToolChoice or Any
    tools: Optional[List[Any]] = None  # Use ToolDefinition or Any
    top_p: Optional[float] = None  # Can be null in response
    truncation: Optional[str] = None  # Use literal or str
    usage: ResponseUsage
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Response metadata might differ slightly


# --- API Router ---

router = APIRouter()


@router.post(
    "/v1/responses",
    response_model=CreateResponseResponse,
    summary="Create a model response",
    response_description="The generated model response",
    tags=["inference"],  # Tag for docs organization
)
async def create_response(request_body: CreateResponseRequest = Body(...)):
    """
    Creates a model response.

    Provide text or image inputs to generate text or JSON outputs.
    Optionally use function calling or built-in tools.
    """
    # --- Placeholder Implementation ---
    # In a real scenario, you would pass the request_body to your inference engine
    # and construct the actual response based on its output.

    # For now, return a dummy response matching the schema
    dummy_output_text = OutputText(type="output_text", text="This is a dummy response.")
    dummy_output_message = OutputMessage(
        type="message",
        id=f"msg_{request_body.model[:5]}_dummy",
        status="completed",
        role="assistant",
        content=[dummy_output_text],
    )
    dummy_usage = ResponseUsage(
        input_tokens=50,  # Dummy value
        output_tokens=10,  # Dummy value
        total_tokens=60,  # Dummy value
    )

    response = CreateResponseResponse(
        id=f"resp_{request_body.model[:5]}_dummy",
        object="response",
        created_at=1741476777,  # Dummy timestamp
        status="completed",
        model=request_body.model,
        output=[dummy_output_message],
        parallel_tool_calls=request_body.parallel_tool_calls
        if request_body.parallel_tool_calls is not None
        else True,
        store=request_body.store if request_body.store is not None else True,
        usage=dummy_usage,
        # Include other fields with defaults or dummy values as needed
        temperature=request_body.temperature,
        top_p=request_body.top_p,
        truncation=request_body.truncation,
    )

    return response


# You might add other related endpoints to this router later
