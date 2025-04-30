import mlx.nn as nn
from pydantic import BaseModel, Field


class RopeConfig(BaseModel):
    """Configuration for Rotary Positional Embeddings (RoPE)."""

    base: float = Field(..., description="The base frequency for RoPE calculations.")
    beta_fast: int | None = Field(
        description="Parameter for YaRN scaling (fast beta component).",
        default=None,
    )
    beta_slow: int | None = Field(
        description="Parameter for YaRN scaling (slow beta component).",
        default=None,
    )
    dimensions: int = Field(..., description="The feature dimension RoPE is applied to.")
    factor: float = Field(
        ..., description="Scaling factor, potentially for NTK or YaRN methods."
    )
    high_freq_factor: float | None = Field(
        None, description="High frequency factor used in Llama 3's RoPE variant."
    )
    low_freq_factor: float | None = Field(
        None, description="Low frequency factor used in Llama 3's RoPE variant."
    )
    max_position_embeddings: int = Field(
        ...,
        description="The maximum sequence length the model is configured to handle after potential scaling.",
    )
    mscale: float | None = Field(
        None, description="Multiplier scale factor for position interpolation."
    )
    mscale_all_dim: float | None = Field(
        None,
        description="Alternative multiplier scale factor applied across all dimensions.",
    )
    original_max_position_embeddings: int = Field(
        ...,
        description="The original maximum sequence length before any scaling or interpolation.",
    )
    traditional: bool = Field(
        False, description="Whether to use the traditional RoPE formulation."
    )
    type: str = Field(
        ..., description="Specifies the RoPE implementation type (e.g., 'llama3')."
    )

def Rope(rope_config: RopeConfig) -> nn.Module:
    """Factory function to create a RoPE module based on the configuration."""
    # Dynamically import to avoid circular dependencies, a necessary entanglement.
    from proxy_inference_engine.models.llama import Llama3RoPE

    if rope_config.type == "llama3":
        return Llama3RoPE(**rope_config.model_dump())
    else:
        # A veritable Gordian knot of unsupported types.
        raise ValueError(f"Unsupported RoPE type: {rope_config.type}")
