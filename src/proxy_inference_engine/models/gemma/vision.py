import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel


class VisionConfig(BaseModel):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    patch_size: int
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False

class AveragePool2D(nn.Module):
    """Applies 4x4 average pooling and reshaping."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __call__(self, x):
        """Applies 4x4 average pooling and reshaping."""
        batch_size, seq_len, channels = x.shape
        width = int(seq_len**0.5)
        if width * width != seq_len:
            raise ValueError(
                f"Sequence length {seq_len} is not a perfect square. Cannot reshape to a square image."
            )
        # Bx(64^2)x1152 -> Bx1152x(64^2) -> Bx1152x64x64
        x = x.transpose(1, 2).reshape(batch_size, channels, width, width)
        # Bx1152x64x64-> Bx1152x16x16
        x = nn.AvgPool2d(kernel_size=4, stride=4)(x)
        # Bx1152x64x64-> Bx1152x256 -> Bx256x1152
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: int | None = None,
        key_input_dims: int | None = None,
        value_input_dims: int | None = None,
        value_dims: int | None = None,
        value_output_dims: int | None = None,
        bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, x, mask=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        output_hidden_states: bool = False,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...] | None]:
        collected_hidden_states = (hidden_states,) if output_hidden_states else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
            if output_hidden_states:
                assert collected_hidden_states is not None
                collected_hidden_states = (*collected_hidden_states, hidden_states)

        if output_hidden_states:
            return hidden_states[0], collected_hidden_states
        else:
            return hidden_states[0], None


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        position_ids = mx.array(mx.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings


class SigLipVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.avgpool = AveragePool2D(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: bool | None = None,
    ) -> mx.array:
        x = self.embeddings(x)

        out, hidden_states = self.encoder(
            hidden_states=x,
            output_hidden_states=output_hidden_states or False,
            mask=None,
        )

        pooler_output = self.post_layernorm(out)
        return self.avgpool(pooler_output)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_model = SigLipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: bool | None = None
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized_weights = {}
        for k, v in weights.items():
            if "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
