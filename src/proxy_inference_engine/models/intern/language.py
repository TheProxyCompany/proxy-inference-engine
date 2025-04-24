import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel

from proxy_inference_engine.cache.kv_cache import BaseCache
from proxy_inference_engine.models.base import create_attention_mask


class TextConfig(BaseModel):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int | None = None
    max_position_embeddings: int | None = 128000
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"mrope_section", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["mrope", "default"]:
                raise ValueError("rope_scaling type must be 'mrope' or 'default'")

class Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rotary_emb = nn.RoPE(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: BaseCache | None = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        offset = cache.offset if cache else 0

        if mask is not None:
            mask = mask[..., : keys.shape[-2]]
            assert mask is not None and mask.shape[-1] >= keys.shape[-2], (
                "Attention mask shorter than seq length"
            )

        queries = self.rotary_emb(queries, offset=offset)
        keys = self.rotary_emb(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: BaseCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen2Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen2VLDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        cache: list[BaseCache] | list[None] | None = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.model = Qwen2Model(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
        cache: list[BaseCache] | list[None] | None = None,
    ):
        if inputs is None and inputs_embeds is None:
            raise ValueError("Either inputs or inputs_embeds must be provided")

        out = self.model(
            inputs,
            mask=mask,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        return logits

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
