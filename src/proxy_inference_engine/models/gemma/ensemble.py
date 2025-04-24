import mlx.core as mx
import mlx.nn as nn

from proxy_inference_engine.cache.kv_cache import BaseCache
from proxy_inference_engine.models.base import BaseModelArgs
from proxy_inference_engine.models.gemma.language import (
    LanguageModel,
    RMSNorm,
    TextConfig,
)
from proxy_inference_engine.models.gemma.vision import VisionConfig, VisionModel


class ModelArgs(BaseModelArgs):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int | None = 0


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.mm_input_projection_weight = mx.ones(
            (config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.text_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def __call__(self, vision_features: mx.array) -> mx.array:
        batch_size, _, hidden_dim = vision_features.shape

        # Reshape from [batch, hidden_dim, seq_len] to [batch, seq_len, patches, patches]
        spatial_features = vision_features.transpose(0, 2, 1)
        spatial_features = spatial_features.reshape(
            batch_size, hidden_dim, self.patches_per_image, self.patches_per_image
        )

        # Transpose to [batch, patches, patches, hidden_dim] for pooling
        spatial_features = spatial_features.transpose(0, 2, 3, 1)
        # Apply average pooling to reduce spatial dimensions
        downsampled_features = self.avg_pool(spatial_features)
        # Reshape to [batch, tokens_per_image, hidden_dim]
        downsampled_features = downsampled_features.transpose(0, 3, 1, 2).flatten(2)
        downsampled_features = downsampled_features.transpose(0, 2, 1)

        # Apply normalization
        normalized_features = self.mm_soft_emb_norm(downsampled_features)

        # Project to language model dimension space
        projected_features = mx.einsum(
            "btm,md->btd", normalized_features, self.mm_input_projection_weight
        )

        # Ensure output has same dtype as input
        return projected_features.astype(vision_features.dtype)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids), None

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_state, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
            output_hidden_states=True,
        )

        image_features = hidden_state[None, :].astype(pixel_values.dtype)
        image_features = self.multi_modal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = (
            self._prepare_inputs_for_multimodal(
                image_features, inputs_embeds, input_ids, mask
            )
        )
        return final_inputs_embeds, final_attention_mask_4d

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        _, _, embed_dim = image_features.shape

        batch_size, sequence_length = input_ids.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))

        pad_token_id = self.config.pad_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == pad_token_id

        # expand masks to match embedding dimension
        text_mask_expanded = mx.expand_dims(text_mask, -1)
        text_mask_expanded = mx.repeat(text_mask_expanded, embed_dim, axis=-1)
        pad_mask_expanded = mx.expand_dims(pad_mask, -1)
        pad_mask_expanded = mx.repeat(pad_mask_expanded, embed_dim, axis=-1)

        # insert padding and text token embeddings
        final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )
        pad_size = final_embedding.shape[1] - scaled_image_features.shape[1]
        scaled_image_features = mx.pad(
            scaled_image_features, ((0, 0), (0, pad_size), (0, 0))
        )
        # insert image embeddings - the image mask is always less or equal to the sentence in length
        image_mask_expanded = mx.expand_dims(image_mask, -1)
        image_mask_expanded = mx.repeat(image_mask_expanded, embed_dim, axis=-1)
        final_embedding = mx.where(
            image_mask_expanded, scaled_image_features, final_embedding
        )

        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )

        attention_mask_expanded_1 = mx.expand_dims(attention_mask, 1)
        attention_mask_expanded_2 = mx.expand_dims(attention_mask, 2)
        final_attention_mask_4d = attention_mask_expanded_1 * attention_mask_expanded_2
        final_attention_mask_4d = final_attention_mask_4d
        final_attention_mask_4d = mx.expand_dims(final_attention_mask_4d, 1)
        final_embedding = mx.array(final_embedding)
        return final_embedding, final_attention_mask_4d

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
        cache: list[BaseCache] | list[None] | None = None,
        **kwargs,
    ):
        input_embeddings, final_attention_mask_4d = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )

        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            # mask=final_attention_mask_4d, # TODO: Fix mask
        )
        return logits

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def head_dim(self):
        return self.language_model.model.head_dim

    @property
    def n_kv_heads(self):
        return self.language_model.model.n_kv_heads
