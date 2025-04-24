from typing import cast

import mlx.core as mx
import mlx.nn as nn

from proxy_inference_engine.cache.kv_cache import BaseCache
from proxy_inference_engine.models.base import BaseModelArgs
from proxy_inference_engine.models.intern.language import LanguageModel, TextConfig
from proxy_inference_engine.models.intern.vision import VisionConfig, VisionModel


class ModelArgs(BaseModelArgs):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        # Identify positions of image tokens (assuming batch size 1)
        image_positions = input_ids == image_token_id
        num_images = mx.sum(image_positions.astype(mx.int32))

        # Fallback to video token if no image token found
        num_images_val: int = cast(int, num_images.item())
        if num_images_val == 0:
            image_positions = input_ids == video_token_id
            num_images = mx.sum(image_positions.astype(mx.int32))
            num_images_val = cast(int, num_images.item())

        # Replace token embeddings with image features if any were found
        if num_images_val > 0:
            seq_len = input_ids.shape[1]
            positions_int = image_positions.astype(mx.int32)
            # Use argsort to find indices: True (1) values end up at the end after sorting
            sorted_indices = mx.argsort(positions_int, axis=1)
            # Extract indices corresponding to image tokens (assumes batch size 1)
            image_indices = sorted_indices[0, seq_len - num_images_val :]

            # Perform in-place update (assumes bs=1 for inputs/features)
            inputs_embeds[0, image_indices] = image_features[0]

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        cache: list[BaseCache] | list[None] | None = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        # second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        # position_ids = kwargs.pop("position_ids", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        logits = self.language_model(None, cache=cache, inputs_embeds=inputs_embeds)
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
