import logging

import mlx.core as mx
import mlx.nn as nn
from PIL import Image
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.proxy_inference_engine.vision.utils import BaseImageProcessor, process_image

logger = logging.getLogger(__name__)

def sanitize_weights(model_obj: nn.Module, weights: dict[str, mx.array], config=None) -> dict[str, mx.array]:
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        assert model_obj.sanitize is not None
        weights = model_obj.sanitize(weights)

    return weights

def set_max_reccomended_device_limit():
    """
    Set the max recommended device limit.
    """
    device_info = mx.metal.device_info()
    safe_max_size = device_info["max_recommended_working_set_size"]
    if isinstance(safe_max_size, int):
        mx.synchronize()
        mx.set_wired_limit(safe_max_size)
        max_rec_gb = safe_max_size / 2**30
        logger.info(f"Set wired memory limit to {max_rec_gb:.2f}GB")
    else:
        logger.warning(f"Max recommended size is not an integer: {safe_max_size}")

def prepare_inputs(
    prompt: str | list[str],
    images: list[Image.Image | str],
    tokenizer: AutoTokenizer,
    image_processor: BaseImageProcessor | None,
    resize_shape: tuple[int, int] | None = None,
) -> dict[str, mx.array]:
    """
    Prepare inputs for the model.

    Args:
        prompt: The text prompt or list of prompts
        images: List of images or image paths
        tokenizer: The tokenizer to use
        image_processor: Optional image processor
        resize_shape: Optional shape to resize images to

    Returns:
        Dictionary of model inputs
    """
    # Process images if image processor is provided
    processed_images = [process_image(img, resize_shape) for img in images]
    if image_processor is not None:
        model_inputs = {
            "input_ids": mx.array(input_ids),
            "pixel_values": mx.stack(
                image_processor.preprocess(images=processed_images)
            ),
            "attention_mask": mx.array(
                [(ids != tokenizer.pad_token_id) for ids in input_ids]
            ).astype(mx.int32)
        }
    else:
        # Use the process_inputs function for more complex processing
        if "images" in inputs:
            inputs["pixel_values"] = inputs["images"]
            inputs.pop("images")

        if isinstance(inputs.get("pixel_values", None), list):
            pixel_values = inputs["pixel_values"]
        else:
            pixel_values = mx.array(inputs["pixel_values"])

        model_inputs = {}
        if "input_ids" in inputs:
            model_inputs["input_ids"] = mx.array(inputs["input_ids"])
        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = pixel_values
        if mask := inputs.get("attention_mask"):
            model_inputs["attention_mask"] = mx.array(mask)
        else:
            model_inputs["attention_mask"] = None

        # Convert remaining inputs to model_inputs with mx.array if present
        for key, value in inputs.items():
            if key not in model_inputs and not isinstance(value, (str, list)):
                model_inputs[key] = mx.array(value)

    return model_inputs
