import logging
import os
import mlx.core as mx
import mlx.nn as nn

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

def load_template(name: str) -> str:
    """
    Load the chat template from the specified file.
    Looks in the current directory for the file.

    Args:
        name: The name of the template file.
        Defaults to "chat_template.jinja".

    Returns:
        The content of the template file.
    """
    name = name or "chat_template.jinja"
    name = f"{name}.jinja" if not name.endswith(".jinja") else name

    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, name)

    # Fall back to default template if specified one doesn't exist
    if not os.path.exists(template_path):
        template_path = os.path.join(current_dir, "chat_template.jinja")

    with open(template_path) as f:
        return f.read()
