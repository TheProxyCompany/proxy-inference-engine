import glob
import importlib
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.proxy_inference_engine.utils import sanitize_weights

logger = logging.getLogger(__name__)


def load(
    path_or_hf_repo: str,
) -> tuple[nn.Module, AutoTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)
    model, _ = load_model(model_path.as_posix())
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def load_model(model_path: str) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
    Returns:
        nn.Module: The loaded and initialized model.
    """
    path = get_model_path(model_path)
    config = load_config(path)
    weight_files = glob.glob(str(path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    architecture = get_model_architecture(config)
    model_args = architecture.ModelArgs.from_dict(config)
    model = architecture.Model(model_args)

    weights = sanitize_weights(model, weights, config)
    if hasattr(architecture, "LanguageModel"):
        weights = sanitize_weights(architecture.LanguageModel, weights, config)
    if hasattr(architecture, "VisionModel"):
        weights = sanitize_weights(architecture.VisionModel, weights, config)

    if (quantization := config.get("quantization", None)) is not None:
        nn.quantize(model, **quantization)

    model.load_weights(list(weights.items()))
    assert isinstance(model, nn.Module)
    mx.eval(model.parameters())
    model.eval()
    return model


def get_model_architecture(config: dict[str, Any]) -> ModuleType:
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = {
        "mistral": "llama",
        "phi-msft": "phixtral",
        "falcon_mamba": "mamba",
        "llama-deepseek": "llama",
    }.get(model_type, model_type)

    try:
        architecture = importlib.import_module(f"proxy_inference_engine.models.{model_type}")
        return architecture
    except ImportError:
        try:
            architecture = importlib.import_module(f"mlx_lm.models.{model_type}")
            return architecture
        except ImportError as e:
            msg = f"Model type {model_type} not supported."
            logging.error(msg)
            raise ValueError(
                "No model architecture found for the given model type."
            ) from e


def load_config(model_path: Path) -> dict:
    """
    Load the model configuration from the given path.

    Args:
        model_path (Path): The path to the model.

    Returns:
        dict: The model configuration.
    """
    with open(model_path / "config.json") as f:
        config = json.load(f)
        return config


def get_model_path(path_or_hf_repo: str, revision: str | None = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except Exception as e:
            raise ValueError(
                f"Model not found for path or HF repo: {path_or_hf_repo}."
            ) from e
    return model_path
