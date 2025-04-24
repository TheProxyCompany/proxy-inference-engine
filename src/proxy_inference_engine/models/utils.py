import glob
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

@dataclass
class LargeLanguageModel:
    model: nn.Module
    hf_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    tokenizer_config: dict[str, Any]


def load(path_or_hf_repo: str) -> LargeLanguageModel:
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
    model, tokenizer_config = load_model(model_path.as_posix())
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    return LargeLanguageModel(
        model=model,
        hf_tokenizer=hf_tokenizer,
        tokenizer_config=tokenizer_config,
    )


def load_model(model_path: str) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
    Returns:
        nn.Module: The loaded and initialized model.
    """
    path = get_model_path(model_path)

    weight_files = glob.glob(str(path / "model*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    with open(path / "config.json") as f:
        model_config = json.load(f)

    architecture = get_model_architecture(model_config)
    if issubclass(architecture.ModelArgs, BaseModel):
        model_args = architecture.ModelArgs(**model_config)
    else:
        model_args = architecture.ModelArgs.from_dict(model_config)
    model: nn.Module = architecture.Model(model_args)

    if hasattr(model, "sanitize") and model.sanitize is not None:
        weights = model.sanitize(weights)

    if (
        hasattr(model, "language_model")
        and model.language_model is not None
        and hasattr(model.language_model, "sanitize")
        and model.language_model.sanitize is not None
    ):
        weights = model.language_model.sanitize(weights)
    if (
        hasattr(model, "vision_tower")
        and model.vision_tower is not None
        and hasattr(model.vision_tower, "sanitize")
        and model.vision_tower.sanitize is not None
    ):
        weights = model.vision_tower.sanitize(weights)

    # Quantization
    quantization: dict[str, Any] = model_config.get("quantization", {})
    if quantization:

        def should_quantize(path: str, module: nn.Module) -> bool:
            valid_weights = (
                hasattr(module, "weight")
                and module.weight is not None
                and module.weight.shape[-1] % 64 == 0
            )
            return (
                hasattr(module, "to_quantized")
                and valid_weights
                and f"{path}.scales" in weights
            )

        nn.quantize(model, **quantization, class_predicate=should_quantize)

    model.load_weights(list(weights.items()))
    assert isinstance(model, nn.Module)
    mx.eval(model.parameters())
    model.eval()

    try:
        with open(path / "tokenizer_config.json") as f:
            tokenizer_config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Tokenizer config not found for model at {path}")
        tokenizer_config = {}

    return model, tokenizer_config


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
        "gemma3": "gemma",
        "mistral": "llama",
        "phi-msft": "phixtral",
        "falcon_mamba": "mamba",
        "llama-deepseek": "llama",
    }.get(model_type, model_type)

    try:
        architecture = importlib.import_module(
            f"proxy_inference_engine.models.{model_type}"
        )
        return architecture
    except ModuleNotFoundError:
        try:
            architecture = importlib.import_module(f"mlx_lm.models.{model_type}")
            return architecture
        except ModuleNotFoundError as e:
            msg = f"Model type {model_type} not supported."
            logging.error(msg)
            raise ValueError(
                "No model architecture found for the given model type."
            ) from e


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


def sanitize_weights(
    model_obj: nn.Module,
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        assert model_obj.sanitize is not None
        weights = model_obj.sanitize(weights)

    return weights
