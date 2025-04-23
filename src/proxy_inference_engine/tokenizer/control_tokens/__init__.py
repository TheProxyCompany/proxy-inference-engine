import json
import os

from pydantic import BaseModel


class Role(BaseModel):
    role_name: str
    role_start_tag: str
    role_end_tag: str
    end_of_message: str | None = None


class RoleTags(BaseModel):
    system: Role | None = None
    agent: Role | None = None
    user: Role | None = None
    tool: Role | None = None


class ControlTokens(BaseModel):
    """Control tokens for different model templates.

    This class defines the structure and access methods for control tokens used in
    various LLM template formats.
    """

    template_type: str
    begin_of_text: str
    end_of_message: str
    end_of_sequence: str


    roles: RoleTags

    def end_tokens(self) -> list[str]:
        """Returns a list of tokens that indicate the end of a sequence.

        Returns:
            A list of end tokens.
        """
        return [self.end_of_sequence, self.end_of_message]

def get_control_tokens(model_path: str, tokenizer_config: dict) -> ControlTokens:
    """Get the control tokens for the model."""
    model_type = _determine_model_type(model_path, tokenizer_config)
    match model_type:
        case "llama":
            return _load_control_tokens("llama")
        case "llama-deepseek":
            return _load_control_tokens("llama-deepseek")
        case "mistral":
            return _load_control_tokens("mistral")
        case "deepseek":
            return _load_control_tokens("deepseek")
        case "hermes":
            return _load_control_tokens("hermes")
        case _:
            return _load_control_tokens("chatml")


def _determine_model_type(model_path: str, tokenizer_config: dict) -> str:
    """Determine the model type from the model path."""
    model_type = tokenizer_config.get("model_type", "chatml")
    eos_token = tokenizer_config.get("eos_token", "<|eot_id|>")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content", "<|eot_id|>")

    if eos_token == "<|eot_id|>":
        model_type = "llama"
    elif eos_token == "</s>":
        model_type = "mistral"
    elif eos_token == "<｜end▁of▁sentence｜>":  # noqa: RUF001
        if tokenizer_config.get("tokenizer_class") == "LlamaTokenizerFast":
            model_type = "llama-deepseek"
        else:
            model_type = "deepseek"
    elif isinstance(eos_token, str) and eos_token.strip() == "<|im_end|>":
        model_type = "chatml"

    if "hermes" in model_path.lower():
        model_type = "hermes"

    return model_type


def _load_control_tokens(model_type: str) -> ControlTokens:
    """Load the control tokens for the model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{model_type}.json")
    with open(file_path) as f:
        data = json.load(f)
        return ControlTokens(**data)
