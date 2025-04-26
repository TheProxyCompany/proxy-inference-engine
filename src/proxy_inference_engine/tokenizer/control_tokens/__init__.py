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

    @property
    def delimiters(self) -> dict[str, tuple[str, str]]:
        """Returns the delimiters for the control tokens."""
        return {
            "end_tokens": (self.end_of_sequence, self.end_of_message),
        }

    def end_tokens(self) -> list[str]:
        """Returns a list of tokens that indicate the end of a sequence.

        Returns:
            A list of end tokens.
        """
        return [self.end_of_sequence, self.end_of_message]

    def get_whitelist_control_tokens(self) -> list[str]:
        """Returns the control tokens used for tokenization.

        Returns:
            A list of the most essential control tokens.
        """
        tokens: list[str] = []
        for delim in self.delimiters.values():
            if delim:
                start, end = delim
                if start.strip():
                    tokens.append(start.strip())
                if end.strip():
                    tokens.append(end.strip())

        return list(set(tokens))

def get_control_tokens(tokenizer_config: dict) -> ControlTokens:
    """Get the control tokens for the model."""
    model_type = _determine_model_type(tokenizer_config)
    match model_type:
        case "gemma":
            return _load_control_tokens("gemma")
        case _: # default to chatml
            return _load_control_tokens("chatml")


def _determine_model_type(tokenizer_config: dict) -> str:
    """Determine the model type from the model path."""
    model_type = tokenizer_config.get("model_type", "chatml")

    eos_token = tokenizer_config.get("eos_token", "<|eot_id|>")

    if isinstance(eos_token, str) and eos_token.strip() == "<|im_end|>":
        model_type = "chatml"

    return model_type


def _load_control_tokens(model_type: str) -> ControlTokens:
    """Load the control tokens for the model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{model_type}.json")
    with open(file_path) as f:
        data = json.load(f)
        return ControlTokens(**data)
