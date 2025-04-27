from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field, model_serializer


class ChatCompletionLogProbs(BaseModel):
    """Represents the log probabilities of each token in the response."""

    class LogProbsContent(BaseModel):
        content: list[ChatCompletionLogProbs] | None = Field(
            default=None,
            description="A list of message content tokens with log probability information.",
        )

    bytes: list[int] | None = Field(
        default=None,
        description="A list of integers representing the UTF-8 bytes representation of the token.",
    )
    token: str = Field(description="The token.")
    logprob: float = Field(description="The log probability of this token.")
    top_logprobs: list[ChatCompletionLogProbs] | None = Field(
        default=None,
        description="The top log probabilities for this token.",
    )

    @model_serializer
    def serialize_model(self):
        result = {
            "token": self.token,
            "logprob": self.logprob,
            "bytes": self.bytes,
        }
        if self.top_logprobs:
            result["top_logprobs"] = self.top_logprobs
        return result

    @staticmethod
    def from_generation(
        tokens: list[int],
        logprobs: list[dict[int, float]] | None,
        decode_func: Callable[[int], str],
    ) -> ChatCompletionLogProbs.LogProbsContent | None:
        if logprobs is None or len(logprobs) == 0:
            return None

        logprobs_content = []
        for token_id, generated_logprobs in zip(tokens, logprobs, strict=True):
            item_map: dict[int, ChatCompletionLogProbs] = {}
            for possible_token_id, logprob_value in generated_logprobs.items():
                token_string = decode_func(possible_token_id)
                utf8_bytes = list(token_string.encode("utf-8"))
                if logprob_value == float("-inf"):
                    logprob_value = -9999.0

                item = ChatCompletionLogProbs(
                    bytes=utf8_bytes, token=token_string, logprob=logprob_value
                )
                item_map[possible_token_id] = item

            sampled_item = item_map.get(token_id)
            sampled_score = sampled_item.logprob if sampled_item else -9999.0
            sampled_token_string = decode_func(token_id)

            selected_item = ChatCompletionLogProbs(
                bytes=list(sampled_token_string.encode("utf-8")),
                token=sampled_token_string,
                logprob=sampled_score,
            )

            if len(item_map) > 1:
                top_logprobs = list(item_map.values())
                top_logprobs.sort(key=lambda x: x.logprob, reverse=True)
                selected_item.top_logprobs = top_logprobs

            logprobs_content.append(selected_item)

        return ChatCompletionLogProbs.LogProbsContent(content=logprobs_content)
