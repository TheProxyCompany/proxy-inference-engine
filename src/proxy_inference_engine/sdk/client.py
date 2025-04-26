from proxy_inference_engine.engine.inference_engine import InferenceEngine
from proxy_inference_engine.interaction.interaction import Interaction
from proxy_inference_engine.sdk.types import GenerationRequest


class InferenceEngineClient:
    """
    A client for the proxy inference engine.
    """

    def __init__(self, model_path: str):
        """
        Initialize the inference engine client.
        """
        self.model_path = model_path
        self.engine = InferenceEngine(model_path)

    async def generate(self, request: GenerationRequest) -> Interaction:
        """
        Generate a chat completion.
        """
        engine_request = request.to_interactions()
        engine_result = await self.engine(
            engine_request,
            **(request.generation_kwargs or {}),
        )
        return engine_result
