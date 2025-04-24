from typing import Annotated

from fastapi import Depends

from proxy_inference_engine.engine import InferenceEngine


async def get_inference_engine() -> InferenceEngine:
    raise NotImplementedError("Dependency provider not configured")

InferenceEngineDep = Annotated[InferenceEngine, Depends(get_inference_engine)]
