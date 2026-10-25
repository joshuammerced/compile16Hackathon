from __future__ import annotations

from ..config import Settings
from .base import VisionProvider


def create_vision_provider(settings: Settings) -> VisionProvider:
    """No silent fallback: if the real model can't load, monitoring reports itself unavailable
    rather than inventing observations about food on a hot stove."""
    if settings.vision_provider == "mock":
        from .mock_provider import MockVisionProvider

        return MockVisionProvider()
    from .qwen_provider import QwenVLMProvider

    return QwenVLMProvider(settings.vlm_inference_path, settings.vlm_model_id, settings.vlm_max_tokens)
