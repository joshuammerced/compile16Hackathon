from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMProvider

log = logging.getLogger(__name__)


def create_provider(settings: Settings) -> LLMProvider:
    from .mock_provider import MockProvider

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            log.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set; using mock LLM")
            return MockProvider()
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            effort=settings.llm_effort,
            fallbacks=settings.llm_fallbacks,
            timeout_s=settings.llm_timeout_s,
        )
    return MockProvider()
