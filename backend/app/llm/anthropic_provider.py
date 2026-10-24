"""Claude vision provider.

Structured output uses ``output_config.format`` (JSON schema) instead of forced tool use: forced
``tool_choice`` is rejected on Claude Sonnet 5.5. The router still validates with Pydantic and
retries once, so a schema mismatch is handled the same way for every provider.
"""

from __future__ import annotations

import base64
import logging

import anthropic

from ..schemas import decision_json_schema
from .base import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-sonnet-5-5",
        effort: str = "low",
        fallbacks: bool = True,
        timeout_s: float = 30.0,
    ):
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=1)
        self._schema = decision_json_schema()

    def _content(self, request: LLMRequest) -> list[dict]:
        content: list[dict] = []
        if request.image_jpeg:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(request.image_jpeg).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": request.user_text})
        if request.retry_feedback:
            content.append(
                {
                    "type": "text",
                    "text": "Your previous answer was rejected by the validator: "
                    f"{request.retry_feedback}\nReturn one corrected JSON object.",
                }
            )
        return content

    async def generate(self, request: LLMRequest) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=4000,
            system=request.system,
            messages=[{"role": "user", "content": self._content(request)}],
            # Latency matters at the stove; low effort is plenty for identify-and-time.
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self._schema}},
        )
        if self.fallbacks:
            # Route safety-classifier refusals to a fallback model server-side.
            response = await self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = await self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"model refused: {getattr(details, 'category', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output truncated (max_tokens)")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block in response (stop_reason={response.stop_reason})")
        log.debug("anthropic usage: %s", response.usage)
        return text
