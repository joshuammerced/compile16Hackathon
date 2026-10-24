"""LLM provider interface and the router that turns raw model output into a CookingDecision."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..schemas import CookingDecision
from . import food_safety
from .prompts import build_system_prompt, build_user_prompt

log = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    system: str
    user_text: str
    image_jpeg: bytes | None = None
    transcript: str | None = None
    # Set on the retry: the validation error from the first attempt.
    retry_feedback: str | None = None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(self, request: LLMRequest) -> str:
        """Return the model's raw output: a JSON object matching CookingDecision."""


class DecisionParseError(ValueError):
    pass


def parse_decision(raw: str) -> CookingDecision:
    """Parse raw model text into a CookingDecision. Tolerates ```json fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise DecisionParseError("no JSON object in model output")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise DecisionParseError(f"invalid JSON: {exc}") from exc
    try:
        return CookingDecision.model_validate(data)
    except ValidationError as exc:
        raise DecisionParseError(f"schema validation failed: {exc}") from exc


def fallback_decision(reason: str) -> CookingDecision:
    return CookingDecision(
        action="ask_clarification",
        confidence=0.0,
        display_text="Say that again?",
        spoken_feedback="Sorry, I didn't catch that. Can you tell me what you're cooking?",
        safety_notes=reason,
    )


def apply_confidence_gate(decision: CookingDecision, min_confidence: float) -> CookingDecision:
    """Downgrade a low-confidence timer start to a clarification question."""
    if decision.action != "start_timer" or decision.confidence >= min_confidence:
        return decision
    log.info(
        "confidence %.2f < %.2f for %r: asking for clarification instead",
        decision.confidence,
        min_confidence,
        decision.food_name,
    )
    guess = f" It looks like {decision.food_name}, is that right?" if decision.food_name else ""
    return decision.model_copy(
        update={
            "action": "ask_clarification",
            "recommended_cook_seconds": None,
            "display_text": "What is it?",
            "spoken_feedback": "I'm not sure what that is." + guess + " Tell me what you're cooking.",
        }
    )


@dataclass
class RouterResult:
    decision: CookingDecision
    raw_outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    provider: str = ""

    def as_debug(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "latency_s": round(self.latency_s, 3),
            "raw_outputs": self.raw_outputs,
            "errors": self.errors,
            "decision": self.decision.model_dump(),
        }


class LLMRouter:
    """Builds the prompt, calls the provider, validates (retrying once), applies guardrails."""

    def __init__(self, provider: LLMProvider, min_confidence: float = 0.55):
        self.provider = provider
        self.min_confidence = min_confidence
        self.system = build_system_prompt(min_confidence)

    async def decide(
        self,
        image_jpeg: bytes | None,
        transcript: str | None,
        state: str,
        timer_label: str | None = None,
        remaining_s: int = 0,
    ) -> RouterResult:
        request = LLMRequest(
            system=self.system,
            user_text=build_user_prompt(transcript, state, timer_label, remaining_s, image_jpeg is not None),
            image_jpeg=image_jpeg,
            transcript=transcript,
        )
        result = RouterResult(decision=fallback_decision("unset"), provider=self.provider.name)
        started = time.perf_counter()
        decision: CookingDecision | None = None
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                raw = await self.provider.generate(request)
            except Exception as exc:  # noqa: BLE001 - provider/network errors become a fallback
                log.exception("LLM call failed (attempt %d)", attempt)
                result.errors.append(f"provider error: {exc}")
                break
            log.info("LLM call %s attempt %d took %.2fs", self.provider.name, attempt, time.perf_counter() - t0)
            result.raw_outputs.append(raw)
            try:
                decision = parse_decision(raw)
                break
            except DecisionParseError as exc:
                log.warning("LLM output failed validation (attempt %d): %s", attempt, exc)
                result.errors.append(str(exc))
                request.retry_feedback = str(exc)

        result.latency_s = time.perf_counter() - started
        if decision is None:
            result.decision = fallback_decision("; ".join(result.errors) or "no output")
            return result
        decision = apply_confidence_gate(decision, self.min_confidence)
        result.decision = food_safety.apply(decision)
        log.info(
            "decision: action=%s food=%r category=%s secs=%s conf=%.2f (%.2fs total)",
            result.decision.action,
            result.decision.food_name,
            result.decision.food_category,
            result.decision.recommended_cook_seconds,
            result.decision.confidence,
            result.latency_s,
        )
        return result
