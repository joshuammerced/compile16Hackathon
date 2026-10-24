import json

import pytest

from app.llm.base import DecisionParseError, LLMProvider, LLMRequest, LLMRouter, parse_decision
from app.llm.mock_provider import MockProvider
from app.schemas import DISPLAY_TEXT_MAX, decision_json_schema

VALID = {
    "action": "start_timer",
    "food_name": "Salmon fillet",
    "food_category": "seafood",
    "cooking_method": "bake",
    "confidence": 0.8,
    "recommended_cook_seconds": 720,
    "adjust_seconds": None,
    "safe_internal_temp_f": 145,
    "safety_notes": None,
    "display_text": "Salmon",
    "spoken_feedback": "Salmon, about 12 minutes.",
}


def test_parse_valid():
    d = parse_decision(json.dumps(VALID))
    assert d.action == "start_timer" and d.recommended_cook_seconds == 720


def test_parse_tolerates_code_fence():
    d = parse_decision("```json\n" + json.dumps(VALID) + "\n```")
    assert d.food_name == "Salmon fillet"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{broken json",
        json.dumps({**VALID, "action": "fry_it"}),
        json.dumps({**VALID, "food_category": "dessert"}),
        json.dumps({**VALID, "confidence": 1.5}),
        json.dumps({k: v for k, v in VALID.items() if k != "spoken_feedback"}),
    ],
)
def test_parse_malformed(raw):
    with pytest.raises(DecisionParseError):
        parse_decision(raw)


def test_display_text_truncated():
    d = parse_decision(json.dumps({**VALID, "display_text": "A very long name for a small screen indeed"}))
    assert len(d.display_text) <= DISPLAY_TEXT_MAX


def test_schema_lists_every_field():
    schema = decision_json_schema()
    assert set(schema["required"]) == set(VALID)
    assert schema["additionalProperties"] is False


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests: list[LLMRequest] = []

    async def generate(self, request):
        self.requests.append(LLMRequest(**vars(request)))
        return self.outputs.pop(0)


async def test_router_retries_once_on_validation_failure():
    provider = ScriptedProvider(["{nope", json.dumps(VALID)])
    result = await LLMRouter(provider).decide(b"jpeg", None, "IDLE")
    assert result.decision.action == "start_timer"
    assert len(provider.requests) == 2
    assert provider.requests[1].retry_feedback  # error fed back on retry
    assert len(result.errors) == 1


async def test_router_falls_back_after_two_failures():
    provider = ScriptedProvider(["{nope", "still nope"])
    result = await LLMRouter(provider).decide(b"jpeg", None, "IDLE")
    assert result.decision.action == "ask_clarification"
    assert len(result.errors) == 2


async def test_router_low_confidence_becomes_clarification():
    provider = ScriptedProvider([json.dumps({**VALID, "confidence": 0.3})])
    result = await LLMRouter(provider, min_confidence=0.55).decide(b"jpeg", None, "IDLE")
    assert result.decision.action == "ask_clarification"
    assert result.decision.recommended_cook_seconds is None


async def test_router_applies_food_safety():
    provider = ScriptedProvider([json.dumps({**VALID, "safe_internal_temp_f": 120, "recommended_cook_seconds": 5})])
    d = (await LLMRouter(provider).decide(b"jpeg", None, "IDLE")).decision
    assert d.safe_internal_temp_f == 145
    assert d.recommended_cook_seconds >= 120


@pytest.mark.parametrize(
    "transcript,action,adjust",
    [
        ("add two minutes", "adjust_timer", 120),
        ("take off 30 seconds", "adjust_timer", -30),
        ("cancel", "cancel_timer", None),
        ("how long for pasta?", "answer_question", None),
    ],
)
async def test_mock_transcript_only_requests(transcript, action, adjust):
    router = LLMRouter(MockProvider(latency_s=0))
    d = (await router.decide(None, transcript, "COOKING", "Egg", 100)).decision
    assert d.action == action
    assert d.adjust_seconds == adjust


async def test_mock_no_frame_no_transcript():
    d = (await LLMRouter(MockProvider(latency_s=0)).decide(None, None, "IDLE")).decision
    assert d.action == "no_food_detected"
