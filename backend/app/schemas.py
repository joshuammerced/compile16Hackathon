"""Pydantic models: the LLM decision schema and the HTTP API models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DISPLAY_TEXT_MAX = 24

FoodCategory = Literal[
    "poultry",
    "ground_meat",
    "whole_cut_meat",
    "pork",
    "seafood",
    "eggs",
    "vegetables",
    "grains_pasta",
    "baked_goods",
    "other",
]

DecisionAction = Literal[
    "start_timer",
    "ask_clarification",
    "no_food_detected",
    "cancel_timer",
    "adjust_timer",
    "answer_question",
]


class CookingDecision(BaseModel):
    action: DecisionAction
    food_name: str | None = None
    food_category: FoodCategory | None = None
    cooking_method: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_cook_seconds: int | None = None
    adjust_seconds: int | None = None
    safe_internal_temp_f: int | None = None
    safety_notes: str | None = None
    display_text: str
    spoken_feedback: str

    @field_validator("display_text")
    @classmethod
    def _truncate_display_text(cls, v: str) -> str:
        # The BOX screen fits ~24 characters in the big font; truncate rather than reject.
        # Plain ASCII only, since microcontroller fonts rarely include an ellipsis glyph.
        v = " ".join(v.split())
        return v[:DISPLAY_TEXT_MAX].rstrip()


def decision_json_schema() -> dict:
    """JSON schema for structured outputs.

    Written by hand (rather than CookingDecision.model_json_schema()) so it stays within the
    subset structured outputs accepts: every property required, no additionalProperties, and
    nullable fields expressed as anyOf with null. Numeric bounds are enforced by Pydantic.
    """

    def nullable(schema: dict) -> dict:
        return {"anyOf": [schema, {"type": "null"}]}

    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(DecisionAction.__args__)},
            "food_name": nullable({"type": "string"}),
            "food_category": nullable({"type": "string", "enum": list(FoodCategory.__args__)}),
            "cooking_method": nullable({"type": "string"}),
            "confidence": {"type": "number"},
            "recommended_cook_seconds": nullable({"type": "integer"}),
            "adjust_seconds": nullable({"type": "integer"}),
            "safe_internal_temp_f": nullable({"type": "integer"}),
            "safety_notes": nullable({"type": "string"}),
            "display_text": {"type": "string"},
            "spoken_feedback": {"type": "string"},
        },
        "required": list(CookingDecision.model_fields.keys()),
        "additionalProperties": False,
    }


class Phase(str, Enum):
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    COOKING = "COOKING"
    DONE = "DONE"
    ERROR = "ERROR"


class DeviceState(BaseModel):
    """Compact state the ESP32-S3-BOX-3 polls. Keep it small."""

    version: int
    state: Phase
    display_text: str
    remaining_s: int
    total_s: int
    audio_id: str | None
    alert: bool
    paused: bool = False
    busy: bool = False


class DeviceStateUnchanged(BaseModel):
    changed: Literal[False] = False
    version: int
    remaining_s: int


class ButtonRequest(BaseModel):
    button: Literal["ack", "cancel", "pause", "resume", "plus_1m", "minus_1m"]


class AnalyzeRequest(BaseModel):
    transcript: str | None = None


class TimerControlRequest(BaseModel):
    action: Literal["start", "cancel", "adjust", "pause", "resume"]
    seconds: int | None = None
    label: str | None = None
