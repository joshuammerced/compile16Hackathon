"""Deterministic food-safety guardrail applied after the LLM.

USDA safe minimum internal temperatures (https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart):
poultry 165°F, ground meats 160°F, whole cuts of beef/pork/veal/lamb 145°F + 3 min rest,
fish & shellfish 145°F, egg dishes 160°F, leftovers & casseroles 165°F.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schemas import CookingDecision

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryRule:
    safe_temp_f: int | None
    min_s: int
    max_s: int
    rest_min: int = 0
    is_meat: bool = False


RULES: dict[str, CategoryRule] = {
    "poultry": CategoryRule(165, 5 * 60, 3 * 3600, is_meat=True),
    "ground_meat": CategoryRule(160, 3 * 60, 90 * 60, is_meat=True),
    "whole_cut_meat": CategoryRule(145, 2 * 60, 5 * 3600, rest_min=3, is_meat=True),
    "pork": CategoryRule(145, 3 * 60, 5 * 3600, rest_min=3, is_meat=True),
    "seafood": CategoryRule(145, 2 * 60, 60 * 60, is_meat=True),
    "eggs": CategoryRule(160, 60, 30 * 60),
    "vegetables": CategoryRule(None, 60, 2 * 3600),
    "grains_pasta": CategoryRule(None, 60, 75 * 60),
    "baked_goods": CategoryRule(None, 5 * 60, 4 * 3600),
    "other": CategoryRule(None, 30, 4 * 3600),
}
DEFAULT_RULE = CategoryRule(None, 30, 4 * 3600)

LEFTOVERS_TEMP_F = 165
LEFTOVER_WORDS = ("leftover", "casserole", "reheat")

THERMOMETER_REMINDER = "A timer is no substitute for a thermometer, so check it reaches {temp} degrees."


def rule_for(category: str | None) -> CategoryRule:
    return RULES.get(category or "", DEFAULT_RULE)


def is_leftovers(decision: CookingDecision) -> bool:
    text = " ".join(filter(None, [decision.food_name, decision.cooking_method])).lower()
    return any(word in text for word in LEFTOVER_WORDS)


def clamp_seconds(seconds: int, category: str | None) -> int:
    rule = rule_for(category)
    clamped = min(max(seconds, rule.min_s), rule.max_s)
    if clamped != seconds:
        log.warning(
            "food_safety: clamped cook time %ds -> %ds for category %s (bounds %d-%ds)",
            seconds,
            clamped,
            category,
            rule.min_s,
            rule.max_s,
        )
    return clamped


def apply(decision: CookingDecision) -> CookingDecision:
    """Return a copy of the decision with authoritative temps, clamped times and reminders."""
    rule = rule_for(decision.food_category)
    update: dict = {}

    safe_temp = rule.safe_temp_f
    if is_leftovers(decision):
        safe_temp = LEFTOVERS_TEMP_F
    if safe_temp is not None and decision.safe_internal_temp_f != safe_temp:
        if decision.safe_internal_temp_f is not None:
            log.warning(
                "food_safety: overriding safe temp %s°F -> %d°F for %s",
                decision.safe_internal_temp_f,
                safe_temp,
                decision.food_category,
            )
        update["safe_internal_temp_f"] = safe_temp

    if decision.action == "start_timer":
        if decision.recommended_cook_seconds is None or decision.recommended_cook_seconds <= 0:
            log.warning("food_safety: start_timer without a cook time; using category minimum")
            update["recommended_cook_seconds"] = rule.min_s
        else:
            clamped = clamp_seconds(decision.recommended_cook_seconds, decision.food_category)
            if clamped != decision.recommended_cook_seconds:
                update["recommended_cook_seconds"] = clamped

    if rule.rest_min and not decision.safety_notes:
        update["safety_notes"] = f"Rest {rule.rest_min} minutes before cutting."

    if rule.is_meat and safe_temp is not None and decision.action in ("start_timer", "answer_question"):
        reminder = THERMOMETER_REMINDER.format(temp=safe_temp)
        if "thermometer" not in decision.spoken_feedback.lower():
            update["spoken_feedback"] = f"{decision.spoken_feedback.rstrip()} {reminder}"

    return decision.model_copy(update=update) if update else decision
