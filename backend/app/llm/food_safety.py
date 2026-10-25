"""Deterministic food-safety knowledge: USDA temperatures, time bounds, typical cook times.

USDA safe minimum internal temperatures (https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart):
poultry 165°F, ground meats 160°F, whole cuts of beef/pork/veal/lamb 145°F + 3 min rest,
fish & shellfish 145°F, egg dishes 160°F, leftovers & casseroles 165°F.

Typical times are *suggestions* for a reminder timer. Neither a timer nor a camera can tell
that food is safe; only a thermometer can, which is what the spoken reminders say.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schemas import TimerSuggestion

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

# (keywords, category, typical stovetop/oven seconds). First match wins, so specific entries
# come before generic ones ("boiled egg" before "egg", "ground beef" before "beef").
FOOD_TABLE: list[tuple[tuple[str, ...], str, int]] = [
    (("boiled egg", "hard-boiled", "hard boiled"), "eggs", 10 * 60),
    (("fried rice",), "grains_pasta", 8 * 60),
    (("chicken breast",), "poultry", 14 * 60),
    (("chicken thigh", "drumstick", "chicken wing"), "poultry", 20 * 60),
    (("chicken", "turkey", "duck", "poultry"), "poultry", 15 * 60),
    (("ground beef", "ground pork", "ground turkey", "burger", "patty", "meatball", "sausage", "mince"),
     "ground_meat", 10 * 60),
    (("bacon",), "pork", 8 * 60),
    (("pork chop", "pork", "ham"), "pork", 10 * 60),
    (("steak", "beef", "lamb", "veal", "roast"), "whole_cut_meat", 8 * 60),
    (("salmon", "fish", "cod", "tuna", "tilapia", "shrimp", "prawn", "scallop", "seafood"), "seafood", 8 * 60),
    (("omelet", "omelette", "scrambled", "fried egg", "egg"), "eggs", 3 * 60),
    (("spaghetti", "pasta", "noodle", "macaroni", "penne", "linguine"), "grains_pasta", 10 * 60),
    (("rice",), "grains_pasta", 18 * 60),
    (("oatmeal", "porridge", "oats"), "grains_pasta", 5 * 60),
    (("pancake", "crepe"), "baked_goods", 4 * 60),
    (("cookie",), "baked_goods", 11 * 60),
    (("cake", "muffin", "bread"), "baked_goods", 30 * 60),
    (("potato",), "vegetables", 20 * 60),
    (("broccoli", "carrot", "vegetable", "onion", "pepper", "spinach", "mushroom", "zucchini", "green bean"),
     "vegetables", 8 * 60),
    (("tofu",), "other", 8 * 60),
]  # fmt: skip


@dataclass(frozen=True)
class FoodInfo:
    name: str
    category: str | None
    typical_s: int | None
    safe_temp_f: int | None
    rest_min: int
    is_meat: bool


def rule_for(category: str | None) -> CategoryRule:
    return RULES.get(category or "", DEFAULT_RULE)


def classify(food_name: str | None) -> FoodInfo:
    name = (food_name or "").strip()
    lower = name.lower()
    category, typical = None, None
    for words, cat, secs in FOOD_TABLE:
        if any(w in lower for w in words):
            category, typical = cat, secs
            break
    rule = rule_for(category)
    temp = LEFTOVERS_TEMP_F if any(w in lower for w in LEFTOVER_WORDS) else rule.safe_temp_f
    return FoodInfo(name, category, typical, temp, rule.rest_min, rule.is_meat)


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


def suggest_timer(food_name: str | None) -> TimerSuggestion | None:
    info = classify(food_name)
    if not info.name or info.typical_s is None:
        return None
    return TimerSuggestion(
        food=info.name,
        seconds=clamp_seconds(info.typical_s, info.category),
        category=info.category,
        safe_temp_f=info.safe_temp_f,
    )


def thermometer_reminder(info: FoodInfo) -> str | None:
    """Spoken reminder for foods where doneness is a safety question."""
    if info.safe_temp_f is None:
        return None
    if info.category == "eggs" and info.safe_temp_f != LEFTOVERS_TEMP_F:
        return f"Cook eggs until the white and yolk are firm; egg dishes should reach {info.safe_temp_f} degrees."
    rest = f", then rest it {info.rest_min} minutes" if info.rest_min else ""
    return (
        "A timer or camera can't tell if it's safe. Use a thermometer: "
        f"it should reach {info.safe_temp_f} degrees{rest}."
    )
