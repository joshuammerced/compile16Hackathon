"""Mock LLM that returns canned decisions, so the whole system runs without an API key.

It understands a few transcript intents (cancel, add/remove time, "how long for X") and
otherwise cycles through canned food identifications for gesture-triggered analyses.
"""

from __future__ import annotations

import asyncio
import json
import re

from .base import LLMProvider, LLMRequest

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60,
}  # fmt: skip

CANNED_FOODS = [
    {
        "food_name": "Chicken breast",
        "food_category": "poultry",
        "cooking_method": "pan-fry",
        "recommended_cook_seconds": 14 * 60,
        "safe_internal_temp_f": 165,
        "spoken_feedback": "That looks like chicken breast in a pan. About 14 minutes, flipping halfway.",
    },
    {
        "food_name": "Fried egg",
        "food_category": "eggs",
        "cooking_method": "pan-fry",
        "recommended_cook_seconds": 3 * 60,
        "safe_internal_temp_f": 160,
        "spoken_feedback": "A fried egg. About 3 minutes, until the white is set and the yolk firms up.",
    },
    {
        "food_name": "Spaghetti",
        "food_category": "grains_pasta",
        "cooking_method": "boil",
        "recommended_cook_seconds": 10 * 60,
        "safe_internal_temp_f": None,
        "spoken_feedback": "Spaghetti in boiling water. About 10 minutes for al dente.",
    },
]

QUESTION_ANSWERS = {
    "pasta": "Most dried pasta takes 8 to 12 minutes in boiling salted water.",
    "rice": "White rice takes about 18 minutes covered on low heat, then rest 5 minutes.",
    "egg": "Hard boiled eggs take about 10 to 12 minutes.",
    "chicken": "Chicken breast takes about 12 to 15 minutes in a pan. It's safe at 165 degrees.",
}


def parse_seconds(text: str) -> int | None:
    """'add two minutes' -> 120, 'take off 30 seconds' -> 30 (unsigned)."""
    m = re.search(r"(\d+|[a-z-]+)\s*(minute|min|second|sec)s?\b", text)
    if not m:
        return None
    word, unit = m.group(1), m.group(2)
    n = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
    if n is None:
        return None
    return n * 60 if unit.startswith("min") else n


def base(**overrides) -> dict:
    d = {
        "action": "answer_question",
        "food_name": None,
        "food_category": None,
        "cooking_method": None,
        "confidence": 0.9,
        "recommended_cook_seconds": None,
        "adjust_seconds": None,
        "safe_internal_temp_f": None,
        "safety_notes": None,
        "display_text": "",
        "spoken_feedback": "",
    }
    d.update(overrides)
    return d


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, latency_s: float = 0.2):
        self.latency_s = latency_s
        self._next_food = 0

    def decide(self, request: LLMRequest) -> dict:
        text = (request.transcript or "").lower().strip()

        if text:
            if re.search(r"\b(cancel|stop|nevermind|never mind)\b", text):
                return base(action="cancel_timer", display_text="Cancelled", spoken_feedback="Okay, timer cancelled.")
            secs = parse_seconds(text)
            if secs and re.search(r"\b(add|plus|more|extra|another)\b", text):
                return base(action="adjust_timer", adjust_seconds=secs, display_text=f"+{secs // 60 or secs}",
                            spoken_feedback=f"Added {self._say(secs)}.")  # fmt: skip
            if secs and re.search(r"\b(remove|minus|less|take off|subtract)\b", text):
                return base(action="adjust_timer", adjust_seconds=-secs, display_text=f"-{secs // 60 or secs}",
                            spoken_feedback=f"Took off {self._say(secs)}.")  # fmt: skip
            if "how long" in text or text.endswith("?"):
                for key, answer in QUESTION_ANSWERS.items():
                    if key in text:
                        return base(action="answer_question", display_text=key.title(), spoken_feedback=answer)
                return base(action="ask_clarification", confidence=0.3, display_text="Which food?",
                            spoken_feedback="Which food are you asking about?")  # fmt: skip

        if request.image_jpeg is None and not text:
            return base(action="no_food_detected", confidence=0.9, display_text="No food seen",
                        spoken_feedback="I can't see any food. Hold it up to the camera.")  # fmt: skip

        food = CANNED_FOODS[self._next_food % len(CANNED_FOODS)]
        self._next_food += 1
        return base(action="start_timer", confidence=0.85, display_text=food["food_name"], **food)

    @staticmethod
    def _say(secs: int) -> str:
        if secs % 60 == 0:
            m = secs // 60
            return f"{m} minute{'s' if m != 1 else ''}"
        return f"{secs} seconds"

    async def generate(self, request: LLMRequest) -> str:
        await asyncio.sleep(self.latency_s)
        return json.dumps(self.decide(request))
