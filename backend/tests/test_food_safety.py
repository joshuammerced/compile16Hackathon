from app.llm import food_safety
from app.schemas import CookingDecision


def decision(**kw) -> CookingDecision:
    base = dict(
        action="start_timer",
        food_name="Chicken breast",
        food_category="poultry",
        cooking_method="pan-fry",
        confidence=0.9,
        recommended_cook_seconds=840,
        safe_internal_temp_f=None,
        display_text="Chicken breast",
        spoken_feedback="About 14 minutes.",
    )
    base.update(kw)
    return CookingDecision(**base)


def test_overrides_safe_temp_for_known_category():
    d = food_safety.apply(decision(safe_internal_temp_f=150))
    assert d.safe_internal_temp_f == 165


def test_category_temps():
    expected = {"ground_meat": 160, "whole_cut_meat": 145, "pork": 145, "seafood": 145, "eggs": 160}
    for cat, temp in expected.items():
        assert food_safety.apply(decision(food_category=cat)).safe_internal_temp_f == temp


def test_leftovers_165():
    d = food_safety.apply(decision(food_name="Leftover rice", food_category="grains_pasta", recommended_cook_seconds=180))
    assert d.safe_internal_temp_f == 165


def test_clamps_too_long_and_too_short(caplog):
    caplog.set_level("WARNING")
    d = food_safety.apply(decision(recommended_cook_seconds=10))
    assert d.recommended_cook_seconds == food_safety.RULES["poultry"].min_s
    d = food_safety.apply(decision(food_category="eggs", recommended_cook_seconds=99999))
    assert d.recommended_cook_seconds == food_safety.RULES["eggs"].max_s
    assert "clamped" in caplog.text


def test_in_bounds_unchanged():
    assert food_safety.apply(decision()).recommended_cook_seconds == 840


def test_missing_time_gets_minimum():
    d = food_safety.apply(decision(food_category="vegetables", recommended_cook_seconds=None))
    assert d.recommended_cook_seconds == food_safety.RULES["vegetables"].min_s


def test_thermometer_reminder_for_meat_only():
    meat = food_safety.apply(decision())
    assert "thermometer" in meat.spoken_feedback and "165" in meat.spoken_feedback
    veg = food_safety.apply(decision(food_name="Broccoli", food_category="vegetables", recommended_cook_seconds=300))
    assert "thermometer" not in veg.spoken_feedback
    assert veg.safe_internal_temp_f is None


def test_rest_note_for_whole_cuts():
    d = food_safety.apply(decision(food_name="Steak", food_category="whole_cut_meat", recommended_cook_seconds=600))
    assert "Rest 3 minutes" in d.safety_notes
