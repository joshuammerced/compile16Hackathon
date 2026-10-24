from app.llm import food_safety


def test_category_temps():
    expected = {
        "Chicken breast": 165,
        "Ground beef patty": 160,
        "Ribeye steak": 145,
        "Pork chop": 145,
        "Salmon fillet": 145,
        "Scrambled eggs": 160,
    }
    for food, temp in expected.items():
        assert food_safety.classify(food).safe_temp_f == temp, food


def test_leftovers_165():
    assert food_safety.classify("Leftover rice").safe_temp_f == 165


def test_specific_entries_win():
    assert food_safety.classify("Hard boiled egg").typical_s == 10 * 60
    assert food_safety.classify("Fried egg").typical_s == 3 * 60
    assert food_safety.classify("Ground beef").category == "ground_meat"


def test_clamps_too_long_and_too_short(caplog):
    caplog.set_level("WARNING")
    assert food_safety.clamp_seconds(10, "poultry") == food_safety.RULES["poultry"].min_s
    assert food_safety.clamp_seconds(99999, "eggs") == food_safety.RULES["eggs"].max_s
    assert food_safety.clamp_seconds(600, "eggs") == 600
    assert "clamped" in caplog.text


def test_suggest_timer_is_table_driven():
    sug = food_safety.suggest_timer("Spaghetti")
    assert sug.seconds == 10 * 60 and sug.category == "grains_pasta" and sug.safe_temp_f is None
    assert food_safety.suggest_timer("Mystery stew") is None
    assert food_safety.suggest_timer(None) is None


def test_thermometer_reminder_for_meat_only():
    meat = food_safety.thermometer_reminder(food_safety.classify("Chicken thigh"))
    assert "thermometer" in meat and "165" in meat
    steak = food_safety.thermometer_reminder(food_safety.classify("Steak"))
    assert "rest it 3 minutes" in steak
    assert food_safety.thermometer_reminder(food_safety.classify("Broccoli")) is None
    egg = food_safety.thermometer_reminder(food_safety.classify("Fried egg"))
    assert "firm" in egg and "160" in egg
