"""The deterministic controller: observations + timer + history -> state, notices, alerts."""

from app.controller import Controller, Rules
from app.schemas import AlertType, CookingStatus as S, Phase, VisualObservation
from app.state import CookingSession

_seq = 0


def obs(clock, status, food="Fried egg"):
    global _seq
    _seq += 1
    return VisualObservation(food=food, status=status, frame_seq=_seq, frame_ts=clock(), observed_at=0.0)


def make(clock, **rules):
    s = CookingSession()
    s.start_session("test")
    return s, Controller(Rules(**rules), clock)


def test_identification_suggests_timer_and_waits_for_confirmation(clock):
    s, c = make(clock)
    fx = c.on_observation(s, obs(clock, S.APPEARS_UNDERCOOKED))
    assert s.phase == Phase.AWAITING_CONFIRMATION
    assert s.suggestion.seconds == 180 and s.timer is None  # a suggestion, not a started timer
    assert "Thumbs up" in fx.say[0] and "doesn't look cooked" in fx.say[0]


def test_auto_start_option_starts_timer(clock):
    s, c = make(clock, auto_start_timer=True)
    c.on_observation(s, obs(clock, S.APPEARS_UNDERCOOKED))
    assert s.phase == Phase.COOKING and s.timer.duration_s == 180


def test_unknown_food_keeps_trying_then_monitors(clock):
    s, c = make(clock, analyze_max_attempts=3)
    for _ in range(2):
        assert not c.on_observation(s, obs(clock, S.CANNOT_DETERMINE, food=None)).say
        assert s.phase == Phase.ANALYZING
    fx = c.on_observation(s, obs(clock, S.CANNOT_DETERMINE, food=None))
    assert s.phase == Phase.MONITORING and "can't see the food" in fx.say[0]


def test_food_without_typical_time_monitors_without_timer(clock):
    s, c = make(clock)
    c.on_observation(s, obs(clock, S.APPEARS_UNDERCOOKED, food="Mystery stew"))
    assert s.phase == Phase.MONITORING and s.suggestion is None


def test_overcook_needs_consecutive_confirmation(clock):
    s, c = make(clock, overcook_confirm=2)
    c.on_observation(s, obs(clock, S.APPEARS_COOKED))
    assert not c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts
    assert not c.on_observation(s, obs(clock, S.APPEARS_COOKED)).alerts  # one-off flicker reset
    assert not c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts
    fx = c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT))
    assert [a.type for a in fx.alerts] == [AlertType.OVERCOOK_WARNING]
    assert "burning" in fx.alerts[0].message


def test_overcook_not_duplicated_while_pending_and_repeats_after_ack(clock):
    s, c = make(clock, overcook_confirm=1, overcook_repeat_s=60)
    first = c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts[0]
    assert not c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts  # still pending
    s.ack(first.id)
    clock.advance(30)
    assert not c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts  # within repeat window
    clock.advance(31)
    again = c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT)).alerts
    assert again and again[0].id != first.id and "still" in again[0].message


def test_overcook_warning_is_independent_of_timer(clock):
    s, c = make(clock, overcook_confirm=1)
    s.start_timer(600, "Fried egg", clock)
    fx = c.on_observation(s, obs(clock, S.POSSIBLY_OVERCOOKED_OR_BURNT))
    assert fx.alerts[0].type == AlertType.OVERCOOK_WARNING
    assert s.phase == Phase.COOKING  # the timer keeps running; the warning is separate


def test_looks_done_early_is_a_single_notice_not_an_alert(clock):
    s, c = make(clock, looks_done_confirm=2)
    s.start_timer(600, "Chicken breast", clock)
    s.food = "Chicken breast"
    c.on_observation(s, obs(clock, S.APPEARS_COOKED, "Chicken breast"))
    fx = c.on_observation(s, obs(clock, S.APPEARS_COOKED, "Chicken breast"))
    assert fx.say and not fx.alerts
    assert "looks cooked" in fx.say[0] and "thermometer" in fx.say[0] and "165" in fx.say[0]
    assert not c.on_observation(s, obs(clock, S.APPEARS_COOKED, "Chicken breast")).say  # once only


def test_timer_done_reports_vision_without_claiming_safety(clock):
    s, c = make(clock)
    s.start_timer(60, "Chicken breast", clock)
    s.food = "Chicken breast"
    clock.advance(55)
    c.on_observation(s, obs(clock, S.APPEARS_UNDERCOOKED, "Chicken breast"))
    clock.advance(5)
    fx = c.on_timer_done(s)
    alert = fx.alerts[0]
    assert s.phase == Phase.DONE and alert.type == AlertType.TIMER_DONE
    assert "still looks undercooked" in alert.message and "thermometer" in alert.message
    assert "safe to eat" not in alert.message


def test_timer_done_ignores_stale_observation(clock):
    s, c = make(clock, observation_fresh_s=30)
    s.start_timer(600, "Rice", clock)
    c.on_observation(s, obs(clock, S.APPEARS_COOKED, "Rice"))
    clock.advance(600)
    msg = c.on_timer_done(s).alerts[0].message
    assert "looks cooked" not in msg  # 10 minutes old: don't quote it


def test_camera_lost_alert_and_recovery(clock):
    s, c = make(clock, camera_lost_after_s=20)
    assert not c.on_camera(s, False).alerts
    clock.advance(21)
    lost = c.on_camera(s, False).alerts
    assert lost and lost[0].type == AlertType.MONITORING_LOST and not s.vision_live
    assert not c.on_camera(s, False).alerts  # not duplicated
    fx = c.on_camera(s, True)
    assert s.vision_live and s.pending(AlertType.MONITORING_LOST) is None and fx.say


def test_food_name_majority_vote(clock):
    s, c = make(clock)
    for food in ["Pancake", "Pancake", "Omelette", "Pancake"]:
        c.on_observation(s, obs(clock, S.APPEARS_UNDERCOOKED, food))
    assert s.food == "Pancake"
