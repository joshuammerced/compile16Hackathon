"""Deterministic cooking controller.

Combines visual observations (what the food looks like), the timer (what the user asked for),
and session history into state changes, spoken notices and alerts. No model output is ever
spoken directly and nothing here concludes that food is safe:

* An overcook warning comes only from vision, needs several consecutive "burnt" frames, and is
  an alert of its own, independent of the timer.
* Timer completion (TIMER_DONE) only means the countdown ended. Its message reports what the
  camera last saw and reminds the cook to check with a thermometer.
* "Looks cooked" before the timer ends is a one-off spoken notice, not an alert.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from .llm import food_safety
from .schemas import Alert, AlertType, CookingStatus, Phase, VisualObservation, display_text
from .state import CookingSession, Clock

log = logging.getLogger(__name__)

S = CookingStatus


@dataclass(frozen=True)
class Rules:
    overcook_confirm: int = 2  # consecutive POSSIBLY_OVERCOOKED_OR_BURNT observations
    overcook_repeat_s: float = 60.0  # re-warn if still burning this long after the last warning
    looks_done_confirm: int = 2  # consecutive APPEARS_COOKED before the early notice
    looks_done_min_remaining_s: int = 60
    analyze_max_attempts: int = 3  # unreadable frames before giving up on identifying the food
    camera_lost_after_s: float = 20.0
    observation_fresh_s: float = 30.0  # how recent an observation must be to be quoted
    auto_start_timer: bool = False


@dataclass
class Effects:
    """What the orchestrator should do after the lock is released."""

    say: list[str] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)


def fmt_duration(seconds: int) -> str:
    """Spoken duration: 90 -> '1 minute 30 seconds'."""
    m, s = divmod(int(seconds), 60)
    parts = []
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s or not m:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts)


def trailing(observations, status: CookingStatus) -> int:
    n = 0
    for obs in reversed(observations):
        if obs.status != status:
            break
        n += 1
    return n


class Controller:
    def __init__(self, rules: Rules, clock: Clock):
        self.rules = rules
        self.clock = clock

    # --- helpers --------------------------------------------------------------------

    def food_name(self, s: CookingSession) -> str:
        return s.food or (s.timer.label if s.timer else None) or "food"

    def fresh_observation(self, s: CookingSession) -> VisualObservation | None:
        if not s.observations:
            return None
        obs = s.observations[-1]
        return obs if self.clock() - obs.frame_ts <= self.rules.observation_fresh_s else None

    @staticmethod
    def _vote_food(s: CookingSession) -> str | None:
        names = [o.food for o in list(s.observations)[-5:] if o.food]
        return Counter(names).most_common(1)[0][0] if names else None

    # --- observations -----------------------------------------------------------------

    def on_observation(self, s: CookingSession, obs: VisualObservation) -> Effects:
        fx = Effects()
        s.observations.append(obs)
        voted = self._vote_food(s)
        if voted and voted != s.food:
            log.info("food identified as %r (was %r)", voted, s.food)
            s.food = voted
            if s.phase == Phase.MONITORING:
                s.display_text = display_text(voted)

        if s.phase == Phase.ANALYZING:
            self._identify(s, obs, fx)
        self._check_overcook(s, fx)
        self._check_looks_done(s, fx)
        return fx

    def _identify(self, s: CookingSession, obs: VisualObservation, fx: Effects) -> None:
        s.analyze_attempts += 1
        if obs.food is None:
            if s.analyze_attempts < self.rules.analyze_max_attempts:
                return  # try the next frame
            s.display_text = "Can't see food"
            s.transition(Phase.MONITORING, "food not identified")
            fx.say.append(
                "I can't see the food clearly. I'll keep watching. "
                "You can also say, set a timer for 5 minutes."
            )
            return
        food = s.food or obs.food
        suggestion = food_safety.suggest_timer(food)
        looks = {
            S.APPEARS_UNDERCOOKED: " It doesn't look cooked yet.",
            S.APPEARS_COOKED: " It already looks cooked.",
            S.POSSIBLY_OVERCOOKED_OR_BURNT: "",  # the overcook check speaks for itself
            S.CANNOT_DETERMINE: "",
        }[obs.status]
        if suggestion is None:
            s.display_text = display_text(food)
            s.transition(Phase.MONITORING, "food identified, no typical time")
            fx.say.append(
                f"I can see {food}.{looks} I'll keep watching it. "
                "If you want a timer, say, set a timer for 5 minutes."
            )
            return
        s.suggestion = suggestion
        mins = fmt_duration(suggestion.seconds)
        if self.rules.auto_start_timer:
            s.start_timer(suggestion.seconds, food, self.clock)
            fx.say.append(
                f"That looks like {food}.{looks} I've set a {mins} reminder, which is a typical time, "
                "and I'll keep watching it."
            )
            return
        s.display_text = display_text(f"{food[:14]}? {max(1, round(suggestion.seconds / 60))}m")
        s.transition(Phase.AWAITING_CONFIRMATION, "timer suggestion awaiting confirmation")
        fx.say.append(
            f"That looks like {food}.{looks} A typical time is about {mins}. "
            "Thumbs up to start a reminder timer, or thumbs down to skip it. I'll keep watching either way."
        )

    def _check_overcook(self, s: CookingSession, fx: Effects) -> None:
        if trailing(s.observations, S.POSSIBLY_OVERCOOKED_OR_BURNT) < self.rules.overcook_confirm:
            return
        if s.pending(AlertType.OVERCOOK_WARNING) is not None:
            return  # still unacknowledged: the device keeps alerting
        now = self.clock()
        if s.last_overcook_alert_at is not None and now - s.last_overcook_alert_at < self.rules.overcook_repeat_s:
            return
        repeat = s.last_overcook_alert_at is not None
        s.last_overcook_alert_at = now
        food = self.food_name(s)
        msg = (
            f"Warning: the {food} {'still ' if repeat else ''}looks like it may be burning. "
            "Please check the pan and turn the heat down."
        )
        fx.alerts.append(s.add_alert(AlertType.OVERCOOK_WARNING, "CHECK PAN!", msg))

    def _check_looks_done(self, s: CookingSession, fx: Effects) -> None:
        if s.phase != Phase.COOKING or s.looks_done_notified or s.timer is None:
            return
        if trailing(s.observations, S.APPEARS_COOKED) < self.rules.looks_done_confirm:
            return
        remaining = int(s.timer.remaining())
        if remaining < self.rules.looks_done_min_remaining_s:
            return
        s.looks_done_notified = True
        info = food_safety.classify(self.food_name(s))
        reminder = food_safety.thermometer_reminder(info)
        fx.say.append(
            f"The {self.food_name(s)} looks cooked, with {fmt_duration(remaining)} left on the timer. "
            "Have a look." + (f" {reminder}" if reminder else "")
        )

    # --- timer ----------------------------------------------------------------------

    def on_timer_done(self, s: CookingSession) -> Effects:
        fx = Effects()
        food = self.food_name(s)
        s.finish()
        obs = self.fresh_observation(s)
        seen = ""
        if obs is not None:
            seen = {
                S.APPEARS_UNDERCOOKED: " It still looks undercooked, so it may need longer.",
                S.APPEARS_COOKED: " It looks cooked.",
                S.POSSIBLY_OVERCOOKED_OR_BURNT: " It may be burning, so check it now.",
                S.CANNOT_DETERMINE: "",
            }[obs.status]
        reminder = food_safety.thermometer_reminder(food_safety.classify(food)) or "Check that it's cooked through."
        msg = f"The timer for the {food} is done.{seen} {reminder}"
        fx.alerts.append(s.add_alert(AlertType.TIMER_DONE, "TIMER DONE", msg))
        return fx

    # --- camera liveness --------------------------------------------------------------

    def on_camera(self, s: CookingSession, ok: bool) -> Effects:
        fx = Effects()
        now = self.clock()
        if ok:
            s.vision_live = True
            if s.camera_stale_since is not None:
                s.camera_stale_since = None
                if s.resolve_alerts(AlertType.MONITORING_LOST):
                    fx.say.append("I can see the stove again.")
            return fx
        s.vision_live = False
        if s.camera_stale_since is None:
            s.camera_stale_since = now
            log.warning("camera frames are stale; visual monitoring paused")
        if (
            now - s.camera_stale_since >= self.rules.camera_lost_after_s
            and s.pending(AlertType.MONITORING_LOST) is None
        ):
            fx.alerts.append(
                s.add_alert(
                    AlertType.MONITORING_LOST,
                    "CAMERA LOST",
                    "I can't see the stove right now, so I'm not watching the food. Please check on it.",
                )
            )
        return fx
