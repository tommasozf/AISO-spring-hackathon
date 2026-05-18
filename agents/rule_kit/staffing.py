"""
Staffing — per-day-of-week base table with weather/trend/walkout modifiers.

Replaces the old `covers / 14 + 1` heuristic that was under-staffing on busy
days (which is the most likely cause of the "Many" walkout penalty stacking
up). Ported from Fabiano's compute_staff_level, with two deliberate changes:

  1. REPUTATION_BAND IS NOT READ HERE. The teammate owns reputation-driven
     actions; staffing reads only DOW + weather + trend + walkout + wait +
     scenario. (Fabiano's REP_ADJ table is removed.)

  2. The cash-safety lid from our prior staffing module is preserved: when
     cash is low (<5000) we never RAISE staff above current; in a cash
     emergency we clamp at 4.

Cap raised from 12 → 14 to allow the Fri/Sat surge crew. API hard cap is 15.
"""
from __future__ import annotations

from typing import Optional


STAFF_BASE = {
    "Monday":    7,
    "Tuesday":   7,
    "Wednesday": 7,
    "Thursday":  8,
    "Friday":   10,
    "Saturday": 11,
    "Sunday":    9,
}

WEATHER_ADJ = {"sunny": 1, "cloudy": 0, "rainy": -1, "stormy": -2}
TREND_ADJ = {"Growing": 1, "Stable": 0, "Declining": -1}

# Hard bounds. API allows 3..15; we cap at 14 because €120/head/day adds up
# and the 15th seat rarely earns its keep.
MIN_STAFF = 3
MAX_STAFF = 14


def compute_staff_level(
    obs: dict,
    state,
    scenario: str = "baseline",
    expected_covers_today: float = 0.0,
    decisions: Optional[dict] = None,
    regime_modifiers: Optional[dict] = None,
) -> int:
    """
    Returns an absolute target staff level for today. Caller emits a
    set_staff_level action only if this differs from obs["staff_level"].

    Reads (in order): DOW base → weather → trend → yesterday's walkout band →
    yesterday's table utilization & peak wait → scenario gating →
    forecast nudge → regime staff_delta → cash safety.

    Phase 3: when `regime_modifiers` is provided, `staff_delta` is added to
    the target before the cash-safety clamp. Final value is still bounded by
    [MIN_STAFF, MAX_STAFF] (= [3, 14]). Brief asks for [4, 14] on the regime
    path; the existing MIN_STAFF=3 only fires in cash emergencies, so the
    regime-applied path effectively respects [4, 14] in normal operation.
    """
    if decisions is None:
        decisions = {}

    current = int(obs.get("staff_level", 8) or 8)
    dow = obs.get("day_of_week", "Monday")
    weather = obs.get("weather_today", "cloudy")
    trend = obs.get("customer_trend", "Stable")

    base = STAFF_BASE.get(dow, 7)
    base += WEATHER_ADJ.get(weather, 0)
    base += TREND_ADJ.get(trend, 0)

    # Yesterday's service signal — bump for any sign we were throughput-bound.
    svc = obs.get("service_summary") or {}
    walkout_band = svc.get("walkout_band", "None")
    if walkout_band == "Many":
        base += 2  # heavy under-staffing penalty — be aggressive
    elif walkout_band == "Some":
        base += 1
    if float(svc.get("table_utilization_peak", 0) or 0) > 0.9:
        base += 1
    if float(svc.get("peak_wait_minutes", 0) or 0) > 15:
        base += 1

    # Forecast nudge: when our covers forecast strongly contradicts the DOW
    # default (eg. a quiet Saturday because rep tanked), let it shave 1.
    if expected_covers_today and expected_covers_today < 60:
        base -= 1

    # Scenario gating.
    if scenario == "renovation":
        # Fewer tables → fewer covers physically possible. Big trim early.
        day_idx = int(obs.get("day", 0) or 0)
        if day_idx <= 12:
            base = max(base - 3, 4)
        else:
            base = max(base - 1, 5)
    elif scenario == "tourist" and trend == "Growing":
        base += 2
    elif scenario == "supply_crisis":
        # If we can't cook half the menu, lean staffing a bit.
        base -= 1

    # Phase 3: regime-driven staff delta. Applied after the scenario gating so
    # that compound regimes (supply_cap- AND demand+) combine additively, then
    # clamped to [4, 14] before cash safety can drag it further.
    rm_staff_delta = 0
    if regime_modifiers:
        try:
            rm_staff_delta = int(regime_modifiers.get("staff_delta", 0) or 0)
        except (TypeError, ValueError):
            rm_staff_delta = 0
    if rm_staff_delta:
        base += rm_staff_delta
        # Clamp the regime-influenced base into the [4, 14] band per Phase 3
        # spec. (Existing MIN_STAFF=3 is a cash-emergency floor preserved below.)
        base = max(4, min(MAX_STAFF, base))
        decisions["regime_staff_delta"] = rm_staff_delta

    # Cash safety — DO NOT let staffing burn the floor.
    cash = float(obs.get("cash", 15000) or 0)
    if cash < 2000:
        base = min(base, 4)
    elif cash < 3000:
        base = min(base, 5)
    elif cash < 5000:
        # don't INCREASE staff in this band — hold or trim only.
        base = min(base, current)

    target = max(MIN_STAFF, min(MAX_STAFF, int(base)))

    decisions["staffing_base_dow"] = STAFF_BASE.get(dow, 7)
    decisions["staffing_weather_adj"] = WEATHER_ADJ.get(weather, 0)
    decisions["staffing_trend_adj"] = TREND_ADJ.get(trend, 0)
    decisions["staffing_walkout_band"] = walkout_band
    decisions["staffing_scenario"] = scenario
    decisions["staffing_final"] = target
    decisions["staffing_current"] = current

    return target
