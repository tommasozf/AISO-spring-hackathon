"""
Demand forecasting.

Forecasts expected covers (customer count) for the next 1-4 days. Reorder
logic multiplies covers × dish mix × recipe yield to get ingredient need.

Start with priors. Replace constants with empirical means as `state.history`
grows. By day 10 the model should be mostly data-driven.
"""
from __future__ import annotations

from typing import List

from .state import AgentState


# ---- priors (REPLACE WITH EMPIRICAL AS DATA ACCUMULATES) ----

WEATHER_MULT = {
    "sunny": 1.00,
    "cloudy": 0.95,
    "rainy": 0.85,
    "stormy": 0.70,
}

DOW_MULT = {
    "Monday": 0.80,
    "Tuesday": 0.85,
    "Wednesday": 0.95,
    "Thursday": 1.00,
    "Friday": 1.20,
    "Saturday": 1.30,
    "Sunday": 1.05,
}

PRIOR_BASE_COVERS = 95.0  # reverted from 110 — let p75-of-recent do the
                          # heavy lifting once we have history. 110 + p75 +
                          # softer rep_mult stacked → over-forecast on busy
                          # days → over-staffed AND over-ordered.
# Reputation multiplier — moderately softened. Original 0.55–1.10 caused a
# death spiral (forecast collapsed → staff cut → walkouts → rep dropped further).
# First fix went too far (0.85 floor) and made us over-confident. Compromise:
# Poor reputation still trims demand, but less aggressively than before.
REP_MULT = {
    "Poor": 0.75, "Fair": 0.85, "Good": 0.95,
    "Very Good": 1.00, "Excellent": 1.05,
}

# Forecast accuracy degrades: 85%, 70%, 55%. Wider safety multiplier on later days.
FORECAST_NOISE_BUFFER = [1.00, 1.10, 1.20, 1.30]


# ---- API ----

def forecast_covers(state: AgentState, day_of_week: str, weather: str, reputation_band: str) -> float:
    base = state.empirical_base_covers() or PRIOR_BASE_COVERS
    return (
        base
        * DOW_MULT.get(day_of_week, 1.0)
        * WEATHER_MULT.get(weather, 1.0)
        * REP_MULT.get(reputation_band, 1.0)
    )


def forecast_multi_day(
    state: AgentState,
    day_of_week_today: str,
    weather_today: str,
    weather_forecast: List[str],
    reputation_band: str,
) -> List[float]:
    """Returns [today, today+1, today+2, today+3] expected covers (best-effort)."""
    cycle = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        i = cycle.index(day_of_week_today)
    except ValueError:
        i = 0

    weathers = [weather_today] + list(weather_forecast or [])
    out: List[float] = []
    for offset, w in enumerate(weathers[:4]):
        dow = cycle[(i + offset) % 7]
        f = forecast_covers(state, dow, w, reputation_band)
        if offset < len(FORECAST_NOISE_BUFFER):
            f *= FORECAST_NOISE_BUFFER[offset]
        out.append(f)
    return out


# ---- TODO ----
# - Replace fixed priors with empirical means by (day_of_week, weather) cell.
# - Add a marketing-spend term once you have data (probably saturating).
# - Add a happy-hour boost term (with decay).
# - Reputation enters non-linearly (spiral) — consider thresholds, not multiplier.
