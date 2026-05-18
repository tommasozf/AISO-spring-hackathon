"""
Tier 2 — policy orchestrator.

Combines forecast + inventory + staffing + pricing + promo into the list of
action dicts to submit, then applies the Tier 1 safety filter.

Returns: list[dict] in the form [{"tool": "...", "args": {...}}, ...]
"""
from __future__ import annotations

from typing import List, Optional

from . import forecast, inventory, safety
from .state import AgentState


# ---- top-level ----

def decide_actions(obs: dict, day: int, state: AgentState) -> List[dict]:
    actions: List[dict] = []

    # ---- 0. forecast horizon ----
    covers_forecast = forecast.forecast_multi_day(
        state,
        obs.get("day_of_week", "Monday"),
        obs.get("weather_today", "sunny"),
        obs.get("weather_forecast", []) or [],
        obs.get("reputation_band", "Very Good"),
    )
    expected_today = covers_forecast[0] if covers_forecast else 90.0

    # ---- 1. phase ----
    state.phase = _phase_for(obs)

    # ---- 2. inventory orders ----
    actions.extend(inventory.reorder_plan(obs, state, expected_today))

    # ---- 3. menu (day 1 only — TODO: also when an active dish becomes un-makeable) ----
    if _should_set_menu(obs, day, state):
        menu = _choose_menu(obs, state)
        if menu:
            actions.append({"tool": "set_menu", "args": {"dishes": menu}})
            state.last_menu_change_day = day

    # ---- 4. staffing ----
    target_staff = _staff_for(expected_today, obs)
    if target_staff != obs.get("staff_level"):
        actions.append({"tool": "set_staff_level", "args": {"level": target_staff}})

    # ---- 5. pricing (skeleton: no-op — TODO experiment) ----
    # for dish in obs.get("active_menu", []):
    #     ...

    # ---- 6. marketing & promos ----
    mkt = _marketing_for(obs, expected_today)
    if mkt > 0:
        actions.append({"tool": "set_marketing_spend", "args": {"amount": mkt}})
    if _should_happy_hour(obs, day, state, expected_today):
        actions.append({"tool": "run_happy_hour", "args": {}})
        state.last_happy_hour_day = day

    # ---- 7. daily special (always — free bonus) ----
    special = _pick_daily_special(obs)
    if special:
        actions.append({"tool": "offer_daily_special", "args": {"dish": special}})

    # ---- 8. Tier 1 safety filter ----
    actions = safety.filter_unsafe(obs, actions)

    return actions


# ---- phase logic ----

def _phase_for(obs: dict) -> str:
    if obs.get("days_remaining", 30) <= 5:
        return "endgame"
    if obs.get("day", 1) <= 7:
        return "build"
    return "optimize"


# ---- menu ----

def _should_set_menu(obs: dict, day: int, state: AgentState) -> bool:
    if not obs.get("active_menu"):
        return True
    if day == 1 and state.last_menu_change_day < 0:
        return True
    return False


def _choose_menu(obs: dict, state: AgentState) -> list:
    """
    Skeleton: pick the first 5 dishes from menu_book.
    TODO: optimize for
      - ingredient diversity (don't have all 5 dishes need chicken)
      - supplier diversity (spread risk across suppliers)
      - margin (base_price - estimated ingredient cost)
      - shelf-life-friendly ingredients
    """
    book = obs.get("menu_book", []) or []
    if len(book) < 5:
        return [d["name"] for d in book]
    return [d["name"] for d in book[:5]]


# ---- staffing ----

def _staff_for(expected_covers_today: float, obs: dict) -> int:
    # Heuristic: ~12 covers per staff per day at default kitchen speed.
    target = int(round(expected_covers_today / 12.0)) + 2
    target = max(3, min(15, target))
    if safety.cash_emergency(obs):
        target = max(3, target - 2)
    return target


# ---- marketing & promos ----

def _marketing_for(obs: dict, expected_today: float) -> float:
    if safety.cash_emergency(obs):
        return 0.0
    # Pulse on weak forecast days only — diminishing returns mean constant
    # spend wastes money.
    if expected_today < 75:
        return 200.0
    return 0.0


def _should_happy_hour(obs: dict, day: int, state: AgentState, expected_today: float) -> bool:
    if safety.cash_emergency(obs):
        return False
    # Decay: don't run on consecutive days
    if day - state.last_happy_hour_day <= 2:
        return False
    return expected_today < 75


# ---- daily special ----

def _pick_daily_special(obs: dict) -> Optional[str]:
    """
    Skeleton: first active dish.
    TODO: find the dish whose recipe draws down our soonest-to-expire inventory.
    """
    menu = obs.get("active_menu") or []
    return menu[0] if menu else None
