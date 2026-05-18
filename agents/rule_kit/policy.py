"""
Tier 2 — policy orchestrator.

Combines forecast + inventory + staffing + pricing + promo into the list of
action dicts to submit, then applies the Tier 1 safety filter.

Returns: list[dict] in the form [{"tool": "...", "args": {...}}, ...]
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import forecast, inventory, safety
from .state import AgentState


# ---- top-level ----

def decide_actions(
    obs: dict,
    day: int,
    state: AgentState,
    record: Optional[Dict] = None,
) -> List[dict]:
    """
    Returns the action list for this turn.

    If `record` is provided (a dict), it is populated in-place with the
    intermediate decisions so telemetry can log *why* we did what we did.
    """
    if record is None:
        record = {}
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
    record["forecast"] = [round(x, 1) for x in covers_forecast]
    record["expected_today"] = round(expected_today, 1)

    # ---- 1. phase ----
    state.phase = _phase_for(obs)
    record["phase"] = state.phase

    # ---- 2. inventory diagnostics + orders ----
    daily_usage = inventory.estimate_daily_usage(obs, expected_today)
    record["daily_usage"] = {k: round(v, 3) for k, v in daily_usage.items()}
    record["days_of_cover"] = {
        k: round(inventory.days_of_cover(k, obs, u), 2)
        for k, u in daily_usage.items()
    }

    proposed_orders = inventory.reorder_plan(obs, state, expected_today)
    record["orders_proposed"] = [
        {"ingredient": a["args"]["ingredient"],
         "supplier": a["args"]["supplier"],
         "quantity_kg": a["args"]["quantity_kg"]}
        for a in proposed_orders
    ]
    actions.extend(proposed_orders)

    # ---- 3. menu (day 1 only — TODO: also when an active dish becomes un-makeable) ----
    if _should_set_menu(obs, day, state):
        menu = _choose_menu(obs, state)
        if menu:
            actions.append({"tool": "set_menu", "args": {"dishes": menu}})
            state.last_menu_change_day = day
            record["menu_set"] = menu
    record.setdefault("menu_set", None)

    # ---- 4. staffing ----
    target_staff = _staff_for(expected_today, obs)
    record["staff_target"] = target_staff
    record["staff_current"] = obs.get("staff_level")
    if target_staff != obs.get("staff_level"):
        actions.append({"tool": "set_staff_level", "args": {"level": target_staff}})

    # ---- 5. pricing (skeleton: no-op — TODO experiment) ----
    record["price_changes"] = []

    # ---- 6. marketing & promos ----
    mkt = _marketing_for(obs, expected_today)
    record["marketing"] = mkt
    if mkt > 0:
        actions.append({"tool": "set_marketing_spend", "args": {"amount": mkt}})

    hh = _should_happy_hour(obs, day, state, expected_today)
    record["happy_hour"] = hh
    if hh:
        actions.append({"tool": "run_happy_hour", "args": {}})
        state.last_happy_hour_day = day

    # ---- 7. daily special (always — free bonus) ----
    special = _pick_daily_special(obs)
    record["daily_special"] = special
    if special:
        actions.append({"tool": "offer_daily_special", "args": {"dish": special}})

    # ---- 8. Tier 1 safety filter ----
    actions_before = len(actions)
    actions = safety.filter_unsafe(obs, actions)
    record["safety_dropped_count"] = actions_before - len(actions)
    record["cash_emergency"] = safety.cash_emergency(obs)
    record["in_endgame"] = safety.in_endgame(obs)

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
    """
    Only set the menu if the server didn't give us one already. The starting
    menu has 8 dishes; replacing it with the first 5 of menu_book shrank
    variety and dropped Salmon/Risotto/Chicken Parmesan on Day 1 of the v1
    run. Bad. Variety attracts a broader customer base — leave it alone unless
    we have a real reason to change it.

    TODO: a smart `_choose_menu` could pick 6-8 dishes optimizing for
    ingredient diversity, supplier diversity, and margin — but only call it
    once, day 1, and only if we believe it beats the default.
    """
    if not obs.get("active_menu"):
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
    """
    Staff sizing heuristic.

    Empirically ~12 covers per staff per day, but +2 was too aggressive — it
    pushed us to 15 staff on busy-day forecasts. €120/staff/day adds up fast.
    Now: covers/14 + 1, capped at 12 (not 15) for normal ops. Hard limit 15
    only via the API cap.
    """
    target = int(round(expected_covers_today / 14.0)) + 1
    target = max(3, min(12, target))
    if safety.cash_emergency(obs):
        target = max(3, target - 2)
    # Defensive cash floor: if cash < 5000, don't INCREASE staff above current
    if (obs.get("cash") or 0) < 5000:
        target = min(target, obs.get("staff_level") or target)
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
    Pick the active dish whose recipe uses our soonest-to-expire ingredient.

    Wins three ways at once:
      - free satisfaction bonus from the special
      - drives demand toward at-risk inventory (reduces waste)
      - rotates daily (variety, instead of "Pizza Margherita every day")

    Falls back to active_menu[0] only if we have no inventory data.
    """
    menu = obs.get("active_menu") or []
    if not menu:
        return None

    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}

    # ingredient -> soonest expiry days, but only ingredients we actually have
    soonest_expiry: dict = {}
    for inv in obs.get("inventory", []) or []:
        if (inv.get("total_kg") or 0) <= 0:
            continue
        batches = inv.get("batches") or []
        if batches:
            soonest_expiry[inv["ingredient"]] = min(
                b.get("expires_in_days", 999) for b in batches
            )

    best_dish = None
    best_urgency = 999
    for dish_name in menu:
        recipe = menu_book.get(dish_name, {}).get("ingredients", [])
        if not recipe:
            continue
        # only pick dishes we can actually make today
        if not all(ing["ingredient"] in soonest_expiry for ing in recipe):
            continue
        urgency = min(soonest_expiry[ing["ingredient"]] for ing in recipe)
        if urgency < best_urgency:
            best_urgency = urgency
            best_dish = dish_name

    return best_dish or menu[0]
