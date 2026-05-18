"""
Inventory + reorder logic.

For each ingredient on the active menu:
  1. estimate daily usage     = sum_dishes( expected_orders * recipe_qty_per_dish )
  2. days_of_cover            = (on_hand + pending) / daily_usage
  3. if days_of_cover < TARGET:
         pick best supplier (by reliability x lead time x price)
         quantity = TARGET * daily_usage * SAFETY - on_hand - pending
         cap at shelf_life - lead_time   (so it can't spoil before use)
         respect min_order_kg, respect cash floor

This is a newsvendor variant. Tune TARGET, SAFETY, supplier weights to taste.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .state import AgentState


# ---- tunables ----

TARGET_DAYS_COVER = 4
MIN_DAYS_COVER = 2          # trip-wire for "panic order"
SAFETY_MULTIPLIER = 1.20    # over-order this much to absorb forecast error
SUPPLIER_RELIABILITY_FLOOR = 0.5  # don't order from suppliers below this


# ---- API ----

def estimate_dish_mix(obs: dict) -> Dict[str, float]:
    """
    Fraction of covers that order each active dish.
    Skeleton: even split across active menu. TODO: weight by recent dishes_sold.
    """
    active = obs.get("active_menu", []) or []
    if not active:
        return {}
    share = 1.0 / len(active)
    return {d: share for d in active}


def estimate_daily_usage(obs: dict, expected_covers_today: float) -> Dict[str, float]:
    """
    For each ingredient: kg consumed today at expected demand.
    Walks the active menu's recipes from `menu_book`.
    """
    mix = estimate_dish_mix(obs)
    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}
    usage: Dict[str, float] = {}
    for dish, share in mix.items():
        recipe = menu_book.get(dish, {}).get("ingredients", [])
        expected_orders = expected_covers_today * share
        for ing in recipe:
            qty = ing.get("quantity_kg", 0.0)
            usage[ing["ingredient"]] = usage.get(ing["ingredient"], 0.0) + expected_orders * qty
    return usage


def days_of_cover(ingredient: str, obs: dict, daily_usage: float) -> float:
    if daily_usage <= 0:
        return float("inf")
    inv = _find(obs.get("inventory", []), "ingredient", ingredient)
    on_hand = (inv or {}).get("total_kg", 0.0)
    pending = sum(
        po.get("quantity_kg", 0.0)
        for po in (obs.get("pending_orders") or [])
        if po.get("ingredient") == ingredient
    )
    return (on_hand + pending) / daily_usage


def reorder_plan(
    obs: dict,
    state: AgentState,
    expected_covers_today: float,
) -> List[dict]:
    """
    Returns a list of place_order action dicts to submit this turn:
        [{"tool": "place_order", "args": {"supplier", "ingredient", "quantity_kg"}}, ...]
    """
    usage = estimate_daily_usage(obs, expected_covers_today)
    orders: List[dict] = []
    cash_remaining = obs.get("cash", 0.0)

    # Reserve a cash floor: 3 days of fixed + staff overhead
    staff_cost = obs.get("staff_level", 8) * obs.get("staff_cost_per_person", 120)
    floor = 3 * (300 + staff_cost)

    # Sort by tightest cover first (most urgent gets first claim on cash)
    by_urgency = sorted(usage.items(), key=lambda kv: days_of_cover(kv[0], obs, kv[1]))

    for ingredient, daily in by_urgency:
        if daily <= 0:
            continue
        cover = days_of_cover(ingredient, obs, daily)
        if cover >= TARGET_DAYS_COVER:
            continue

        need_kg = (TARGET_DAYS_COVER * daily * SAFETY_MULTIPLIER) - (cover * daily)

        # Shelf-life cap: never order more than you can plausibly use before expiry
        inv = _find(obs.get("inventory", []), "ingredient", ingredient) or {}
        shelf = inv.get("shelf_life_days") or 14
        max_safely_usable = daily * max(1, shelf - 1)
        need_kg = min(need_kg, max_safely_usable)

        if need_kg <= 0:
            continue

        supplier = pick_supplier(ingredient, obs, state)
        if not supplier:
            continue

        unit_price = supplier["ingredients"][ingredient]
        min_order = supplier.get("min_order_kg", 0.0)
        qty = max(need_kg, min_order)
        cost = qty * unit_price

        if cost > cash_remaining - floor:
            # try shrinking to min_order; skip entirely if even that breaks the floor
            qty = min_order
            cost = qty * unit_price
            if cost > cash_remaining - floor or qty <= 0:
                continue

        orders.append({
            "tool": "place_order",
            "args": {
                "supplier": supplier["name"],
                "ingredient": ingredient,
                "quantity_kg": round(qty, 2),
            },
        })
        cash_remaining -= cost

    return orders


def pick_supplier(ingredient: str, obs: dict, state: AgentState) -> Optional[dict]:
    """Best supplier for an ingredient: reliability > lead time > price."""
    candidates = [
        s for s in (obs.get("supplier_catalog") or [])
        if ingredient in (s.get("ingredients") or {})
    ]
    if not candidates:
        return None

    def score(s: dict) -> float:
        rel = state.suppliers.get(s["name"])
        reliability = rel.delivered_ratio if rel else 1.0
        if reliability < SUPPLIER_RELIABILITY_FLOOR:
            return -1.0  # actively avoid
        lead = s.get("lead_time_days", 2)
        price = s["ingredients"][ingredient]
        return reliability * 10 - lead - price * 0.05

    return max(candidates, key=score)


# ---- helpers ----

def _find(seq: list, key: str, value) -> Optional[dict]:
    for x in seq or []:
        if x.get(key) == value:
            return x
    return None


# ---- TODO ----
# - estimate_dish_mix: pull last 7 days of service_summary.dishes_sold and
#   compute empirical mix per (day_of_week, weather) cell.
# - reorder_plan: account for delivery DAY too — a Wed-only supplier might
#   deliver in 1 day or 6 depending on what day-of-week you order on.
# - Diversification rule: if any one supplier is >60% of your orders by value,
#   intentionally route some volume to a secondary even at higher price.
# - Stockout history: if `dishes_unavailable_at` fired yesterday, bump that
#   ingredient's safety multiplier for the next few days.
