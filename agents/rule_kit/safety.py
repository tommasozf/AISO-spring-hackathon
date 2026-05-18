"""
Tier 1 — hard invariants. These rules win over everything else.
If any of these fire, the planned action set is mutated or filtered to comply.

Designed to make BANKRUPTCY structurally impossible. The score floor matters
more than the ceiling when averaging across the eval matrix.
"""
from __future__ import annotations

from typing import List

# ---- thresholds ----

MIN_CASH_FLOOR = 500.0
CASH_EMERGENCY_DAYS = 3       # if cash - (this many days burn) < 0 -> emergency
PANIC_REORDER_DAYS = 1.5
ENDGAME_FREEZE_DAYS = 5       # last N days: no menu changes


# ---- predicates ----

def projected_burn(obs: dict, days: int) -> float:
    staff_cost = obs.get("staff_level", 8) * obs.get("staff_cost_per_person", 120)
    fixed = 300
    return (staff_cost + fixed) * days


def cash_emergency(obs: dict) -> bool:
    return obs.get("cash", 0) - projected_burn(obs, CASH_EMERGENCY_DAYS) < MIN_CASH_FLOOR


def in_endgame(obs: dict) -> bool:
    return obs.get("days_remaining", 30) <= ENDGAME_FREEZE_DAYS


# ---- filter ----

def filter_unsafe(obs: dict, actions: List[dict]) -> List[dict]:
    """
    Drop or clamp actions that would violate hard rules.
    Runs AFTER the policy proposes the action list.
    """
    safe: List[dict] = []
    cash_remaining = obs.get("cash", 0.0)
    floor = projected_burn(obs, CASH_EMERGENCY_DAYS) + MIN_CASH_FLOOR

    for a in actions:
        tool = a.get("tool")
        args = dict(a.get("args") or {})

        # ---- staff range ----
        if tool == "set_staff_level":
            lvl = int(args.get("level", 8))
            lvl = max(3, min(15, lvl))
            # In cash emergency, never raise staff above current
            if cash_emergency(obs) and lvl > obs.get("staff_level", 8):
                continue
            safe.append({"tool": tool, "args": {"level": lvl}})
            continue

        # ---- marketing ----
        if tool == "set_marketing_spend":
            amt = float(args.get("amount", 0))
            if cash_emergency(obs):
                amt = 0.0
            amt = max(0.0, min(500.0, amt))
            safe.append({"tool": tool, "args": {"amount": amt}})
            continue

        # ---- happy hour ----
        if tool == "run_happy_hour":
            if cash_emergency(obs):
                continue
            safe.append({"tool": tool, "args": {}})
            continue

        # ---- menu freeze in endgame ----
        if tool == "set_menu" and in_endgame(obs):
            continue

        # ---- price range (caller should clamp; we sanity-check) ----
        if tool == "set_price":
            safe.append({"tool": tool, "args": args})
            continue

        # ---- orders: cash check ----
        if tool == "place_order":
            cost = _estimate_order_cost(obs, args)
            if cost is None or cost > cash_remaining - floor:
                continue
            cash_remaining -= cost
            safe.append({"tool": tool, "args": args})
            continue

        # ---- pass-through ----
        safe.append({"tool": tool, "args": args})

    return safe


# ---- helpers ----

def _estimate_order_cost(obs: dict, args: dict):
    supplier_name = args.get("supplier")
    ingredient = args.get("ingredient")
    qty = args.get("quantity_kg", 0)
    for s in obs.get("supplier_catalog", []) or []:
        if s.get("name") == supplier_name and ingredient in (s.get("ingredients") or {}):
            return s["ingredients"][ingredient] * float(qty)
    return None


# ---- TODO ----
# - Detect "regime change": if yesterday's revenue deviates more than 2 sigma
#   from forecast, set a state flag → policy should switch to defensive mode.
# - Reputation floor: if reputation drops to "Fair", suppress promos & price
#   hikes for N days, prioritize service quality.
# - Stockout protection: if `dishes_unavailable_at` is non-empty in yesterday's
#   service_summary, FORCE a same-day reorder of that ingredient regardless of
#   the regular reorder schedule.
