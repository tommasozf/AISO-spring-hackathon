"""
Algorithmic pricing — emits set_price actions for active-menu dishes.

Five composing signals (multiplicative on base price, then clamped):

  1. Walkout band      — "Many" raises price to cool demand
                         "None" lowers slightly to attract
  2. Inventory pressure — raise on dishes whose recipe ingredients are running
                         low (preserves stock for higher-margin sales)
  3. Day-of-week        — bump on Fri/Sat (peak demand), trim on Mon/Tue
  4. Reputation cap     — don't squeeze prices when reputation is struggling;
                         caps the upper bound by rep band
  5. Margin floor       — never price below 2× marginal ingredient cost,
                         so we always preserve gross margin

Final clamp: [0.80, 1.20] × base_price (the API hard limit).
Only emits set_price if the change is ≥ PRICE_CHANGE_THRESHOLD (default €0.10)
to avoid spamming actions for noise-level adjustments.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .inventory import days_of_cover, estimate_daily_usage
from .state import AgentState


# ---- tunables ----

# Walkout-driven cooling. "Many" walkouts means we're throughput-bound, so
# raise prices to ration capacity to higher-margin customers.
WALKOUT_ADJUST = {
    "Many": 1.05,
    "Some": 1.03,
    "Few":  1.01,
    "None": 0.99,  # gentle drop attracts price-sensitive customers
}

# Day-of-week — peaks vs valleys for Italian restaurant patterns
DOW_ADJUST = {
    "Monday":    0.98,
    "Tuesday":   0.98,
    "Wednesday": 1.00,
    "Thursday":  1.01,
    "Friday":    1.03,
    "Saturday":  1.03,
    "Sunday":    1.00,
}

# Upper-cap multiplier on base price keyed by reputation band.
# Below "Very Good", squeeze less — preserve margin on customers we still have
# rather than risk losing them entirely.
REP_UPPER_CAP = {
    "Poor":      0.95,
    "Fair":      1.05,
    "Good":      1.15,
    "Very Good": 1.20,
    "Excellent": 1.20,
}

# Inventory-pressure thresholds (days of cover at recipe-level)
INV_PRESSURE_CRITICAL = 1.0  # < 1 day → bump 8%
INV_PRESSURE_WARNING  = 2.0  # < 2 days → bump 4%

# Never price below this multiple of marginal ingredient cost
MARGIN_FLOOR_MULTIPLIER = 2.0

# Don't emit set_price for tiny price moves
PRICE_CHANGE_THRESHOLD = 0.10  # €


# ---- API ----

def pricing_actions(
    obs: dict,
    state: AgentState,
    expected_covers_today: float,
    record: Optional[Dict] = None,
    regime_modifiers: Optional[Dict] = None,
) -> List[dict]:
    """
    Return a list of {"tool": "set_price", "args": {dish, price}} actions.
    Populates `record["pricing"]` with per-dish rationale if provided.

    Phase 3: `regime_modifiers` (optional) recognises two pricing knobs:
        suppress_price_drops   : when True, target prices are floored at the
                                 current price (no downward moves).
        allow_capacity_premium : when True, the per-dish target is bumped 5%
                                 (still clamped by the API hard bound).
    """
    if record is None:
        record = {}
    actions: List[dict] = []
    rm = regime_modifiers or {}
    rm_suppress_drops = bool(rm.get("suppress_price_drops", False))
    rm_capacity_premium = bool(rm.get("allow_capacity_premium", False))
    rm_pricing_applied: Dict[str, object] = {}
    if rm_suppress_drops:
        rm_pricing_applied["suppress_price_drops"] = True
    if rm_capacity_premium:
        rm_pricing_applied["allow_capacity_premium"] = True

    active_menu = obs.get("active_menu") or []
    if not active_menu:
        record["pricing"] = {}
        return actions

    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}
    supplier_catalog = obs.get("supplier_catalog") or []

    svc = obs.get("service_summary") or {}
    walkout_band = svc.get("walkout_band", "None")
    day_of_week = obs.get("day_of_week", "Wednesday")
    reputation = obs.get("reputation_band", "Very Good")

    # Fix 4: gate price changes by ≥10% delta unless yesterday's walkout band
    # indicates real demand pressure (Few/Some/Many). Composed AFTER Phase 3
    # suppress_price_drops / allow_capacity_premium modify the target; this
    # gate sits on emission, not on the price math itself.
    walkout_urgency = walkout_band in ("Few", "Some", "Many")
    PRICE_CHANGE_DELTA_FRAC = 0.10
    changes_emitted = 0
    changes_suppressed_small_delta = 0

    daily_usage = estimate_daily_usage(obs, expected_covers_today)

    walkout_factor = WALKOUT_ADJUST.get(walkout_band, 1.0)
    dow_factor = DOW_ADJUST.get(day_of_week, 1.0)
    rep_upper_factor = REP_UPPER_CAP.get(reputation, 1.20)

    per_dish: Dict[str, dict] = {}

    for dish_name in active_menu:
        dish = menu_book.get(dish_name)
        if not dish:
            continue

        base = dish.get("base_price")
        current = dish.get("current_price", base)
        if base is None or current is None:
            continue

        # ---- inventory pressure for THIS dish's recipe ----
        min_cover = float("inf")
        for ing in dish.get("ingredients") or []:
            usage = daily_usage.get(ing["ingredient"], 0.0)
            if usage > 0:
                cov = days_of_cover(ing["ingredient"], obs, usage)
                if cov < min_cover:
                    min_cover = cov

        if min_cover < INV_PRESSURE_CRITICAL:
            inv_factor = 1.08
        elif min_cover < INV_PRESSURE_WARNING:
            inv_factor = 1.04
        else:
            inv_factor = 1.00

        # ---- compose target ----
        target = base * walkout_factor * dow_factor * inv_factor

        # ---- margin floor ----
        marginal_cost = _marginal_cost(dish, supplier_catalog)
        margin_floor = marginal_cost * MARGIN_FLOOR_MULTIPLIER
        if target < margin_floor:
            target = margin_floor

        # ---- reputation upper cap ----
        upper_cap = base * rep_upper_factor
        if target > upper_cap:
            target = upper_cap

        # ---- Phase 3 regime overrides ----
        # Capacity premium: we're capacity-bound (e.g. renovation), so charge
        # a small premium to ration demand to higher-margin covers.
        if rm_capacity_premium:
            target = target * 1.05
        # Suppress price drops: in a demand+ regime, never lower the asking
        # price below where we already sit — the demand is there to absorb it.
        if rm_suppress_drops and target < current:
            target = current

        # ---- API hard bounds [0.80, 1.20] × base ----
        target = max(base * 0.80, min(base * 1.20, target))
        target = round(target, 2)

        per_dish[dish_name] = {
            "current": current,
            "target": target,
            "base": base,
            "walkout_factor": walkout_factor,
            "dow_factor": dow_factor,
            "inv_factor": inv_factor,
            "min_cover_days": (round(min_cover, 2)
                               if min_cover != float("inf") else None),
            "marginal_cost": round(marginal_cost, 2),
            "margin_floor": round(margin_floor, 2),
            "rep_upper_cap": round(upper_cap, 2),
        }

        if abs(target - current) >= PRICE_CHANGE_THRESHOLD:
            # Fix 4: gate by ≥10% delta unless walkout band shows demand
            # pressure. Tiny price moves churn the action stream without
            # affecting customer behaviour at the band-resolution the
            # simulator exposes.
            current_safe = current if current > 0 else 1.0
            delta_frac = abs(target - current) / current_safe
            if delta_frac >= PRICE_CHANGE_DELTA_FRAC or walkout_urgency:
                actions.append({
                    "tool": "set_price",
                    "args": {"dish": dish_name, "price": target},
                })
                changes_emitted += 1
            else:
                changes_suppressed_small_delta += 1

    record["pricing"] = per_dish
    record["price_changes_emitted"] = changes_emitted
    record["price_changes_suppressed_small_delta"] = changes_suppressed_small_delta
    record["price_gate_walkout_urgency"] = walkout_urgency
    if rm_pricing_applied:
        record["regime_pricing_clamp_applied"] = rm_pricing_applied
    return actions


# ---- helpers ----

def _cheapest_unit_price(ingredient: str, supplier_catalog: list) -> Optional[float]:
    """Lowest €/kg available for this ingredient across all suppliers."""
    prices = []
    for s in supplier_catalog or []:
        p = (s.get("ingredients") or {}).get(ingredient)
        if p is not None:
            prices.append(p)
    return min(prices) if prices else None


def _marginal_cost(dish: dict, supplier_catalog: list) -> float:
    """Sum of (recipe_qty × cheapest_unit_price) for one serving of the dish."""
    total = 0.0
    for ing in dish.get("ingredients", []) or []:
        unit = _cheapest_unit_price(ing["ingredient"], supplier_catalog)
        if unit is None:
            continue
        total += unit * ing.get("quantity_kg", 0.0)
    return total


# ---- TODO ----
# - Track empirical price elasticity per dish from (price, covers) history.
#   Once we have ~10 days of varied prices we can estimate dE[q]/dp.
# - Cross-dish substitution: raising pizza price might shift demand to pasta.
#   Hard without more data, but worth flagging in telemetry.
# - Hysteresis: don't oscillate prices day-to-day even if signal flips. Track
#   last change day per dish.
