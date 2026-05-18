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
SAFETY_MULTIPLIER = 1.20    # reverted from 1.50 — combined with the pending
                            # window fix and higher forecasts, 1.50 caused
                            # over-ordering (1139 kg vs 871 kg in v1) →
                            # cash drained to €54 by day 28.
SUPPLIER_RELIABILITY_FLOOR = 0.5  # don't order from suppliers below this

# Only count pending orders that arrive within this many days when computing
# days_of_cover. Pending arriving later can't help today's service even if the
# paper total looks healthy. Was the root cause of Day 8's "no orders placed"
# bug: 44 kg of Flour was pending but most arrived 3+ days later, so on-hand
# was 2.2 kg while the policy reported 8 days of cover.
PENDING_ARRIVAL_WINDOW = 1  # days

# Task 5: stockout-history feedback.
STOCKOUT_WINDOW_DAYS = 3      # if ingredient hit 0 within this many days...
STOCKOUT_SAFETY_BUMP = 1.5    # ...inflate effective SAFETY_MULTIPLIER by ×1.5
STOCKOUT_DECAY_DAYS = 5       # full decay back to 1.0 after this many clean days

# Task 8: diversification — if any supplier hoards more than this fraction of
# total kg in the recent window, try to route at least one order this turn to
# an alternate supplier when one exists.
DIVERSIFICATION_THRESHOLD = 0.60
DIVERSIFICATION_LOOKBACK = 7

# Task 7: weekday cycle used to convert (today + lead_time) into actual
# delivery day given a supplier's delivery_days schedule.
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---- API ----

def estimate_dish_mix(obs: dict, state: Optional[AgentState] = None) -> Dict[str, float]:
    """
    Task 2: empirical fraction of covers ordering each active dish.

    Strategy:
      - For each active dish, look at the last 7 days of `dishes_sold` and
        compute `sum(dishes_sold[d]) / sum(total_covers)`.
      - When we have <3 recorded days OR a brand-new dish absent from history,
        fall back to (or smooth toward) a uniform-prior weight `1/N`.
      - Shares are renormalized over the CURRENT active menu so they sum to 1.
    """
    active = obs.get("active_menu", []) or []
    if not active:
        return {}

    uniform = 1.0 / len(active)
    recent = (state.history[-7:] if state and state.history else [])

    if len(recent) < 3:
        return {d: uniform for d in active}

    total_covers = sum(r.covers for r in recent) or 0
    sold: Dict[str, int] = {}
    for r in recent:
        for d, c in (r.dishes_sold or {}).items():
            sold[d] = sold.get(d, 0) + c

    if total_covers <= 0 or not sold:
        return {d: uniform for d in active}

    # Empirical share per active dish. For dishes with no history (e.g. just
    # added to menu), use uniform-prior smoothing.
    raw: Dict[str, float] = {}
    for d in active:
        observed = sold.get(d, 0)
        if observed > 0:
            raw[d] = observed / total_covers
        else:
            raw[d] = uniform * 0.5  # smaller-than-uniform prior for cold-start

    s = sum(raw.values())
    if s <= 0:
        return {d: uniform for d in active}
    return {d: v / s for d, v in raw.items()}


def estimate_daily_usage(
    obs: dict,
    expected_covers_today: float,
    state: Optional[AgentState] = None,
) -> Dict[str, float]:
    """
    For each ingredient: kg consumed today at expected demand.
    Walks the active menu's recipes from `menu_book`.
    """
    mix = estimate_dish_mix(obs, state)
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
    """
    On-hand + soon-arriving pending, expressed as days of forward cover.

    "Soon-arriving" = delivers within PENDING_ARRIVAL_WINDOW days. Anything
    later is excluded from the trigger calculation so we don't suppress new
    orders when the pending queue is full of late-arriving items.
    """
    if daily_usage <= 0:
        return float("inf")
    inv = _find(obs.get("inventory", []), "ingredient", ingredient)
    on_hand = (inv or {}).get("total_kg", 0.0)
    today = obs.get("day", 0)
    pending_soon = sum(
        po.get("quantity_kg", 0.0)
        for po in (obs.get("pending_orders") or [])
        if po.get("ingredient") == ingredient
        and (po.get("delivery_day", 999) - today) <= PENDING_ARRIVAL_WINDOW
    )
    return (on_hand + pending_soon) / daily_usage


def reorder_plan(
    obs: dict,
    state: AgentState,
    expected_covers_horizon,
    decisions: Optional[Dict] = None,
    scenario: str = "baseline",
    regime_modifiers: Optional[Dict] = None,
) -> List[dict]:
    """
    Returns a list of place_order action dicts to submit this turn:
        [{"tool": "place_order", "args": {"supplier", "ingredient", "quantity_kg"}}, ...]

    Task 1: `expected_covers_horizon` is a list of forecasted covers for
    [today, today+1, today+2, ...]. We size orders against the BUSIEST day
    in the horizon (weighted by ingredient shelf life — if shelf is short we
    fall back to today's demand to avoid spoilage). Scalar input is tolerated
    for backwards compatibility but the policy passes a list of >=3 entries.

    Task 3: per-ingredient TARGET_DAYS_COVER = max(global, lead_time + 2).
    Task 5: ingredients that recently stocked out get an inflated SAFETY.
    Task 8: if a supplier hogs more than DIVERSIFICATION_THRESHOLD of recent
    kg volume, we try to route at least one order this turn to an alternate.

    Phase 3: `regime_modifiers` is the dict produced by
    `regime.compose_modifiers(...)`. Optional — when omitted, the function
    behaves exactly as before. Recognized keys:
        inventory_safety_multiplier  : multiplies the global SAFETY_MULTIPLIER
        inventory_target_days_delta  : added to TARGET_DAYS_COVER
        diversification_force        : force a split order on any ingredient
                                       with >=2 viable suppliers
        single_supplier_share_cap    : redirect order so no supplier exceeds
                                       this fraction of today's pending volume

    `decisions` (optional dict) is populated with telemetry data such as
    `target_per_ingredient`, `effective_safety`, and per-supplier scores.
    """
    if decisions is None:
        decisions = {}

    # ---- Phase 3 regime modifier knobs (with safe defaults) ----
    rm = regime_modifiers or {}
    rm_safety_scale = float(rm.get("inventory_safety_multiplier", 1.0) or 1.0)
    rm_target_delta = int(rm.get("inventory_target_days_delta", 0) or 0)
    rm_div_force = bool(rm.get("diversification_force", False))
    rm_share_cap = float(rm.get("single_supplier_share_cap", 1.0) or 1.0)
    # Effective global tunables for this turn — overrides are local, not
    # mutating the module constants.
    eff_safety_base = SAFETY_MULTIPLIER * rm_safety_scale
    eff_target_global = TARGET_DAYS_COVER + rm_target_delta
    # Telemetry for what was actually applied.
    rm_applied: Dict[str, object] = {}
    if abs(rm_safety_scale - 1.0) > 1e-9:
        rm_applied["safety_scale"] = round(rm_safety_scale, 3)
    if rm_target_delta:
        rm_applied["target_days_delta"] = rm_target_delta
    if rm_div_force:
        rm_applied["diversification_force"] = True
    if rm_share_cap < 1.0:
        rm_applied["single_supplier_share_cap"] = round(rm_share_cap, 3)
    if rm_applied:
        decisions["regime_modifiers_applied"] = rm_applied

    # Normalize horizon input.
    if isinstance(expected_covers_horizon, (int, float)):
        horizon: List[float] = [float(expected_covers_horizon)]
    else:
        horizon = [float(x) for x in (expected_covers_horizon or [])]
    if not horizon:
        horizon = [0.0]
    today_covers = horizon[0]
    max_horizon = max(horizon)

    decisions["forecast_max_horizon"] = round(max_horizon, 1)
    decisions["forecast_horizon"] = [round(x, 1) for x in horizon]

    # Usage at today's demand (used for days_of_cover) and at the busy-day
    # peak (used for reorder sizing).
    usage_today = estimate_daily_usage(obs, today_covers, state)
    usage_peak = estimate_daily_usage(obs, max_horizon, state)

    # Telemetry: empirical dish mix for visibility.
    decisions["dish_mix"] = {
        k: round(v, 3) for k, v in estimate_dish_mix(obs, state).items()
    }

    orders: List[dict] = []
    cash_remaining = obs.get("cash", 0.0)
    # Phase 3: track per-supplier kg placed THIS turn so we can enforce
    # single_supplier_share_cap as orders accrue.
    turn_supplier_kg: Dict[str, float] = {}
    rm_share_redirects: List[Dict] = []
    # Fix 5: collect each capped order so telemetry can show what was
    # trimmed from `requested_kg` → `capped_kg`. Appended-to by all four
    # order paths (primary, split, Saturday floor, Day-1 starter).
    order_qty_capped: List[Dict] = []

    # Compute the day context early so Fix-5's _apply_qty_cap closure can
    # reference `days_remaining` from any order path (Day-1 starter included).
    today_index = int(obs.get("day", 0))
    dow_today = obs.get("day_of_week", "Monday")
    days_remaining = int(obs.get("days_remaining", 30 - today_index) or 0)

    def _apply_qty_cap(ingredient_name: str, daily_use: float, lead_time: int,
                      requested_kg: float) -> float:
        """
        Fix 5: cap single-order qty at daily_use * min(horizon, days_remaining+1)
        where horizon = lead_time + 5. Returns the capped qty and appends a
        telemetry entry when the cap fires. Caller decides what to do with
        the returned value (e.g. drop the order if it's now below min_order).
        """
        try:
            lt = int(lead_time or 2)
        except (TypeError, ValueError):
            lt = 2
        horizon_days = lt + 5
        if days_remaining > 0:
            cap_days = min(horizon_days, days_remaining + 1)
        else:
            cap_days = horizon_days
        cap_kg = max(0.0, float(daily_use) * float(cap_days))
        if cap_kg <= 0 or requested_kg <= cap_kg:
            return requested_kg
        order_qty_capped.append({
            "ingredient": ingredient_name,
            "requested_kg": round(requested_kg, 2),
            "capped_kg": round(cap_kg, 2),
            "cap_days": cap_days,
        })
        return cap_kg

    # ---- Fix 3: Day-1 baseline starter orders ----
    # On day 1 with no history, the main loop's days_of_cover check often
    # reports "infinite cover" because daily_usage estimates are based on an
    # uninformed prior and on_hand is the starter pack. Result: Day 1 ships
    # zero orders and we're behind from Day 2 onward. Force a starter pass
    # ordering one batch per active-menu ingredient targeting ≥7-day usage
    # at the busy-day peak. Bypasses the main-loop "skip if covered" check.
    day1_starter_orders: List[Dict] = []
    today_for_day1 = int(obs.get("day", 0) or 0)
    is_day1_starter = (today_for_day1 == 1) and (not state.history)
    if is_day1_starter:
        for ingredient, daily in usage_today.items():
            if daily <= 0:
                continue
            sup = pick_supplier(
                ingredient, obs, state,
                today_index=today_for_day1,
                dow_today=obs.get("day_of_week", "Monday"),
            )
            if not sup or ingredient not in (sup.get("ingredients") or {}):
                continue
            unit_price = float(sup["ingredients"][ingredient])
            min_order = float(sup.get("min_order_kg", 0.0) or 0.0)
            # Target ≥7 days of expected usage at the peak in the horizon.
            peak_daily = max(daily, usage_peak.get(ingredient, daily))
            want_kg = max(min_order, peak_daily * 7.0)
            # Respect shelf cap so we don't spoil.
            inv_for_shelf = _find(obs.get("inventory", []), "ingredient", ingredient) or {}
            shelf = inv_for_shelf.get("shelf_life_days") or 14
            max_safely_usable = peak_daily * max(1, shelf - 1)
            want_kg = min(want_kg, max_safely_usable)
            # Fix 5: cap by lead_time + 5 horizon.
            lt_d1 = int(sup.get("lead_time_days", 2) or 2)
            want_kg = _apply_qty_cap(ingredient, peak_daily, lt_d1, want_kg)
            # Cap may have shrunk below min_order — only keep order if still viable.
            want_kg = max(want_kg, min_order if want_kg >= min_order * 0.5 else 0.0)
            if want_kg <= 0:
                continue
            cost = want_kg * unit_price
            staff_cost_d1 = obs.get("staff_level", 8) * obs.get("staff_cost_per_person", 120)
            floor_d1 = 3 * (300 + staff_cost_d1)
            if cost > cash_remaining - floor_d1:
                want_kg = min_order
                cost = want_kg * unit_price
                if cost > cash_remaining - floor_d1 or want_kg <= 0:
                    continue
            orders.append({
                "tool": "place_order",
                "args": {
                    "supplier": sup["name"],
                    "ingredient": ingredient,
                    "quantity_kg": round(want_kg, 2),
                },
            })
            cash_remaining -= cost
            turn_supplier_kg[sup["name"]] = turn_supplier_kg.get(sup["name"], 0.0) + want_kg
            try:
                state.record_placed_order(today_for_day1, sup["name"], ingredient, want_kg)
            except Exception:
                pass
            day1_starter_orders.append({
                "ingredient": ingredient,
                "supplier": sup["name"],
                "quantity_kg": round(want_kg, 2),
            })
    decisions["day1_starter_orders"] = day1_starter_orders
    # Track ingredients with day-1 starter orders so the main loop below
    # doesn't double-up on the same ingredient (the starter already covers
    # ~7 days of usage which dwarfs the main pass's 4-6 day target).
    day1_ordered_ings = {o["ingredient"] for o in day1_starter_orders}

    # Reserve a cash floor: 3 days of fixed + staff overhead
    staff_cost = obs.get("staff_level", 8) * obs.get("staff_cost_per_person", 120)
    floor = 3 * (300 + staff_cost)

    # Sort by tightest cover first (most urgent gets first claim on cash)
    by_urgency = sorted(
        usage_today.items(),
        key=lambda kv: days_of_cover(kv[0], obs, kv[1]),
    )

    target_per_ing: Dict[str, int] = {}
    safety_per_ing: Dict[str, float] = {}
    supplier_scores_telem: Dict[str, list] = {}

    # Task 8: precompute current supplier share for diversification check.
    share_map = state.supplier_volume_share(DIVERSIFICATION_LOOKBACK)
    hogging_suppliers = {s for s, frac in share_map.items() if frac > DIVERSIFICATION_THRESHOLD}
    decisions["supplier_share_recent"] = {k: round(v, 3) for k, v in share_map.items()}
    decisions["hogging_suppliers"] = sorted(hogging_suppliers)
    diversified_routed = False  # we only need to satisfy this once per turn

    # Diff 1: Sunday-closure pre-stock window.
    # No supplier delivers Sunday, so Sat orders are too late. Thu/Fri are the
    # last actionable days. For shelf-stable ingredients (shelf >= 4 days) we
    # force per_ing_target to cover Sat + Sun + Mon recovery buffer.
    try:
        dow_today_idx = _WEEKDAYS.index(dow_today)
    except ValueError:
        dow_today_idx = 0
    days_to_sunday = (6 - dow_today_idx) % 7
    is_prestock_window = dow_today in ("Thursday", "Friday")
    decisions["days_to_sunday"] = days_to_sunday
    decisions["prestock_sunday"] = is_prestock_window
    prestock_ingredients_list: List[str] = []

    # Port-fabiano-A: ingredients whose dishes ran OUT during yesterday's
    # service get top-priority same-day reorder. service_summary supplies
    # `dishes_unavailable_at` as a {dish: time-string} map; we expand each
    # dish to its recipe ingredients.
    urgent_ings: set = set()
    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}
    svc = obs.get("service_summary") or {}
    for dish in (svc.get("dishes_unavailable_at") or {}):
        for ing in menu_book.get(dish, {}).get("ingredients", []) or []:
            urgent_ings.add(ing["ingredient"])
    decisions["urgent_ingredients"] = sorted(urgent_ings)

    # Port-fabiano-B: demand-surge factor. If the last day was busier than the
    # rolling average we widen the safety buffer by that ratio (caps at 2x).
    covers_hist = [r.covers for r in state.history]
    avg_cov = sum(covers_hist) / len(covers_hist) if covers_hist else 0.0
    last_cov = covers_hist[-1] if covers_hist else 0.0
    if avg_cov > 10:
        demand_surge = max(1.0, min(2.0, last_cov / avg_cov))
    else:
        demand_surge = 1.0
    decisions["demand_surge_factor"] = round(demand_surge, 2)

    # Port-fabiano-B: scenario gating on horizon (longer buffer during supply
    # crisis, shorter during renovation since we can't serve as many).
    horizon_bias = 1.0
    if scenario == "supply_crisis":
        horizon_bias = 1.2
    elif scenario == "renovation":
        horizon_bias = 0.7
    decisions["order_scenario"] = scenario
    decisions["order_horizon_bias"] = horizon_bias

    for ingredient, daily in by_urgency:
        if daily <= 0:
            continue
        # Fix 3: Day-1 starter already covered this ingredient — skip the
        # main pass to avoid double-ordering 14+ days of stock on Day 1.
        if is_day1_starter and ingredient in day1_ordered_ings:
            continue
        cover = days_of_cover(ingredient, obs, daily)

        # Task 6/7: pick supplier first so we know its lead time + delivery day.
        # We may swap to an alternate later (Task 8 diversification).
        primary = pick_supplier(
            ingredient, obs, state,
            today_index=today_index, dow_today=dow_today,
            scores_out=supplier_scores_telem,
        )
        if not primary:
            continue

        # Task 3: per-ingredient target = max(global, lead_time + 2).
        # Phase 3: eff_target_global already incorporates the regime delta.
        nominal_lead = int(primary.get("lead_time_days", 2) or 2)
        per_ing_target = max(eff_target_global, nominal_lead + 2)
        # Port-fabiano-B: scale by scenario horizon bias.
        per_ing_target = max(2, int(round(per_ing_target * horizon_bias)))

        # Diff 1: Sunday-closure pre-stock. If today is Thu/Fri AND the
        # ingredient is shelf-stable (>= 4d), force lead_time + 3 cover
        # to bridge Sat + Sun (no delivery) + Mon recovery buffer.
        inv_for_shelf = _find(obs.get("inventory", []), "ingredient", ingredient) or {}
        shelf_for_check = inv_for_shelf.get("shelf_life_days") or 14
        if is_prestock_window and shelf_for_check >= 4:
            prestock_target = nominal_lead + 3
            if prestock_target > per_ing_target:
                per_ing_target = prestock_target
                prestock_ingredients_list.append(ingredient)

        target_per_ing[ingredient] = per_ing_target

        # Port-fabiano-A: if yesterday's service ran OUT of dishes using this
        # ingredient, force a reorder even if days_of_cover looks adequate.
        is_urgent = ingredient in urgent_ings
        if (not is_urgent) and cover >= per_ing_target:
            continue

        # Task 5: stockout-history feedback. Bump SAFETY for recent stockouts,
        # decaying back to baseline over STOCKOUT_DECAY_DAYS clean days.
        # Phase 3: eff_safety_base already includes the regime-driven scale.
        eff_safety = eff_safety_base
        ds = state.days_since_stockout(ingredient)
        if ds <= STOCKOUT_WINDOW_DAYS:
            eff_safety *= STOCKOUT_SAFETY_BUMP
        elif ds <= STOCKOUT_WINDOW_DAYS + STOCKOUT_DECAY_DAYS:
            # linear decay from full bump → 1.0 across STOCKOUT_DECAY_DAYS
            clean_days = ds - STOCKOUT_WINDOW_DAYS
            decay_frac = 1.0 - (clean_days / STOCKOUT_DECAY_DAYS)
            eff_safety *= 1.0 + (STOCKOUT_SAFETY_BUMP - 1.0) * decay_frac
        # Port-fabiano-B: demand surge widens the safety buffer.
        eff_safety *= demand_surge
        # Port-fabiano-A: urgent ingredients get an additional bump.
        if is_urgent:
            eff_safety *= 1.3
        safety_per_ing[ingredient] = round(eff_safety, 3)

        # Task 1: size against the busiest day in the horizon, but cap by what
        # we can plausibly use before the shortest plausible shelf life.
        peak_daily = max(daily, usage_peak.get(ingredient, daily))
        need_kg = (per_ing_target * peak_daily * eff_safety) - (cover * daily)

        # Shelf-life cap: never order more than you can plausibly use before expiry
        inv = _find(obs.get("inventory", []), "ingredient", ingredient) or {}
        shelf = inv.get("shelf_life_days") or 14
        # When shelf life is short, anchor max usable to peak_daily (so a busy
        # day forecast still allows a respectable order). For longer-shelf
        # items shelf bound stays loose either way.
        max_safely_usable = peak_daily * max(1, shelf - 1)
        need_kg = min(need_kg, max_safely_usable)
        # Fix 5: cap single-order qty at peak_daily * min(lead_time+5,
        # days_remaining+1). Applied after shelf cap and before end-of-game
        # cap so we don't oversize from a single bad horizon forecast.
        need_kg = _apply_qty_cap(
            ingredient, peak_daily,
            int(primary.get("lead_time_days", 2) or 2),
            need_kg,
        )

        # Port-fabiano-C: end-of-game cap. Don't order more kg than we can
        # realistically use in the days remaining (plus one for slack).
        if days_remaining > 0:
            on_hand_now = float((inv or {}).get("total_kg", 0.0) or 0.0)
            pending_now = sum(
                float(po.get("quantity_kg") or 0.0)
                for po in (obs.get("pending_orders") or [])
                if po.get("ingredient") == ingredient
            )
            max_useful = peak_daily * (days_remaining + 1) - (on_hand_now + pending_now)
            if max_useful > 0:
                need_kg = min(need_kg, max_useful)
            else:
                # already covered through end-of-game; only urgent stockouts justify a buy
                if not is_urgent:
                    continue

        if need_kg <= 0:
            continue

        # Task 8: if primary is a hogging supplier and an alternate exists, try
        # routing this single order to a secondary (only once per turn).
        chosen = primary
        if (
            not diversified_routed
            and primary["name"] in hogging_suppliers
        ):
            alt = pick_supplier(
                ingredient, obs, state,
                today_index=today_index, dow_today=dow_today,
                exclude={primary["name"]},
            )
            if alt and ingredient in (alt.get("ingredients") or {}):
                chosen = alt
                diversified_routed = True
                decisions.setdefault("diversification_routed", []).append({
                    "ingredient": ingredient,
                    "from": primary["name"],
                    "to": alt["name"],
                })

        # Phase 3: single_supplier_share_cap. If routing this order to `chosen`
        # would push their share of this turn's total kg above the cap, try an
        # alternate first. We check share against (cumulative + this order).
        if rm_share_cap < 1.0:
            current_total = sum(turn_supplier_kg.values())
            chosen_after = turn_supplier_kg.get(chosen["name"], 0.0) + max(need_kg, 0.0)
            new_total = current_total + max(need_kg, 0.0)
            if new_total > 0 and chosen_after / new_total > rm_share_cap:
                alt_cap = pick_supplier(
                    ingredient, obs, state,
                    today_index=today_index, dow_today=dow_today,
                    exclude={chosen["name"]},
                )
                if alt_cap and ingredient in (alt_cap.get("ingredients") or {}):
                    rm_share_redirects.append({
                        "ingredient": ingredient,
                        "from": chosen["name"],
                        "to": alt_cap["name"],
                        "cap": round(rm_share_cap, 3),
                    })
                    chosen = alt_cap

        unit_price = chosen["ingredients"][ingredient]
        min_order = chosen.get("min_order_kg", 0.0)
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
                "supplier": chosen["name"],
                "ingredient": ingredient,
                "quantity_kg": round(qty, 2),
            },
        })
        cash_remaining -= cost
        turn_supplier_kg[chosen["name"]] = turn_supplier_kg.get(chosen["name"], 0.0) + qty
        # Task 8: log placed order for diversification stats next turn.
        try:
            state.record_placed_order(today_index, chosen["name"], ingredient, qty)
        except Exception:
            pass

        # Diff 3: allow a second order to a *different* supplier when we have
        # slack (cover >= MIN_DAYS_COVER) OR we're in the initial stock-up
        # phase (day <= 3). Diagnostic showed Day 5 lettuce shipped only
        # 6.4 kg from a single supplier when ~12 kg was needed for Saturday.
        # Phase 3: rm_div_force forces the split whenever an alternate exists,
        # regardless of slack/early-game (e.g. during a supply_ingredient-
        # regime we want to spread risk across suppliers).
        not_critical = cover >= MIN_DAYS_COVER
        early_game = today_index <= 3
        if not_critical or early_game or rm_div_force:
            alt2 = pick_supplier(
                ingredient, obs, state,
                today_index=today_index, dow_today=dow_today,
                exclude={chosen["name"]},
            )
            if alt2 and ingredient in (alt2.get("ingredients") or {}):
                alt_price = float(alt2["ingredients"][ingredient])
                alt_min = float(alt2.get("min_order_kg", 0.0) or 0.0)
                # Aim for ~half the primary order, but never less than alt_min.
                alt_qty = max(alt_min, qty * 0.5)
                # Respect shelf cap (use same max_safely_usable bound).
                alt_qty = min(alt_qty, max_safely_usable)
                # Fix 5: cap the split-order qty too.
                alt_qty = _apply_qty_cap(
                    ingredient, peak_daily,
                    int(alt2.get("lead_time_days", 2) or 2),
                    alt_qty,
                )
                # Respect end-of-game cap when applicable.
                if days_remaining > 0:
                    on_hand_now = float((inv or {}).get("total_kg", 0.0) or 0.0)
                    pending_now = sum(
                        float(po.get("quantity_kg") or 0.0)
                        for po in (obs.get("pending_orders") or [])
                        if po.get("ingredient") == ingredient
                    )
                    remaining_cap = peak_daily * (days_remaining + 1) - (
                        on_hand_now + pending_now + qty
                    )
                    if remaining_cap > 0:
                        alt_qty = min(alt_qty, remaining_cap)
                    else:
                        alt_qty = 0.0
                alt_cost = alt_qty * alt_price
                if alt_qty > 0 and alt_cost <= cash_remaining - floor:
                    orders.append({
                        "tool": "place_order",
                        "args": {
                            "supplier": alt2["name"],
                            "ingredient": ingredient,
                            "quantity_kg": round(alt_qty, 2),
                        },
                    })
                    cash_remaining -= alt_cost
                    turn_supplier_kg[alt2["name"]] = (
                        turn_supplier_kg.get(alt2["name"], 0.0) + alt_qty
                    )
                    try:
                        state.record_placed_order(
                            today_index, alt2["name"], ingredient, alt_qty
                        )
                    except Exception:
                        pass
                    split_reason = (
                        "regime_diversification" if rm_div_force
                        else ("slack" if not_critical else "early_game")
                    )
                    decisions.setdefault("split_orders", []).append({
                        "ingredient": ingredient,
                        "primary": chosen["name"],
                        "secondary": alt2["name"],
                        "primary_qty": round(qty, 2),
                        "secondary_qty": round(alt_qty, 2),
                        "reason": split_reason,
                    })

    # ---- Fix 2: Saturday pre-stock floor ----
    # Diff 1 already pre-stocks on Thu/Fri (days_to_sunday <= 2) for shelf-
    # stable ingredients, but the head-to-head vs Fabiano showed Saturday
    # itself is still a black hole: any ingredient we forgot to order Thu/Fri
    # has nowhere to recover from before Sunday closure (no delivery) and
    # Monday service. We force a minimum order on Saturday for every active-
    # menu ingredient whose days_of_cover < 2, regardless of the main loop's
    # decisions. Cash gate: skip ONLY if cash < €1500 (true emergency).
    sat_floor_orders: List[Dict] = []
    sat_floor_skipped_low_cash = False
    if dow_today == "Saturday":
        if cash_remaining < 1500.0:
            sat_floor_skipped_low_cash = True
        else:
            already_ordered_today = {
                a["args"]["ingredient"] for a in orders
            }
            for ingredient, daily in usage_today.items():
                if daily <= 0:
                    continue
                if ingredient in already_ordered_today:
                    continue
                cover = days_of_cover(ingredient, obs, daily)
                if cover >= 2.0:
                    continue
                sup = pick_supplier(
                    ingredient, obs, state,
                    today_index=today_index, dow_today=dow_today,
                )
                if not sup or ingredient not in (sup.get("ingredients") or {}):
                    continue
                unit_price = float(sup["ingredients"][ingredient])
                min_order = float(sup.get("min_order_kg", 0.0) or 0.0)
                # Aim for 2 days of cover at peak demand. Saturday demand is
                # high so peak_daily here is from usage_peak.
                peak_daily = max(daily, usage_peak.get(ingredient, daily))
                want_kg = max(min_order, peak_daily * 2.0)
                # Fix 5: cap by lead_time + 5 horizon.
                want_kg = _apply_qty_cap(
                    ingredient, peak_daily,
                    int(sup.get("lead_time_days", 2) or 2),
                    want_kg,
                )
                # End-of-game cap.
                if days_remaining > 0:
                    inv_for_cap = _find(obs.get("inventory", []), "ingredient", ingredient) or {}
                    on_hand_now = float(inv_for_cap.get("total_kg", 0.0) or 0.0)
                    pending_now = sum(
                        float(po.get("quantity_kg") or 0.0)
                        for po in (obs.get("pending_orders") or [])
                        if po.get("ingredient") == ingredient
                    )
                    max_useful = peak_daily * (days_remaining + 1) - (on_hand_now + pending_now)
                    if max_useful <= 0:
                        continue
                    want_kg = min(want_kg, max_useful)
                if want_kg <= 0:
                    continue
                cost = want_kg * unit_price
                if cost > cash_remaining - 500.0:  # tighter floor than main loop
                    want_kg = min_order
                    cost = want_kg * unit_price
                    if cost > cash_remaining - 500.0 or want_kg <= 0:
                        continue
                orders.append({
                    "tool": "place_order",
                    "args": {
                        "supplier": sup["name"],
                        "ingredient": ingredient,
                        "quantity_kg": round(want_kg, 2),
                    },
                })
                cash_remaining -= cost
                turn_supplier_kg[sup["name"]] = (
                    turn_supplier_kg.get(sup["name"], 0.0) + want_kg
                )
                try:
                    state.record_placed_order(today_index, sup["name"], ingredient, want_kg)
                except Exception:
                    pass
                sat_floor_orders.append({
                    "ingredient": ingredient,
                    "supplier": sup["name"],
                    "quantity_kg": round(want_kg, 2),
                    "cover_before": round(cover, 2),
                })
    decisions["saturday_floor_orders"] = sat_floor_orders
    decisions["saturday_floor_skipped_low_cash"] = sat_floor_skipped_low_cash

    decisions["prestock_ingredients"] = prestock_ingredients_list
    decisions["target_per_ingredient"] = target_per_ing
    decisions["effective_safety"] = safety_per_ing
    decisions["supplier_scores"] = supplier_scores_telem
    # Fix 5 telemetry — list of (ingredient, requested_kg, capped_kg).
    decisions["order_qty_capped"] = order_qty_capped
    # Phase 3 telemetry: record post-hoc evidence of regime-driven actions.
    if rm_share_redirects:
        decisions["regime_share_cap_redirects"] = rm_share_redirects
        if "regime_modifiers_applied" in decisions:
            decisions["regime_modifiers_applied"]["share_cap_redirects"] = len(
                rm_share_redirects
            )

    return orders


def pick_supplier(
    ingredient: str,
    obs: dict,
    state: AgentState,
    today_index: Optional[int] = None,
    dow_today: Optional[str] = None,
    exclude: Optional[set] = None,
    scores_out: Optional[Dict] = None,
) -> Optional[dict]:
    """
    Best supplier for an ingredient. Combines:
      - delivered_ratio (reliability)            — Task 6 carry-over
      - on_time_rate                             — Task 6: penalize chronic lateness
      - effective wait = days until first valid  — Task 7: delivery_days schedule
        delivery_day >= today + lead_time
      - unit price                               — light penalty

    `exclude` lets a caller skip already-routed suppliers (used by Task 8
    diversification). `scores_out` is an optional telemetry sink keyed by
    ingredient: each entry is a list of (supplier_name, score) tuples.
    """
    candidates = [
        s for s in (obs.get("supplier_catalog") or [])
        if ingredient in (s.get("ingredients") or {})
        and (not exclude or s.get("name") not in exclude)
    ]
    if not candidates:
        return None

    if dow_today is None:
        dow_today = obs.get("day_of_week", "Monday")
    if today_index is None:
        today_index = int(obs.get("day", 0))

    scored: List[tuple] = []
    for s in candidates:
        rel_stat = state.suppliers.get(s["name"])
        reliability = rel_stat.delivered_ratio if rel_stat else 1.0
        on_time = rel_stat.on_time if rel_stat else 1.0
        if reliability < SUPPLIER_RELIABILITY_FLOOR:
            scored.append((s, -1.0))
            continue

        nominal_lead = float(s.get("lead_time_days", 2) or 2)
        # Task 6: penalize chronic lateness by stretching the lead time.
        effective_lead = nominal_lead / max(on_time, 0.5)

        # Task 7: convert effective_lead to actual wait given delivery_days.
        actual_wait = _effective_wait_days(
            dow_today=dow_today,
            lead_days=effective_lead,
            delivery_days=s.get("delivery_days") or [],
        )

        price = float(s["ingredients"][ingredient])
        score = reliability * 10.0 - actual_wait - price * 0.05
        scored.append((s, score))

    if scores_out is not None:
        scores_out[ingredient] = [
            {"supplier": s["name"], "score": round(sc, 3)} for s, sc in scored
        ]

    # Pick best by score. If all are -1.0 (all unreliable), fall back to highest
    # delivered_ratio so we still attempt SOMETHING.
    best = max(scored, key=lambda x: x[1])
    return best[0] if best else None


def _effective_wait_days(
    dow_today: str,
    lead_days: float,
    delivery_days: list,
) -> float:
    """
    Task 7: given today's weekday, a (possibly fractional) lead time, and the
    supplier's allowed delivery weekdays, return the number of days from today
    until the first valid delivery.

    If `delivery_days` is empty we fall back to the raw lead time (suppliers
    that deliver any day get no schedule penalty).
    """
    if not delivery_days:
        return lead_days
    try:
        today_idx = _WEEKDAYS.index(dow_today)
    except ValueError:
        return lead_days
    allowed = set()
    for d in delivery_days:
        if d in _WEEKDAYS:
            allowed.add(_WEEKDAYS.index(d))
    if not allowed:
        return lead_days
    # Earliest day-offset at which the order CAN arrive: ceil(lead_days)
    import math
    earliest_offset = max(1, int(math.ceil(lead_days)))
    # Scan up to 14 days ahead for the first delivery_day weekday match.
    for offset in range(earliest_offset, earliest_offset + 14):
        if (today_idx + offset) % 7 in allowed:
            return float(offset)
    return float(earliest_offset)


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
