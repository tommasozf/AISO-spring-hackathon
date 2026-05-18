"""
Tier 2 — policy orchestrator.

Combines forecast + inventory + staffing + pricing + promo into the list of
action dicts to submit, then applies the Tier 1 safety filter.

Returns: list[dict] in the form [{"tool": "...", "args": {...}}, ...]
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import (
    forecast,
    inventory,
    pricing,
    regime as regime_mod,
    safety,
    scenario as scenario_mod,
    staffing,
)
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

    # ---- 0pre. regime detection (Phase 3: feeds modifiers into actions) ----
    # Pure function — populates state.{day1_baseline, residuals, regime_days,
    # regime_streaks, regime_signals} as a side effect. Complementary to the
    # scenario tag from scenario.py: scenario gives a mode label, regimes
    # give per-axis quantitative signals. Both coexist in telemetry.
    try:
        regimes = regime_mod.detect_regimes(obs, state)
        record["regimes"] = {k: v.to_dict() for k, v in regimes.items()}
    except Exception as e:
        # Detection must never crash the agent — log and continue.
        record["regimes"] = {}
        record["regime_error"] = repr(e)
        regimes = {}

    # ---- 0pre.b. compose modifiers (Phase 3 wiring) ----
    # Passed through to inventory/staffing/pricing so a confirmed regime
    # produces real action deltas, not just telemetry.
    try:
        regime_modifiers = regime_mod.compose_modifiers(state.regime_signals)
    except Exception as e:
        regime_modifiers = {}
        record["regime_modifiers_error"] = repr(e)
    record["regime_modifiers"] = regime_modifiers

    # ---- 0a. scenario detection (alert TEXT only — not reputation) ----
    scenario_decisions: Dict = {}
    scenario_from_alerts = scenario_mod.detect_scenario(
        obs,
        state,
        cover_history=[r.covers for r in state.history],
        decisions=scenario_decisions,
    )

    # Phase 3 (Change 2): if the alert-based scenario has decayed back to
    # "baseline" but the regime detector still believes a per-axis signal is
    # active, hold the scenario at the regime-implied tag so that downstream
    # consumers (inventory.scenario gating, staffing.scenario gating,
    # marketing/happy-hour gates) keep doing the right thing across the long
    # tail of a crisis the alert text only mentions once.
    #
    # Rules: only UPGRADE baseline → specific. We never overwrite a more
    # specific scenario fired from alerts with a regime-derived tag.
    scenario_source = "alert"
    final_scenario = scenario_from_alerts
    if final_scenario == "baseline":
        sigs = state.regime_signals or {}

        def _sig_sign(axis: str) -> int:
            try:
                return int((sigs.get(axis) or {}).get("sign", 0) or 0)
            except (TypeError, ValueError):
                return 0

        if _sig_sign(regime_mod.AXIS_SUPPLY_ING) == -1:
            final_scenario = "supply_crisis"
            scenario_source = "regime_hold"
        elif _sig_sign(regime_mod.AXIS_SUPPLY_CAP) == -1:
            final_scenario = "renovation"
            scenario_source = "regime_hold"
        elif _sig_sign(regime_mod.AXIS_DEMAND) == +1:
            final_scenario = "tourist"
            scenario_source = "regime_hold"
        elif _sig_sign(regime_mod.AXIS_DEMAND) == -1:
            final_scenario = "health_scare"
            scenario_source = "regime_hold"
        elif _sig_sign(regime_mod.AXIS_COST) == +1:
            final_scenario = "inflation"
            scenario_source = "regime_hold"

    state.scenario = final_scenario
    record["scenario"] = state.scenario
    record["scenario_decisions"] = scenario_decisions
    record["scenario_source"] = scenario_source

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

    # ---- 1b. narrowed (low-stock) dishes (Diff 2) ----
    # Any active dish with fewer than `narrow_threshold` servings available
    # on-hand is excluded from the active menu for the day. This stops the
    # kitchen from trying to serve 8 dishes on 0.1 kg of staples (root cause
    # of the Sunday/Monday walkout cascade in the diagnostic run).
    #
    # Phase 3: when a regime modifier raises the threshold (e.g. 8.0 during a
    # supply_ingredient- regime) we narrow MORE aggressively, pre-empting the
    # collapse that follows a multi-ingredient simultaneous depletion.
    narrow_threshold = float(regime_modifiers.get("menu_narrow_threshold", 5.0) or 5.0)
    narrowed = _compute_narrowed_dishes(obs, threshold=narrow_threshold)
    record["narrowed_dishes"] = sorted(narrowed)
    record["narrow_threshold"] = narrow_threshold

    # ---- 1c. menu re-expand (Fix 1: hysteresis against the one-way ratchet) ----
    # Once dishes are dropped from active_menu (via narrow-then-_choose_menu),
    # there was no signal to add them BACK once ingredients restocked. In the
    # supply_crisis seed-42 run, Days 25-30 had 277kg flour / 25kg mozz / 38kg
    # tomato unused because the menu had locked to a narrow 5-dish set whose
    # ingredients were at zero — the well-stocked dishes never re-entered.
    #
    # Hysteresis: narrow at <narrow_threshold (=5 default), re-expand at
    # re_expand_threshold (>=8). The +3 gap prevents flapping even when the
    # regime modifier bumps narrow_threshold up to 8.0.
    re_expand_threshold = max(8.0, narrow_threshold + 3.0)
    inactive_high_stock = _compute_inactive_high_stock_dishes(
        obs, threshold=re_expand_threshold,
    )
    record["menu_reexpand_threshold"] = re_expand_threshold
    record["menu_reexpanded"] = sorted(inactive_high_stock.keys())
    record["menu_reexpand_min_servings"] = {
        k: round(v, 1) for k, v in inactive_high_stock.items()
    }

    # ---- 2. inventory diagnostics + orders ----
    daily_usage = inventory.estimate_daily_usage(obs, expected_today, state)
    record["daily_usage"] = {k: round(v, 3) for k, v in daily_usage.items()}
    record["days_of_cover"] = {
        k: round(inventory.days_of_cover(k, obs, u), 2)
        for k, u in daily_usage.items()
    }

    # Task 1: pass the multi-day forecast horizon (today + next 2 days) so
    # reorder_plan can size against the busiest day, not just today.
    horizon = covers_forecast[:3] if covers_forecast else [expected_today]
    inv_decisions: Dict = {}
    proposed_orders = inventory.reorder_plan(
        obs, state, horizon,
        decisions=inv_decisions,
        scenario=state.scenario,
        regime_modifiers=regime_modifiers,
    )
    record["inventory_decisions"] = inv_decisions
    record["orders_proposed"] = [
        {"ingredient": a["args"]["ingredient"],
         "supplier": a["args"]["supplier"],
         "quantity_kg": a["args"]["quantity_kg"]}
        for a in proposed_orders
    ]
    actions.extend(proposed_orders)

    # ---- 3. menu (day 1 / unmakeable / narrowed / Fix 1 re-expand) ----
    if _should_set_menu(
        obs, day, state,
        narrowed_dishes=narrowed,
        inactive_high_stock=inactive_high_stock,
    ):
        menu = _choose_menu(
            obs, state,
            narrowed_dishes=narrowed,
            reexpand_dishes=set(inactive_high_stock.keys()),
        )
        if menu:
            actions.append({"tool": "set_menu", "args": {"dishes": menu}})
            state.last_menu_change_day = day
            record["menu_set"] = menu
    record.setdefault("menu_set", None)

    # ---- 4. staffing (port-fabiano: DOW table + weather/trend/walkout) ----
    staff_decisions: Dict = {}
    target_staff = staffing.compute_staff_level(
        obs, state,
        scenario=state.scenario,
        expected_covers_today=expected_today,
        decisions=staff_decisions,
        regime_modifiers=regime_modifiers,
    )
    record["staff_target"] = target_staff
    record["staff_current"] = obs.get("staff_level")
    record["staffing_decisions"] = staff_decisions
    if target_staff != obs.get("staff_level"):
        actions.append({"tool": "set_staff_level", "args": {"level": target_staff}})

    # ---- 5. pricing (algorithmic: walkout × inv-pressure × DOW × rep-cap × margin) ----
    price_actions = pricing.pricing_actions(
        obs, state, expected_today, record=record,
        regime_modifiers=regime_modifiers,
    )
    actions.extend(price_actions)
    record["price_changes"] = [
        {"dish": a["args"]["dish"], "price": a["args"]["price"]}
        for a in price_actions
    ]

    # ---- 6. marketing & promos ----
    mkt = _marketing_for(obs, expected_today, scenario=state.scenario, day=day)
    record["marketing"] = mkt
    if mkt > 0:
        actions.append({"tool": "set_marketing_spend", "args": {"amount": mkt}})

    hh = _should_happy_hour(obs, day, state, expected_today, scenario=state.scenario)
    record["happy_hour"] = hh
    if hh:
        actions.append({"tool": "run_happy_hour", "args": {}})
        state.last_happy_hour_day = day
        state.happy_hour_days.append(day)
        if len(state.happy_hour_days) > 10:
            state.happy_hour_days = state.happy_hour_days[-10:]

    # ---- 7. daily special (Diff 4: round-robin instead of soonest-expiry) ----
    special = _pick_daily_special(obs, day=day, narrowed_dishes=narrowed)
    record["daily_special"] = special
    record["daily_special_strategy"] = "round_robin"
    record["daily_special_dish"] = special
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

def _should_set_menu(
    obs: dict,
    day: int,
    state: AgentState,
    narrowed_dishes: Optional[set] = None,
    inactive_high_stock: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Set the menu in four cases:
      a) Day 1 / fresh game with no active_menu (legacy bootstrap).
      b) Port-fabiano-2: an active dish has become un-makeable — every
         ingredient in its recipe is at zero on-hand AND there is no
         pending order arriving in the next 2 days. In that case we drop
         the broken dishes (keeping the rest) so the kitchen stops
         walking customers out for an item we can't actually serve.
      c) Diff 2: any active dish has fewer than 5 servings on-hand. We
         temporarily narrow the menu (will re-include on a subsequent day
         when stock returns — this fn fires again when active_menu changes).
      d) Fix 1: an inactive dish (i.e. one we previously dropped from
         active_menu) now has >= re_expand_threshold servings reachable.
         The diagnostic run showed we have a one-way ratchet without this
         — once Diff 2 narrows the menu, dishes never get added back even
         when ingredients restock.

    Safety.filter_unsafe still freezes set_menu in the endgame window.
    """
    if not obs.get("active_menu"):
        return True
    # Skip in endgame regardless — safety will drop it anyway.
    if safety.in_endgame(obs):
        return False
    if _has_unmakeable_active_dish(obs):
        return True
    # Diff 2: trigger when any active dish is in the narrowed set.
    if narrowed_dishes:
        active = obs.get("active_menu") or []
        if any(d in narrowed_dishes for d in active):
            return True
    # Fix 1: trigger when any inactive dish is fully restocked.
    if inactive_high_stock:
        return True
    return False


def _has_unmakeable_active_dish(obs: dict) -> bool:
    active = obs.get("active_menu") or []
    if not active:
        return False
    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}
    inv_map = {
        str(i.get("ingredient")): float(i.get("total_kg") or 0.0)
        for i in (obs.get("inventory") or [])
    }
    today = int(obs.get("day", 0) or 0)
    pending_soon: Dict[str, float] = {}
    for po in obs.get("pending_orders") or []:
        ing = po.get("ingredient")
        if ing and (po.get("delivery_day", 999) - today) <= 2:
            pending_soon[ing] = pending_soon.get(ing, 0.0) + float(po.get("quantity_kg") or 0.0)

    for dish_name in active:
        recipe = menu_book.get(dish_name, {}).get("ingredients") or []
        if not recipe:
            continue
        if all(
            inv_map.get(ing["ingredient"], 0.0) <= 0.0
            and pending_soon.get(ing["ingredient"], 0.0) <= 0.0
            for ing in recipe
        ):
            return True
    return False


def _choose_menu(
    obs: dict,
    state: AgentState,
    narrowed_dishes: Optional[set] = None,
    reexpand_dishes: Optional[set] = None,
) -> list:
    """
    Port-fabiano-2: pick the largest viable subset of the menu_book.

    Viable = at least one serving's worth of every ingredient is reachable
    (on-hand or arriving within ~3 days). Falls back to the first 5 dishes
    of menu_book if no viable subset exists.

    Diff 2: `narrowed_dishes` are dishes with <5 servings on-hand today and
    are filtered out of the active set (they may still appear in the
    fill-from-book step if they're reachable, but only after non-narrowed
    candidates are exhausted).

    Fix 1: `reexpand_dishes` are inactive dishes whose ingredients are now
    fully restocked (>= re_expand_threshold servings on-hand). These get
    prepended to the fill-pass candidate list so a restocked dish is
    *preferred* over an active-but-narrowed dish.

    We aim for 5-8 dishes for variety. Active_menu is preserved verbatim
    when every active dish is still viable; otherwise we keep the viable
    subset of `active_menu`, then fill from menu_book until we hit 5.
    """
    narrowed_dishes = narrowed_dishes or set()
    reexpand_dishes = reexpand_dishes or set()
    book = obs.get("menu_book", []) or []
    if not book:
        return []
    active = obs.get("active_menu", []) or []

    inv_map = {
        str(i.get("ingredient")): float(i.get("total_kg") or 0.0)
        for i in (obs.get("inventory") or [])
    }
    today = int(obs.get("day", 0) or 0)
    pending_soon: Dict[str, float] = {}
    for po in obs.get("pending_orders") or []:
        ing = po.get("ingredient")
        if ing and (po.get("delivery_day", 999) - today) <= 3:
            pending_soon[ing] = pending_soon.get(ing, 0.0) + float(po.get("quantity_kg") or 0.0)

    def _viable(dish: dict) -> bool:
        recipe = dish.get("ingredients") or []
        if not recipe:
            return False
        for ing in recipe:
            qty_needed = float(ing.get("quantity_kg") or 0.0)
            if qty_needed <= 0:
                continue
            reachable = inv_map.get(ing["ingredient"], 0.0) + pending_soon.get(ing["ingredient"], 0.0)
            if reachable < qty_needed:
                return False
        return True

    name_to_dish = {d["name"]: d for d in book}

    # Diff 2: narrowed dishes are excluded from the preferred active set.
    viable_active = [
        d for d in active
        if d in name_to_dish
        and _viable(name_to_dish[d])
        and d not in narrowed_dishes
    ]
    if viable_active and len(viable_active) >= 5:
        # All-but-broken: drop the ones that can't be made (or are low-stock).
        return viable_active

    chosen = list(viable_active)
    # Fix 1: prefer re-expand candidates first — these are inactive dishes
    # with fully-restocked ingredients that should come back online.
    for d in book:
        name = d["name"]
        if name in chosen:
            continue
        if name not in reexpand_dishes:
            continue
        if name in narrowed_dishes:
            continue
        if _viable(d):
            chosen.append(name)
        if len(chosen) >= 8:
            break
    # First fill pass: prefer non-narrowed candidates.
    for d in book:
        name = d["name"]
        if name in chosen:
            continue
        if name in narrowed_dishes:
            continue
        if _viable(d):
            chosen.append(name)
        if len(chosen) >= 8:
            break
    # Second fill pass: if still short of 5, let narrowed dishes back in (they
    # may at least be partially servable — better than an empty menu).
    if len(chosen) < 5:
        for d in book:
            name = d["name"]
            if name in chosen:
                continue
            if _viable(d):
                chosen.append(name)
            if len(chosen) >= 5:
                break
    if len(chosen) >= 5:
        return chosen

    # Last-ditch fallback so we never emit an empty menu.
    fallback = [d["name"] for d in book[:5]]
    return chosen if chosen else fallback


def _compute_narrowed_dishes(obs: dict, threshold: float = 5.0) -> set:
    """
    Diff 2: return the set of active dishes with fewer than `threshold`
    servings available from current on-hand inventory. Pending orders are
    NOT counted — they don't help today's service.

    min_servings = min(on_hand[ing] / per_serving_kg[ing] for ing in recipe).
    Dishes with empty recipes return inf (treated as not narrowed).
    """
    active = obs.get("active_menu") or []
    if not active:
        return set()
    menu_book = {d["name"]: d for d in obs.get("menu_book", []) or []}
    inv_map = {
        str(i.get("ingredient")): float(i.get("total_kg") or 0.0)
        for i in (obs.get("inventory") or [])
    }
    narrowed: set = set()
    for dish in active:
        recipe = menu_book.get(dish, {}).get("ingredients") or []
        if not recipe:
            continue
        min_serv = float("inf")
        for ing in recipe:
            per = float(ing.get("quantity_kg") or 0.0)
            if per <= 0:
                continue
            on_hand = inv_map.get(ing["ingredient"], 0.0)
            servings = on_hand / per
            if servings < min_serv:
                min_serv = servings
        if min_serv < threshold:
            narrowed.add(dish)
    return narrowed


def _compute_inactive_high_stock_dishes(
    obs: dict, threshold: float = 8.0,
) -> Dict[str, float]:
    """
    Fix 1 (menu re-expand hysteresis): return inactive dishes from the
    menu_book whose ingredients are now fully restocked (>= threshold
    servings reachable from on-hand + soon-arriving pending orders).

    This is the inverse of `_compute_narrowed_dishes`: where that finds
    active dishes that should be DROPPED, this finds inactive dishes that
    should be ADDED BACK. The +3 gap between narrow_threshold (5 default)
    and re_expand_threshold (>=8 by construction at the call site)
    prevents flapping.

    Returns a dict {dish_name: min_servings} so telemetry can log which
    dishes are coming back online and at what stock depth.
    """
    book = obs.get("menu_book", []) or []
    if not book:
        return {}
    active = set(obs.get("active_menu") or [])
    inv_map = {
        str(i.get("ingredient")): float(i.get("total_kg") or 0.0)
        for i in (obs.get("inventory") or [])
    }
    # Include soon-arriving pending orders (≤3d delivery) since re-expand
    # is a forward-looking move; if a delivery is hitting Monday we want
    # the dish active Monday morning.
    today = int(obs.get("day", 0) or 0)
    pending_soon: Dict[str, float] = {}
    for po in obs.get("pending_orders") or []:
        ing = po.get("ingredient")
        if ing and (po.get("delivery_day", 999) - today) <= 3:
            pending_soon[ing] = pending_soon.get(ing, 0.0) + float(po.get("quantity_kg") or 0.0)

    out: Dict[str, float] = {}
    for dish in book:
        name = dish.get("name")
        if not name or name in active:
            continue
        recipe = dish.get("ingredients") or []
        if not recipe:
            continue
        min_serv = float("inf")
        for ing in recipe:
            per = float(ing.get("quantity_kg") or 0.0)
            if per <= 0:
                continue
            reachable = inv_map.get(ing["ingredient"], 0.0) + pending_soon.get(
                ing["ingredient"], 0.0
            )
            servings = reachable / per
            if servings < min_serv:
                min_serv = servings
        if min_serv == float("inf"):
            continue
        if min_serv >= threshold:
            out[name] = min_serv
    return out


# ---- staffing ----
# Active staffing now lives in rule_kit/staffing.py (port-fabiano: DOW table
# + weather/trend/walkout bumps + scenario gating + our cash safety).
# This wrapper is kept only for backward compatibility with any caller that
# still imports _staff_for directly.

def _staff_for(expected_covers_today: float, obs: dict, state: Optional[AgentState] = None) -> int:
    """Backwards-compat shim. Prefer staffing.compute_staff_level()."""
    return staffing.compute_staff_level(
        obs,
        state or AgentState(),
        scenario=getattr(state, "scenario", "baseline") if state else "baseline",
        expected_covers_today=expected_covers_today,
    )


# ---- marketing & promos ----

def _marketing_for(
    obs: dict,
    expected_today: float,
    scenario: str = "baseline",
    day: int = 0,
) -> float:
    """
    Port-fabiano-3: tiered marketing instead of a single weak-day pulse.

    Spend more when the demand picture is bad (Declining trend or weak
    forecast), less when we're already busy. Stays out of the
    reputation-band lane — that's the teammate's domain — so we only
    read trend and forecast.

    Port-fabiano-5: renovation scenario suppresses marketing for the
    early stretch of construction (can't seat customers we'd attract).
    """
    if safety.cash_emergency(obs):
        return 0.0
    # Renovation gate — don't pay to attract customers we can't seat.
    if scenario == "renovation" and day <= 12:
        return 0.0
    cash = float(obs.get("cash") or 0.0)
    days_left = int(obs.get("days_remaining") or 0)
    trend = obs.get("customer_trend", "Stable")

    if trend == "Declining":
        base = 200.0
    elif expected_today < 75:
        base = 200.0
    elif trend == "Growing":
        base = 80.0
    else:
        base = 60.0

    # Taper into the final days — paid marketing rarely pays back in <=3 days.
    if 0 < days_left <= 3:
        base = max(0.0, base - 30.0)

    # Never spend more than 3% of cash on a single day's promo.
    cap = max(0.0, cash * 0.03)
    return float(min(base, 300.0, cap))


_SLOW_WEEKDAYS = ("Monday", "Tuesday", "Wednesday")
_BAD_WEATHER = ("rainy", "stormy")


def _should_happy_hour(
    obs: dict,
    day: int,
    state: AgentState,
    expected_today: float,
    scenario: str = "baseline",
) -> bool:
    """
    Port-fabiano-4: trigger Happy Hour on slow-weekday OR bad-weather days,
    in addition to the existing weak-forecast trigger. Keeps the 2-day
    cooldown so we don't spam it.

    Port-fabiano-5: renovation gate (no point drawing crowd when seating cut),
    and a 3-per-7-day cap so we don't spam discounts even when conditions
    keep ticking the trigger.
    """
    if safety.cash_emergency(obs):
        return False
    # Renovation gate.
    if scenario == "renovation" and day <= 12:
        return False
    # Decay: don't run on consecutive days
    if day - state.last_happy_hour_day <= 2:
        return False
    # 3-per-7-day cap from tracked happy_hour_days list.
    recent_hh = sum(1 for d in (state.happy_hour_days or []) if (day - d) <= 7 and (day - d) > 0)
    if recent_hh >= 3:
        return False
    if expected_today < 75:
        return True
    dow = obs.get("day_of_week", "")
    weather = obs.get("weather_today", "")
    # Diff 5: drop the `expected_today < 95` gate on slow weekdays. The
    # 3-per-7-day cap above already prevents spam, and the diagnostic showed
    # we were under-firing happy hours that should have lifted soft demand.
    if dow in _SLOW_WEEKDAYS:
        return True
    if weather in _BAD_WEATHER:
        return True
    return False


# ---- daily special ----

def _pick_daily_special(
    obs: dict,
    day: int = 1,
    narrowed_dishes: Optional[set] = None,
) -> Optional[str]:
    """
    Diff 4: round-robin through the active menu by day.

    The previous strategy picked the dish using the soonest-to-expire
    ingredient — but that directs demand AT scarce ingredients. Diagnostic
    showed Day 7 special pointed at Salmon while Salmon was at 0.1 kg.

    Round-robin: `active_menu[(day - 1) % len(active_menu)]`. If that index
    lands on a narrowed (low-stock) dish, fall through to the next index.
    """
    narrowed_dishes = narrowed_dishes or set()
    menu = obs.get("active_menu") or []
    if not menu:
        return None
    n = len(menu)
    base = (day - 1) % n
    for offset in range(n):
        idx = (base + offset) % n
        candidate = menu[idx]
        if candidate not in narrowed_dishes:
            return candidate
    # Every dish is narrowed — fall back to the round-robin index anyway so
    # we still emit a special (a free satisfaction bump is better than none).
    return menu[base]
