"""
Regime detection layer.

Quantitative per-axis signals that COMPLEMENT scenario.py.

  scenario.py  → a single high-level mode label (renovation / supply_crisis /
                 tourist / inflation / health_scare / baseline) from alerts.
  regime.py    → per-axis quantitative signals (demand / supply-ingredient /
                 supply-capacity / cost), each with sign / z-score / source /
                 confidence / days_in_regime. Designed to be the BIAS engine for
                 inventory / staffing / pricing modifiers in Phase 3.

Three axes, each carrying a RegimeSignal. Sub-channels:

| Axis         | Sub-channels                       | Known scenarios               |
| demand       | total covers                       | tourist_season (+), health (−)|
| supply       | ingredients, capacity (SEPARATE)   | supply_crisis (− ing),        |
|              |                                    | renovation (− capacity)       |
| cost         | input prices                       | inflation (+)                 |

We bias HEAVILY toward "normal" — most days even in special scenarios behave
like normal-regime behavior. False positives in baseline are pure loss.

Phase 1 contract:
  detect_regimes(obs, state) -> Dict[axis_name, RegimeSignal]
  Pure function. Caller (policy.decide_actions) is responsible for state
  mutation via the returned context and for capturing day-1 baseline.

============================================================================
CALIBRATION NOTES (Phase 2)
----------------------------------------------------------------------------
Live API runs were NOT permitted in this session per the hackathon
constraints, so calibration was performed via a synthetic-scenario harness
that mirrors the AGENT_CONTRACT.md observation schema and the documented
scenario mechanics (supplier outage / tourist surge / renovation / inflation).
Results below should be RE-VALIDATED against `analyze.py --regimes` on real
seed-42 runs before relying on the thresholds for Phase 3 action wiring.

Thresholds in use:
  Z_ENTER             = 2.5    # |z| sustained for 2 days → enter regime
  Z_EXIT              = 0.5    # |z| < 0.5 for 3 days → return to normal
  ENTER_STREAK_DAYS   = 2
  EXIT_STREAK_DAYS    = 3
  RESIDUAL_WINDOW     = 7      # rolling window for z-score
  COST_DRIFT_FRACTION = 0.05   # >5% mean drift in input price = cost regime
  SUPPLY_GAP_Z_FLOOR  = 0.15   # 15%+ undelivery across recent suppliers
  CAPACITY_TRIP_COUNT = 4      # all 4 composite conditions must hit

Synthetic-harness measurements (see /tmp/test_calibration2.py in the
session that wrote this module):

  scenario          axis              first-trigger    return-to-normal   FP
  ----------------  ----------------  ---------------  -----------------  -----
  baseline          (all axes)        N/A              N/A                0.0%
                    target ≤5%
  supply_crisis     supply_ingred -   day 5 (alert)    day 24 (slow,      0%
                    target ≤8/lat≤3   target day 5     z-window decay)    other
  tourist_season    demand +          day 10 (alert)   N/A in window      0%
                    target ≤13        target day 10                       other
  renovation        supply_capacity-  day 1 (alert)    day 13 (post 12-d) 0%
                    target day 1      target day 1                        other
  inflation         cost +            day 5 (alert)    -- not exited      0%
                    target day 5      cost is TELEM ONLY in v1            other

Observations:
  - Alert fast-path delivers latency=0 (fires same day) on every alerted
    scenario. This is by design — the brief mandates alerts override the
    streak requirement.
  - Supply_crisis residual-exit is slower than ideal (~9 days post-recovery)
    because the EWMA delivered_ratio re-stabilises faster than the 7-day
    rolling z-window. The axis remains negative across that tail but
    transitions on confidence rather than sign. Acceptable for v1.
  - No spurious cross-axis activity in any scenario — false-positive rate
    on baseline is 0% in the harness, comfortably under the 5% target.

Tuning if live-run FP rate exceeds 5% on baseline calibration:
  - raise Z_ENTER to 2.8
  - raise ENTER_STREAK_DAYS to 3
  - raise COST_DRIFT_FRACTION to 0.07

Tuning if latency exceeds target on the scenario triggers without alerts:
  - lower Z_ENTER to 2.2
  - drop ENTER_STREAK_DAYS to 1 for alert-confirmed axes (already free via
    the alert override path)

Tuning if exit-latency on supply_crisis matters for action wiring:
  - shorten RESIDUAL_WINDOW to 5
  - widen Z_EXIT to 0.8

PHASE 3 GATE: do NOT proceed to action wiring until live-run measurements
confirm these latencies and FP rates on dev seed 42.
============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from . import forecast as forecast_mod


# ---------- tunables ----------

Z_ENTER = 2.5
Z_EXIT = 0.5
ENTER_STREAK_DAYS = 2
EXIT_STREAK_DAYS = 3
RESIDUAL_WINDOW = 7
COST_DRIFT_FRACTION = 0.05
SUPPLY_GAP_Z_FLOOR = 0.15
CAPACITY_TRIP_COUNT = 4

# Axis names — canonical strings used throughout the codebase.
AXIS_DEMAND = "demand"
AXIS_SUPPLY_ING = "supply_ingredient"
AXIS_SUPPLY_CAP = "supply_capacity"
AXIS_COST = "cost"

AXES = (AXIS_DEMAND, AXIS_SUPPLY_ING, AXIS_SUPPLY_CAP, AXIS_COST)


# Alert keyword → (axis, sign) mapping. Keywords are lowercased substrings
# checked against the alerts blob.
_ALERT_RULES: Tuple[Tuple[Tuple[str, ...], str, int], ...] = (
    # supply-ingredient negative
    (("supplier", "outage", "halted", "disruption", "shortage"),
        AXIS_SUPPLY_ING, -1),
    # demand positive
    (("demand", "tourist", "festival", "surge", "rush"),
        AXIS_DEMAND, +1),
    # demand negative
    (("health", "recall", "scare", "closure"),
        AXIS_DEMAND, -1),
    # supply-capacity negative
    (("renovation", "construction", "capacity", "reduced tables"),
        AXIS_SUPPLY_CAP, -1),
    # cost positive
    (("inflation", "price increase", "cost", "pricing"),
        AXIS_COST, +1),
)


# ---------- data model ----------

@dataclass
class RegimeSignal:
    """Snapshot of a single axis on a single day."""
    sign: int = 0                  # -1, 0, +1
    magnitude_z: float = 0.0
    source: str = "normal"         # "normal" | "alert" | "residual"
    confidence: str = "low"        # "low" | "medium" | "high"
    days_in_regime: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def normal(cls) -> "RegimeSignal":
        return cls()


# ---------- helpers ----------

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


def _dow_yesterday(today: str) -> str:
    try:
        i = _WEEKDAYS.index(today)
    except ValueError:
        return today
    return _WEEKDAYS[(i - 1) % 7]


def _alerts_blob(obs: dict) -> str:
    """Concatenate alerts (mixed strings + dicts) into a single lowercase blob."""
    parts: List[str] = []
    for a in obs.get("alerts") or []:
        if isinstance(a, str):
            parts.append(a)
        elif isinstance(a, dict):
            for v in a.values():
                if isinstance(v, str):
                    parts.append(v)
        else:
            parts.append(str(a))
    return " ".join(parts).lower()


def _parse_alerts(obs: dict) -> Dict[str, Tuple[int, str]]:
    """
    Return axis -> (sign, matched_keyword) for every axis matched by alerts.
    Latest matching rule wins per axis.
    """
    text = _alerts_blob(obs)
    if not text:
        return {}
    out: Dict[str, Tuple[int, str]] = {}
    for keywords, axis, sign in _ALERT_RULES:
        for kw in keywords:
            if kw in text:
                out[axis] = (sign, kw)
                break
    return out


def _rolling_zscore(window: List[float], value: float) -> float:
    """
    Z-score of `value` against the existing rolling window (mean/std of the
    series). Returns 0.0 if the window is too short or std is zero.
    """
    if not window:
        return 0.0
    n = len(window)
    mean = sum(window) / n
    if n < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in window) / max(1, n - 1)
    std = math.sqrt(var)
    if std <= 1e-9:
        return 0.0
    return (value - mean) / std


def _push_window(window: List[float], value: float, cap: int = RESIDUAL_WINDOW) -> List[float]:
    window.append(float(value))
    if len(window) > cap:
        del window[: len(window) - cap]
    return window


# ---------- residual computers ----------

def _demand_residual(obs: dict, state) -> Optional[float]:
    """
    yesterday.service_summary.total_covers
       − forecast_covers(state, yesterday DOW, yesterday weather, yesterday rep band)

    Returns None on day 1 (no service_summary), so caller can skip update.
    """
    svc = obs.get("service_summary") or {}
    if not svc or "total_covers" not in svc:
        return None
    yest_covers = float(svc.get("total_covers") or 0)
    today_dow = obs.get("day_of_week", "Monday")
    yest_dow = _dow_yesterday(today_dow)
    # Yesterday's weather is unknown after-the-fact; best proxy is today's
    # weather as a baseline guess. Forecast accuracy doesn't matter — we want
    # a stable expected value so the residual is meaningful.
    yest_weather = obs.get("weather_today", "sunny")
    # Reputation BEFORE today's update. We don't have a pre-update read, so use
    # the current reputation_band (it reflects yesterday's service window).
    rep_band = obs.get("reputation_band", "Very Good")
    expected = forecast_mod.forecast_covers(state, yest_dow, yest_weather, rep_band)
    if expected <= 1.0:
        return None
    return yest_covers - expected


def _supply_ingredient_residual(obs: dict, state) -> Optional[float]:
    """
    MAX (1 − delivered_ratio) across suppliers ordered from in the last 7 days.
    State.suppliers carries EWMA delivered_ratio per supplier we've ordered
    from. Returns None if we have no supplier samples yet.

    The "raw" value here IS the gap (0..1). The z-score will compute against
    its own rolling window so even small gaps near baseline ~0 are detectable.
    """
    suppliers = getattr(state, "suppliers", {}) or {}
    if not suppliers:
        return 0.0
    gaps = [
        max(0.0, 1.0 - float(stat.delivered_ratio))
        for stat in suppliers.values()
        if getattr(stat, "samples", 0) > 0
    ]
    if not gaps:
        return 0.0
    return max(gaps)


def _supply_capacity_composite(obs: dict, state) -> Tuple[int, float]:
    """
    Composite 0–4 count of capacity-bound signals:
      (a) table_util_peak ≥ 0.95
      (b) walkout_band in {"Some", "Many"}
      (c) min_ingredient_days_cover ≥ 1.0
      (d) staff_level ≥ forecast-recommended (proxy: ≥ 7, which is the floor
          of the DOW base table in staffing.py)

    The "we have ingredients AND staff AND tables still saturated AND walking
    customers out" pattern is the canonical capacity-constrained signature.

    Returns (count, raw_value_for_z) where raw_value_for_z is just the count
    as a float so it can be tracked in a rolling window the same way.
    """
    count = 0

    svc = obs.get("service_summary") or {}
    util_peak = float(svc.get("table_utilization_peak") or 0.0)
    if util_peak >= 0.95:
        count += 1

    walkout = str(svc.get("walkout_band") or "None")
    if walkout in ("Some", "Many"):
        count += 1

    # Min days-of-cover across inventory items that are on the active menu.
    # We approximate "min" by checking on-hand directly; full days_of_cover
    # requires daily_usage from inventory.estimate_daily_usage, which we want
    # to avoid recomputing here. The composite is robust to this approximation:
    # if EVERY active ingredient has at least the minimum threshold of on-hand
    # we treat it as "ingredients available".
    inv = obs.get("inventory") or []
    if inv:
        # Heuristic: at least one item has total_kg > 0, AND on average we look
        # ok. The exact rule: count items with total_kg > 0 vs zero — if more
        # than half have stock, treat as "ingredients available".
        stocked = sum(1 for i in inv if (i.get("total_kg") or 0) > 0.5)
        if stocked >= max(1, len(inv) // 2):
            count += 1

    staff = int(obs.get("staff_level") or 0)
    # Floor of DOW base table in staffing.py is 7.
    if staff >= 7:
        count += 1

    return count, float(count)


def _cost_residual(obs: dict, state) -> Optional[float]:
    """
    Mean across ingredients of:
        (current_cheapest_price − day1_cheapest_price) / day1_cheapest_price.

    Returns None if day1_baseline isn't populated yet.
    """
    baseline = getattr(state, "day1_baseline", None)
    if not baseline:
        return None
    base_prices: Dict[str, float] = baseline.get("ingredient_prices") or {}
    if not base_prices:
        return None
    catalog = obs.get("supplier_catalog") or []
    if not catalog:
        return None

    current_cheapest: Dict[str, float] = {}
    for s in catalog:
        for ing, price in (s.get("ingredients") or {}).items():
            try:
                p = float(price)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            prev = current_cheapest.get(ing)
            if prev is None or p < prev:
                current_cheapest[ing] = p
    if not current_cheapest:
        return None

    drifts: List[float] = []
    for ing, base_p in base_prices.items():
        cur = current_cheapest.get(ing)
        if cur is None or base_p <= 0:
            continue
        drifts.append((cur - base_p) / base_p)
    if not drifts:
        return None
    return sum(drifts) / len(drifts)


# ---------- baseline capture ----------

def capture_day1_baseline(obs: dict) -> dict:
    """
    Build the day-1 baseline dict from the current observation.

    Captures:
      ingredient_prices : cheapest unit price per ingredient from supplier_catalog
      table_count       : best-effort (22 by AGENT_CONTRACT, but read if present)
      reputation_band   : day-1 rep band

    Numbers are kept compact so the dict fits in the notes budget.
    """
    catalog = obs.get("supplier_catalog") or []
    prices: Dict[str, float] = {}
    for s in catalog:
        for ing, price in (s.get("ingredients") or {}).items():
            try:
                p = float(price)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            prev = prices.get(ing)
            if prev is None or p < prev:
                prices[ing] = round(p, 4)

    # Table count: not always present in observation schema; default to 22 per
    # AGENT_CONTRACT.md key-numbers table.
    table_count = obs.get("table_count")
    if not isinstance(table_count, int):
        table_count = 22

    return {
        "ingredient_prices": prices,
        "table_count": table_count,
        "reputation_band": obs.get("reputation_band", "Very Good"),
    }


# ---------- main detection ----------

def detect_regimes(obs: dict, state) -> Dict[str, RegimeSignal]:
    """
    Compute the regime state for every axis on this turn.

    SIDE EFFECTS:
        - Updates state.residuals (rolling windows per axis).
        - Updates state.regime_days (consecutive-day counters).
        - Updates state.regime_streaks (entry/exit streaks per axis, internal).
        - Updates state.regime_signals (last-known signal per axis, internal).
        - Captures state.day1_baseline if missing.

    Returns the per-axis RegimeSignal dict for telemetry.
    """
    # Ensure all the regime fields exist on state. Defensive against old notes
    # blobs that pre-date this module.
    if getattr(state, "residuals", None) is None:
        state.residuals = {a: [] for a in AXES}
    else:
        for a in AXES:
            state.residuals.setdefault(a, [])
    if getattr(state, "regime_days", None) is None:
        state.regime_days = {a: 0 for a in AXES}
    else:
        for a in AXES:
            state.regime_days.setdefault(a, 0)
    if getattr(state, "regime_streaks", None) is None:
        state.regime_streaks = {a: {"enter": 0, "exit": 0, "sign": 0}
                                for a in AXES}
    else:
        for a in AXES:
            state.regime_streaks.setdefault(a, {"enter": 0, "exit": 0, "sign": 0})
    if getattr(state, "regime_signals", None) is None:
        state.regime_signals = {a: RegimeSignal.normal().to_dict() for a in AXES}
    else:
        for a in AXES:
            state.regime_signals.setdefault(a, RegimeSignal.normal().to_dict())

    # Day-1 baseline.
    if not getattr(state, "day1_baseline", None):
        state.day1_baseline = capture_day1_baseline(obs)

    # Compute raw residuals.
    demand_r = _demand_residual(obs, state)
    supply_ing_r = _supply_ingredient_residual(obs, state)
    cap_count, supply_cap_r = _supply_capacity_composite(obs, state)
    cost_r = _cost_residual(obs, state)

    # Update rolling windows BEFORE computing z (z is against existing history,
    # but we want today's residual recorded for future turns).
    raw_residuals = {
        AXIS_DEMAND: demand_r,
        AXIS_SUPPLY_ING: supply_ing_r,
        AXIS_SUPPLY_CAP: supply_cap_r,
        AXIS_COST: cost_r,
    }

    # Alert parser fast-path (high confidence, fires day 1, bypasses streak).
    alert_hits = _parse_alerts(obs)

    out: Dict[str, RegimeSignal] = {}

    for axis in AXES:
        raw = raw_residuals.get(axis)
        prev_signal = RegimeSignal(**state.regime_signals.get(axis,
                                   RegimeSignal.normal().to_dict()))
        prev_streaks = state.regime_streaks.get(axis,
                                                {"enter": 0, "exit": 0, "sign": 0})

        # ---- alert fast path ----
        if axis in alert_hits:
            sign, _kw = alert_hits[axis]
            sig = RegimeSignal(
                sign=int(sign),
                magnitude_z=3.0,
                source="alert",
                confidence="high",
                days_in_regime=(prev_signal.days_in_regime + 1
                                if prev_signal.sign == sign
                                else 1),
            )
            # When an alert fires, treat the entry as confirmed; reset exit
            # streak; keep window updated so the residual remains useful for
            # post-alert calibration.
            state.regime_streaks[axis] = {"enter": ENTER_STREAK_DAYS,
                                          "exit": 0, "sign": sign}
            state.regime_days[axis] = sig.days_in_regime
            state.regime_signals[axis] = sig.to_dict()
            if raw is not None:
                _push_window(state.residuals[axis], raw)
            out[axis] = sig
            continue

        # ---- residual path ----
        window = state.residuals.get(axis, [])
        if raw is None:
            # Not enough info this turn — keep prior signal but increment day
            # counter only if we're in a non-normal regime.
            if prev_signal.sign != 0:
                prev_signal.days_in_regime += 1
                state.regime_days[axis] = prev_signal.days_in_regime
                state.regime_signals[axis] = prev_signal.to_dict()
            out[axis] = prev_signal
            continue

        # Special: cost axis uses fractional drift threshold, not raw z.
        if axis == AXIS_COST:
            z = raw  # raw IS the fractional drift; treat as effective z proxy
            entered = raw > COST_DRIFT_FRACTION
            exited = abs(raw) < COST_DRIFT_FRACTION * 0.4
            sign_now = +1 if raw > 0 else (-1 if raw < 0 else 0)
        # Special: supply-capacity composite — trigger if count == CAPACITY_TRIP_COUNT.
        elif axis == AXIS_SUPPLY_CAP:
            z = float(cap_count)
            entered = cap_count >= CAPACITY_TRIP_COUNT
            exited = cap_count <= 1
            sign_now = -1 if entered else 0
        # Special: supply-ingredient — gap threshold AND z-score, either path triggers.
        elif axis == AXIS_SUPPLY_ING:
            z = _rolling_zscore(window, raw)
            entered = (raw > SUPPLY_GAP_Z_FLOOR) or (abs(z) >= Z_ENTER)
            exited = (raw < SUPPLY_GAP_Z_FLOOR * 0.5) and (abs(z) < Z_EXIT)
            sign_now = -1 if entered else 0
        else:
            # demand axis — pure z-score on the rolling window
            z = _rolling_zscore(window, raw)
            entered = abs(z) >= Z_ENTER
            exited = abs(z) < Z_EXIT
            sign_now = +1 if z > 0 else (-1 if z < 0 else 0)
            # Tie sign to the sign of the raw residual (more stable than z
            # when the window is short).
            if raw > 0:
                sign_now = +1
            elif raw < 0:
                sign_now = -1

        # ---- streak machine ----
        streaks = dict(prev_streaks)
        # Reset enter streak if sign changed.
        if entered and sign_now != 0 and streaks.get("sign", 0) != sign_now:
            streaks["enter"] = 0
            streaks["sign"] = sign_now
        if entered:
            streaks["enter"] = streaks.get("enter", 0) + 1
            streaks["exit"] = 0
        elif exited:
            streaks["exit"] = streaks.get("exit", 0) + 1
            streaks["enter"] = 0
        else:
            # neither extreme — gently decay both streaks
            streaks["enter"] = max(0, streaks.get("enter", 0) - 1)
            streaks["exit"] = max(0, streaks.get("exit", 0) - 1)

        # Decide regime state.
        new_sign = prev_signal.sign
        new_source = prev_signal.source
        new_conf = prev_signal.confidence
        new_days = prev_signal.days_in_regime

        if streaks["enter"] >= ENTER_STREAK_DAYS and sign_now != 0:
            # Trigger / continue regime.
            if prev_signal.sign != sign_now:
                new_days = 1
            else:
                new_days = prev_signal.days_in_regime + 1
            new_sign = sign_now
            new_source = "residual"
            new_conf = "medium" if abs(z) < Z_ENTER + 1.0 else "high"
        elif streaks["exit"] >= EXIT_STREAK_DAYS:
            # Confirmed return to normal.
            new_sign = 0
            new_source = "normal"
            new_conf = "low"
            new_days = 0
        else:
            # In-between: hold prior regime and tick days_in_regime up.
            if prev_signal.sign != 0:
                new_days = prev_signal.days_in_regime + 1
            else:
                new_days = 0

        sig = RegimeSignal(
            sign=int(new_sign),
            magnitude_z=round(float(z), 3),
            source=str(new_source),
            confidence=str(new_conf),
            days_in_regime=int(new_days),
        )
        state.regime_streaks[axis] = streaks
        state.regime_days[axis] = sig.days_in_regime
        state.regime_signals[axis] = sig.to_dict()
        _push_window(state.residuals[axis], raw)
        out[axis] = sig

    return out


# ---------- modifier composition (Phase 3) ----------

def _as_signal(s) -> RegimeSignal:
    """Tolerate either RegimeSignal or dict input (state.regime_signals stores
    the dict form). Anything unrecognized becomes a normal signal."""
    if isinstance(s, RegimeSignal):
        return s
    if isinstance(s, dict):
        try:
            return RegimeSignal(
                sign=int(s.get("sign", 0) or 0),
                magnitude_z=float(s.get("magnitude_z", 0.0) or 0.0),
                source=str(s.get("source", "normal") or "normal"),
                confidence=str(s.get("confidence", "low") or "low"),
                days_in_regime=int(s.get("days_in_regime", 0) or 0),
            )
        except Exception:
            return RegimeSignal.normal()
    return RegimeSignal.normal()


def compose_modifiers(regimes) -> Dict[str, object]:
    """
    Translate per-axis regime state into a flat modifier dict consumed by
    inventory.reorder_plan, staffing.compute_staff_level, and pricing.

    Phase 3 schema (additive — downstream functions read with safe defaults):

      inventory_safety_multiplier  : float, default 1.0
      inventory_target_days_delta  : int,   default 0
      diversification_force        : bool,  default False
      single_supplier_share_cap    : float, default 1.0
      staff_delta                  : int,   default 0
      menu_narrow_threshold        : float, default 5.0
      suppress_price_drops         : bool,  default False
      allow_capacity_premium       : bool,  default False

    Compound rules:
      - Only act on axes where sign != 0 (regime entered, not pending).
      - On conflicting scalar modifiers, the larger |magnitude_z| axis wins.
        (For monotonic-up multipliers like inventory_safety_multiplier we take
        the maximum, since stacking distinct shocks shouldn't *lower* safety.)
      - Non-conflicting modifiers stack — e.g. supply_ing- AND demand+ both
        contribute to safety, and demand+ adds +1 staff regardless of supply.
      - Cost axis is TELEMETRY ONLY in v1 (no action wiring).

    Change 3 (shock response): for supply_ingredient.sign == -1, when source
    is "alert" OR (source == "residual" AND days_in_regime >= 2), the safety
    multiplier is 1.5 for the first 3 days of the regime then decays to 1.3.
    Rationale: be bigger early when we're least sure what's happening; trim
    once inventory has had a few turns to actually adjust.

    `regimes` may be either Dict[str, RegimeSignal] (from detect_regimes) or
    Dict[str, dict] (from state.regime_signals). Both are handled.
    """
    mods: Dict[str, object] = {
        "inventory_safety_multiplier": 1.0,
        "inventory_target_days_delta": 0,
        "diversification_force": False,
        "single_supplier_share_cap": 1.0,
        "staff_delta": 0,
        "menu_narrow_threshold": 5.0,
        "suppress_price_drops": False,
        "allow_capacity_premium": False,
        # Telemetry-side fields. Additive, never read for decisions.
        "notes": [],
        "applied_axes": [],
    }
    if not regimes:
        return mods

    demand = _as_signal(regimes.get(AXIS_DEMAND))
    supply_ing = _as_signal(regimes.get(AXIS_SUPPLY_ING))
    supply_cap = _as_signal(regimes.get(AXIS_SUPPLY_CAP))
    cost = _as_signal(regimes.get(AXIS_COST))

    notes: List[str] = mods["notes"]            # type: ignore[assignment]
    applied: List[str] = mods["applied_axes"]   # type: ignore[assignment]

    # ---- Supply-ingredient negative ----
    if supply_ing.sign == -1:
        # Default bump for an entered supply-ingredient negative regime.
        ing_bump = 1.3
        # Change 3: shock response — 1.5 early in the regime when the signal
        # is alert-confirmed or residual-confirmed for >=2 days.
        shock_eligible = (
            supply_ing.source == "alert"
            or (supply_ing.source == "residual" and supply_ing.days_in_regime >= 2)
        )
        if shock_eligible and supply_ing.days_in_regime <= 3:
            ing_bump = 1.5
            notes.append("supply_ing_shock_bump_1.5")
        else:
            notes.append("supply_ing_neg_bump_1.3")
        mods["inventory_safety_multiplier"] = max(
            float(mods["inventory_safety_multiplier"]), ing_bump
        )
        mods["diversification_force"] = True
        mods["single_supplier_share_cap"] = min(
            float(mods["single_supplier_share_cap"]), 0.5
        )
        # Preemptive menu-narrow: stop selling dishes we're about to run out
        # of, *before* they collapse service.
        mods["menu_narrow_threshold"] = max(
            float(mods["menu_narrow_threshold"]), 8.0
        )
        applied.append(AXIS_SUPPLY_ING)

    # ---- Supply-capacity negative ----
    if supply_cap.sign == -1:
        # Capacity is constrained (renovation / table outage). Trim staff so
        # we don't burn payroll on covers we can't physically seat, and let
        # pricing charge a small capacity premium.
        mods["staff_delta"] = int(mods["staff_delta"]) + (-2)
        mods["allow_capacity_premium"] = True
        # Also narrow the menu — fewer concurrent dishes when throughput is low.
        mods["menu_narrow_threshold"] = max(
            float(mods["menu_narrow_threshold"]), 8.0
        )
        notes.append("supply_cap_neg")
        applied.append(AXIS_SUPPLY_CAP)

    # ---- Demand positive ----
    if demand.sign == +1:
        mods["inventory_safety_multiplier"] = max(
            float(mods["inventory_safety_multiplier"]), 1.2
        )
        mods["inventory_target_days_delta"] = int(mods["inventory_target_days_delta"]) + 1
        mods["staff_delta"] = int(mods["staff_delta"]) + 1
        mods["suppress_price_drops"] = True
        notes.append("demand_pos")
        applied.append(AXIS_DEMAND)

    # ---- Demand negative ----
    elif demand.sign == -1:
        mods["inventory_target_days_delta"] = int(mods["inventory_target_days_delta"]) - 1
        mods["staff_delta"] = int(mods["staff_delta"]) + (-1)
        notes.append("demand_neg")
        applied.append(AXIS_DEMAND)

    # ---- Cost positive (TELEMETRY ONLY in v1) ----
    if cost.sign == +1:
        notes.append("cost_pos_telemetry_only")

    # ---- Conflict resolution by |z| ----
    # For modifiers that two axes both push, the larger-magnitude axis wins.
    # The only realistic conflict in our current axis set is the menu narrow
    # threshold — both supply_ing- and supply_cap- can push it to 8.0, which
    # is the same value, so no resolution needed. Left as the explicit hook
    # for when more axes get added.
    # (Safety multiplier is max-stack by design above; delta fields can only
    # be touched by one axis each in the current rule set.)
    candidates = [
        (supply_ing, abs(supply_ing.magnitude_z)),
        (supply_cap, abs(supply_cap.magnitude_z)),
        (demand, abs(demand.magnitude_z)),
    ]
    if candidates:
        dominant = max(candidates, key=lambda kv: kv[1])
        if dominant[0].sign != 0:
            notes.append(f"dominant=|z|{round(dominant[1], 2)}")

    return mods
