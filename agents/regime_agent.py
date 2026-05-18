"""Regime-based restaurant agent — rule-driven core with LLM oversight.

Each day the situation is classified into a small set of *regimes* (tourist,
supply_crisis, capacity_limited, reputation_crisis, cost_inflation,
demand_drought, unknown_anomaly), each carrying a 0.0-1.0 severity. Per-domain
ownership rules pick which regime's rule set drives each decision
(inventory / staffing / pricing / marketing / happy_hour / daily_special /
menu). An LLM oversight pass adjusts severities and picks discrete
low/med/high tiers per rule-set knob, with strict JSON validation and a
deterministic fallback. When >=2 regimes co-occur at severity >=0.5 a richer
"synthesis" prompt invites the LLM to specify per-domain owner+tier overrides.

Observability:
  - Default: one terse line per day on stdout (prefix `[regime]`).
  - REGIME_DEBUG=1: full classifier breakdown, LLM I/O, action list, state diff.
  - REGIME_LOG=path.jsonl: machine-readable per-day record.

State persists across turns via `save_notes` (compact JSON, ~3.8 KB cap).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import litellm

from agents.runner import run_game

# ---------------------------------------------------------------------------
# LLM proxy config

_DEFAULT_BASE = "http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com"
LLM_API_BASE = os.getenv("OPENAI_API_BASE", _DEFAULT_BASE)
MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
MAX_USER_MSG_CHARS = 5500

# ---------------------------------------------------------------------------
# Constants

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DOW_INDEX = {d: i for i, d in enumerate(DAYS_OF_WEEK)}
WEATHER_DEMAND = {"sunny": 1.15, "cloudy": 1.0, "rainy": 0.80, "stormy": 0.55}
REPUTATION_OCC = {"Poor": 0.30, "Fair": 0.55, "Good": 0.80, "Very Good": 1.00, "Excellent": 1.15}
TOTAL_SEATS = 108
PEAK_COVERS_CEILING = TOTAL_SEATS * 2

STAFF_DOW_BASE = {
    "Monday": 7, "Tuesday": 7, "Wednesday": 7,
    "Thursday": 8, "Friday": 10, "Saturday": 11, "Sunday": 9,
}

REGIMES = (
    "tourist",
    "demand_drought",
    "supply_crisis",
    "capacity_limited",
    "reputation_crisis",
    "cost_inflation",
    "unknown_anomaly",
)

# Per-decision priority lists. First active regime (severity >= LOW threshold)
# in the list owns the decision. "(suppress)" marks regimes that, when owner,
# force the action off regardless of tier.
PRIORITY = {
    "inventory":      ["supply_crisis", "cost_inflation", "tourist", "capacity_limited", "demand_drought"],
    "staffing":       ["capacity_limited", "reputation_crisis", "tourist", "demand_drought"],
    "pricing":        ["cost_inflation", "reputation_crisis", "tourist", "demand_drought"],
    "marketing":      ["capacity_limited", "reputation_crisis", "tourist", "demand_drought"],
    "happy_hour":     ["reputation_crisis", "demand_drought", "supply_crisis", "tourist"],
    "daily_special":  ["supply_crisis", "reputation_crisis"],
    "menu":           ["supply_crisis", "capacity_limited"],
}
SUPPRESSES = {
    ("marketing", "capacity_limited"),
    ("marketing", "tourist"),
    ("happy_hour", "supply_crisis"),
    ("happy_hour", "tourist"),
}

# Keyword sets (extensible at module level per plan).
KEYWORDS = {
    "tourist": ("tourist", "festival", "surge", "event", "influx", "boom"),
    "demand_drought": ("off-season", "low season", "downturn", "slow", "quiet period"),
    "supply_crisis": ("outage", "shortage", "strike", "halt", "halted", "disruption", "supplier"),
    "capacity_limited": ("renovation", "construction", "tables unavailable", "tables are unavailable", "fewer table"),
    "reputation_crisis": ("health", "scare", "inspection", "food safety", "complaint", "violation"),
    "cost_inflation": ("inflation", "price increase", "cost rise", "rising cost", "price hike"),
}

# Tier bands (lower-bound inclusive).
TIER_BANDS = (("high", 0.70), ("med", 0.45), ("low", 0.20), ("off", 0.0))

# ---------------------------------------------------------------------------
# Observability

DEBUG = os.environ.get("REGIME_DEBUG", "") not in ("", "0", "false")
LOG_PATH = os.environ.get("REGIME_LOG", "")


def _log_jsonl(record: dict) -> None:
    if not LOG_PATH:
        return
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"[regime] log write failed: {e}", file=sys.stderr)


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[regime:dbg] {msg}")


# ---------------------------------------------------------------------------
# State

DEFAULT_STATE = {
    "ema_burn": {},
    "peak_burn": {},
    "covers_ema": 0.0,
    "revenue_ema": 0.0,
    "covers_var": 0.0,            # variance proxy for anomaly detection
    "last_hh_day": -10,
    "consec_hh": 0,
    "outage_suppliers": [],
    "regime_history": [],         # list of {"d": day, "r": {regime: sev}}
    "supplier_price_snapshot": {},  # "<sup>:<ing>" -> price (first seen)
    "delivery_shortfall_log": [],  # [{"d": day, "sup": name, "frac": delivered/ordered}]
}


def load_state(notes: str | None) -> dict:
    if not notes:
        return _clone_default()
    try:
        if notes.lstrip().startswith("{"):
            s = json.loads(notes)
            merged = _clone_default()
            merged.update(s)
            return merged
    except (json.JSONDecodeError, TypeError):
        pass
    return _clone_default()


def _clone_default() -> dict:
    out = {}
    for k, v in DEFAULT_STATE.items():
        out[k] = list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)
    return out


def serialize_state(state: dict) -> str:
    payload = {
        "ema_burn": {k: round(v, 3) for k, v in state.get("ema_burn", {}).items()},
        "peak_burn": {k: round(v, 3) for k, v in state.get("peak_burn", {}).items()},
        "covers_ema": round(state.get("covers_ema", 0), 1),
        "revenue_ema": round(state.get("revenue_ema", 0), 1),
        "covers_var": round(state.get("covers_var", 0), 1),
        "last_hh_day": state.get("last_hh_day", -10),
        "consec_hh": state.get("consec_hh", 0),
        "outage_suppliers": state.get("outage_suppliers", [])[:8],
        "regime_history": state.get("regime_history", [])[-7:],
        "supplier_price_snapshot": state.get("supplier_price_snapshot", {}),
        "delivery_shortfall_log": state.get("delivery_shortfall_log", [])[-5:],
    }
    raw = json.dumps(payload, separators=(",", ":"))
    if len(raw) > 3800:
        payload["regime_history"] = payload["regime_history"][-3:]
        payload["delivery_shortfall_log"] = payload["delivery_shortfall_log"][-3:]
        raw = json.dumps(payload, separators=(",", ":"))
    return raw[:3900]


def update_emas(state: dict, observation: dict, menu_dict: dict) -> None:
    summary = observation.get("service_summary") or {}
    dishes_sold = summary.get("dishes_sold") or {}

    daily_burn: dict[str, float] = {}
    for dish, count in dishes_sold.items():
        recipe = menu_dict.get(dish)
        if not recipe:
            continue
        for ing in recipe["ingredients"]:
            daily_burn[ing["ingredient"]] = (
                daily_burn.get(ing["ingredient"], 0.0) + ing["quantity_kg"] * count
            )

    ema_burn = state.get("ema_burn", {})
    peak_burn = state.get("peak_burn", {})
    alpha = 0.4
    for ing, burn in daily_burn.items():
        prev = ema_burn.get(ing, burn)
        ema_burn[ing] = alpha * burn + (1 - alpha) * prev
        peak_burn[ing] = max(burn, peak_burn.get(ing, 0.0) * 0.85)
    state["ema_burn"] = ema_burn
    state["peak_burn"] = peak_burn

    covers = summary.get("total_covers") or 0
    revenue = summary.get("total_revenue") or 0.0
    prev_covers = state.get("covers_ema", 0.0)
    if prev_covers <= 0:
        state["covers_ema"] = float(covers)
        state["revenue_ema"] = float(revenue)
        state["covers_var"] = 0.0
    else:
        new_ema = 0.4 * covers + 0.6 * prev_covers
        delta = covers - prev_covers
        state["covers_var"] = 0.3 * (delta * delta) + 0.7 * state.get("covers_var", 0.0)
        state["covers_ema"] = new_ema
        state["revenue_ema"] = 0.4 * revenue + 0.6 * state["revenue_ema"]

    # Stockout aggressive bump.
    unavail = summary.get("dishes_unavailable_at") or {}
    for dish in unavail:
        recipe = menu_dict.get(dish)
        if not recipe:
            continue
        for ing in recipe["ingredients"]:
            cur = ema_burn.get(ing["ingredient"], 0.5)
            ema_burn[ing["ingredient"]] = max(cur, cur * 1.5)


def detect_outages(state: dict, observation: dict, day: int) -> None:
    outage = set(state.get("outage_suppliers") or [])
    shortfall_log = state.get("delivery_shortfall_log", [])
    for entry in observation.get("delivery_history") or []:
        ordered = entry.get("ordered_kg", 0) or 0
        delivered = entry.get("delivered_kg", 0) or 0
        if ordered > 0 and delivered < ordered * 0.5:
            outage.add(entry.get("supplier"))
            shortfall_log.append({
                "d": day,
                "sup": entry.get("supplier"),
                "frac": round(delivered / max(ordered, 1), 2),
            })
    for alert in observation.get("alerts") or []:
        text = (alert or "").lower()
        if any(k in text for k in KEYWORDS["supply_crisis"]):
            for sup in observation.get("supplier_catalog") or []:
                if sup["name"].lower() in text:
                    outage.add(sup["name"])
    state["outage_suppliers"] = sorted(outage)
    state["delivery_shortfall_log"] = shortfall_log[-10:]


def snapshot_supplier_prices(state: dict, observation: dict) -> dict[str, float]:
    """Update first-seen price snapshot, return current prices keyed `<sup>:<ing>`."""
    snap = state.get("supplier_price_snapshot", {})
    current: dict[str, float] = {}
    for sup in observation.get("supplier_catalog") or []:
        for ing, price in (sup.get("ingredients") or {}).items():
            key = f"{sup['name']}:{ing}"
            current[key] = price
            snap.setdefault(key, price)
    state["supplier_price_snapshot"] = snap
    return current


# ---------------------------------------------------------------------------
# Tier helpers

def tier(severity: float) -> str:
    for name, lb in TIER_BANDS:
        if severity >= lb:
            return name
    return "off"


def tier_value(tier_name: str, low: float, med: float, high: float, off: float = 0.0) -> float:
    return {"off": off, "low": low, "med": med, "high": high}.get(tier_name, off)


# ---------------------------------------------------------------------------
# Regime classifiers — each returns severity in [0,1]

def _kw_score(text: str, keywords) -> float:
    text = (text or "").lower()
    hits = sum(1 for k in keywords if k in text)
    if not hits:
        return 0.0
    return min(1.0, 0.4 + 0.2 * hits)


def _alerts_text(observation: dict) -> str:
    return " ".join(str(a) for a in (observation.get("alerts") or []))


def classify_tourist(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["tourist"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    covers_ema = state.get("covers_ema", 0.0)
    yest_covers = (obs.get("service_summary") or {}).get("total_covers", 0)
    if covers_ema > 20 and yest_covers > 0:
        ratio = yest_covers / max(covers_ema, 1.0)
        if ratio > 1.25:
            r_score = min(1.0, (ratio - 1.0) * 1.2)
            reasons.append(f"covers_ratio={ratio:.2f}->{r_score:.2f}")
            score = max(score, r_score)
    trend = obs.get("customer_trend") or "Stable"
    if trend == "Growing":
        score = min(1.0, score + 0.1)
        reasons.append("trend=Growing(+0.10)")
    return score, reasons


def classify_demand_drought(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["demand_drought"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    covers_ema = state.get("covers_ema", 0.0)
    yest_covers = (obs.get("service_summary") or {}).get("total_covers", 0)
    if covers_ema > 30 and day > 3:
        ratio = yest_covers / max(covers_ema, 1.0)
        if ratio < 0.8:
            r_score = min(1.0, (0.8 - ratio) * 1.6)
            reasons.append(f"covers_ratio={ratio:.2f}->{r_score:.2f}")
            score = max(score, r_score)
    trend = obs.get("customer_trend") or "Stable"
    weather_fc = obs.get("weather_forecast") or []
    bad_days = sum(1 for w in weather_fc[:3] if w in ("rainy", "stormy"))
    if trend == "Declining":
        score = min(1.0, score + 0.15)
        reasons.append("trend=Declining(+0.15)")
    if bad_days >= 2:
        score = min(1.0, score + 0.10)
        reasons.append(f"bad_weather_fc={bad_days}(+0.10)")
    return score, reasons


def classify_supply_crisis(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["supply_crisis"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    recent_shortfalls = [s for s in state.get("delivery_shortfall_log", []) if s["d"] >= day - 5]
    if recent_shortfalls:
        sf_score = min(1.0, 0.4 + 0.2 * len(recent_shortfalls))
        reasons.append(f"shortfalls5d={len(recent_shortfalls)}->{sf_score:.2f}")
        score = max(score, sf_score)
    if state.get("outage_suppliers"):
        score = max(score, 0.6)
        reasons.append(f"outages={len(state['outage_suppliers'])}")
    return score, reasons


def classify_capacity_limited(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["capacity_limited"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    # If util peak is at/over 1.0 with lots of walkouts, that's pressure, not capacity
    # constraint per se — true capacity constraint is observed in covers ceiling vs EMA.
    return score, reasons


def classify_reputation_crisis(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["reputation_crisis"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    rep = obs.get("reputation_band") or "Good"
    rep_score = {"Poor": 0.9, "Fair": 0.6, "Good": 0.0, "Very Good": 0.0, "Excellent": 0.0}.get(rep, 0.0)
    if rep_score > 0:
        reasons.append(f"rep_band={rep}->{rep_score:.2f}")
        score = max(score, rep_score)
    reviews = obs.get("recent_reviews") or []
    last = reviews[-5:]
    if len(last) >= 2:
        avg = sum(r["stars"] for r in last) / len(last)
        if avg < 3.0:
            rs = min(1.0, (3.0 - avg) * 0.5)
            reasons.append(f"reviews_avg={avg:.1f}->{rs:.2f}")
            score = max(score, rs)
    return score, reasons


def classify_cost_inflation(obs: dict, state: dict, day: int) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    kw = _kw_score(_alerts_text(obs), KEYWORDS["cost_inflation"])
    if kw > 0:
        reasons.append(f"alert_kw={kw:.2f}")
        score = max(score, kw)
    snap = state.get("supplier_price_snapshot", {})
    if snap:
        current = {}
        for sup in obs.get("supplier_catalog") or []:
            for ing, price in (sup.get("ingredients") or {}).items():
                current[f"{sup['name']}:{ing}"] = price
        ratios = []
        for k, base in snap.items():
            if k in current and base > 0:
                ratios.append(current[k] / base)
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            if avg_ratio > 1.05:
                rs = min(1.0, (avg_ratio - 1.0) * 4)
                reasons.append(f"price_ratio_avg={avg_ratio:.2f}->{rs:.2f}")
                score = max(score, rs)
    return score, reasons


def classify_unknown_anomaly(
    obs: dict, state: dict, day: int, named_severities: dict[str, float]
) -> tuple[float, list[str]]:
    """Fires when something weird happens that no named regime explains."""
    reasons = []
    score = 0.0
    # 1. Alerts present but no named regime crossed LOW threshold.
    alerts = obs.get("alerts") or []
    if alerts and max(named_severities.values(), default=0.0) < 0.2:
        reasons.append(f"alerts={len(alerts)}_unmatched")
        score = max(score, 0.5)
    # 2. >2 sigma cover swing without tourist/drought claiming it.
    covers_var = state.get("covers_var", 0.0)
    covers_ema = state.get("covers_ema", 0.0)
    yest_covers = (obs.get("service_summary") or {}).get("total_covers", 0)
    sigma = covers_var ** 0.5
    if sigma > 5 and covers_ema > 30:
        z = abs(yest_covers - covers_ema) / max(sigma, 1.0)
        if z > 2.0 and named_severities.get("tourist", 0) < 0.3 and named_severities.get("demand_drought", 0) < 0.3:
            zs = min(1.0, (z - 2.0) * 0.4 + 0.4)
            reasons.append(f"covers_z={z:.1f}->{zs:.2f}")
            score = max(score, zs)
    # 3. Walkouts spike without reputation_crisis claiming it.
    wko = (obs.get("service_summary") or {}).get("walkout_band") or "None"
    if wko in ("Some", "Many") and named_severities.get("reputation_crisis", 0) < 0.3:
        reasons.append(f"walkouts={wko}_unexplained")
        score = max(score, 0.5)
    return score, reasons


def classify_regimes(obs: dict, state: dict, day: int) -> tuple[dict[str, float], dict[str, list[str]]]:
    severities: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    severities["tourist"], reasons["tourist"] = classify_tourist(obs, state, day)
    severities["demand_drought"], reasons["demand_drought"] = classify_demand_drought(obs, state, day)
    severities["supply_crisis"], reasons["supply_crisis"] = classify_supply_crisis(obs, state, day)
    severities["capacity_limited"], reasons["capacity_limited"] = classify_capacity_limited(obs, state, day)
    severities["reputation_crisis"], reasons["reputation_crisis"] = classify_reputation_crisis(obs, state, day)
    severities["cost_inflation"], reasons["cost_inflation"] = classify_cost_inflation(obs, state, day)
    severities["unknown_anomaly"], reasons["unknown_anomaly"] = classify_unknown_anomaly(
        obs, state, day, {k: v for k, v in severities.items()}
    )
    return severities, reasons


# ---------------------------------------------------------------------------
# Composer

def pick_owner(domain: str, severities: dict[str, float]) -> str:
    for regime in PRIORITY.get(domain, []):
        if severities.get(regime, 0.0) >= 0.20:
            return regime
    return "baseline"


def is_suppressed(domain: str, owner: str) -> bool:
    return (domain, owner) in SUPPRESSES


# ---------------------------------------------------------------------------
# Discrete-calendar lead-time engine (from luigi)

def days_until_next_delivery(
    today_dow: str, lead_time: int, delivery_days: list[str], from_offset: int = 0
) -> int:
    if today_dow not in DOW_INDEX:
        return lead_time + 1
    today_idx = DOW_INDEX[today_dow]
    delivery_idx_set = {DOW_INDEX[d] for d in delivery_days if d in DOW_INDEX}
    if not delivery_idx_set:
        return 30
    start = max(lead_time, from_offset + 1)
    for offset in range(start, start + 14):
        candidate_idx = (today_idx + offset) % 7
        if candidate_idx in delivery_idx_set:
            return offset
    return 30


# ---------------------------------------------------------------------------
# Rule sets per (regime, domain)
#
# Each rule set takes (obs, state, tier_name, knobs) and returns a dict of
# parameters consumed by build_orders / staffing / etc. Keeping these as
# parameter producers (not action emitters) lets the central translator stay
# uniform.

# ---- inventory orders ----

def inv_params(owner: str, t: str, knobs: dict, obs: dict, state: dict) -> dict:
    """Return safety_days, service_level, supplier_diversify (bool) for build_orders."""
    p = {"safety_days": 2, "service_level": 1.5, "supplier_diversify": False}
    if owner == "supply_crisis":
        sd_tier = knobs.get("safety_days_tier") or t
        p["safety_days"] = int(tier_value(sd_tier, low=3, med=4, high=6))
        p["supplier_diversify"] = True
        p["service_level"] = 1.7
    elif owner == "cost_inflation":
        # Buy bigger lots while prices still low; cap to shelf life.
        p["safety_days"] = 3 if t in ("med", "high") else 2
        p["service_level"] = 1.5
    elif owner == "tourist":
        ex = knobs.get("extra_buffer_tier") or t
        p["safety_days"] = 2 + int(tier_value(ex, low=1, med=2, high=3))
        p["service_level"] = 1.7
    elif owner == "capacity_limited":
        # Reduce inventory pressure during reduced capacity.
        p["safety_days"] = 1
        p["service_level"] = 1.2
    elif owner == "demand_drought":
        p["safety_days"] = 1
        p["service_level"] = 1.2
    return p


# ---- staffing ----

def staffing_target(owner: str, t: str, knobs: dict, obs: dict, state: dict, day: int) -> int:
    dow = obs.get("day_of_week") or "Monday"
    base = STAFF_DOW_BASE.get(dow, 8)
    # Walkout / capacity-pressure bumps regardless of owner.
    summary = obs.get("service_summary") or {}
    wko = summary.get("walkout_band") or "None"
    if wko == "Many":
        base += 2
    elif wko == "Some":
        base += 1
    if summary.get("table_utilization_peak", 0) > 0.9:
        base += 1

    if owner == "capacity_limited":
        # Cut hard during construction; recover later.
        cut = int(tier_value(t, low=2, med=3, high=4))
        base = max(base - cut, 4)
    elif owner == "tourist":
        ex = knobs.get("extra_staff_tier") or t
        base += int(tier_value(ex, low=1, med=2, high=3))
    elif owner == "reputation_crisis":
        # Over-staff for quality / pace.
        base += int(tier_value(t, low=1, med=2, high=2))
    elif owner == "demand_drought":
        base -= int(tier_value(t, low=1, med=2, high=2))
    elif owner == "baseline":
        weather = obs.get("weather_today") or "cloudy"
        base += {"sunny": 1, "cloudy": 0, "rainy": -1, "stormy": -2}.get(weather, 0)
        rep = obs.get("reputation_band") or "Good"
        base += {"Excellent": 1, "Very Good": 0, "Good": 0, "Fair": -1, "Poor": -2}.get(rep, 0)

    # Cash floor.
    cash = obs.get("cash", 15000)
    if cash < 2000:
        base = min(base, 4)
    elif cash < 3000:
        base = min(base, 5)

    return max(3, min(15, int(base)))


# ---- pricing ----

def pricing_multipliers(
    owner: str, t: str, knobs: dict, obs: dict, state: dict
) -> dict[str, float]:
    """Return {dish: multiplier} in [0.81, 1.19]."""
    out: dict[str, float] = {}
    active = obs.get("active_menu") or []
    if owner == "cost_inflation":
        mk = knobs.get("markup_tier") or t
        m = 1.0 + tier_value(mk, low=0.05, med=0.10, high=0.15)
    elif owner == "reputation_crisis":
        d = knobs.get("discount_tier") or t
        m = 1.0 - tier_value(d, low=0.05, med=0.10, high=0.15)
    elif owner == "tourist":
        m = 1.0 + tier_value(t, low=0.04, med=0.08, high=0.12)
    elif owner == "demand_drought":
        m = 1.0 - tier_value(t, low=0.04, med=0.08, high=0.12)
    else:
        return {}
    m = max(0.81, min(1.19, m))
    for dish in active:
        out[dish] = m
    return out


# ---- marketing ----

def marketing_amount(owner: str, t: str, knobs: dict, obs: dict, state: dict, day: int) -> int:
    if is_suppressed("marketing", owner):
        return 0
    cash = obs.get("cash", 15000)
    if cash < 2000:
        return 0
    if owner == "reputation_crisis":
        amt = int(tier_value(t, low=100, med=200, high=300))
    elif owner == "demand_drought":
        sp = knobs.get("spend_tier") or t
        amt = int(tier_value(sp, low=80, med=160, high=250))
    elif owner == "baseline":
        rep = obs.get("reputation_band") or "Good"
        amt = 150 if rep in ("Very Good", "Excellent") else 80
    else:
        amt = 50
    # End-game taper.
    days_remaining = obs.get("days_remaining", 30)
    if days_remaining <= 3:
        amt = max(0, amt - 50)
    return max(0, min(500, amt))


# ---- happy hour ----

def should_happy_hour(owner: str, t: str, knobs: dict, obs: dict, state: dict, day: int) -> bool:
    if is_suppressed("happy_hour", owner):
        return False
    if obs.get("cash", 15000) < 2000:
        return False
    if state.get("consec_hh", 0) >= 2 and owner != "reputation_crisis":
        return False  # diminishing returns
    if owner == "reputation_crisis":
        return True
    if owner == "demand_drought":
        dow = obs.get("day_of_week") or ""
        weather = obs.get("weather_today") or "cloudy"
        return dow in ("Monday", "Tuesday", "Wednesday") or weather in ("rainy", "stormy")
    if owner == "baseline":
        days_remaining = obs.get("days_remaining", 30)
        return days_remaining <= 4 and state.get("consec_hh", 0) < 3
    return False


# ---- daily special ----

def pick_daily_special(owner: str, t: str, knobs: dict, obs: dict, state: dict) -> str | None:
    active = obs.get("active_menu") or []
    if not active:
        return None
    summary = obs.get("service_summary") or {}
    sold = summary.get("dishes_sold") or {}
    unavail = summary.get("dishes_unavailable_at") or {}
    candidates = [d for d in active if d not in unavail]
    if not candidates:
        return None
    menu_book = {m["name"]: m for m in (obs.get("menu_book") or [])}

    if owner == "supply_crisis":
        # Push a dish that uses ingredients we have plenty of.
        inv = {row["ingredient"]: row.get("total_kg", 0.0) for row in (obs.get("inventory") or [])}
        ema = state.get("ema_burn", {})

        def headroom(dish: str) -> float:
            recipe = menu_book.get(dish)
            if not recipe:
                return 0.0
            return min(
                (inv.get(ing["ingredient"], 0.0) / max(ema.get(ing["ingredient"], 0.5), 0.1))
                for ing in recipe["ingredients"]
            )

        candidates.sort(key=headroom, reverse=True)
        return candidates[0]
    if owner == "reputation_crisis":
        # Best-rated proxy: top seller (acts as anchor).
        candidates.sort(key=lambda d: -sold.get(d, 0))
        return candidates[0]
    # baseline: rotate by day.
    candidates.sort(key=lambda d: -sold.get(d, 0))
    return candidates[0]


# ---- menu ----

def menu_choice(
    owner: str, t: str, knobs: dict, obs: dict, state: dict, projections: dict
) -> list[str] | None:
    menu_book = {m["name"]: m for m in (obs.get("menu_book") or [])}
    active = obs.get("active_menu") or []
    if not menu_book or not active:
        return None

    if owner == "supply_crisis":
        # Drop dishes whose ingredients are critically low (and we can't restock).
        viable = []
        for name, dish in menu_book.items():
            ok = True
            for ing in dish.get("ingredients", []):
                proj = projections.get(ing["ingredient"], {})
                if proj.get("days_effective", 999) < 2:
                    ok = False
                    break
            if ok:
                viable.append(name)
        if len(viable) >= 5 and set(viable) != set(active):
            return viable[:10]
        return None
    if owner == "capacity_limited":
        # Lean menu — keep top sellers only.
        sold = (obs.get("service_summary") or {}).get("dishes_sold") or {}
        ranked = sorted(menu_book.keys(), key=lambda d: -sold.get(d, 0))
        lean = ranked[:6]
        if len(lean) >= 5 and set(lean) != set(active):
            return lean
        return None
    return None


# ---------------------------------------------------------------------------
# Inventory order builder (regime-aware) — adapted from luigi.

def build_orders(
    observation: dict,
    state: dict,
    menu_dict: dict,
    inv_p: dict,
    phase: str,
) -> list[dict]:
    actions: list[dict] = []
    today_dow = observation.get("day_of_week") or "Monday"
    days_remaining = observation.get("days_remaining", 30)
    safety_days = int(inv_p.get("safety_days", 2))
    service_level = float(inv_p.get("service_level", 1.5))
    diversify = bool(inv_p.get("supplier_diversify", False))

    inventory = {row["ingredient"]: row for row in (observation.get("inventory") or [])}
    pending_by_ing: dict[str, float] = {}
    for po in observation.get("pending_orders") or []:
        pending_by_ing[po["ingredient"]] = (
            pending_by_ing.get(po["ingredient"], 0.0) + po["quantity_kg"]
        )

    catalog = observation.get("supplier_catalog") or []
    outage = set(state.get("outage_suppliers") or [])
    available = [s for s in catalog if s["name"] not in outage] or catalog

    overhead_per_day = 300 + observation.get("staff_level", 8) * 120 + 50
    reserve_days = 6 if phase == "early" else 4
    if phase == "endgame":
        reserve_days = 3
    if state.get("outage_suppliers") or observation.get("cash", 0) < 6000:
        reserve_days += 2
    reserve = overhead_per_day * reserve_days
    budget = max(observation["cash"] - reserve, 0.0)

    covers_ema = state.get("covers_ema", 0.0)
    demand_scalar = 1.0
    if covers_ema > 0 and covers_ema < 90:
        demand_scalar = max(covers_ema / 140.0, 0.5)

    ema_burn = state.get("ema_burn", {})
    bootstrap_burn = _bootstrap_burn(observation, menu_dict)

    seen_ings: set[str] = set()
    for sup in catalog:
        seen_ings.update(sup["ingredients"].keys())

    shelf_life_by_ing = {
        row["ingredient"]: row.get("shelf_life_days", 7)
        for row in (observation.get("inventory") or [])
    }

    # Track per-ingredient suppliers used today (for diversify).
    used_suppliers_per_ing: dict[str, set[str]] = {}

    for ingredient in sorted(seen_ings):
        avg_burn = max(
            ema_burn.get(ingredient, 0.0),
            bootstrap_burn.get(ingredient, 0.0),
            0.3,
        ) * demand_scalar
        burn = avg_burn * service_level

        on_hand = inventory.get(ingredient, {}).get("total_kg", 0.0)
        pending = pending_by_ing.get(ingredient, 0.0)

        candidate_quotes = []
        for sup in available:
            if ingredient not in sup["ingredients"]:
                continue
            d1 = days_until_next_delivery(today_dow, sup["lead_time_days"], sup["delivery_days"])
            d2 = days_until_next_delivery(
                today_dow, sup["lead_time_days"], sup["delivery_days"], from_offset=d1
            )
            candidate_quotes.append({
                "name": sup["name"],
                "price": sup["ingredients"][ingredient],
                "min": sup["min_order_kg"],
                "d1": d1,
                "d2": d2,
            })
        if not candidate_quotes:
            continue

        days_to_empty = on_hand / max(burn, 0.1)
        sufficient = [q for q in candidate_quotes if q["d1"] <= max(days_to_empty, 2)]
        if sufficient:
            sufficient.sort(key=lambda q: q["price"])
            primary = sufficient[0]
        else:
            candidate_quotes.sort(key=lambda q: (q["d1"], q["price"]))
            primary = candidate_quotes[0]

        target_kg = burn * (primary["d2"] + safety_days)
        projected_stock = on_hand + pending - burn * primary["d1"]
        if projected_stock >= target_kg:
            continue

        shortfall = target_kg - projected_stock
        order_qty = max(shortfall, primary["min"])

        shelf_life = shelf_life_by_ing.get(ingredient, 7)
        spoil_cap = avg_burn * shelf_life * 0.9
        if spoil_cap >= primary["min"]:
            order_qty = min(order_qty, spoil_cap)

        usable_days = max(days_remaining - primary["d1"], 0)
        if usable_days <= 0:
            continue
        if phase == "endgame":
            cap_qty = burn * usable_days
            if cap_qty < primary["min"]:
                continue
            order_qty = min(order_qty, cap_qty)

        cost = order_qty * primary["price"]
        if cost > budget:
            order_qty = budget / primary["price"]
            if order_qty < primary["min"]:
                continue
            cost = order_qty * primary["price"]

        actions.append({
            "tool": "place_order",
            "args": {
                "supplier": primary["name"],
                "ingredient": ingredient,
                "quantity_kg": round(order_qty, 1),
            },
        })
        budget -= cost
        used_suppliers_per_ing.setdefault(ingredient, set()).add(primary["name"])

        # Diversification: place a smaller backup order from a different supplier.
        if diversify and len(candidate_quotes) >= 2 and budget > 0:
            backups = [
                q for q in candidate_quotes
                if q["name"] not in used_suppliers_per_ing[ingredient]
            ]
            if backups:
                backups.sort(key=lambda q: q["d1"])
                b = backups[0]
                qty = max(b["min"], shortfall * 0.4)
                bcost = qty * b["price"]
                if bcost <= budget and qty * avg_burn * 0.0 + qty <= spoil_cap * 1.2:
                    actions.append({
                        "tool": "place_order",
                        "args": {
                            "supplier": b["name"],
                            "ingredient": ingredient,
                            "quantity_kg": round(qty, 1),
                        },
                    })
                    budget -= bcost
                    used_suppliers_per_ing[ingredient].add(b["name"])

    return actions


def _bootstrap_burn(observation: dict, menu_dict: dict) -> dict[str, float]:
    active = observation.get("active_menu") or list(menu_dict.keys())
    if not active:
        return {}
    summary = observation.get("service_summary") or {}
    observed_covers = summary.get("total_covers") if summary else None
    expected_covers = max(observed_covers or 0, 110)
    per_dish = expected_covers / max(len(active), 1)
    burn: dict[str, float] = {}
    for dish in active:
        recipe = menu_dict.get(dish)
        if not recipe:
            continue
        for ing in recipe["ingredients"]:
            burn[ing["ingredient"]] = burn.get(ing["ingredient"], 0.0) + ing["quantity_kg"] * per_dish
    return burn


# ---------------------------------------------------------------------------
# Inventory projections (for menu rule set).

def project_inventory(observation: dict, ema_burn: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for inv in observation.get("inventory") or []:
        name = inv["ingredient"]
        usable = sum(
            b["quantity_kg"]
            for b in inv.get("batches", [])
            if b.get("expires_in_days", 0) > 1
        )
        pending = sum(
            po["quantity_kg"]
            for po in observation.get("pending_orders") or []
            if po["ingredient"] == name
        )
        eff = usable + pending
        dr = ema_burn.get(name, 0.0)
        out[name] = {
            "usable": usable,
            "effective": eff,
            "daily_rate": dr,
            "days_effective": round(eff / dr, 1) if dr > 0.01 else 999,
        }
    return out


# ---------------------------------------------------------------------------
# LLM oversight

LLM_SYSTEM_STD = """You are the Oversight Brain of a regime-based restaurant agent.

A deterministic Python core has already:
  1) computed `severities` for each regime (tourist, demand_drought, supply_crisis,
     capacity_limited, reputation_crisis, cost_inflation, unknown_anomaly).
  2) picked an OWNER regime per decision domain via fixed priority lists.

Your job is to SANITY-CHECK the categorization and pick discrete tiers for
rule-set knobs.

Output ONLY a JSON object with this exact shape (no markdown, no explanation):
{
  "regime_overrides": {"<regime_name>": <float 0..1>, ...},
  "tier_choices": {"<knob_name>": "off"|"low"|"med"|"high", ...}
}

Rules:
- regime_overrides may adjust any regime's severity; values are clamped 0..1.
  Only override if you have a strong reason from the signals.
- Valid knob names are listed in the user message under `knobs`.
- Tiers MUST be one of off/low/med/high (any other value is rejected).
- Empty objects are valid (means "trust the deterministic defaults").
"""

LLM_SYSTEM_SYNTH = """You are the Oversight Brain of a regime-based restaurant agent.

Multiple regimes are active simultaneously (severity >= 0.5). The default
per-domain ownership rule may be too crude; you may specify per-domain
OVERRIDES that pick a different active regime to own that decision.

Output ONLY a JSON object with this exact shape (no markdown, no explanation):
{
  "regime_overrides": {"<regime_name>": <float 0..1>, ...},
  "tier_choices": {"<knob_name>": "off"|"low"|"med"|"high", ...},
  "domain_overrides": {
    "<domain>": {"regime": "<regime_name>", "tier": "off"|"low"|"med"|"high", "note": "..."}
  }
}

Domains: inventory, staffing, pricing, marketing, happy_hour, daily_special, menu.
Domain overrides accepted only if the named regime is currently active.
Empty objects are valid.
"""


def _knob_names(severities: dict[str, float]) -> list[str]:
    """Which knobs are relevant given current owners?"""
    relevant = []
    owners = {dom: pick_owner(dom, severities) for dom in PRIORITY}
    if owners["inventory"] == "supply_crisis":
        relevant += ["safety_days_tier", "supplier_diversify_tier"]
    if owners["inventory"] == "tourist":
        relevant += ["extra_buffer_tier"]
    if owners["staffing"] == "tourist":
        relevant += ["extra_staff_tier"]
    if owners["pricing"] == "reputation_crisis":
        relevant += ["discount_tier"]
    if owners["pricing"] == "cost_inflation":
        relevant += ["markup_tier"]
    if owners["marketing"] == "demand_drought":
        relevant += ["spend_tier"]
    return relevant


def call_llm_oversight(
    observation: dict, state: dict, day: int, severities: dict[str, float]
) -> dict | None:
    co_severe = [r for r, s in severities.items() if s >= 0.5 and r != "unknown_anomaly"]
    synth = len(co_severe) >= 2

    summary = observation.get("service_summary") or {}
    context = {
        "day": day,
        "days_remaining": observation.get("days_remaining"),
        "cash": round(observation.get("cash", 0)),
        "reputation": observation.get("reputation_band"),
        "trend": observation.get("customer_trend"),
        "walkout_band": summary.get("walkout_band"),
        "covers_yesterday": summary.get("total_covers"),
        "covers_ema": round(state.get("covers_ema", 0), 1),
        "weather_today": observation.get("weather_today"),
        "weather_forecast": (observation.get("weather_forecast") or [])[:3],
        "alerts": (observation.get("alerts") or [])[:4],
        "severities": {k: round(v, 2) for k, v in severities.items()},
        "regime_history": state.get("regime_history", [])[-3:],
        "knobs": _knob_names(severities),
    }
    if synth:
        context["owners_default"] = {dom: pick_owner(dom, severities) for dom in PRIORITY}
        context["rule_summary"] = {
            r: _RULE_SUMMARY[r] for r in co_severe if r in _RULE_SUMMARY
        }

    user_msg = f"Day {day}/30. Context:\n{json.dumps(context, separators=(',', ':'))}"
    if len(user_msg) > MAX_USER_MSG_CHARS:
        slim = {k: v for k, v in context.items() if k not in ("regime_history", "rule_summary")}
        user_msg = f"Day {day}/30. Context:\n{json.dumps(slim, separators=(',', ':'))}"
    if len(user_msg) > MAX_USER_MSG_CHARS:
        return None

    system = LLM_SYSTEM_SYNTH if synth else LLM_SYSTEM_STD
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            api_base=LLM_API_BASE,
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        parsed = json.loads(content)
        parsed["_synth"] = synth
        parsed["_raw"] = content
        return parsed
    except Exception as e:
        _dbg(f"LLM day {day}: {type(e).__name__}: {str(e)[:120]}")
        return None


_RULE_SUMMARY = {
    "tourist": "Extra staff and bigger order buffers; raise prices slightly; suppress marketing/HH (already busy).",
    "demand_drought": "Cut staff; happy hour on slow days; bump marketing; trim prices.",
    "supply_crisis": "Order earlier with larger safety buffers; diversify suppliers; drop menu items at risk; daily special on abundant ingredients.",
    "capacity_limited": "Cut staff; lean menu; no marketing; conservative orders.",
    "reputation_crisis": "Over-staff for quality; discount prices; aggressive marketing + happy hour; daily special on top dish.",
    "cost_inflation": "Raise menu prices to maintain margin; switch to cheaper suppliers when delivery permits.",
}


def apply_llm_overrides(
    raw: dict | None, severities: dict[str, float]
) -> tuple[dict[str, float], dict[str, str], dict[str, tuple[str, str]]]:
    """Returns (new_severities, knob_tiers, domain_overrides)."""
    new_sev = dict(severities)
    knobs: dict[str, str] = {}
    domain_overrides: dict[str, tuple[str, str]] = {}
    if not isinstance(raw, dict):
        return new_sev, knobs, domain_overrides
    ro = raw.get("regime_overrides") or {}
    if isinstance(ro, dict):
        for r, v in ro.items():
            if r in REGIMES:
                try:
                    new_sev[r] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    pass
    tc = raw.get("tier_choices") or {}
    if isinstance(tc, dict):
        for k, v in tc.items():
            if isinstance(v, str) and v in ("off", "low", "med", "high"):
                knobs[k] = v
    do = raw.get("domain_overrides") or {}
    if isinstance(do, dict):
        for dom, spec in do.items():
            if dom not in PRIORITY or not isinstance(spec, dict):
                continue
            reg = spec.get("regime")
            tier_name = spec.get("tier")
            if (
                reg in REGIMES
                and new_sev.get(reg, 0.0) >= 0.20
                and isinstance(tier_name, str)
                and tier_name in ("off", "low", "med", "high")
            ):
                domain_overrides[dom] = (reg, tier_name)
    return new_sev, knobs, domain_overrides


# ---------------------------------------------------------------------------
# Strategy

def _phase(day: int, days_remaining: int) -> str:
    if days_remaining <= 4:
        return "endgame"
    if day <= 5:
        return "early"
    return "mid"


def _format_severities_line(sev: dict[str, float]) -> str:
    parts = []
    for r, s in sorted(sev.items(), key=lambda kv: -kv[1]):
        if s >= 0.2:
            parts.append(f"{r}:{s:.2f}({tier(s)[0].upper()})")
    return " ".join(parts) or "baseline"


def strategy(observation: dict, day: int) -> list[dict]:
    actions: list[dict] = []
    menu_dict = {m["name"]: m for m in (observation.get("menu_book") or [])}
    state = load_state(observation.get("notes"))

    # 1. State update
    update_emas(state, observation, menu_dict)
    detect_outages(state, observation, day)
    snapshot_supplier_prices(state, observation)

    # 2. Classify
    severities, reasons = classify_regimes(observation, state, day)
    if DEBUG:
        for r, rs in reasons.items():
            if rs:
                _dbg(f"day={day} regime={r} sev={severities[r]:.2f} reasons={rs}")

    # 3. LLM oversight
    llm_raw = call_llm_oversight(observation, state, day, severities)
    severities, knobs, domain_overrides = apply_llm_overrides(llm_raw, severities)

    # 4. Compose: pick owner per domain (with LLM domain overrides applied)
    owners: dict[str, tuple[str, str]] = {}
    for dom in PRIORITY:
        if dom in domain_overrides:
            reg, t = domain_overrides[dom]
            owners[dom] = (reg, t)
        else:
            reg = pick_owner(dom, severities)
            t = tier(severities.get(reg, 0.0)) if reg != "baseline" else "off"
            owners[dom] = (reg, t)

    phase = _phase(day, observation.get("days_remaining", 30))

    # 5. Build action list per domain.

    # Inventory
    inv_owner, inv_t = owners["inventory"]
    inv_p = inv_params(inv_owner, inv_t, knobs, observation, state)
    actions.extend(build_orders(observation, state, menu_dict, inv_p, phase))

    # Menu (do this before pricing/special so we don't price dropped dishes)
    menu_owner, menu_t = owners["menu"]
    projections = project_inventory(observation, state.get("ema_burn", {}))
    new_menu = menu_choice(menu_owner, menu_t, knobs, observation, state, projections) if menu_owner != "baseline" else None
    if new_menu and set(new_menu) != set(observation.get("active_menu") or []):
        actions.append({"tool": "set_menu", "args": {"dishes": new_menu}})

    # Staffing
    staff_owner, staff_t = owners["staffing"]
    target_staff = staffing_target(staff_owner, staff_t, knobs, observation, state, day)
    if target_staff != observation.get("staff_level"):
        actions.append({"tool": "set_staff_level", "args": {"level": target_staff}})

    # Pricing
    price_owner, price_t = owners["pricing"]
    multipliers = pricing_multipliers(price_owner, price_t, knobs, observation, state)
    for dish, mult in multipliers.items():
        if dish not in menu_dict:
            continue
        base = menu_dict[dish]["base_price"]
        new_price = round(base * mult, 2)
        current = next(
            (m["current_price"] for m in (observation.get("menu_book") or []) if m["name"] == dish),
            base,
        )
        if abs(new_price - current) >= 0.3:
            actions.append({"tool": "set_price", "args": {"dish": dish, "price": new_price}})

    # Marketing
    mkt_owner, mkt_t = owners["marketing"]
    mkt = marketing_amount(mkt_owner, mkt_t, knobs, observation, state, day)
    actions.append({"tool": "set_marketing_spend", "args": {"amount": int(mkt)}})

    # Happy hour
    hh_owner, hh_t = owners["happy_hour"]
    if should_happy_hour(hh_owner, hh_t, knobs, observation, state, day):
        actions.append({"tool": "run_happy_hour", "args": {}})
        state["consec_hh"] = state.get("consec_hh", 0) + 1
        state["last_hh_day"] = day
    else:
        state["consec_hh"] = 0

    # Daily special
    ds_owner, ds_t = owners["daily_special"]
    special = pick_daily_special(ds_owner, ds_t, knobs, observation, state)
    if special:
        actions.append({"tool": "offer_daily_special", "args": {"dish": special}})

    # 6. Record regime history
    history = state.setdefault("regime_history", [])
    history.append({"d": day, "r": {k: round(v, 2) for k, v in severities.items() if v >= 0.2}})
    state["regime_history"] = history[-7:]

    # 7. Observability — terse stdout line
    own_str = " ".join(
        f"{dom}={reg}({t[0].upper()}{',suppr' if is_suppressed(dom, reg) else ''})"
        for dom, (reg, t) in owners.items()
        if reg != "baseline"
    )
    if not own_str:
        own_str = "all=baseline"
    llm_state = (
        "skip" if llm_raw is None
        else (f"synth({len(domain_overrides)}d,{len(knobs)}k)" if llm_raw.get("_synth")
              else f"ok({len(knobs)}k)")
    )
    print(
        f"[regime] d={day} ({observation.get('day_of_week','?')[:3]}) "
        f"sev=[{_format_severities_line(severities)}] | own: {own_str} | llm:{llm_state}"
    )

    # JSONL log
    if LOG_PATH:
        _log_jsonl({
            "day": day,
            "dow": observation.get("day_of_week"),
            "severities": {k: round(v, 3) for k, v in severities.items()},
            "owners": {dom: {"regime": r, "tier": t} for dom, (r, t) in owners.items()},
            "knobs": knobs,
            "domain_overrides": {dom: {"regime": r, "tier": t} for dom, (r, t) in domain_overrides.items()},
            "llm_raw": (llm_raw or {}).get("_raw") if isinstance(llm_raw, dict) else None,
            "llm_synth": bool(isinstance(llm_raw, dict) and llm_raw.get("_synth")),
            "actions": actions,
            "signals": {
                "cash": observation.get("cash"),
                "reputation": observation.get("reputation_band"),
                "trend": observation.get("customer_trend"),
                "covers_yesterday": (observation.get("service_summary") or {}).get("total_covers"),
                "walkout_band": (observation.get("service_summary") or {}).get("walkout_band"),
                "weather_today": observation.get("weather_today"),
                "alerts": observation.get("alerts") or [],
            },
        })

    # 8. Persist state
    actions.append({"tool": "save_notes", "args": {"text": serialize_state(state)}})

    return actions


# ---------------------------------------------------------------------------
# Entry

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first (the hackathon proxy key).")
        sys.exit(1)
    print(f"Using model: {MODEL} via {LLM_API_BASE}")
    if DEBUG:
        print("[regime] REGIME_DEBUG enabled")
    if LOG_PATH:
        print(f"[regime] logging to {LOG_PATH}")
    run_game(strategy, team_name="regime_agent", seed=42)
