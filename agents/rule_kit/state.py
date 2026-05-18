"""
Internal agent state.

The runner spawns a fresh `strategy()` call each turn, and `evaluate.py` runs
parallel games in threads. So module-level state would corrupt across games.

Our ONLY memory is the `notes` field (<= 4000 chars), which the server echoes
back inside each observation. We serialize state as compact JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------- records ----------

@dataclass
class DayRecord:
    day: int
    day_of_week: str
    weather: str
    covers: int
    revenue: float
    walkout_band: str
    reputation_band: str
    cash_end: float = 0.0
    # Task 2: per-day empirical mix data. dishes_sold maps dish_name -> count.
    dishes_sold: Dict[str, int] = field(default_factory=dict)
    # Task 5: list of ingredients that ran out during yesterday's service.
    # Populated from service_summary.dishes_unavailable_at AND from active
    # dish recipes whose ingredient is at 0 kg on-hand. The simulator's
    # stockout signal is unreliable when the whole kitchen is empty.
    stockout_ingredients: List[str] = field(default_factory=list)


@dataclass
class SupplierStat:
    """EWMA of supplier delivery performance."""
    delivered_ratio: float = 1.0   # delivered_kg / ordered_kg
    on_time: float = 1.0
    samples: int = 0


# ---------- state ----------

@dataclass
class AgentState:
    phase: str = "build"  # "build" | "optimize" | "endgame"
    history: List[DayRecord] = field(default_factory=list)
    suppliers: Dict[str, SupplierStat] = field(default_factory=dict)
    last_promo_day: int = -10
    last_menu_change_day: int = -10
    last_happy_hour_day: int = -10
    last_logged_day: int = 0  # so update_from_observation is idempotent
    # Task 8: rolling log of orders we PLACED. Each entry:
    #   {"day": int, "supplier": str, "ingredient": str, "kg": float}
    # Kept to last ~30 entries; we look at the last 7 days for diversification.
    recent_orders: List[Dict] = field(default_factory=list)
    # Port-fabiano: scenario tag fired by alerts text parsing.
    # "baseline" | "renovation" | "supply_crisis" | "tourist" | "inflation" | "health_scare"
    scenario: str = "baseline"
    # Tail of the last few alert strings the server has surfaced, used by
    # scenario.detect_scenario to keep classification sticky across days
    # where the alert isn't re-emitted.
    alerts_recent: List[str] = field(default_factory=list)
    # Port-fabiano: days we fired run_happy_hour. Used for 3-per-7-day cap.
    happy_hour_days: List[int] = field(default_factory=list)
    # ----- regime detection (Phase 1) -----
    # Captured on first observation. Carries cheapest unit price per
    # ingredient (from day-1 supplier_catalog), table_count, day-1 rep band.
    day1_baseline: Optional[Dict] = None
    # Rolling 7-day windows per axis of the raw residual signal.
    residuals: Dict[str, List[float]] = field(default_factory=dict)
    # Consecutive days in the current regime, per axis.
    regime_days: Dict[str, int] = field(default_factory=dict)
    # Hysteresis streak counters per axis: {"enter": int, "exit": int, "sign": int}.
    regime_streaks: Dict[str, Dict] = field(default_factory=dict)
    # Last-emitted RegimeSignal per axis (as dict) so we can round-trip.
    regime_signals: Dict[str, Dict] = field(default_factory=dict)
    # add other rolling stats here as you grow the agent

    MAX_HISTORY: int = 21
    MAX_RECENT_ORDERS: int = 60
    MAX_ALERTS_RECENT: int = 5
    NOTES_BUDGET: int = 4000

    # ---- update from observations ----

    def update_from_observation(self, obs: dict) -> None:
        """
        Roll yesterday's service_summary into history (idempotent — safe to
        call multiple times for the same observation).
        """
        svc = obs.get("service_summary") or {}
        yesterday = obs.get("day", 1) - 1
        if (
            svc
            and "total_covers" in svc
            and yesterday > 0
            and yesterday > self.last_logged_day
        ):
            # Task 2: capture dishes_sold so estimate_dish_mix can use empirical
            # fractions instead of a uniform 1/N split.
            raw_sold = svc.get("dishes_sold") or {}
            dishes_sold: Dict[str, int] = {}
            for k, v in raw_sold.items():
                try:
                    dishes_sold[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue

            # Task 5: collect stockout ingredients. Two signals:
            #  (a) service_summary.dishes_unavailable_at — explicit, but spotty.
            #  (b) any active dish whose recipe ingredient is at 0 kg on-hand
            #      RIGHT NOW (post-service). Catches whole-kitchen-empty days
            #      the simulator misses.
            stockouts: List[str] = []
            unavail = svc.get("dishes_unavailable_at") or {}
            inv_map = {
                str(i.get("ingredient")): float(i.get("total_kg") or 0)
                for i in (obs.get("inventory") or [])
            }
            recipe_map = {
                d.get("name"): d.get("ingredients") or []
                for d in (obs.get("menu_book") or [])
            }
            for dish_name in unavail.keys():
                for ing in recipe_map.get(dish_name, []):
                    ing_name = ing.get("ingredient")
                    if ing_name and ing_name not in stockouts:
                        stockouts.append(ing_name)
            for dish_name in obs.get("active_menu") or []:
                for ing in recipe_map.get(dish_name, []):
                    ing_name = ing.get("ingredient")
                    if not ing_name:
                        continue
                    if inv_map.get(ing_name, 0.0) <= 0.0 and ing_name not in stockouts:
                        stockouts.append(ing_name)

            self.history.append(
                DayRecord(
                    day=yesterday,
                    day_of_week=_dow_yesterday(obs.get("day_of_week", "Monday")),
                    weather=obs.get("weather_today", "?"),
                    covers=int(svc.get("total_covers", 0)),
                    revenue=float(svc.get("total_revenue", 0)),
                    walkout_band=str(svc.get("walkout_band", "?")),
                    reputation_band=str(obs.get("reputation_band", "?")),
                    cash_end=float(obs.get("cash", 0.0)),
                    dishes_sold=dishes_sold,
                    stockout_ingredients=stockouts,
                )
            )
            self.history = self.history[-self.MAX_HISTORY:]
            self.last_logged_day = yesterday

        # supplier reliability EWMA
        for d in obs.get("delivery_history", []) or []:
            self._record_delivery(d)

        # Port-fabiano: tail recent ALERT TEXT (strings only) for scenario
        # detection across the days where the same alert isn't re-emitted.
        # Reputation/event semantics are NOT consumed here — we just keep the
        # raw strings around so scenario.detect_scenario can keyword-match them.
        for a in obs.get("alerts") or []:
            if isinstance(a, str) and a.strip():
                self.alerts_recent.append(a.strip())
            elif isinstance(a, dict):
                for v in a.values():
                    if isinstance(v, str) and v.strip():
                        self.alerts_recent.append(v.strip())
                        break
        if len(self.alerts_recent) > self.MAX_ALERTS_RECENT:
            self.alerts_recent = self.alerts_recent[-self.MAX_ALERTS_RECENT:]

    def _record_delivery(self, d: dict, alpha: float = 0.3) -> None:
        name = d.get("supplier")
        if not name:
            return
        stat = self.suppliers.setdefault(name, SupplierStat())
        ordered = max(d.get("ordered_kg") or 0, 1e-6)
        delivered = d.get("delivered_kg") or 0
        ratio = min(delivered / ordered, 1.0)
        on_time = 1.0 if d.get("on_time") else 0.0
        stat.delivered_ratio = (1 - alpha) * stat.delivered_ratio + alpha * ratio
        stat.on_time = (1 - alpha) * stat.on_time + alpha * on_time
        stat.samples += 1

    # ---- empirical features ----


    def empirical_base_covers(self) -> float:
        """
        Base demand estimate from history.

        Uses the 75th percentile of the last 7 days rather than the simple
        average. The average gets dragged down by stockout-induced low-cover
        days — using a high percentile means we forecast (and prep) for the
        demand we COULD serve, not the demand our crashes let through.
        """
        if not self.history:
            return 0.0
        recent = sorted(r.covers for r in self.history[-7:])
        if not recent:
            return 0.0
        # p75: index = ceil(0.75 * n) - 1, clamped
        idx = max(0, min(len(recent) - 1, int(round(0.75 * (len(recent) - 1)))))
        return float(recent[idx])

    def days_since_stockout(self, ingredient: str) -> int:
        """
        Task 5: how many days since `ingredient` last appeared on a stockout
        list. Returns a large number (999) if we've never logged one.
        """
        if not self.history:
            return 999
        today = self.history[-1].day + 1  # the day we're currently planning
        last_seen: Optional[int] = None
        for r in self.history:
            if ingredient in (r.stockout_ingredients or []):
                last_seen = r.day
        if last_seen is None:
            return 999
        return today - last_seen

    def supplier_volume_share(self, lookback: int = 7) -> Dict[str, float]:
        """
        Task 8: fraction of total kg ordered per supplier over the last
        `lookback` days. Computed from `recent_orders`.
        """
        if not self.recent_orders:
            return {}
        # Anchor "today" to the largest day we've recorded.
        max_day = max((o.get("day", 0) for o in self.recent_orders), default=0)
        cutoff = max_day - lookback + 1
        totals: Dict[str, float] = {}
        for o in self.recent_orders:
            if o.get("day", 0) < cutoff:
                continue
            totals[o["supplier"]] = totals.get(o["supplier"], 0.0) + float(o.get("kg", 0.0))
        total = sum(totals.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in totals.items()}

    def record_placed_order(self, day: int, supplier: str, ingredient: str, kg: float) -> None:
        """Append a placed order to the rolling log for diversification stats."""
        self.recent_orders.append(
            {"day": day, "supplier": supplier, "ingredient": ingredient, "kg": float(kg)}
        )
        if len(self.recent_orders) > self.MAX_RECENT_ORDERS:
            self.recent_orders = self.recent_orders[-self.MAX_RECENT_ORDERS:]

    # ---- serialize ----

    @classmethod
    def from_notes(cls, notes_text: str) -> "AgentState":
        if not notes_text:
            return cls()
        try:
            data = json.loads(notes_text)
        except Exception:
            return cls()
        try:
            history = []
            for r in data.get("history", []):
                # Tolerate older notes blobs that don't include the new fields.
                r = dict(r)
                r.setdefault("dishes_sold", {})
                r.setdefault("stockout_ingredients", [])
                history.append(DayRecord(**r))
            suppliers = {k: SupplierStat(**v) for k, v in data.get("suppliers", {}).items()}
            return cls(
                phase=data.get("phase", "build"),
                history=history,
                suppliers=suppliers,
                last_promo_day=data.get("last_promo_day", -10),
                last_menu_change_day=data.get("last_menu_change_day", -10),
                last_happy_hour_day=data.get("last_happy_hour_day", -10),
                last_logged_day=data.get("last_logged_day", 0),
                recent_orders=list(data.get("recent_orders", []) or []),
                scenario=str(data.get("scenario", "baseline") or "baseline"),
                alerts_recent=list(data.get("alerts_recent", []) or []),
                happy_hour_days=[int(x) for x in (data.get("happy_hour_days") or [])],
                day1_baseline=data.get("day1_baseline") or None,
                residuals={k: [float(x) for x in (v or [])]
                           for k, v in (data.get("residuals") or {}).items()},
                regime_days={k: int(v) for k, v in (data.get("regime_days") or {}).items()},
                regime_streaks={k: dict(v or {}) for k, v in (data.get("regime_streaks") or {}).items()},
                regime_signals={k: dict(v or {}) for k, v in (data.get("regime_signals") or {}).items()},
            )
        except Exception:
            return cls()

    def to_notes(self) -> str:
        """Serialize, shrinking history until we fit under NOTES_BUDGET.

        Regime fields (residuals, regime_days, regime_streaks, regime_signals,
        day1_baseline) are kept additively. Residual windows are clipped to
        7 entries (the working window) and price baselines are kept compact.
        If we still overflow, we shrink history first, then drop residual
        history (keeping only signals + day1_baseline), then drop day1
        baseline prices.
        """
        # Trim recent_orders to bounded size before serializing.
        if len(self.recent_orders) > self.MAX_RECENT_ORDERS:
            self.recent_orders = self.recent_orders[-self.MAX_RECENT_ORDERS:]

        # Clip per-axis residual windows to the working window size (7).
        clipped_residuals = {
            k: [round(float(x), 4) for x in (v or [])[-7:]]
            for k, v in (self.residuals or {}).items()
        }
        clipped_streaks = {
            k: {kk: int(vv) if isinstance(vv, (int, float)) else vv
                for kk, vv in (v or {}).items()}
            for k, v in (self.regime_streaks or {}).items()
        }
        clipped_signals = {
            k: dict(v or {})
            for k, v in (self.regime_signals or {}).items()
        }

        text = ""
        # Fallback regime payloads — full → residual-only-signals → minimal.
        regime_levels = [
            {
                "day1_baseline": self.day1_baseline,
                "residuals": clipped_residuals,
                "regime_days": dict(self.regime_days or {}),
                "regime_streaks": clipped_streaks,
                "regime_signals": clipped_signals,
            },
            {
                # drop residual history first — windows can rebuild in ~7 days
                "day1_baseline": self.day1_baseline,
                "regime_days": dict(self.regime_days or {}),
                "regime_streaks": clipped_streaks,
                "regime_signals": clipped_signals,
            },
            {
                # last-ditch: keep only the day1 baseline + final signals
                "day1_baseline": self._compact_day1_baseline(),
                "regime_signals": clipped_signals,
            },
            {},
        ]

        for cap in [self.MAX_HISTORY, 14, 10, 7, 5, 3]:
            for regime_payload in regime_levels:
                payload = {
                    "phase": self.phase,
                    "history": [asdict(r) for r in self.history[-cap:]],
                    "suppliers": {k: asdict(v) for k, v in self.suppliers.items()},
                    "last_promo_day": self.last_promo_day,
                    "last_menu_change_day": self.last_menu_change_day,
                    "last_happy_hour_day": self.last_happy_hour_day,
                    "last_logged_day": self.last_logged_day,
                    "recent_orders": self.recent_orders[-self.MAX_RECENT_ORDERS:],
                    "scenario": self.scenario,
                    "alerts_recent": self.alerts_recent[-self.MAX_ALERTS_RECENT:],
                    "happy_hour_days": self.happy_hour_days[-10:],
                }
                payload.update(regime_payload)
                text = json.dumps(payload, separators=(",", ":"), default=str)
                if len(text) <= self.NOTES_BUDGET:
                    return text
        return text[: self.NOTES_BUDGET]

    def _compact_day1_baseline(self) -> Optional[Dict]:
        """Compact form of day1_baseline that drops the per-ingredient prices."""
        if not self.day1_baseline:
            return None
        return {
            "table_count": self.day1_baseline.get("table_count"),
            "reputation_band": self.day1_baseline.get("reputation_band"),
        }


# ---------- helpers ----------

_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _dow_yesterday(today: str) -> str:
    try:
        i = _DOW.index(today)
    except ValueError:
        return today
    return _DOW[(i - 1) % 7]
