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
    # add other rolling stats here as you grow the agent

    MAX_HISTORY: int = 21
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
                )
            )
            self.history = self.history[-self.MAX_HISTORY:]
            self.last_logged_day = yesterday

        # supplier reliability EWMA
        for d in obs.get("delivery_history", []) or []:
            self._record_delivery(d)

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
            history = [DayRecord(**r) for r in data.get("history", [])]
            suppliers = {k: SupplierStat(**v) for k, v in data.get("suppliers", {}).items()}
            return cls(
                phase=data.get("phase", "build"),
                history=history,
                suppliers=suppliers,
                last_promo_day=data.get("last_promo_day", -10),
                last_menu_change_day=data.get("last_menu_change_day", -10),
                last_happy_hour_day=data.get("last_happy_hour_day", -10),
                last_logged_day=data.get("last_logged_day", 0),
            )
        except Exception:
            return cls()

    def to_notes(self) -> str:
        """Serialize, shrinking history until we fit under NOTES_BUDGET."""
        text = ""
        for cap in [self.MAX_HISTORY, 14, 10, 7, 5, 3]:
            payload = {
                "phase": self.phase,
                "history": [asdict(r) for r in self.history[-cap:]],
                "suppliers": {k: asdict(v) for k, v in self.suppliers.items()},
                "last_promo_day": self.last_promo_day,
                "last_menu_change_day": self.last_menu_change_day,
                "last_happy_hour_day": self.last_happy_hour_day,
                "last_logged_day": self.last_logged_day,
            }
            text = json.dumps(payload, separators=(",", ":"))
            if len(text) <= self.NOTES_BUDGET:
                return text
        return text[: self.NOTES_BUDGET]


# ---------- helpers ----------

_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _dow_yesterday(today: str) -> str:
    try:
        i = _DOW.index(today)
    except ValueError:
        return today
    return _DOW[(i - 1) % 7]
