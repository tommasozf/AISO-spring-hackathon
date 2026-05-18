"""
Per-turn telemetry.

Each strategy() call appends one JSON line to a run file. One file per game
(per thread, so `evaluate.py`'s parallel games each get their own log).

Files: runs/{team}_{timestamp}_t{thread_id}.jsonl
Rotated on `day == 1` (new game starting in this thread).

Override the run dir with the RUN_DIR env var.

Read the logs with `agents.rule_kit.analyze`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List


RUN_DIR = os.environ.get("RUN_DIR", "runs")
_LOCAL = threading.local()


def _ensure_dir() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)


def _new_path(team: str, thread_id: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_team = "".join(c if c.isalnum() or c in "_-" else "_" for c in team)
    return os.path.join(RUN_DIR, f"{safe_team}_{ts}_t{thread_id}.jsonl")


def record_turn(
    observation: Dict[str, Any],
    actions: List[Dict[str, Any]],
    decisions: Dict[str, Any],
    team: str = "agent",
) -> str:
    """
    Append one compact turn record. Returns the path written to.
    Swallows nothing — caller can try/except if it wants to ignore failures.
    """
    day = int(observation.get("day", 0))
    thread_id = threading.get_ident()

    # rotate on new game
    if day <= 1 or not getattr(_LOCAL, "path", None):
        _ensure_dir()
        _LOCAL.path = _new_path(team, thread_id)

    line = {
        "day": day,
        "ts": time.time(),
        "summary": _summarize_observation(observation),
        "decisions": decisions,
        "actions": actions,
    }
    with open(_LOCAL.path, "a") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return _LOCAL.path


def _summarize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary — strip the big repetitive fields (menu_book, supplier_catalog)."""
    inv = []
    for i in obs.get("inventory") or []:
        batches = i.get("batches") or []
        soonest = min((b.get("expires_in_days", 999) for b in batches), default=None)
        inv.append({
            "ingredient": i.get("ingredient"),
            "total_kg": i.get("total_kg"),
            "shelf_life_days": i.get("shelf_life_days"),
            "soonest_expiry_days": soonest,
            "n_batches": len(batches),
        })

    return {
        "day_of_week": obs.get("day_of_week"),
        "days_remaining": obs.get("days_remaining"),
        "cash": obs.get("cash"),
        "yesterday_revenue": obs.get("yesterday_revenue"),
        "yesterday_total_costs": obs.get("yesterday_total_costs"),
        "cost_breakdown": obs.get("cost_breakdown"),
        "reputation_band": obs.get("reputation_band"),
        "customer_trend": obs.get("customer_trend"),
        "weather_today": obs.get("weather_today"),
        "weather_forecast": obs.get("weather_forecast"),
        "alerts": obs.get("alerts"),
        "active_menu": obs.get("active_menu"),
        "staff_level": obs.get("staff_level"),
        "service_summary": obs.get("service_summary"),
        "inventory_summary": inv,
        "pending_orders": obs.get("pending_orders"),
        "delivery_history_tail": (obs.get("delivery_history") or [])[-5:],
    }
