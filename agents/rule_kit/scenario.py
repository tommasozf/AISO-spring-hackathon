"""
Scenario detection — read the ALERT text only.

We classify the current scenario by keyword-matching on `observation.alerts`
plus a short tail of recent alerts we persist in state. The output is a
short tag ("renovation", "supply_crisis", "tourist", "inflation",
"health_scare", or "baseline") that policy/staffing/ordering can gate on.

This is text parsing on the alerts stream — NOT reputation logic. The
teammate owns reputation_band reads / reputation-driven actions; scenarios
fired from alerts are fair game.

Ported from Fabiano's smart_rule.detect_scenario, hardened slightly:
  - Defensive against non-string alerts (some alerts arrive as dicts).
  - Lightweight "tourist surge" override from covers history when alerts
    are silent but covers are growing fast.
"""
from __future__ import annotations

from typing import List, Optional


SCENARIOS = (
    "baseline",
    "renovation",
    "supply_crisis",
    "tourist",
    "inflation",
    "health_scare",
)


_RENOVATION_KEYS = (
    "renovation", "construction", "tables are unavailable",
    "tables unavailable", "fewer table",
)
_SUPPLY_KEYS = (
    "supplier", "outage", "halted", "disruption", "shortage",
    "out of stock", "delivery delay",
)
_TOURIST_KEYS = (
    "tourist", "surge", "festival", "event", "influx", "boom",
)
_INFLATION_KEYS = (
    "inflation", "price increase", "cost rise", "prices rising",
)
_HEALTH_KEYS = (
    "health", "scare", "inspection", "food safety",
)


def _alerts_to_text(alerts) -> str:
    """Concatenate alerts (mixed strings + dicts) into a single lowercase blob."""
    out: List[str] = []
    for a in alerts or []:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            for v in a.values():
                if isinstance(v, str):
                    out.append(v)
        else:
            out.append(str(a))
    return " ".join(out).lower()


def detect_scenario(
    observation: dict,
    state,
    cover_history: Optional[List[int]] = None,
    decisions: Optional[dict] = None,
) -> str:
    """
    Return a scenario tag. `state` is our AgentState (used to read the
    persisted alerts tail and prior scenario as a hysteresis bias).

    `decisions` (optional) is populated with the matched keyword cluster for
    telemetry visibility.
    """
    current = getattr(state, "scenario", "baseline") or "baseline"
    alerts_now = observation.get("alerts") or []
    alerts_tail = getattr(state, "alerts_recent", []) or []
    text = _alerts_to_text(list(alerts_now) + list(alerts_tail))

    matched: Optional[str] = None
    if any(k in text for k in _RENOVATION_KEYS):
        matched = "renovation"
    elif any(k in text for k in _SUPPLY_KEYS):
        matched = "supply_crisis"
    elif any(k in text for k in _TOURIST_KEYS):
        matched = "tourist"
    elif any(k in text for k in _INFLATION_KEYS):
        matched = "inflation"
    elif any(k in text for k in _HEALTH_KEYS):
        matched = "health_scare"

    # Fallback: covers-history surge → tourist (only when no other scenario set).
    if matched is None and not current:
        covers = list(cover_history or [])
        if len(covers) >= 5:
            recent = sum(covers[-3:]) / 3.0
            older_window = covers[-6:-3]
            older = sum(older_window) / max(len(older_window), 1)
            if older > 10 and recent / older > 1.5:
                matched = "tourist"

    if decisions is not None:
        decisions["scenario_matched_keyword"] = matched
        decisions["scenario_alerts_seen"] = (alerts_now[-3:] if alerts_now else [])

    return matched or current or "baseline"
