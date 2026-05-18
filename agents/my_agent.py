"""
RestBench rule-based agent.

Architecture:
  Tier 1 (safety) — hard invariants (cash floor, valid ranges, endgame freeze)
  Tier 2 (policy) — quantitative optimizers (forecast, newsvendor reorder, staffing, promos)
  Tier 3 (memory) — state persisted via save_notes (4000-char JSON blob)

No LLM — just rules + math.

Run a single game:
    python -m agents.my_agent

Evaluate across scenarios and seeds:
    python -m agents.evaluate agents.my_agent
    python -m agents.evaluate agents.my_agent --scenarios baseline,supply_crisis --seeds 42,88,123
"""
from __future__ import annotations

import os

from agents.runner import run_game
from agents.rule_kit.policy import decide_actions
from agents.rule_kit.state import AgentState

# CHANGE THIS to your real team name (or set TEAM_NAME env var).
# The leaderboard groups by team name — keep it consistent across every game.
DEFAULT_TEAM_NAME = "CHANGE_ME"


def strategy(observation: dict, day: int) -> list[dict]:
    """
    Called once per simulated day by the runner. Stateless: state must be
    reloaded from observation["notes"] each turn, then re-serialized back via
    save_notes at the end of the action list.
    """
    # 1) Reload memory from the notes field (server echoes it back each turn)
    state = AgentState.from_notes(observation.get("notes", "") or "")

    # 2) Roll yesterday's service_summary into history (idempotent)
    state.update_from_observation(observation)

    # 3) Decide actions for today
    actions = decide_actions(observation, day, state)

    # 4) Always overwrite notes last so it captures the latest state.
    actions = [a for a in actions if a.get("tool") != "save_notes"]
    actions.append({"tool": "save_notes", "args": {"text": state.to_notes()}})

    return actions


if __name__ == "__main__":
    team_name = os.environ.get("TEAM_NAME", DEFAULT_TEAM_NAME)
    if team_name == "CHANGE_ME":
        raise SystemExit(
            "Set TEAM_NAME env var (or edit DEFAULT_TEAM_NAME in agents/my_agent.py).\n"
            "  PowerShell: $env:TEAM_NAME = 'your_team_name'\n"
            "  cmd:        set TEAM_NAME=your_team_name"
        )
    seed = int(os.environ.get("SEED", "42"))
    scenario = os.environ.get("SCENARIO", "baseline")
    run_game(strategy, team_name=team_name, seed=seed, scenario=scenario)
