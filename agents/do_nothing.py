"""Do-nothing agent — submits zero actions every turn.

Establishes the scoring floor. Expected: bankruptcy around day 16, score = -100,000.
"""

from __future__ import annotations

from agents.runner import run_game


def strategy(observation: dict, day: int) -> list[dict]:
    return []


if __name__ == "__main__":
    result = run_game(strategy, team_name="do_nothing", seed=42)
