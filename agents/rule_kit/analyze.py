"""
Read & summarize telemetry JSONL produced by `telemetry.record_turn`.

Usage:
    # list all recorded runs in a directory
    python -m agents.rule_kit.analyze --list runs/

    # per-day table for one run
    python -m agents.rule_kit.analyze runs/myteam_20260518_141322_t12345.jsonl

    # listing every stockout event in a run
    python -m agents.rule_kit.analyze --stockouts runs/<file>.jsonl

    # action counts per day
    python -m agents.rule_kit.analyze --actions runs/<file>.jsonl

    # inventory + days-of-cover history for one ingredient
    python -m agents.rule_kit.analyze --inventory Chicken runs/<file>.jsonl

    # rejections + safety filter drops per day
    python -m agents.rule_kit.analyze --decisions runs/<file>.jsonl

    # aggregate across many runs (e.g. evaluate output)
    python -m agents.rule_kit.analyze --aggregate runs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List


# ---------- loading ----------

def load(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- views ----------

def view_summary(rows: List[Dict[str, Any]]) -> None:
    """Per-day one-line summary."""
    print(
        f"{'Day':>3} {'DoW':<4} {'Cash':>8} {'Rev':>7} "
        f"{'Cov':>4} {'Walk':<5} {'Rep':<10} {'Trend':<10} "
        f"{'Weath':<7} {'Staff':>5} {'Stockouts':<30}"
    )
    print("-" * 100)
    for r in rows:
        s = r["summary"]
        svc = s.get("service_summary") or {}
        stockouts = svc.get("dishes_unavailable_at") or {}
        stockout_str = ",".join(f"{k}@h{v}" for k, v in stockouts.items())[:30]
        dow = (s.get("day_of_week") or "?")[:3]
        weather = (s.get("weather_today") or "?")[:6]
        covers = svc.get("total_covers")
        print(
            f"{r['day']:>3} {dow:<4} "
            f"{(s.get('cash') or 0):>8.0f} "
            f"{(s.get('yesterday_revenue') or 0):>7.0f} "
            f"{(covers if covers is not None else 0):>4} "
            f"{(svc.get('walkout_band') or '-'):<5} "
            f"{(s.get('reputation_band') or '-'):<10} "
            f"{(s.get('customer_trend') or '-'):<10} "
            f"{weather:<7} "
            f"{(s.get('staff_level') or 0):>5} "
            f"{stockout_str:<30}"
        )


def view_stockouts(rows: List[Dict[str, Any]]) -> None:
    """List every recorded stockout."""
    found = False
    for r in rows:
        svc = (r["summary"].get("service_summary") or {})
        so = svc.get("dishes_unavailable_at") or {}
        if so:
            found = True
            dow = r["summary"].get("day_of_week")
            print(f"Day {r['day']:>2} ({dow}): {so}")
    if not found:
        print("No stockouts recorded.")


def view_actions(rows: List[Dict[str, Any]]) -> None:
    """Per-day action-tool counts."""
    print(f"{'Day':>3} {'Total':>5} {'Orders':>6} {'Menu':>4} {'Staff':>5} {'Price':>5} "
          f"{'Mkt':>4} {'HH':>3} {'Special':>7} {'Notes':>5}")
    print("-" * 70)
    for r in rows:
        actions = r.get("actions") or []
        counts: Dict[str, int] = {}
        for a in actions:
            counts[a.get("tool")] = counts.get(a.get("tool"), 0) + 1
        print(
            f"{r['day']:>3} {len(actions):>5} "
            f"{counts.get('place_order', 0):>6} "
            f"{counts.get('set_menu', 0):>4} "
            f"{counts.get('set_staff_level', 0):>5} "
            f"{counts.get('set_price', 0):>5} "
            f"{counts.get('set_marketing_spend', 0):>4} "
            f"{counts.get('run_happy_hour', 0):>3} "
            f"{counts.get('offer_daily_special', 0):>7} "
            f"{counts.get('save_notes', 0):>5}"
        )


def view_inventory(rows: List[Dict[str, Any]], ingredient: str) -> None:
    """Stock history for one ingredient over the run."""
    print(f"Ingredient: {ingredient}\n")
    print(f"{'Day':>3} {'DoW':<4} {'on_hand':>8} {'pending':>8} {'soonest_exp':>11} "
          f"{'days_cover':>11} {'ordered_today':>14}")
    print("-" * 75)
    for r in rows:
        s = r["summary"]
        inv = next(
            (i for i in s.get("inventory_summary", []) if i["ingredient"] == ingredient),
            None,
        )
        pending = sum(
            (po.get("quantity_kg") or 0)
            for po in (s.get("pending_orders") or [])
            if po.get("ingredient") == ingredient
        )
        dec = r.get("decisions") or {}
        cover = (dec.get("days_of_cover") or {}).get(ingredient)
        on_hand = (inv or {}).get("total_kg")
        soonest = (inv or {}).get("soonest_expiry_days")
        # find orders placed today for this ingredient
        ordered_today = sum(
            (a.get("args", {}).get("quantity_kg") or 0)
            for a in (r.get("actions") or [])
            if a.get("tool") == "place_order"
            and a.get("args", {}).get("ingredient") == ingredient
        )
        dow = (s.get("day_of_week") or "?")[:3]
        print(
            f"{r['day']:>3} {dow:<4} "
            f"{(on_hand if on_hand is not None else 0):>8.1f} "
            f"{pending:>8.1f} "
            f"{(soonest if soonest is not None else -1):>11} "
            f"{(cover if cover is not None else 0):>11.2f} "
            f"{ordered_today:>14.1f}"
        )


def view_decisions(rows: List[Dict[str, Any]]) -> None:
    """Per-day forecast + key decision rationale."""
    print(f"{'Day':>3} {'DoW':<4} {'Phase':<8} {'Fcst':>6} {'Fcst+1':>7} {'StaffTgt':>8} "
          f"{'Mkt':>5} {'HH':<4} {'Special':<18}")
    print("-" * 80)
    for r in rows:
        d = r.get("decisions") or {}
        s = r["summary"]
        fcst = d.get("forecast") or []
        f0 = fcst[0] if fcst else None
        f1 = fcst[1] if len(fcst) > 1 else None
        dow = (s.get("day_of_week") or "?")[:3]
        print(
            f"{r['day']:>3} {dow:<4} "
            f"{(d.get('phase') or '-'):<8} "
            f"{(f0 if f0 is not None else 0):>6.0f} "
            f"{(f1 if f1 is not None else 0):>7.0f} "
            f"{(d.get('staff_target') or 0):>8} "
            f"{(d.get('marketing') or 0):>5.0f} "
            f"{('Y' if d.get('happy_hour') else 'N'):<4} "
            f"{(d.get('daily_special') or '-'):<18}"
        )


def view_list(directory: str) -> None:
    """List runs in a directory with brief metadata."""
    paths = sorted(glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        print(f"No .jsonl files in {directory}")
        return
    print(f"{'File':<50} {'Turns':>5} {'LastDay':>7} {'FinalCash':>10} {'Walk':<5} {'Rep':<10}")
    print("-" * 95)
    for p in paths:
        try:
            rows = load(p)
        except Exception as e:
            print(f"{os.path.basename(p):<50} ERROR: {e}")
            continue
        if not rows:
            print(f"{os.path.basename(p):<50} (empty)")
            continue
        last = rows[-1]
        s = last["summary"]
        svc = s.get("service_summary") or {}
        print(
            f"{os.path.basename(p):<50} "
            f"{len(rows):>5} "
            f"{last['day']:>7} "
            f"{(s.get('cash') or 0):>10.0f} "
            f"{(svc.get('walkout_band') or '-'):<5} "
            f"{(s.get('reputation_band') or '-'):<10}"
        )


def view_aggregate(directory: str) -> None:
    """Counts across many runs in a directory."""
    paths = sorted(glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        print(f"No .jsonl files in {directory}")
        return

    total_stockout_days = 0
    stockouts_by_ingredient: Dict[str, int] = {}
    stockouts_by_dow: Dict[str, int] = {}
    zero_cover_days_by_dow: Dict[str, int] = {}
    walkout_counts: Dict[str, int] = {}
    runs = 0

    for p in paths:
        try:
            rows = load(p)
        except Exception:
            continue
        if not rows:
            continue
        runs += 1
        for r in rows:
            s = r["summary"]
            svc = s.get("service_summary") or {}
            so = svc.get("dishes_unavailable_at") or {}
            dow = s.get("day_of_week") or "?"
            if so:
                total_stockout_days += 1
                stockouts_by_dow[dow] = stockouts_by_dow.get(dow, 0) + 1
                for ing in so:
                    stockouts_by_ingredient[ing] = stockouts_by_ingredient.get(ing, 0) + 1
            if svc.get("total_covers") == 0:
                zero_cover_days_by_dow[dow] = zero_cover_days_by_dow.get(dow, 0) + 1
            wb = svc.get("walkout_band") or "-"
            walkout_counts[wb] = walkout_counts.get(wb, 0) + 1

    print(f"Runs analyzed: {runs}")
    print(f"Total stockout-days: {total_stockout_days}")
    print(f"\nStockouts by day-of-week:")
    for k, v in sorted(stockouts_by_dow.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v}")
    print(f"\nStockouts by ingredient:")
    for k, v in sorted(stockouts_by_ingredient.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v}")
    print(f"\nZero-cover days by day-of-week:")
    for k, v in sorted(zero_cover_days_by_dow.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v}")
    print(f"\nWalkout band distribution:")
    for k, v in sorted(walkout_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v}")


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="JSONL file or directory")
    p.add_argument("--list", action="store_true", help="List runs in a dir")
    p.add_argument("--stockouts", action="store_true", help="List stockout events")
    p.add_argument("--actions", action="store_true", help="Action counts per day")
    p.add_argument("--decisions", action="store_true", help="Forecast + decisions per day")
    p.add_argument("--inventory", help="Track one ingredient's stock over time")
    p.add_argument("--aggregate", action="store_true", help="Aggregate across all runs in dir")
    args = p.parse_args()

    if args.list:
        view_list(args.path)
        return
    if args.aggregate:
        view_aggregate(args.path)
        return

    if not os.path.isfile(args.path):
        print(f"Not a file: {args.path}")
        sys.exit(1)

    rows = load(args.path)
    if not rows:
        print("Empty file.")
        return

    if args.stockouts:
        view_stockouts(rows)
    elif args.actions:
        view_actions(rows)
    elif args.decisions:
        view_decisions(rows)
    elif args.inventory:
        view_inventory(rows, args.inventory)
    else:
        view_summary(rows)


if __name__ == "__main__":
    main()
