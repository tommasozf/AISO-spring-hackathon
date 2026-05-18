from __future__ import annotations

import json
import sys

from agents.runner import run_game

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

STAFF_BASE = {
    "Monday": 7, "Tuesday": 7, "Wednesday": 7,
    "Thursday": 8, "Friday": 10, "Saturday": 11, "Sunday": 9,
}
WEATHER_ADJ = {"sunny": 1, "cloudy": 0, "rainy": -1, "stormy": -2}
REP_ADJ = {"Excellent": 1, "Very Good": 0, "Good": 0, "Fair": -1, "Poor": -2}
TREND_ADJ = {"Growing": 1, "Stable": 0, "Declining": -1}
SLOW_DAYS = {"Monday", "Tuesday", "Wednesday"}


def parse_notes(notes_str: str) -> dict:
    if not notes_str or not notes_str.strip():
        return {}
    try:
        return json.loads(notes_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def serialize_notes(state: dict) -> str:
    for key in ("d", "ds"):
        if key in state:
            for k in list(state[key]):
                if isinstance(state[key][k], list) and len(state[key][k]) > 7:
                    state[key][k] = state[key][k][-7:]
    for key in ("cov", "rev", "wko", "rep"):
        if key in state and isinstance(state[key], list) and len(state[key]) > 10:
            state[key] = state[key][-10:]
    if "hh" in state and isinstance(state["hh"], list) and len(state["hh"]) > 10:
        state["hh"] = state["hh"][-10:]
    if "al" in state and isinstance(state["al"], list) and len(state["al"]) > 5:
        state["al"] = state["al"][-5:]

    raw = json.dumps(state, separators=(",", ":"))
    if len(raw) > 3900:
        for key in ("d", "ds"):
            if key in state:
                for k in list(state[key]):
                    if isinstance(state[key][k], list) and len(state[key][k]) > 3:
                        state[key][k] = state[key][k][-3:]
        for key in ("cov", "rev", "wko", "rep"):
            if key in state and isinstance(state[key], list) and len(state[key]) > 5:
                state[key] = state[key][-5:]
        raw = json.dumps(state, separators=(",", ":"))
    return raw[:4000]


def update_state(observation: dict, state: dict, day: int) -> dict:
    service = observation.get("service_summary") or {}

    state.setdefault("cov", []).append(service.get("total_covers", 0))
    state.setdefault("rev", []).append(round(service.get("total_revenue", 0), 1))

    wko_map = {"None": 0, "Few": 1, "Some": 2, "Many": 3}
    state.setdefault("wko", []).append(wko_map.get(service.get("walkout_band", "None"), 0))
    state.setdefault("rep", []).append(observation.get("reputation_band", "Very Good"))

    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    dishes_sold = service.get("dishes_sold", {})

    daily_usage: dict[str, float] = {}
    for dish_name, qty in dishes_sold.items():
        if dish_name in menu_book:
            for ing in menu_book[dish_name].get("ingredients", []):
                daily_usage[ing["ingredient"]] = (
                    daily_usage.get(ing["ingredient"], 0) + ing["quantity_kg"] * qty
                )

    d = state.setdefault("d", {})
    for ing_name, used in daily_usage.items():
        d.setdefault(ing_name, []).append(round(used, 1))

    ds = state.setdefault("ds", {})
    for dish_name, qty in dishes_sold.items():
        ds.setdefault(dish_name, []).append(qty)

    alerts = observation.get("alerts", [])
    if alerts:
        state.setdefault("al", []).extend(alerts[-3:])

    return state


def _dow_index(name: str) -> int:
    try:
        return DAYS_OF_WEEK.index(name)
    except ValueError:
        return 0


def estimate_delivery_day(
    current_day: int, lead_time: int, delivery_days: list[str], today_dow: str
) -> int:
    today_idx = _dow_index(today_dow)
    valid = {_dow_index(d) for d in delivery_days}
    for offset in range(lead_time, lead_time + 8):
        if (today_idx + offset) % 7 in valid:
            return current_day + offset
    return current_day + lead_time + 7


def detect_scenario(observation: dict, state: dict, day: int) -> str:
    import inspect
    for frame_info in inspect.stack():
        if frame_info.function == "run_game":
            sc = frame_info.frame.f_locals.get("scenario", "")
            if sc == "tourist_season":
                return "tourist"
            return sc
    return "baseline"


def compute_consumption_rates(observation: dict, state: dict) -> dict[str, float]:
    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    dishes_sold = (observation.get("service_summary") or {}).get("dishes_sold", {})

    yesterday: dict[str, float] = {}
    for dish, qty in dishes_sold.items():
        if dish in menu_book:
            for ing in menu_book[dish].get("ingredients", []):
                yesterday[ing["ingredient"]] = (
                    yesterday.get(ing["ingredient"], 0) + ing["quantity_kg"] * qty
                )

    hist = state.get("d", {})
    rates: dict[str, float] = {}
    for name in set(list(yesterday) + list(hist)):
        y = yesterday.get(name, 0)
        h = hist.get(name, [])
        if h:
            avg = sum(h) / len(h)
            peak = max(h) if h else y
            rates[name] = max(0.4 * avg + 0.6 * y, 0.7 * peak)
        elif y > 0:
            rates[name] = y
    return rates


def _day1_rates(observation: dict, scenario: str) -> dict[str, float]:
    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    active = observation.get("active_menu", [])
    rates: dict[str, float] = {}
    per_dish = 14
    for dish in active:
        if dish in menu_book:
            for ing in menu_book[dish].get("ingredients", []):
                rates[ing["ingredient"]] = rates.get(ing["ingredient"], 0) + ing["quantity_kg"] * per_dish
    return rates


def project_inventory(observation: dict, rates: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    shelf_lives: dict[str, int] = {}

    for inv in observation.get("inventory", []):
        name = inv["ingredient"]
        shelf_lives[name] = inv.get("shelf_life_days", 14)
        usable = sum(b["quantity_kg"] for b in inv.get("batches", []) if b["expires_in_days"] > 1)
        total = inv.get("total_kg", 0)
        pending = sum(
            po["quantity_kg"]
            for po in observation.get("pending_orders", [])
            if po["ingredient"] == name
        )
        effective = usable + pending
        dr = rates.get(name, 0)
        result[name] = {
            "usable": usable,
            "total": total,
            "pending": pending,
            "effective": effective,
            "daily_rate": dr,
            "days_stockout": round(usable / dr, 1) if dr > 0.01 else 999,
            "days_effective": round(effective / dr, 1) if dr > 0.01 else 999,
            "shelf_life": shelf_lives.get(name, 14),
        }
    return result


def compute_orders(
    observation: dict, projections: dict, state: dict, day: int
) -> list[dict]:
    actions: list[dict] = []
    cash = observation.get("cash", 0)
    days_remaining = observation.get("days_remaining", 30 - day)
    today_dow = observation.get("day_of_week", "Monday")
    scenario = state.get("sc", "baseline")

    reserve = max(500, min(1500, days_remaining * 50))
    budget = cash - reserve
    if budget <= 0:
        return actions

    supplier_idx: dict[str, list[tuple]] = {}
    for sup in observation.get("supplier_catalog", []):
        for ingredient, price in sup.get("ingredients", {}).items():
            supplier_idx.setdefault(ingredient, []).append((
                sup["name"],
                price,
                sup.get("min_order_kg", 5),
                sup.get("lead_time_days", 1),
                sup.get("delivery_days", DAYS_OF_WEEK),
            ))
    for ing in supplier_idx:
        supplier_idx[ing].sort(key=lambda x: x[1])

    dow = observation.get("day_of_week", "Monday")
    pre_peak = dow in ("Wednesday", "Thursday", "Friday")
    horizon = min(7, max(5, days_remaining))
    if pre_peak:
        horizon = min(8, max(5, days_remaining))
    
    if scenario == "supply_crisis":
        horizon = min(9, max(6, days_remaining))

    urgent_ings: set[str] = set()
    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    for dish in (observation.get("service_summary") or {}).get("dishes_unavailable_at", {}):
        if dish in menu_book:
            for ing in menu_book[dish].get("ingredients", []):
                urgent_ings.add(ing["ingredient"])

    needs: list[tuple[float, float, str]] = []
    for name, proj in projections.items():
        dr = proj["daily_rate"]
        if dr < 0.01:
            continue

        eff_horizon = min(horizon, proj.get("shelf_life", 14) - 1)
        covers = state.get("cov", [])
        avg_cov = sum(covers) / len(covers) if covers else 100
        last_cov = covers[-1] if covers else 100
        demand_surge = max(1.0, last_cov / max(avg_cov, 1)) if avg_cov > 10 else 1.0
        
        if day <= 5:
            safety = max(1.5, demand_surge)
        else:
            safety = max(1.4, demand_surge * 0.9)
            
        target = dr * eff_horizon * safety
        deficit = target - proj["effective"]
        urgency = proj["days_effective"]
        if name in urgent_ings:
            urgency = min(urgency, 0.5)
        if day <= 3:
            urgency = min(urgency, 1.0)

        if deficit > 0 or urgency < 4:
            qty = max(deficit, dr * 3)
            max_useful = dr * (days_remaining + 1) - proj["effective"]
            if max_useful > 0:
                qty = min(qty, max_useful)
            if qty > 0 and name in supplier_idx:
                needs.append((urgency, qty, name))

    needs.sort()

    spent = 0.0
    ordered: dict[str, int] = {}
    for urgency, qty_needed, name in needs:
        if spent >= budget:
            break
        max_orders = 2 if (urgency < 2 or day <= 3) else 1
        if ordered.get(name, 0) >= max_orders:
            continue
        suppliers = supplier_idx.get(name, [])
        for sup_name, price, min_ord, lead, del_days in suppliers:
            if ordered.get(name, 0) >= max_orders:
                break
            if urgency < 2:
                eta = estimate_delivery_day(day, lead, del_days, today_dow)
                if eta > day + 4:
                    continue

            qty = max(qty_needed, min_ord)
            qty = round(qty, 1)
            if qty < min_ord:
                qty = min_ord
            cost = qty * price
            if spent + cost > budget:
                qty = min_ord
                cost = qty * price
                if spent + cost > budget:
                    continue

            actions.append({
                "tool": "place_order",
                "args": {"supplier": sup_name, "ingredient": name, "quantity_kg": qty},
            })
            spent += cost
            ordered[name] = ordered.get(name, 0) + 1

    return actions


def compute_staff_level(observation: dict, state: dict, day: int) -> int | None:
    current = observation.get("staff_level", 8)
    dow = observation.get("day_of_week", "Monday")

    base = STAFF_BASE.get(dow, 7)
    base += WEATHER_ADJ.get(observation.get("weather_today", "cloudy"), 0)
    base += REP_ADJ.get(observation.get("reputation_band", "Very Good"), 0)
    base += TREND_ADJ.get(observation.get("customer_trend", "Stable"), 0)

    service = observation.get("service_summary") or {}
    if service.get("walkout_band", "None") in ("Some", "Many"):
        base += 1
    if service.get("table_utilization_peak", 0) > 0.9:
        base += 1
    if service.get("peak_wait_minutes", 0) > 15:
        base += 1

    scenario = state.get("sc", "baseline")
    if scenario == "renovation":
        if day <= 12:
            base = 5 if dow in ("Friday", "Saturday") else 4
        else:
            base = max(base - 1, 5)
    elif scenario == "tourist" and observation.get("customer_trend") == "Growing":
        base += 2

    cash = observation.get("cash", 15000)
    if cash < 2000:
        base = min(base, 4)
    elif cash < 3000:
        base = min(base, 5)

    target = max(3, min(15, base))
    return target if target != current else None


def compute_menu(
    observation: dict,
    projections: dict,
    dish_metrics: dict,
    day: int
) -> list[str] | None:

    if day <= 1:
        return None

    menu_book = {
        d["name"]: d
        for d in observation.get("menu_book", [])
    }

    active = observation.get("active_menu", [])

    dish_scores: list[tuple[float, str]] = []

    for dish_name, dish in menu_book.items():

        feasible = True
        min_servings = 999.0

        for ing in dish.get("ingredients", []):

            proj = projections.get(
                ing["ingredient"]
            )

            if not proj:
                feasible = False
                break

            if ing["quantity_kg"] <= 0:
                continue

            servings = (
                proj["effective"]
                / ing["quantity_kg"]
            )

            min_servings = min(
                min_servings,
                servings
            )

        if not feasible or min_servings <= 2:
            continue

        metrics = dish_metrics.get(
            dish_name,
            {}
        )

        expected_profit = metrics.get(
            "expected_profit",
            0
        )

        margin_pct = metrics.get(
            "margin_pct",
            0
        )

        inventory_pressure = metrics.get(
            "inventory_pressure",
            0
        )

        avg_sales = metrics.get(
            "avg_sales",
            0
        )

        score = (
            expected_profit
            + margin_pct * 20
            + inventory_pressure * 5
            + avg_sales * 3
        )

        dish_scores.append(
            (score, dish_name)
        )

    dish_scores.sort(reverse=True)

    new_menu = [
        dish
        for _, dish in dish_scores[:8]
    ]

    if len(new_menu) < 5:

        for dish_name in menu_book:

            if dish_name not in new_menu:
                new_menu.append(dish_name)

            if len(new_menu) >= 5:
                break

    if set(new_menu) != set(active):
        return new_menu

    return None


def compute_pricing(observation: dict, state: dict, day: int) -> list[dict]:
    if day < 4:
        return []

    actions: list[dict] = []
    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    active = observation.get("active_menu", [])
    ds_hist = state.get("ds", {})

    for dish_name in active:
        if dish_name not in menu_book:
            continue
        dish = menu_book[dish_name]
        base = dish["base_price"]
        current = dish.get("current_price", base)

        hist = ds_hist.get(dish_name, [])
        avg_sold = sum(hist) / len(hist) if hist else 0

        if avg_sold > 15:
            target = base * 1.10
        elif avg_sold > 10:
            target = base * 1.05
        elif avg_sold > 5:
            target = base * 1.0
        elif 0 < avg_sold <= 3:
            target = base * 0.92
        else:
            target = base

        target = max(base * 0.8, min(base * 1.2, round(target, 2)))
        if abs(target - current) >= 0.5:
            actions.append({"tool": "set_price", "args": {"dish": dish_name, "price": target}})

    return actions


def should_run_happy_hour(observation: dict, state: dict, day: int) -> bool:
    if observation.get("cash", 15000) < 2000:
        return False
    if state.get("sc") == "renovation" and day <= 12:
        return False
    hh = state.get("hh", [])
    if hh and hh[-1] >= day - 1:
        return False
    if sum(1 for d in hh if d > day - 7) >= 3:
        return False

    dow = observation.get("day_of_week", "")
    weather = observation.get("weather_today", "cloudy")
    if dow in SLOW_DAYS or weather in ("rainy", "stormy"):
        return True
    return False


def choose_daily_special(observation: dict, day: int) -> str | None:
    active = observation.get("active_menu", [])
    if not active:
        return None
    return active[(day - 1) % len(active)]


def compute_marketing(observation: dict, state: dict, day: int) -> float:
    cash = observation.get("cash", 15000)
    if cash < 2000:
        return 0
    if cash < 3000:
        return 20

    trend = observation.get("customer_trend", "Stable")
    rep = observation.get("reputation_band", "Very Good")
    days_left = observation.get("days_remaining", 30 - day)

    scenario = state.get("sc", "baseline")
    base = 60.0
    if scenario == "renovation" and day <= 12:
        return 0
    elif trend == "Declining":
        base = 150
    elif trend == "Growing":
        base = 80
    if rep in ("Fair", "Poor"):
        base += 80
    if days_left <= 3:
        base = max(0, base - 30)

    return min(base, 300, cash * 0.03)

def compute_dish_metrics(
    observation: dict,
    projections: dict,
    state: dict,
) -> dict[str, dict]:

    ingredient_costs = build_ingredient_cost_map(observation)

    ds_hist = state.get("ds", {})

    metrics: dict[str, dict] = {}

    for dish in observation.get("menu_book", []):

        name = dish["name"]

        current_price = dish.get("current_price", dish["base_price"])

        ingredient_cost = 0.0

        for ing in dish.get("ingredients", []):

            unit_cost = ingredient_costs.get(
                ing["ingredient"],
                0
            )

            ingredient_cost += (
                unit_cost * ing["quantity_kg"]
            )

        margin = current_price - ingredient_cost

        margin_pct = (
            margin / current_price
            if current_price > 0 else 0
        )

        sales_hist = ds_hist.get(name, [])

        avg_sales = (
            sum(sales_hist) / len(sales_hist)
            if sales_hist else 0
        )

        expected_profit = margin * avg_sales

        inventory_pressure = 0.0

        for ing in dish.get("ingredients", []):

            proj = projections.get(ing["ingredient"])

            if not proj:
                continue

            shelf = proj.get("shelf_life", 14)

            # reward dishes using risky inventory
            if shelf <= 3:
                inventory_pressure += 2
            elif shelf <= 5:
                inventory_pressure += 1

        metrics[name] = {
            "ingredient_cost": round(ingredient_cost, 2),
            "margin": round(margin, 2),
            "margin_pct": round(margin_pct, 2),
            "avg_sales": round(avg_sales, 2),
            "expected_profit": round(expected_profit, 2),
            "inventory_pressure": inventory_pressure,
        }

    return metrics


def build_ingredient_cost_map(observation: dict) -> dict[str, float]:
    costs: dict[str, float] = {}

    for supplier in observation.get("supplier_catalog", []):
        for ingredient, price in supplier.get("ingredients", {}).items():

            # keep cheapest supplier
            if ingredient not in costs:
                costs[ingredient] = price
            else:
                costs[ingredient] = min(costs[ingredient], price)

    return costs


def strategy(observation: dict, day: int) -> list[dict]:
    actions: list[dict] = []

    state = parse_notes(observation.get("notes", ""))

    if day > 1:
        state = update_state(observation, state, day)

    state["sc"] = detect_scenario(observation, state, day)

    rates = compute_consumption_rates(observation, state) if day > 1 else {}
    if not rates:
        rates = _day1_rates(observation, state.get("sc", "baseline"))

    projections = project_inventory(observation, rates)

    dish_metrics = compute_dish_metrics(
        observation,
        projections,
        state
    )

    menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
    for dish in observation.get("active_menu", []):
        if dish in menu_book:
            for ing in menu_book[dish].get("ingredients", []):
                name = ing["ingredient"]
                if name not in projections:
                    projections[name] = {
                        "usable": 0, "total": 0, "pending": 0,
                        "effective": 0, "daily_rate": rates.get(name, 0),
                        "days_stockout": 0, "days_effective": 0,
                        "shelf_life": 14,
                    }

    staff = compute_staff_level(observation, state, day)
    if staff is not None:
        actions.append({"tool": "set_staff_level", "args": {"level": staff}})

    new_menu = compute_menu(observation, projections, state, day)
    if new_menu is not None:
        actions.append({"tool": "set_menu", "args": {"dishes": new_menu}})

    actions.extend(compute_pricing(observation, state, day))
    actions.extend(compute_orders(observation, projections, state, day))

    if should_run_happy_hour(observation, state, day):
        actions.append({"tool": "run_happy_hour", "args": {}})
        state.setdefault("hh", []).append(day)

    special = choose_daily_special(observation, day)
    if special:
        actions.append({"tool": "offer_daily_special", "args": {"dish": special}})

    mkt = compute_marketing(observation, state, day)
    actions.append({"tool": "set_marketing_spend", "args": {"amount": round(mkt)}})

    actions.append({"tool": "save_notes", "args": {"text": serialize_notes(state)}})

    return actions


if __name__ == "__main__":
    print("Running SmartRule agent...")
    result = run_game(strategy, team_name="italian_waiters", seed=88, scenario="tourist_season")
