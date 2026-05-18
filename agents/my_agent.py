"""Mixed Rule-Based and LLM Agent

This agent combines deterministic rules for survival (inventory, staffing)
with an LLM for strategic decisions (pricing, marketing, handling crises).
"""

from __future__ import annotations

import json
import os
import sys

import litellm

from agents.runner import run_game

# Use gpt-4o-mini by default unless overridden
MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = """\
You are the strategic manager of an Italian restaurant. You have 30 days to maximize profit and reputation.
A rule-based system already handles basic inventory reordering and baseline staffing.
Your job is to read the daily summary and make strategic decisions by returning a JSON array of tool calls.

Available tools for you:
- set_price: {"tool": "set_price", "args": {"dish": "...", "price": N}} (Range: 0.8x to 1.2x of base price)
- set_marketing_spend: {"tool": "set_marketing_spend", "args": {"amount": N}} (Range: 0-500 EUR. Boosts demand.)
- run_happy_hour: {"tool": "run_happy_hour", "args": {}} (Boosts demand, lowers prices. Diminishing returns if used consecutively.)
- offer_daily_special: {"tool": "offer_daily_special", "args": {"dish": "..."}} (Must be an active dish. Slight satisfaction boost.)
- set_menu: {"tool": "set_menu", "args": {"dishes": [...]}} (Min 5 dishes. Change if ingredients are consistently unavailable or a supplier is down.)
- save_notes: {"tool": "save_notes", "args": {"text": "..."}} (Leave notes for yourself for tomorrow. Max 4000 chars.)

Guidelines:
1. Pay close attention to "alerts". If a supplier is down, consider removing dishes that rely on them. DO NOT change the menu otherwise. Changing the menu ruins kitchen efficiency.
2. If "dishes_unavailable_at" shows dishes ran out, it means we lost sales. The rule-based system is trying to reorder, but you might want to raise prices on those dishes to slow demand, or remove them temporarily.
3. If reputation is dropping, consider running a happy hour or a daily special to recover.
4. If walkout_band is 'Some' or 'Many', your prices are too low! You should gradually raise prices on active dishes (by 0.50 to 1.50 EUR) to gently cool demand. Do NOT jump straight to the maximum allowed limit, as this causes demand to drop to zero! If walkouts are 'None', you can safely lower prices slightly to build reputation.
5. Respond ONLY with a JSON array of tool calls. No markdown block formatting, no explanation. Just the array.
"""

def summarize_observation(obs: dict) -> dict:
    """Summarize the raw observation into a cleaner format for the LLM."""
    active_menu_details = {}
    menu_book = {dish["name"]: dish for dish in obs.get("menu_book", [])}
    for dish_name in obs.get("active_menu", []):
        if dish_name in menu_book:
            base_price = menu_book[dish_name]["base_price"]
            active_menu_details[dish_name] = {
                "current_price": menu_book[dish_name]["current_price"],
                "allowed_range": [round(base_price * 0.81, 2), round(base_price * 1.19, 2)]
            }

    summary = {
        "day_info": f"Day {obs['day']}/30 ({obs['day_of_week']})",
        "financials": {
            "cash": obs["cash"],
            "yesterday_revenue": obs["yesterday_revenue"],
            "yesterday_costs": obs["yesterday_total_costs"]
        },
        "service_summary": obs.get("service_summary", {}),
        "reputation": f"{obs.get('reputation_band', 'Unknown')} (Trend: {obs.get('customer_trend', 'Unknown')})",
        "weather": f"Today: {obs.get('weather_today', 'Unknown')}, Forecast: {obs.get('weather_forecast', [])}",
        "active_menu_with_price_bounds": active_menu_details,
        "alerts": obs.get("alerts", []),
        "notes_from_yesterday": obs.get("notes", ""),
    }
    return summary


def llm_strategy(obs: dict, day: int) -> list[dict]:
    """Use the LLM for strategic decisions."""
    summary = summarize_observation(obs)
    user_msg = f"Here is the summary for today:\n\n{json.dumps(summary, indent=2)}\n\nWhat are your strategic actions? Return ONLY a JSON array."

    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            api_base=os.getenv("OPENAI_API_BASE", "http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com"),
            temperature=0.4,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()

        # Clean up markdown if the LLM ignored instructions
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        tool_calls = json.loads(content)
        if not isinstance(tool_calls, list):
            return []
        
        # Filter out tool calls that the LLM shouldn't be making (rule-based system's job)
        allowed_tools = {"set_price", "set_marketing_spend", "run_happy_hour", "offer_daily_special", "set_menu", "save_notes"}
        filtered_calls = [tc for tc in tool_calls if tc.get("tool") in allowed_tools]
        
        return filtered_calls

    except Exception as e:
        print(f"  [LLM Engine Error] Day {day}: {e}")
        return []

def rule_based_strategy(obs: dict, day: int) -> list[dict]:
    """Deterministic rules for survival: inventory and basic staffing."""
    actions = []
    
    cash = obs["cash"]
    inventory = {inv["ingredient"]: inv["total_kg"] for inv in obs.get("inventory", [])}
    pending = {}
    for po in obs.get("pending_orders", []):
        pending[po["ingredient"]] = pending.get(po["ingredient"], 0) + po["quantity_kg"]

    # 1. Staffing Logic
    base_staff = 8
    service_summary = obs.get("service_summary") or {}
    walkout_band = service_summary.get("walkout_band", "None")
    if walkout_band == "Many":
        base_staff += 2
    elif walkout_band == "Some":
        base_staff += 1
    elif walkout_band == "None":
        base_staff -= 1
        
    # Cap staff to save money if we are poor
    if cash < 3000:
        base_staff = min(base_staff, 6)
        
    # Ensure within bounds
    base_staff = max(3, min(15, base_staff))
    actions.append({"tool": "set_staff_level", "args": {"level": base_staff}})

    # 2. Inventory Logic
    # Build a map of the cheapest available supplier for each ingredient
    cheapest_supplier = {}
    for sup in obs.get("supplier_catalog", []):
        for ingredient, price in sup["ingredients"].items():
            if ingredient not in cheapest_supplier or price < cheapest_supplier[ingredient][1]:
                cheapest_supplier[ingredient] = (sup["name"], price, sup["min_order_kg"])

    # Determine which ingredients we actually need (based on active menu)
    active_menu = obs.get("active_menu", [])
    menu_book = {dish["name"]: dish for dish in obs.get("menu_book", [])}
    required_ingredients = set()
    for dish_name in active_menu:
        if dish_name in menu_book:
            for ing in menu_book[dish_name].get("ingredients", []):
                required_ingredients.add(ing["ingredient"])

    # If an ingredient is required but not in the catalog (e.g., supplier crisis), we can't order it.
    
    # Keep a safety reserve of cash
    budget = cash - 1500  
    
    # Target stock levels
    service_summary = obs.get("service_summary") or {}
    covers_yesterday = service_summary.get("total_covers", 0)
    if covers_yesterday > 100:
        TARGET_STOCK = 10.0
    elif covers_yesterday < 50:
        TARGET_STOCK = 4.0
    else:
        TARGET_STOCK = 6.0 

    for ingredient in required_ingredients:
        if ingredient not in cheapest_supplier:
            continue # Can't order this right now
            
        supplier, price, min_qty = cheapest_supplier[ingredient]
        
        stock = inventory.get(ingredient, 0) + pending.get(ingredient, 0)
        
        if stock < TARGET_STOCK and budget > min_qty * price:
            # Order enough to reach target, but respect min_qty
            qty_needed = TARGET_STOCK - stock
            qty_to_order = max(qty_needed, min_qty)
            
            # Make sure we don't blow the budget
            if qty_to_order * price <= budget:
                actions.append({
                    "tool": "place_order",
                    "args": {"supplier": supplier, "ingredient": ingredient, "quantity_kg": round(qty_to_order, 1)},
                })
                budget -= qty_to_order * price

    return actions

def strategy(observation: dict, day: int) -> list[dict]:
    """Main entrypoint for the agent."""
    rule_actions = rule_based_strategy(observation, day)
    llm_actions = llm_strategy(observation, day)
    
    # Filter LLM actions to prevent bankruptcy
    safe_llm_actions = []
    cash = observation.get("cash", 0)
    for action in llm_actions:
        tool = action.get("tool")
        if tool in ["set_marketing_spend", "run_happy_hour"]:
            # Only allow marketing/promotions if we have a healthy cash reserve
            if cash > 4000:
                safe_llm_actions.append(action)
            else:
                print(f"  [Rule Engine] Blocked {tool} to save cash (Cash: {cash})")
        elif tool == "set_price":
            # Just verify it's within strict bounds just in case
            args = action.get("args", {})
            dish_name = args.get("dish")
            price = args.get("price")
            menu_book = {d["name"]: d for d in observation.get("menu_book", [])}
            if dish_name in menu_book and isinstance(price, (int, float)):
                base_price = menu_book[dish_name]["base_price"]
                # Clamp to safe bounds
                safe_price = max(base_price * 0.81, min(base_price * 1.19, price))
                action["args"]["price"] = round(safe_price, 2)
                safe_llm_actions.append(action)
        else:
            safe_llm_actions.append(action)

    # Merge actions. Let LLM actions append to rule actions.
    return rule_actions + safe_llm_actions

if __name__ == "__main__":
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY first.")
        sys.exit(1)
        
    print(f"Starting Mixed Agent with model: {MODEL}")
    result = run_game(strategy, team_name="mixed_agent", seed=42)
