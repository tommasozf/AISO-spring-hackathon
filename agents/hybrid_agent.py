import json
import os
import sys
import litellm

from agents.runner import run_game

# Set up the hackathon LiteLLM proxy
litellm.api_base = os.getenv("OPENAI_API_BASE", "http://litellm-production.eba-pvykax23.eu-west-1.elasticbeanstalk.com")

# Default to Claude 3.5 Sonnet via OpenAI proxy as requested
MODEL = os.getenv("AGENT_MODEL", "openai/claude-3-5-sonnet")

SYSTEM_PROMPT = """You are the Executive Brain of an AI Italian restaurant manager.
You manage pricing, menu, marketing, and promotions. Another system manages inventory and staff.
Your goal is to maximize long-term profit and keep reputation high.
Respond with ONLY a JSON array of tool calls.

Available tools:
- set_price: {"tool": "set_price", "args": {"dish": "...", "price": N}}
- set_menu: {"tool": "set_menu", "args": {"dishes": [...]}} (min 5 dishes)
- set_marketing_spend: {"tool": "set_marketing_spend", "args": {"amount": N}} (0-500)
- run_happy_hour: {"tool": "run_happy_hour", "args": {}}
- offer_daily_special: {"tool": "offer_daily_special", "args": {"dish": "..."}}

Guidelines:
1. Do not use place_order, set_staff_level, or save_notes. The Logistics Core handles those.
2. Watch reputation carefully. If it dips, reduce prices, run happy hour, or offer specials.
3. Keep the menu focused if demand is stable, but rotate if needed.
4. IMPORTANT: Ensure dish names match exactly with the active_menu or menu_book provided.
5. ONLY return valid JSON array. No explanations or markdown.
"""

def strategy(observation: dict, day: int) -> list[dict]:
    actions = []
    
    # 1. Parse State
    notes_str = observation.get("notes") or ""
    state = {}
    try:
        if notes_str.startswith("{"):
            state = json.loads(notes_str)
    except:
        pass
        
    ema_burn = state.get("ema_burn", {})
    
    # 2. Logistics Core
    # Calculate yesterday's consumption
    menu_dict = {dish["name"]: dish for dish in (observation.get("menu_book") or [])}
    daily_consumption = {}
    service_summary = observation.get("service_summary") or {}
    dishes_sold = service_summary.get("dishes_sold") or {}
    
    for dish_name, count in dishes_sold.items():
        if dish_name in menu_dict:
            for ing in menu_dict[dish_name]["ingredients"]:
                ing_name = ing["ingredient"]
                qty = ing["quantity_kg"] * count
                daily_consumption[ing_name] = daily_consumption.get(ing_name, 0.0) + qty
                
    # Update EMA
    ALPHA = 0.3
    for ing_name in daily_consumption:
        if ing_name in ema_burn:
            ema_burn[ing_name] = ALPHA * daily_consumption[ing_name] + (1 - ALPHA) * ema_burn[ing_name]
        else:
            ema_burn[ing_name] = daily_consumption[ing_name]
            
    # Handle dishes_unavailable_at (bump EMA aggressively for missing ingredients)
    unavailable = service_summary.get("dishes_unavailable_at") or {}
    for dish_name in unavailable:
        if dish_name in menu_dict:
            for ing in menu_dict[dish_name]["ingredients"]:
                ing_name = ing["ingredient"]
                ema_burn[ing_name] = ema_burn.get(ing_name, 1.0) * 1.5

    # Determine safety stock target (days of stock to keep)
    days_remaining = observation.get("days_remaining", 30)
    safety_days = 3 if days_remaining > 5 else 0

    # Lookahead predictor
    inventory = {inv["ingredient"]: inv["total_kg"] for inv in (observation.get("inventory") or [])}
    pending = {}
    for po in (observation.get("pending_orders") or []):
        pending[po["ingredient"]] = pending.get(po["ingredient"], 0) + po["quantity_kg"]
        
    catalog = observation.get("supplier_catalog") or []
    
    # Find cheapest suppliers
    cheapest_supplier = {}
    for sup in catalog:
        for ingredient, price in sup["ingredients"].items():
            if ingredient not in cheapest_supplier or price < cheapest_supplier[ingredient][1]:
                cheapest_supplier[ingredient] = (sup["name"], price, sup["min_order_kg"], sup["lead_time_days"], sup["delivery_days"])

    day_of_week = observation.get("day_of_week")
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today_idx = days_of_week.index(day_of_week) if day_of_week in days_of_week else 0
    
    budget = observation["cash"] - 3000 # Keep a safety reserve
    
    # Endgame: stop ordering if <= 5 days
    if days_remaining > 5:
        for ingredient, (sup_name, price, min_qty, lead_time, delivery_days) in cheapest_supplier.items():
            earliest_delivery_idx = (today_idx + lead_time) % 7
            
            days_to_delivery = lead_time
            curr_idx = earliest_delivery_idx
            while days_of_week[curr_idx] not in delivery_days:
                days_to_delivery += 1
                curr_idx = (curr_idx + 1) % 7
                
            burn_rate = max(ema_burn.get(ingredient, 1.0), 0.5) 
            
            stock = inventory.get(ingredient, 0) + pending.get(ingredient, 0)
            forecasted_stock_at_delivery = stock - (burn_rate * days_to_delivery)
            
            days_to_next_delivery = 1
            curr_idx = (curr_idx + 1) % 7
            while days_of_week[curr_idx] not in delivery_days:
                days_to_next_delivery += 1
                curr_idx = (curr_idx + 1) % 7
                
            target_stock = burn_rate * (days_to_next_delivery + safety_days)
            
            if forecasted_stock_at_delivery < target_stock and budget > 0:
                order_amount = target_stock - forecasted_stock_at_delivery
                order_amount = max(order_amount, min_qty)
                order_amount = min(order_amount, budget / price)
                
                if order_amount >= min_qty:
                    actions.append({
                        "tool": "place_order",
                        "args": {
                            "supplier": sup_name,
                            "ingredient": ingredient,
                            "quantity_kg": round(order_amount, 1)
                        }
                    })
                    budget -= order_amount * price

    # 3. Staffing
    reputation = observation.get("reputation_band") or "Good"
    walkouts = service_summary.get("walkout_band") or "None"
    
    if walkouts in ["Some", "Many"]:
        actions.append({"tool": "set_staff_level", "args": {"level": 11}})
    elif reputation in ["Very Good", "Excellent"] and walkouts == "None":
        actions.append({"tool": "set_staff_level", "args": {"level": 9}})
    else:
        actions.append({"tool": "set_staff_level", "args": {"level": 10}})

    # 4. LLM Executive Brain
    # Reduce size of observation sent to LLM
    llm_obs = {
        "day": day,
        "cash": observation.get("cash", 0),
        "reputation_band": observation.get("reputation_band"),
        "recent_reviews": observation.get("recent_reviews") or [],
        "customer_trend": observation.get("customer_trend"),
        "weather_today": observation.get("weather_today"),
        "weather_forecast": observation.get("weather_forecast") or [],
        "alerts": observation.get("alerts") or [],
        "active_menu": observation.get("active_menu") or [],
        "service_summary_yesterday": observation.get("service_summary") or {},
        "menu_book_prices": {m["name"]: m["current_price"] for m in (observation.get("menu_book") or [])}
    }
    
    user_msg = f"Day {day}/30.\nObservation:\n{json.dumps(llm_obs, indent=2)}"
    
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        if content:
            llm_tools = json.loads(content)
            if isinstance(llm_tools, list):
                actions.extend(llm_tools)
    except Exception as e:
        print(f"  LLM error on day {day}: {e}")

    # 5. Save State
    state["ema_burn"] = ema_burn
    state["last_day"] = day
    actions.append({"tool": "save_notes", "args": {"text": json.dumps(state)}})
    
    return actions

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first.")
        sys.exit(1)
    print(f"Using model: {MODEL}")
    result = run_game(strategy, team_name="hybrid_team", seed=42)
