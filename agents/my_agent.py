"""Master Profitability Agent

This agent implements a highly optimized, 100% deterministic mathematical
control system that dynamically scales prices, staff, marketing, and inventory
to maximize profit while maintaining near-perfect reputation.
"""

from __future__ import annotations

import json
import os
import sys
import math

from agents.runner import run_game


def strategy(observation: dict, day: int) -> list[dict]:
    actions = []
    
    # 1. READ OBSERVATION
    cash = observation.get("cash", 0.0)
    inventory = {inv["ingredient"]: inv["total_kg"] for inv in observation.get("inventory", [])}
    pending = {}
    for po in observation.get("pending_orders", []):
        pending[po["ingredient"]] = pending.get(po["ingredient"], 0.0) + po["quantity_kg"]
        
    dow = observation.get("day_of_week", "Thursday")
    weather = observation.get("weather_today", "cloudy")
    rep = observation.get("reputation_band", "Good")
    
    # 2. Sunday closed-day check
    is_sunday = (dow == "Sunday")
    
    # 3. FORECAST DEMAND (expected_covers)
    if is_sunday:
        expected_covers = 0
    else:
        base_covers = 80
        # Day of week adjustments
        if dow in ["Friday", "Saturday", "Sunday"]:
            base_covers += 40
        elif dow in ["Monday", "Tuesday", "Wednesday"]:
            base_covers -= 30
            
        # Weather adjustments
        if weather == "sunny":
            base_covers += 20
        elif weather == "rainy":
            base_covers -= 20
        elif weather == "stormy":
            base_covers -= 40
            
        # Reputation adjustments
        if rep == "Excellent":
            base_covers += 20
        elif rep == "Very Good":
            base_covers += 10
        elif rep == "Fair":
            base_covers -= 10
        elif rep == "Poor":
            base_covers -= 30
            
        expected_covers = max(10, base_covers)
    
    # 4. STAFFING OPTIMIZER
    # Scale staff dynamically to fully capture high-demand revenue while keeping labor costs low.
    if is_sunday:
        staff_level = 3  # Closed day, absolute minimum staff to save cash
    else:
        # Calibrated peak staffing to prevent walkouts on high-demand days
        if expected_covers > 150:
            staff_level = 11
        elif expected_covers > 120:
            staff_level = 9
        elif expected_covers > 80:
            staff_level = 7
        else:
            staff_level = 6
            
        # Emergency cash-saving staffing caps
        if cash < 3000:
            staff_level = min(staff_level, 5)
        if cash < 1500:
            staff_level = min(staff_level, 4)
            
    actions.append({"tool": "set_staff_level", "args": {"level": staff_level}})
    
    # 5. MARKETING & PROMOTIONS OPTIMIZER
    marketing_spend = 0.0
    if not is_sunday and cash > 4000 and dow in ["Monday", "Tuesday", "Wednesday", "Thursday"]:
        if expected_covers < 80:
            marketing_spend = 150.0
            
    if marketing_spend > 0:
        actions.append({"tool": "set_marketing_spend", "args": {"amount": marketing_spend}})
        
    # Happy Hour Logic: Only run if reputation is low (Good, Fair, Poor) to boost it
    if not is_sunday and rep in ["Good", "Fair", "Poor"] and cash > 4000:
        actions.append({"tool": "run_happy_hour", "args": {}})
        
    # 6. DYNAMIC PRICING OPTIMIZER
    # Safe dynamic pricing to maximize margin without crashing demand
    price_multiplier = 1.12
    if expected_covers > 120:
        price_multiplier = 1.18
    elif expected_covers < 60:
        price_multiplier = 1.00
        
    # Safety Valve: If yesterday's covers were low (< 60), reset prices to 1.0x to fill the restaurant
    service_summary = observation.get("service_summary") or {}
    yesterday_covers = service_summary.get("total_covers", 0)
    if yesterday_covers < 60 and day > 1 and not is_sunday:
        price_multiplier = 1.00
        
    # Set menu prices
    menu_book = {dish["name"]: dish for dish in observation.get("menu_book", [])}
    active_menu = observation.get("active_menu", [])
    
    for dish_name in active_menu:
        if dish_name in menu_book:
            base_price = menu_book[dish_name]["base_price"]
            target_price = round(base_price * price_multiplier, 2)
            # Strict safety clamp to 1.19x base price to prevent any float precision rejections from the server
            target_price = min(target_price, round(base_price * 1.19, 2))
            actions.append({"tool": "set_price", "args": {"dish": dish_name, "price": target_price}})
            
    # 7. DAILY SPECIAL
    # Always set the daily special to the active dish with the highest base price to boost high-margin sales
    highest_price_dish = None
    highest_price = 0.0
    for dish_name in active_menu:
        if dish_name in menu_book:
            base_price = menu_book[dish_name]["base_price"]
            if base_price > highest_price:
                highest_price = base_price
                highest_price_dish = dish_name
                
    if highest_price_dish and not is_sunday:
        actions.append({"tool": "offer_daily_special", "args": {"dish": highest_price_dish}})
        
    # 8. INVENTORY ORDERING OPTIMIZER
    # We must keep a high target stock (15.0kg) to completely eliminate 0-cover days caused by delayed deliveries!
    if is_sunday:
        TARGET_STOCK = 5.0  # Closed tomorrow, keep inventory low
    else:
        TARGET_STOCK = 15.0  # Safe robust buffer to prevent running out of food
    
    # Cheapest supplier mapping
    cheapest_supplier = {}
    for sup in observation.get("supplier_catalog", []):
        for ingredient, price in sup["ingredients"].items():
            if ingredient not in cheapest_supplier or price < cheapest_supplier[ingredient][1]:
                cheapest_supplier[ingredient] = (sup["name"], price, sup["min_order_kg"])
                
    # Determine required ingredients based on active menu
    required_ingredients = set()
    for dish_name in active_menu:
        if dish_name in menu_book:
            for ing in menu_book[dish_name].get("ingredients", []):
                required_ingredients.add(ing["ingredient"])
                
    # Reorder ingredients to reach TARGET_STOCK
    budget = cash - 1500  # Cash safety reserve
    
    # Sort ingredients by price to prioritize cheaper/critical ingredients first
    order_queue = []
    for ingredient in required_ingredients:
        if ingredient not in cheapest_supplier:
            continue
        supplier, price, min_qty = cheapest_supplier[ingredient]
        stock = inventory.get(ingredient, 0.0) + pending.get(ingredient, 0.0)
        if stock < TARGET_STOCK:
            qty_needed = TARGET_STOCK - stock
            qty_to_order = max(qty_needed, min_qty)
            cost = qty_to_order * price
            order_queue.append((price, ingredient, supplier, qty_to_order, cost))
            
    order_queue.sort() # Prioritize cheaper ingredients
    
    for price, ingredient, supplier, qty, cost in order_queue:
        if cost <= budget:
            actions.append({
                "tool": "place_order",
                "args": {"supplier": supplier, "ingredient": ingredient, "quantity_kg": round(qty, 1)},
            })
            budget -= cost
            
    return actions


if __name__ == "__main__":
    print("Starting Master Profitability Agent...")
    result = run_game(strategy, team_name="master_profit_agent", seed=42)
