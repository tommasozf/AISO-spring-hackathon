# Agent Contract

Your restaurant is losing money. You have 30 days.

You manage a 22-table Italian restaurant. Each simulated day, you receive an **observation** via the REST API describing the current state — financials, inventory, weather, customer signals, supplier catalog. You respond with **tool calls** specifying your decisions: what to order, which dishes to serve, how many staff to schedule, whether to run promotions. Invalid actions are rejected with a reason message.

---

## Game Flow

All interaction happens over HTTP. Your agent creates a game, submits zero or more tool calls per turn, then ends the turn to advance the day.

```
POST /games                        → create game, get first observation
  ↓
LOOP (30 days):
  POST /games/{id}/action          → submit a tool call (repeat 0+ times)
  POST /games/{id}/end-turn        → advance the day, get new observation
  ↓
  if status != "in_progress": break
  ↓
GET /games/{id}/score              → get final score breakdown
```

**Creating a game:**
```json
POST /games
{
  "team_name": "my-team",
  "scenario": "baseline",
  "seed": 42
}
```

**Submitting a tool call:**
```json
POST /games/{id}/action
{
  "tool": "place_order",
  "args": {
    "supplier": "Fresh Farms NL",
    "ingredient": "Tomato Sauce",
    "quantity_kg": 15.0
  }
}
```

**Ending a turn:**
```json
POST /games/{id}/end-turn
→ { "observation": {...}, "day": 2, "status": "in_progress", "day_result": {...} }
```

---

## How a Day Works

When you end your turn, the simulator runs a full day of restaurant operation:

1. **Your actions are applied** — orders placed, menu/prices/staff updated, promotions activated
2. **Deliveries arrive** — orders reaching their delivery day are fulfilled (supplier reliability affects whether you receive everything you ordered)
3. **Spoiled inventory is removed** — batches at expiry are discarded (waste cost counted)
4. **Service runs hour by hour (11:00–22:00)** — customers arrive based on demand, are seated at available tables, order dishes from your menu, consume ingredients, and leave satisfied or not
5. **Reputation updates** — customer reviews are generated (including negative "ghost reviews" from walkouts) and your reputation adjusts
6. **End-of-day accounting** — revenue, staff costs, fixed costs, marketing spend, and waste are tallied. If cash goes negative, you're bankrupt.
7. **Weather advances** — tomorrow's weather is determined

Key insight: everything you do happens *before* service. You cannot react to today's customers — only prepare for them.

---

## Observation Schema

Every day you receive an observation. Here is an abbreviated example from day 5:

```json
{
  "day": 5,
  "day_of_week": "Friday",
  "days_remaining": 25,

  "cash": 12580.45,
  "yesterday_revenue": 2145.80,
  "yesterday_total_costs": 1263.50,
  "cost_breakdown": {
    "staff": 960.0,
    "fixed": 300.0,
    "marketing": 0.0,
    "waste": 3.50
  },

  "inventory": [
    {
      "ingredient": "Flour",
      "total_kg": 42.5,
      "shelf_life_days": 14,
      "batches": [
        {"quantity_kg": 22.5, "expires_in_days": 9},
        {"quantity_kg": 20.0, "expires_in_days": 12}
      ]
    },
    {
      "ingredient": "Tomato Sauce",
      "total_kg": 8.3,
      "shelf_life_days": 7,
      "batches": [
        {"quantity_kg": 8.3, "expires_in_days": 2}
      ]
    }
  ],

  "service_summary": {
    "total_covers": 103,
    "total_revenue": 2145.80,
    "walkout_band": "Few",
    "hourly_covers": [3, 14, 12, 5, 2, 3, 6, 12, 18, 16, 9, 3],
    "avg_wait_minutes": 4.2,
    "peak_wait_minutes": 14.7,
    "dishes_sold": {"Pizza Margherita": 18, "Chicken Parmesan": 12},
    "dishes_unavailable_at": {},
    "substitution_count": 2,
    "table_utilization_peak": 0.77,
    "kitchen_bottleneck_hours": []
  },

  "supplier_catalog": [
    {
      "name": "Fresh Farms NL",
      "lead_time_days": 1,
      "delivery_days": ["Monday", "Wednesday", "Friday"],
      "min_order_kg": 5.0,
      "ingredients": {"Tomato Sauce": 3.1, "Mushrooms": 4.2, "Lettuce": 2.8, "Chicken": 8.5}
    }
  ],

  "pending_orders": [
    {"supplier": "Fresh Farms NL", "ingredient": "Tomato Sauce",
     "quantity_kg": 15.0, "delivery_day": 7}
  ],

  "delivery_history": [
    {"supplier": "Fresh Farms NL", "ingredient": "Tomato Sauce",
     "ordered_kg": 15.0, "delivered_kg": 15.0, "order_day": 2,
     "delivery_day": 3, "on_time": true}
  ],

  "menu_book": [
    {"name": "Pizza Margherita", "category": "Pizza",
     "base_price": 14.5, "current_price": 14.5, "is_active": true,
     "ingredients": [{"ingredient": "Flour", "quantity_kg": 0.25}]}
  ],
  "active_menu": ["Pizza Margherita", "Chicken Parmesan", "Mushroom Tagliatelle", "Spaghetti Carbonara", "Grilled Salmon"],

  "staff_level": 8,
  "staff_cost_per_person": 120.0,

  "reputation_band": "Very Good",
  "recent_reviews": [
    {"stars": 4.2, "day_of_visit": 3, "day_posted": 5}
  ],

  "customer_trend": "Stable",

  "weather_today": "sunny",
  "weather_forecast": ["cloudy", "rainy", "sunny"],

  "alerts": [],
  "notes": "Day 4: ordered Tomato Sauce. Watching Flour levels.",
  "tick_budget_ms": 30000
}
```

---

## Observation Field Reference

### Exact Visibility

These fields reflect the true state of your restaurant.

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current simulation day (1-30) |
| `day_of_week` | str | "Monday" through "Sunday" |
| `days_remaining` | int | Days left in simulation |
| `cash` | float | Current cash balance in EUR |
| `yesterday_revenue` | float | Previous day's total revenue |
| `yesterday_total_costs` | float | Previous day's total costs |
| `cost_breakdown` | dict | Breakdown: `staff`, `fixed`, `marketing`, `waste` |
| `inventory` | list | Batch-level inventory with `ingredient`, `total_kg`, `shelf_life_days`, `batches` |
| `staff_level` | int | Current number of staff |
| `staff_cost_per_person` | float | EUR 120/day per staff member |
| `active_menu` | list[str] | Currently offered dish names |
| `menu_book` | list | All recipes with `name`, `category`, `base_price`, `current_price`, `is_active`, `ingredients` |
| `pending_orders` | list | Orders placed but not delivered |
| `delivery_history` | list | Recent deliveries with ordered vs. delivered quantities |
| `supplier_catalog` | list | Available suppliers with prices, delivery schedules, lead times |
| `weather_today` | str | Exact: sunny/cloudy/rainy/stormy |
| `weather_forecast` | list[str] | 3-day forecast (accuracy degrades: 85%, 70%, 55%) |
| `alerts` | list[str] | Scenario-injected warnings |
| `notes` | str | Your persisted scratchpad from previous day (max 4000 chars) |
| `tick_budget_ms` | int | 30,000 ms (30 seconds) to submit your tool calls |

### Approximate Visibility

These fields are coarsened. You see bands, not exact values.

| Field | Type | Values |
|-------|------|--------|
| `reputation_band` | str | "Poor", "Fair", "Good", "Very Good", "Excellent" |
| `walkout_band` | str | "None" (0), "Few" (1-5), "Some" (6-20), "Many" (21+) |
| `customer_trend` | str | "Declining", "Stable", "Growing" |

### Delayed Visibility

Reviews are posted days after the visit. Today's reviews reflect decisions from 1-4 days ago.

| Field | Type | Description |
|-------|------|-------------|
| `recent_reviews` | list | Reviews posted recently. Each: `stars`, `day_of_visit`, `day_posted` |
| `service_summary` | dict | Yesterday's service metrics: covers, revenue, dishes sold, wait times, walkout band |

---

## Tool Reference

Submit tool calls via `POST /games/{id}/action`. Each call is `{"tool": "<name>", "args": {...}}`.

| Tool | Args | Effect |
|------|------|--------|
| `place_order` | `supplier`, `ingredient`, `quantity_kg` | Order ingredients. Cost deducted at end of turn. Delivered after lead time on a valid delivery day. |
| `set_menu` | `dishes` (list of names) | Set active menu. Minimum 5 dishes. |
| `set_price` | `dish`, `price` | Set dish price. Must be within 80%-120% of base price. |
| `set_staff_level` | `level` (int) | Hire or fire. Range: 3-15. |
| `set_marketing_spend` | `amount` (float) | Marketing budget for today. Range: 0-500 EUR. |
| `run_happy_hour` | *(no args)* | Activate happy hour (15:00-18:00). Boosts demand, discounts prices. |
| `offer_daily_special` | `dish` | Today's special — must be on active menu. Small satisfaction bonus. |
| `save_notes` | `text` | Persist up to 4000 chars to your next observation. |

---

## Validation Rules

Actions are validated when submitted. Invalid actions return `{"status": "rejected", "reason": "..."}` with a specific error message explaining what went wrong.

- **Orders:** Supplier must exist. Ingredient must be in that supplier's catalog. Quantity must be positive and meet `min_order_kg`. Total cost must not exceed remaining cash.
- **Menu:** All dish names must exist. Minimum 5 dishes after deduplication.
- **Prices:** Dish must exist. Price must be between 0.8x and 1.2x the base price.
- **Staff:** Integer between 3 and 15. Out of range is rejected.
- **Marketing:** Between 0 and 500 EUR. Out of range is rejected.
- **Daily special:** Dish must exist in the recipe book.
- **Notes:** Truncated to 4,000 characters if longer.

**Names are case-sensitive.** Use the exact names from the observation (e.g., `"Fresh Farms NL"`, `"Tomato Sauce"`, `"Pizza Margherita"`).

---

## Scoring

Your composite score combines profit with quality metrics. The exact formula penalizes:

- **Low satisfaction** — quadratic penalty below threshold
- **Low reputation** — quadratic penalty below threshold
- **Walkouts** — linear penalty per walkout
- **Food waste** — penalty for excessive waste rate

Going bankrupt results in a catastrophic score of -100,000. Survival first, optimization second.

The exact coefficients are not disclosed. Build a restaurant you would eat at.

### Scoring Priorities

In rough order of impact:
1. **Don't go bankrupt** — instant -100,000 score. Nothing else matters if cash hits zero.
2. **Keep quality metrics above their thresholds** — some penalties are steep and non-linear. Falling a little below a threshold costs much more than you'd expect.
3. **Minimize walkouts** — every walkout has direct and indirect costs. Walkouts generate negative reviews that compound.
4. **Control waste** — moderate waste is acceptable; excessive waste is penalized.
5. **Maximize net profit** — what remains after all penalties.

The simulation cares about how you finish, not just how you average.

---

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Starting cash | 15,000 EUR |
| Fixed daily cost | 300 EUR |
| Staff cost | 120 EUR/day per person |
| Starting staff | 8 |
| Tables | 22 (4×2-seat, 8×4-seat, 6×6-seat, 4×8-seat) |
| Simulation length | 30 days |
| Service hours | 11:00-22:00 |
| Day 1 | Monday |
