# RestBench — AI Restaurant Management Challenge

You manage an Italian restaurant for 30 simulated days. Your AI agent makes daily decisions via a REST API: ordering ingredients, setting prices, managing staff, and running promotions. The agent with the highest composite score wins.

**Full specification:** See [AGENT_CONTRACT.md](AGENT_CONTRACT.md) for the complete game contract and scoring priorities.
**Strategy hints:** See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) for thinking about the problem.

---

## Hackathon Server

The competition server is running at:

```bash
export RESTBENCH_URL=http://52.48.183.209:8001
```

Set this environment variable and all agent scripts will use it automatically.

> **Interactive docs:** Visit http://52.48.183.209:8001/docs for Swagger UI where you can try every endpoint from your browser.

---

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (just `httpx`)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a game

```bash
curl -X POST $RESTBENCH_URL/games \
  -H 'Content-Type: application/json' \
  -d '{"team_name": "my_team", "scenario": "baseline", "seed": 42}'
```

Response:
```json
{
  "game_id": "abc-123",
  "day": 1,
  "status": "in_progress",
  "observation": { ... }
}
```

Save the `game_id` — you'll need it for every subsequent request.

### 3. Read the observation

The observation tells you everything your agent can see: cash, inventory, supplier prices, yesterday's service results, weather, reputation, and more.

```bash
curl $RESTBENCH_URL/games/{game_id}/observe
```

### 4. Submit actions

Submit one tool call at a time. You can submit multiple actions per turn.

```bash
# Order ingredients
curl -X POST $RESTBENCH_URL/games/{game_id}/action \
  -H 'Content-Type: application/json' \
  -d '{"tool": "place_order", "args": {"supplier": "Fresh Farms NL", "ingredient": "Chicken", "quantity_kg": 8}}'

# Set staff level
curl -X POST $RESTBENCH_URL/games/{game_id}/action \
  -H 'Content-Type: application/json' \
  -d '{"tool": "set_staff_level", "args": {"level": 5}}'
```

Each action returns `{"status": "accepted"}` or `{"status": "rejected", "reason": "..."}`.

### 5. End the turn

```bash
curl -X POST $RESTBENCH_URL/games/{game_id}/end-turn
```

This advances the simulation by one day and returns the new observation plus yesterday's service results.

### 6. Get your score (after day 30 or bankruptcy)

```bash
curl $RESTBENCH_URL/games/{game_id}/score
```

### 7. Check the leaderboard

```bash
curl $RESTBENCH_URL/leaderboard
```

---

## Building Your Agent

### Python (recommended)

Copy the starter template and edit the `strategy()` function:

```bash
cp agents/starter_template.py agents/my_agent.py
```

```python
def strategy(observation: dict, day: int) -> list[dict]:
    actions = []

    # Check inventory and order what's low
    for inv in observation["inventory"]:
        if inv["total_kg"] < 5.0:
            actions.append({
                "tool": "place_order",
                "args": {
                    "supplier": "Fresh Farms NL",
                    "ingredient": inv["ingredient"],
                    "quantity_kg": 8.0
                }
            })

    return actions
```

Run it:
```bash
python -m agents.my_agent
```

The included `agents/runner.py` handles the HTTP loop for you. Your strategy function receives the observation dict and returns a list of tool calls.

### Any language

The API is plain HTTP + JSON. Here's the game loop in pseudocode:

```
POST /games              -> { game_id, observation }
loop 30 times:
    analyze observation
    for each decision:
        POST /games/{id}/action  -> { status }
    POST /games/{id}/end-turn    -> { observation, day_result, status }
    if status != "in_progress": break
GET /games/{id}/score    -> { score }
```

---

## Scoring

Your composite score combines profit with quality metrics:

```
total_score = net_profit - penalties
```

Penalties are assessed for low satisfaction, low reputation, walkouts, and food waste. Going bankrupt results in a catastrophic score of -100,000.

**Higher is better.** The included baselines score around -15,000 to -19,000. A well-designed agent can score positive. Aim to beat the baselines first, then optimize.

See [AGENT_CONTRACT.md](AGENT_CONTRACT.md) for scoring priorities. The exact penalty formulas, thresholds, and coefficients are not disclosed.

---

## Available Tools

### place_order
Order ingredients from a supplier. Delivery takes 1-2 days and only arrives on the supplier's delivery days.

```json
{"tool": "place_order", "args": {"supplier": "Fresh Farms NL", "ingredient": "Chicken", "quantity_kg": 8}}
```

Cost is deducted when the turn is processed. Delivery is not guaranteed — suppliers can have disruptions.

### set_staff_level
Adjust staff between 3 and 15. Each staff member costs 120 EUR/day. More staff = faster kitchen, fewer delays. Too few = long waits, walkouts.

```json
{"tool": "set_staff_level", "args": {"level": 6}}
```

### set_menu
Change which dishes are served. Minimum 5 dishes. New dishes have a learning curve in the kitchen. Narrow menus reduce demand.

```json
{"tool": "set_menu", "args": {"dishes": ["Pizza Margherita", "Chicken Parmesan", "Grilled Salmon", "Mushroom Risotto", "Spaghetti Carbonara"]}}
```

### set_price
Adjust a dish's price between 0.8x and 1.2x its base price. Pricing affects demand.

```json
{"tool": "set_price", "args": {"dish": "Grilled Salmon", "price": 22.0}}
```

### set_marketing_spend
Spend 0-500 EUR/day on marketing. Boosts demand with diminishing returns.

```json
{"tool": "set_marketing_spend", "args": {"amount": 200}}
```

### run_happy_hour
Activates happy hour for the afternoon. Boosts demand, discounts prices, small satisfaction bonus. Consecutive use has diminishing returns.

```json
{"tool": "run_happy_hour", "args": {}}
```

### offer_daily_special
Pick one menu dish as today's special. Gives a satisfaction bonus when ordered.

```json
{"tool": "offer_daily_special", "args": {"dish": "Mushroom Risotto"}}
```

### save_notes
Save up to 4,000 characters of text that persists between turns. Use it to track state across days.

```json
{"tool": "save_notes", "args": {"text": "Day 3: ordered salmon, low on cream"}}
```

You can read saved notes via `GET /games/{game_id}/notes`.

---

## Observation Reference

The observation is returned when you create a game, call `/observe`, or call `/end-turn`. Here are the key fields:

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current day (1-30) |
| `day_of_week` | string | "Monday" through "Sunday" |
| `days_remaining` | int | Days left in the game |
| `cash` | float | Current cash balance (EUR) |
| `yesterday_revenue` | float | Revenue from yesterday's service |
| `yesterday_total_costs` | float | Total costs yesterday |
| `cost_breakdown` | dict | Breakdown: staff, fixed, marketing, waste |
| `inventory` | list | Per-ingredient: total_kg, batches with expiry, shelf_life_days |
| `service_summary` | object | Yesterday's service: covers, revenue, walkout band, dishes sold, wait times, table utilization, kitchen bottlenecks, stockout info |
| `supplier_catalog` | list | Per-supplier: name, lead time, delivery days, min order, ingredient prices |
| `pending_orders` | list | Orders in transit: supplier, ingredient, quantity, delivery day |
| `delivery_history` | list | Last 14 days of deliveries: ordered vs delivered, on-time |
| `menu_book` | list | All recipes: name, category, base price, current price, ingredients, is_active |
| `active_menu` | list | Currently active dish names |
| `staff_level` | int | Current staff count |
| `reputation_band` | string | "Poor", "Fair", "Good", "Very Good", or "Excellent" |
| `recent_reviews` | list | Reviews from the last 14 days: stars, visit day, post day |
| `customer_trend` | string | "Declining", "Stable", or "Growing" |
| `weather_today` | string | "sunny", "cloudy", "rainy", or "stormy" |
| `weather_forecast` | list | 3-day forecast (accuracy degrades with distance) |
| `alerts` | list | Scenario-injected alerts (supplier issues, events, etc.) |
| `notes` | string | Your saved notes from `save_notes` |

### Service Summary Detail

After day 1, `service_summary` contains:

| Field | What it tells you |
|-------|-------------------|
| `total_covers` | Customers served |
| `total_revenue` | Revenue earned |
| `walkout_band` | "None", "Few", "Some", or "Many" (approximate) |
| `hourly_covers` | Array of 12 values (11:00-22:00) — see demand patterns |
| `avg_wait_minutes` / `peak_wait_minutes` | How long customers waited |
| `dishes_sold` | Dict of dish name -> count |
| `dishes_unavailable_at` | Dict of dish -> hour when it ran out (**critical signal**) |
| `substitution_count` | How many times kitchen substituted ingredients |
| `table_utilization_peak` | Peak table usage (0-1) |
| `kitchen_bottleneck_hours` | Hours when kitchen was overwhelmed |

**`dishes_unavailable_at` is the most important field.** If a dish ran out at hour 14, you lost 8 hours of potential sales for that dish.

---

## Game Mechanics

### Economics
- **Starting cash:** 15,000 EUR
- **Daily fixed cost:** 300 EUR (rent, utilities)
- **Staff cost:** 120 EUR/person/day (default 8 staff = 960/day)
- **Total daily overhead at 8 staff:** 1,260 EUR

### Supply Chain
- 5-7 suppliers with different ingredients, prices, and delivery schedules
- **Lead time:** 1-2 days after ordering, then delivery only on the supplier's delivery days
- **Example:** Order from a Wed-only supplier on Thursday with 1-day lead -> delivers next Wednesday (6 days)
- Ingredients are perishable (3-14 day shelf life depending on type)
- **Suppliers can have disruptions** — deliveries may fail during outages
- FIFO consumption — oldest batches are used first

### Demand
- Varies by hour (lunch and dinner peaks)
- Varies by day of week (weekdays are quieter, weekends are busier)
- Affected by weather, reputation, marketing, menu variety, and price levels

### Tables
- 22 tables of various sizes (2, 4, 6, 8 seats)
- Customers are assigned the smallest table that fits their party
- If no table is available, they wait briefly, then walk out

### Reputation
- Starts at "Very Good"
- Updated daily as a moving average from customer reviews
- Negative experiences have outsized impact
- Reputation affects how many customers show up
- **Reputation spirals are real** — a bad week can take many days to recover

### What You Don't See
Your agent does NOT have access to:
- Exact satisfaction scores
- Exact reputation value (only the band)
- Exact walkout count (only the band)
- Supplier reliability ratings
- Customer cohort sizes
- Other teams' games

---

## Scenarios

Scenarios change the game conditions. Some have tuning changes (different starting cash, different costs), others inject mid-game events (supplier outages, demand surges, price shocks).

### Known scenarios (you can test against these)

| Scenario | What happens |
|----------|-------------|
| `baseline` | Standard 30-day game. No events |
| `supply_crisis` | A major supplier goes into outage mid-game |
| `tourist_season` | Large demand swings: surge then drop |
| `inflation` | Ingredient prices, rent, and wages escalate over time |
| `renovation` | Reduced table capacity for first 12 days |
| `health_scare` | Viral negative reviews tank your reputation |

### Hidden scenarios

There are additional hidden scenarios used for final evaluation. Your agent won't know the scenario name — it must adapt to whatever happens based on the observation and alerts.

**Alerts are your friend.** When a scenario event fires, it often comes with an alert message in `observation.alerts`. Read them.

### Playing a specific scenario

```bash
curl -X POST $RESTBENCH_URL/games \
  -H 'Content-Type: application/json' \
  -d '{"team_name": "my_team", "scenario": "supply_crisis", "seed": 42}'
```

### Listing available scenarios

```bash
curl $RESTBENCH_URL/scenarios
```

---

## API Reference

### POST /games
Create a new game.

**Request:**
```json
{"team_name": "my_team", "scenario": "baseline", "seed": 42}
```
- `team_name` (required): Your team identifier
- `scenario` (optional, default "baseline"): Scenario name
- `seed` (optional, default random): RNG seed for reproducibility

**Response:** `200` with `{game_id, day, status, observation}`

### GET /games/{game_id}/observe
Get the current observation without advancing the game.

**Response:** `200` with `{observation, day, status}`

### POST /games/{game_id}/action
Submit a single tool call. Call multiple times per turn for multiple actions.

**Request:**
```json
{"tool": "place_order", "args": {"supplier": "Fresh Farms NL", "ingredient": "Chicken", "quantity_kg": 8}}
```

**Response:** `200` with `{"status": "accepted"}` or `{"status": "rejected", "reason": "..."}`

### POST /games/{game_id}/end-turn
Advance the simulation by one day.

**Response:** `200` with:
```json
{
  "observation": { ... },
  "day": 2,
  "status": "in_progress",
  "day_result": {
    "total_covers": 133,
    "total_revenue": 2415.0,
    "walkout_band": "None",
    "dishes_sold": {"Pizza Margherita": 11, "Chicken Parmesan": 25},
    "substitutions": 0
  }
}
```

`status` is one of: `"in_progress"`, `"completed"`, `"bankrupt"`.

### GET /games/{game_id}/score
Get the final score. Only available after game completes or goes bankrupt.

**Response:** `200` with score breakdown including net_profit, penalties, and total_score.

### GET /games/{game_id}/status
Quick status check.

**Response:** `200` with `{game_id, day, cash, status}`

### GET /games/{game_id}/notes
Read saved notes.

**Response:** `200` with `{"notes": "..."}`

### GET /leaderboard
Ranked scores. Best score per team.

**Query params:** `?scenario=baseline` (optional filter)

### GET /scenarios
List available scenarios (hidden scenarios excluded).

### GET /games
List games. **Query params:** `?team_name=my_team`

### DELETE /games/{game_id}
Abandon a game.

### GET /health
Server health check.

---

## Tips

1. **Don't run out of ingredients.** A single stockout day costs significant revenue plus reputation damage that compounds for days. Order early and often.

2. **Watch delivery schedules.** A supplier that only delivers on Wednesdays with 1-day lead means orders placed Thursday won't arrive until next Wednesday. Plan ahead.

3. **Check `dishes_unavailable_at`.** This tells you exactly which dish ran out and when. If Grilled Salmon ran out at hour 14, you lost 8 hours of salmon sales.

4. **Monitor `pending_orders`.** Don't double-order ingredients that are already in transit.

5. **Staff costs add up.** 8 staff = 960 EUR/day. If you're not filling the restaurant, consider reducing.

6. **Reputation is sticky.** It takes many good days to recover from a bad one. Avoid bad days rather than trying to have great days.

7. **Use `save_notes` wisely.** Track what you ordered, what ran out, and any patterns you notice. Your notes persist between turns.

8. **Read the alerts.** Scenario events often announce themselves. "Supplier X has halted operations" means you need to find an alternative supplier fast.

9. **Determinism is your friend.** Same seed + same scenario + same actions = same result. Use this to debug and iterate.

10. **Start simple.** A basic "keep everything stocked" strategy beats a clever strategy that occasionally runs out of food.

---

## Running the Baselines

```bash
export RESTBENCH_URL=http://52.48.183.209:8001
pip install -r requirements.txt

python -m agents.do_nothing        # Bankrupt by day ~16, score: -100,000
python -m agents.naive_rule         # Survives 30 days
python -m agents.starter_template   # Your starting point
python -m agents.compare            # Run all baselines side by side
```
