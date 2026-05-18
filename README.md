# RestBench — AI Restaurant Management Hackathon

**Build an AI agent that runs an Italian restaurant for 30 days. Order ingredients, set prices, manage staff, run promotions — all through a REST API. The team with the highest score wins.**

Your restaurant starts in the red. You have 30 simulated days to turn it around. Every decision matters: order too much and food expires, order too little and customers walk out, cut staff to save money and your reputation tanks. The best agents balance dozens of competing tradeoffs — and adapt when things go wrong.

---

## Get Started (5 minutes)

### 1. Clone and install

```bash
git clone <this-repo>
cd restbench-starter-kit
pip install -r requirements.txt
```

### 2. Set the server URL

```bash
export RESTBENCH_URL=http://52.48.183.209:8001
```

> **Explore the API interactively:** http://52.48.183.209:8001/docs (Swagger UI)

### 3. Run a baseline to see how the game works

```bash
python -m agents.naive_rule
```

This runs a simple rule-based agent that survives all 30 days but scores around -15,000. Your job: beat it.

### 4. Start building your agent

**Option A — LLM-based agent (recommended starting point):**
```bash
cp agents/llm_template.py agents/my_agent.py
export OPENAI_API_KEY=sk-...            # or ANTHROPIC_API_KEY for Claude
export AGENT_MODEL=openai/gpt-4.1-mini  # any litellm-supported model
python -m agents.my_agent
```

**Option B — Rule-based agent:**
```bash
cp agents/starter_template.py agents/my_agent.py
# Edit the strategy() function in agents/my_agent.py
python -m agents.my_agent
```

**Option C — Any language:** The API is plain HTTP + JSON. Build your agent in whatever you like.

### 5. Check the leaderboard

```bash
curl $RESTBENCH_URL/leaderboard
```

---

## The Challenge

You manage a 22-table Italian restaurant. Each day, your agent receives an **observation** (cash, inventory, suppliers, weather, customer feedback, reputation) and responds with **actions** (order food, set prices, adjust staff, run promotions). After 30 days, you get a composite score.

```
total_score = net_profit - penalties
```

Penalties are assessed for low satisfaction, low reputation, walkouts, and food waste. Going bankrupt = score of **-100,000**.

**Higher is better.** The baselines score -15,000 to -19,000. A well-designed agent can score positive.

> **Full game specification:** [AGENT_CONTRACT.md](AGENT_CONTRACT.md)
> **Strategy thinking:** [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)

---

## Game Loop

All interaction happens over HTTP:

```
POST /games                       -> create game, get first observation
loop 30 days:
    GET  /games/{id}/observe      -> read current state (optional)
    POST /games/{id}/action       -> submit one action (repeat as needed)
    POST /games/{id}/end-turn     -> advance the day, get results
GET  /games/{id}/score            -> final score
```

### Create a game

```bash
curl -X POST $RESTBENCH_URL/games \
  -H 'Content-Type: application/json' \
  -d '{"team_name": "my_team", "scenario": "baseline", "seed": 42}'
```

Returns `{game_id, day, status, observation}`. Save the `game_id`.

### Submit actions (one at a time, as many as you want per turn)

```bash
curl -X POST $RESTBENCH_URL/games/{game_id}/action \
  -H 'Content-Type: application/json' \
  -d '{"tool": "place_order", "args": {"supplier": "Fresh Farms NL", "ingredient": "Chicken", "quantity_kg": 8}}'
```

Returns `{"status": "accepted"}` or `{"status": "rejected", "reason": "..."}`.

### End the turn

```bash
curl -X POST $RESTBENCH_URL/games/{game_id}/end-turn
```

Advances the simulation by one day. Returns the new observation plus yesterday's service results.

### Get your score (after day 30 or bankruptcy)

```bash
curl $RESTBENCH_URL/games/{game_id}/score
```

---

## Available Actions

| Action | What it does |
|--------|-------------|
| **place_order** | Order ingredients from a supplier. Delivery takes 1-2 days and only on that supplier's delivery days. |
| **set_staff_level** | Adjust staff between 3 and 15. Each costs 120 EUR/day. More staff = faster kitchen, fewer walkouts. |
| **set_menu** | Change active dishes (min 5). New dishes have a kitchen learning curve. Narrow menus reduce demand. |
| **set_price** | Adjust a dish's price between 0.8x and 1.2x its base price. |
| **set_marketing_spend** | Spend 0-500 EUR/day on marketing. Diminishing returns. |
| **run_happy_hour** | Boosts demand, discounts prices, small satisfaction bonus. Diminishing returns on consecutive use. |
| **offer_daily_special** | Pick one menu dish as today's special for a satisfaction bonus. |
| **save_notes** | Save up to 4,000 chars that persist between turns. Read via `GET /games/{id}/notes`. |

<details>
<summary>Action examples (JSON)</summary>

```json
{"tool": "place_order", "args": {"supplier": "Fresh Farms NL", "ingredient": "Chicken", "quantity_kg": 8}}
{"tool": "set_staff_level", "args": {"level": 6}}
{"tool": "set_menu", "args": {"dishes": ["Pizza Margherita", "Chicken Parmesan", "Grilled Salmon", "Mushroom Risotto", "Spaghetti Carbonara"]}}
{"tool": "set_price", "args": {"dish": "Grilled Salmon", "price": 22.0}}
{"tool": "set_marketing_spend", "args": {"amount": 200}}
{"tool": "run_happy_hour", "args": {}}
{"tool": "offer_daily_special", "args": {"dish": "Mushroom Risotto"}}
{"tool": "save_notes", "args": {"text": "Day 3: ordered salmon, low on cream"}}
```

</details>

---

## What Your Agent Can See

The observation is returned when you create a game, call `/observe`, or call `/end-turn`.

| Field | What it tells you |
|-------|-------------------|
| `day`, `day_of_week`, `days_remaining` | Where you are in the 30-day game |
| `cash` | Current balance (EUR) |
| `yesterday_revenue`, `yesterday_total_costs` | Yesterday's P&L |
| `cost_breakdown` | Staff, fixed, marketing, waste costs |
| `inventory` | Per-ingredient: total kg, batches with expiry dates, shelf life |
| `service_summary` | Yesterday's covers, revenue, walkouts, dishes sold, wait times, stockouts |
| `supplier_catalog` | All suppliers: prices, lead times, delivery days, min orders |
| `pending_orders` | Orders in transit with expected delivery day |
| `delivery_history` | Last 14 days: ordered vs delivered, on-time status |
| `menu_book` | All recipes: ingredients, base price, current price, active status |
| `active_menu` | Currently served dishes |
| `staff_level` | Current staff count |
| `reputation_band` | "Poor" / "Fair" / "Good" / "Very Good" / "Excellent" |
| `recent_reviews` | Reviews from last 14 days: stars, visit day |
| `customer_trend` | "Declining" / "Stable" / "Growing" |
| `weather_today`, `weather_forecast` | Today's weather + 3-day forecast (accuracy degrades) |
| `alerts` | Scenario events (supplier issues, demand changes, etc.) |
| `notes` | Your saved notes |

### The most important field: `dishes_unavailable_at`

Inside `service_summary`, this tells you exactly which dish ran out and when. If Grilled Salmon ran out at hour 14, you lost 8 hours of salmon sales. This is the #1 signal for inventory management.

### What you DON'T see

- Exact satisfaction scores (only reputation band)
- Exact walkout counts (only band: "None" / "Few" / "Some" / "Many")
- Supplier reliability ratings
- Customer cohort sizes
- Other teams' games

---

## Game Mechanics

### Economics
- **Starting cash:** 15,000 EUR
- **Daily fixed cost:** 300 EUR (rent, utilities)
- **Staff cost:** 120 EUR/person/day (default 8 staff = 960/day)
- **Total daily overhead at 8 staff:** 1,260 EUR

### Supply Chain
- 5-7 suppliers with different ingredients, prices, and delivery schedules
- Lead time: 1-2 days, then delivery only on supplier's specific days
- **Example:** Order from a Wed-only supplier on Thursday with 1-day lead = delivers next Wednesday (6 days!)
- Ingredients are perishable (3-14 day shelf life)
- Suppliers can have disruptions — deliveries may fail during outages
- Oldest batches are consumed first (FIFO)

### Demand
- Varies by hour (lunch and dinner peaks), day of week (weekends busier), weather, reputation, marketing, menu variety, and pricing

### Tables
- 22 tables of various sizes (2, 4, 6, 8 seats)
- Customers get the smallest table that fits; if none available, they wait briefly then leave

### Reputation
- Starts at "Very Good", updated daily as a moving average
- Negative experiences have outsized impact
- **Reputation spirals are real** — a bad week can take many days to recover

---

## Scenarios

Test your agent against different conditions. The final evaluation includes **hidden scenarios** your agent hasn't seen — it must adapt based on observations and alerts.

| Scenario | What happens |
|----------|-------------|
| `baseline` | Standard 30-day game, no events |
| `supply_crisis` | A major supplier goes into outage mid-game |
| `tourist_season` | Large demand swings: surge then drop |
| `renovation` | Reduced table capacity for first 12 days |

Additional hidden scenarios will be used for final evaluation. Your agent must adapt to unseen conditions based on observations and alerts.

### Recommended seeds

Use these seeds during development for reproducible, comparable results:

| Seed | Purpose |
|------|---------|
| `42` | Primary development seed |
| `88` | Alternate seed for variety |
| `123` | Stress-test seed |

```bash
# Play a specific scenario
curl -X POST $RESTBENCH_URL/games \
  -H 'Content-Type: application/json' \
  -d '{"team_name": "my_team", "scenario": "supply_crisis", "seed": 42}'

# List all scenarios
curl $RESTBENCH_URL/scenarios
```

**Read the alerts.** When scenario events fire, they come with alert messages in `observation.alerts`.

---

## Tips for Winning

1. **Don't run out of ingredients.** A single stockout day costs revenue + reputation damage that compounds for days.
2. **Watch delivery schedules.** A Wed-only supplier with 1-day lead means orders placed Thursday arrive next Wednesday.
3. **Check `dishes_unavailable_at` every turn.** It's the clearest signal for what to reorder.
4. **Don't double-order.** Check `pending_orders` before placing new ones.
5. **Reputation is sticky.** It takes many good days to recover from one bad one. Avoid bad days rather than chasing great ones.
6. **Read the alerts.** "Supplier X halted operations" means find an alternative fast.
7. **Use `save_notes`.** Track orders, stockouts, and patterns. Your agent has no memory between turns otherwise.
8. **Same seed + same scenario + same actions = same result.** Use determinism to debug and iterate.
9. **Start simple.** A boring "keep everything stocked" strategy beats a clever one that occasionally runs out of food.
10. **Test across scenarios.** An agent that aces `baseline` but crashes on `supply_crisis` will lose on the hidden scenarios.

---

## Baselines

Run these to understand the scoring range and validate your setup:

```bash
python -m agents.do_nothing        # Bankrupt by day ~16, score: -100,000
python -m agents.naive_rule         # Survives 30 days, score: ~-15,000
python -m agents.starter_template   # Rule-based starting point
python -m agents.llm_template       # LLM starting point (needs API key)
python -m agents.compare            # Run all baselines side by side

# 🚀 The main hackathon submission agent
python -m agents.hybrid_agent       # Hybrid rule/LLM agent (requires OPENAI_API_KEY)
```

### Evaluate across scenarios and seeds

```bash
python -m agents.evaluate agents.my_agent                          # all scenarios, seeds 42/88/123
python -m agents.evaluate agents.my_agent --scenarios baseline,supply_crisis
python -m agents.evaluate agents.my_agent --seeds 42,88
python -m agents.evaluate agents.my_agent --parallel 5             # control concurrency (default: 10)
python -m agents.evaluate agents.my_agent --quiet                  # summary table only
```

Runs your agent against every (scenario, seed) combination in parallel and prints a summary report.

### 🌟 Running the Custom Hybrid Agent

You must configure the server URL and OpenAI API key first, then run using the custom conda environment (`ai-spring`):
```bash
export OPENAI_API_KEY=sk-... # Add your active key
export RESTBENCH_URL=http://52.48.183.209:8001
conda run -n ai-spring python -m agents.evaluate agents.hybrid_agent --scenarios baseline,supply_crisis,tourist_season,renovation --seeds 42,88,123
```
*Note: Make sure your API key is active. The agent will still survive if the LLM fails, but optimization will suffer.*

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/games` | Create a game. Body: `{team_name, scenario?, seed?}` |
| `GET` | `/games/{id}/observe` | Get current observation |
| `POST` | `/games/{id}/action` | Submit one action. Body: `{tool, args}` |
| `POST` | `/games/{id}/end-turn` | Advance to next day |
| `GET` | `/games/{id}/score` | Final score (after game ends) |
| `GET` | `/games/{id}/status` | Quick status: `{game_id, day, cash, status}` |
| `GET` | `/games/{id}/notes` | Read saved notes |
| `GET` | `/leaderboard` | Ranked scores. Filter: `?scenario=baseline` |
| `GET` | `/scenarios` | List available scenarios |
| `GET` | `/games` | List games. Filter: `?team_name=my_team` |
| `DELETE` | `/games/{id}` | Abandon a game |
| `GET` | `/health` | Server health check |

### Rate Limits

Per team: max **10 concurrent games** and **60 games per hour**. Exceeding either returns `429 Too Many Requests`. The evaluate harness handles parallelism automatically.

---

Good luck. Go feed some customers.
