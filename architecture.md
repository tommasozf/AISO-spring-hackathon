# architecture aka claude's plan

# Regime-Based Restaurant Agent (`regime_agent.py`)

## Context

The AISO hackathon is RestBench: a 30-day Italian-restaurant sim scored on `net_profit − quadratic_penalties(satisfaction, reputation, walkouts, waste)`. Evaluation runs across 4 known scenarios (`baseline`, `supply_crisis`, `tourist_season`, `renovation`) and 6 **unseen** scenarios revealed at test time. Robustness across diverse conditions outranks excelling at one scenario.

Two reference agents already exist:
- [agents/smart_rule.py](agents/smart_rule.py) — pure deterministic, ad-hoc per-scenario tweaks scattered across the code.
- [agents/luigi_agent.py](agents/luigi_agent.py) — hybrid deterministic + LLM brain, but the scenario model is a single `crisis_mode` boolean.

This new agent makes **scenario categorization a first-class concept**: each day, the situation is classified into a small set of **regimes**, each with a 0.0–1.0 severity. Each regime has a deterministic rule set; multi-regime composition is handled by per-decision priority. The LLM acts as **daily oversight** — it sees the deterministic categorization and can adjust severities and pick discrete tiers per rule-set knob, with strict JSON validation and rule fallback. Goal: an agent that generalizes to unseen scenarios because the regimes describe the *underlying disturbance*, not the named story.

User-confirmed design choices (locked):
- **Multi-label regimes** with continuous severity weights.
- **Daily LLM oversight** (~30 calls/game) — deterministic categorization first, LLM adjusts.
- **Discrete tier tuning only** (low / med / high) — no raw numbers from the LLM.
- **New file**, no changes to luigi or smart_rule.

### Iterative workflow this plan supports

This implementation is **phase 1 of two**:
- **Phase 1 (this plan)**: build the framework — regime classifiers, composer, LLM oversight, observability. Each rule set starts as a thin stub that just produces sane defaults (lifted from luigi/smart_rule defaults). Goal: prove the architecture beats the baselines on the 4 known scenarios.
- **Phase 2 (after this lands)**: deepen each regime's rule set one at a time, leveraging tuned playbooks the user already has (e.g., the existing tourist-season strategy). Per-regime tuning is intentionally isolated to single functions so it doesn't ripple.

The file layout below puts each regime's rule sets in dedicated functions so Phase 2 work is local.

---

## Regime set

Seven regimes (six named + implicit baseline). Each carries severity ∈ [0,1].

| Regime | Trigger signals (deterministic) | Maps to known scenarios |
|---|---|---|
| `tourist` | Recent covers / EMA > 1.4; alert keywords (`tourist`, `festival`, `surge`, `event`); growing trend | `tourist_season` |
| `demand_drought` | Recent covers / EMA < 0.7; declining trend; bad weather forecast | (hidden off-season) |
| `supply_crisis` | Delivery shortfall (ordered vs delivered < 50%); alert keywords (`outage`, `shortage`, `strike`, supplier name); pending orders stuck | `supply_crisis` |
| `capacity_limited` | Alert keywords (`renovation`, `tables unavailable`, `construction`); observed capacity drop | `renovation` |
| `reputation_crisis` | reputation_band ∈ {Fair, Poor}; recent reviews avg < 3 stars; alert keywords (`health`, `scare`, `inspection`) | (hidden food-safety) |
| `cost_inflation` | Supplier catalog prices > stored snapshot * 1.10; alert keywords (`inflation`, `price increase`, `cost rise`) | (hidden inflation) |
| `unknown_anomaly` | An alert text doesn't match any keyword set, OR a key metric (cash burn rate, walkout band, rep drop, delivery shortfall) shifts >2σ from EMA without a named regime explaining it | (any hidden) |
| `baseline` | implicit — all named regimes have severity < 0.2 | `baseline` |

Severities are clamped to [0,1]. A `tier` helper maps to discrete bands: `off` (<0.2), `low` (0.2–0.45), `med` (0.45–0.7), `high` (>0.7) — these are what rule sets consume.

### Flexibility for unseen scenarios

Three layers of generalization to the 6 hidden scenarios:
1. **Regimes describe underlying disturbances**, not named stories. A hidden "ingredient recall" scenario manifests as `supply_crisis` severity ↑ via delivery shortfalls. A hidden "competitor opens nearby" scenario manifests as `demand_drought` severity ↑ via covers / EMA ratio.
2. **`unknown_anomaly` regime** fires when signals diverge from expectation without a named regime claiming credit. When this regime is the top severity, the LLM oversight call switches modes: it gets the full anomaly context (recent observations diff, alert text) and is asked to either (a) re-assign severity to a named regime if it now sees the pattern, or (b) recommend a conservative posture (defensive ordering, no marketing splurges, hold pricing). The deterministic fallback is "act baseline-conservative".
3. **Keyword sets are extensible.** Each regime's classifier reads its keyword list from a module-level constant, so a quick edit covers new alert wording observed during eval runs without touching logic.

---

## Architecture

```
[ Observation ]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. State load + EMA update + outage detection               │
│    (reuse luigi: load_state, update_emas, detect_outages)   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Deterministic regime classifier                          │
│    classify_regimes(obs, state) -> {regime: severity}       │
│    — per-regime scorer function, each emits 0.0–1.0         │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM oversight (daily, ~500 tokens)                       │
│    Input: deterministic severities, last 3 days of regime   │
│    history, key signals, recent alerts.                     │
│    Output JSON: {regime_overrides, tier_choices}            │
│    Validate; fall back to deterministic on parse/error.     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Rule-set composer                                        │
│    For each decision domain, pick the OWNER regime (the     │
│    highest-severity regime among that domain's priority     │
│    list) and apply its rule set at the chosen tier.         │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Translate to validated tool calls                        │
│    Reuse luigi's order builder + staffing helpers.          │
│    Add price/marketing/HH/special calls from rule sets.     │
│    Save notes (regime history + EMAs).                      │
└─────────────────────────────────────────────────────────────┘
```

### Per-decision regime ownership

Each decision domain has a fixed priority list. The owner is the first regime in the list whose severity ≥ low-tier threshold; falls through to baseline if none.

| Domain | Priority order |
|---|---|
| `inventory_orders` | `supply_crisis` → `cost_inflation` → `tourist` → `capacity_limited` → `demand_drought` → baseline |
| `staffing` | `capacity_limited` → `reputation_crisis` → `tourist` → `demand_drought` → baseline |
| `pricing` | `cost_inflation` → `reputation_crisis` → `tourist` → `demand_drought` → baseline |
| `marketing` | `capacity_limited` (suppresses) → `reputation_crisis` → `tourist` (suppresses; we're already full) → `demand_drought` → baseline |
| `happy_hour` | `reputation_crisis` → `demand_drought` → `supply_crisis` (suppresses) → `tourist` (suppresses) → baseline |
| `daily_special` | `supply_crisis` (use abundant ingredient) → `reputation_crisis` (top dish) → baseline |
| `menu` | `supply_crisis` (drop dishes whose ingredients are constrained) → `capacity_limited` (lean menu) → baseline |

"Suppresses" means: if that regime is the owner, the action is *off* regardless of tier.

### Handling co-severe regimes (synthesis path)

The default per-domain ownership rule is **predictable and cheap** but ignores secondary regimes. When this is too crude, a synthesis path kicks in:

**Trigger:** ≥2 named regimes simultaneously have severity ≥ 0.5.

**Behavior:** the daily LLM oversight call switches to a richer prompt that includes:
- The full severity table.
- The rule-set summary for each co-severe regime (1–2 sentences each, hard-coded strings).
- The per-domain owner under default rules.
- Explicit ask: "for each domain, either confirm the default owner's tier OR specify a synthesized override (e.g., 'use supply_crisis ordering but with tourist's larger order qty'). Output JSON only."

The synthesis output schema extends the standard tier_choices with optional `domain_overrides: {domain: {regime: regime_name, tier: tier_name, note: string}}`. The validation layer accepts overrides only if the named regime is among the active regimes and the tier is in {off,low,med,high}. Invalid overrides drop silently and the default owner applies.

This keeps the architecture predictable (default ownership) but admits LLM-driven combination exactly when it matters (genuine multi-shock days). Cost stays roughly the same since this replaces the standard oversight call, not adds to it.

### Rule-set knobs (LLM-tunable, discrete tiers)

Each rule set exposes 1–3 knobs that the LLM may set to low/med/high (default: derived from severity tier). Code maps tier → number; out-of-range → ignored.

| Rule set | Knobs |
|---|---|
| `supply_crisis.orders` | `safety_days_tier`, `supplier_diversify_tier` |
| `tourist.staffing` | `extra_staff_tier` |
| `reputation_crisis.pricing` | `discount_tier` |
| `cost_inflation.pricing` | `markup_tier` |
| `demand_drought.marketing` | `spend_tier` |
| (etc.) | |

---

## File layout: `agents/regime_agent.py`

Single file, ~600–700 lines. Section order:

1. **Imports + constants** — DOW, weather/rep maps (reuse from luigi).
2. **State helpers** — `load_state`, `update_emas`, `detect_outages` (copy from luigi, add `regime_history` and `supplier_price_snapshot` fields to `DEFAULT_STATE`).
3. **Regime classifiers** — one scorer per regime, each returning 0.0–1.0 from `(obs, state)`. Module-level functions for testability.
4. **Tier helper** — `tier(severity) -> "off"|"low"|"med"|"high"`.
5. **LLM oversight** — system prompt fixes the output JSON schema (`{"regime_overrides": {...}, "tier_choices": {...}}`); validation clamps overrides to [0,1] and rejects unknown tier values.
6. **Rule sets** — one function per (regime, domain) pair. Each takes `(obs, state, tier, knobs)` and returns either action dicts or parameters consumed by the composer.
7. **Composer** — `pick_owner(domain, severities)`; `apply_rules(domain, owner, obs, state, knobs)`.
8. **Strategy** — `strategy(observation, day)` that wires everything and emits the action list, ending with `save_notes`.

### Files referenced / reused

| What | Source |
|---|---|
| EMA burn tracking | [luigi_agent.py:90-136](agents/luigi_agent.py#L90-L136) |
| Discrete-calendar `days_until_next_delivery` | [luigi_agent.py:166-185](agents/luigi_agent.py#L166-L185) |
| Outage detection | [luigi_agent.py:139-160](agents/luigi_agent.py#L139-L160) |
| Order builder (adapt with regime-aware `safety_days`) | [luigi_agent.py:229-393](agents/luigi_agent.py#L229-L393) |
| LLM call + JSON parsing pattern | [luigi_agent.py:492-523](agents/luigi_agent.py#L492-L523) |
| Alert-keyword detector (extend per regime) | [smart_rule.py:111-134](agents/smart_rule.py#L111-L134) |
| DOW staffing base | [smart_rule.py:10-13](agents/smart_rule.py#L10-L13) |
| Notes serializer with size cap | [smart_rule.py:29-54](agents/smart_rule.py#L29-L54) |

Helpers can be **copied** (single-file agent is the convention here) rather than imported, to keep the agent self-contained. Run module is [agents/runner.py](agents/runner.py).

### Notes JSON layout (≤3800 chars)

```json
{
  "ema_burn": {"<ing>": kg/day, ...},
  "peak_burn": {"<ing>": kg/day, ...},
  "covers_ema": 0.0, "revenue_ema": 0.0,
  "consec_hh": 0, "last_hh_day": -10,
  "outage_suppliers": [],
  "regime_history": [
    {"d": 7, "r": {"tourist": 0.7, "supply_crisis": 0.4}}
  ],          // last 7 days
  "supplier_price_snapshot": {"<sup>:<ing>": price},   // for inflation detection
  "delivery_shortfall_log": [{"d": 5, "sup": "...", "frac": 0.4}]   // last 5
}
```

---

## Observability

Critical for local testing — we need to be able to look at any day of any run and reconstruct *why* the agent did what it did.

Three layers:

1. **Default stdout (always on, terse).** One line per day, prefixed with `[regime]`:
   ```
   [regime] d=7 (Sun) | tourist:0.72(H) supply_crisis:0.40(M) | own: inv=supply_crisis(M) staff=tourist(M) price=baseline mkt=tourist(H,suppr) | llm: ok (3 tier adj)
   ```
   Compact enough to scan a 30-day run; tells you regime severities, tiers, per-domain owner, LLM result. The runner streams stdout so this shows up live.

2. **REGIME_DEBUG=1 (verbose).** Same as above plus, for each day:
   - The full classifier breakdown (per regime: which signals contributed, raw score before clamping).
   - The full LLM input JSON + raw LLM output.
   - The full tool-call list emitted.
   - The state diff vs. yesterday (which EMAs moved, which knobs the LLM changed).

3. **REGIME_LOG=path.jsonl (machine-readable).** When set, writes one JSON object per day to the path:
   ```json
   {"day": 7, "dow": "Sun", "severities": {...}, "tiers": {...}, "owners": {...},
    "llm_input": {...}, "llm_output_raw": "...", "llm_output_parsed": {...},
    "actions": [...], "obs_signals": {"cash": ..., "rep": ..., ...}}
   ```
   Lets us post-mortem any run, diff across seeds, or feed analysis scripts later.

All three are mutually independent and zero-cost when off (default stdout is one f-string per day, the others are no-ops without env vars).

The agent's `notes` field (visible in every observation) also carries `regime_history` for the last 7 days, so a partial post-mortem is possible even without the JSONL file by reading the notes at game end.

---

## Verification

Run the existing evaluator against the 4 known scenarios with multiple seeds and compare avg score against the two existing baselines:

```bash
python -m agents.evaluate agents.regime_agent \
  --scenarios baseline,supply_crisis,tourist_season,renovation \
  --seeds 42,88,123 \
  --parallel 5
```

Target: average score ≥ `agents.luigi_agent` baseline; specifically beat smart_rule on `renovation` (where the existing per-scenario hack is brittle).

For debugging:
- Add a `--verbose` analogue that prints day-by-day regime severities and chosen owners (via stdout in `strategy`, since the runner streams it). Once stable, gate behind `os.getenv("REGIME_DEBUG")`.
- The `notes` field is observable in the per-turn observation, so after a run we can inspect the regime history without extra plumbing.
- Manual A/B: run `python -m agents.compare` (if present in [agents/compare.py](agents/compare.py)) between regime_agent and luigi_agent.

End-to-end smoke test before any tuning:
```bash
OPENAI_API_KEY=... python -m agents.regime_agent     # single game, seed 42
```
Confirm: agent survives 30 days, notes round-trip cleanly, no JSON parse failures spam the log.

---

## Risks and mitigations

1. **Regime detector misclassifies** an unseen scenario into the wrong bucket → LLM oversight catches it daily; even if LLM is offline, the worst case is the deterministic baseline behavior (which is what smart_rule does today).
2. **Composition produces conflicting actions** (e.g., `supply_crisis` says cut menu, `tourist` says push specials) → fixed per-domain ownership eliminates this; the table above is authoritative.
3. **LLM hallucinates tiers** outside {low,med,high} → validation rejects and falls back to severity-derived tier.
4. **Notes overflow 4 KB** → reuse smart_rule's `serialize_notes` truncation pattern, cap regime_history at 7 days.
5. **LLM is rate-limited or 5xx** → existing pattern in luigi (catch-all `except Exception`, return None, deterministic fallback applies).

---

## What's explicitly out of scope (Phase 1)

- No new heavy math (no XGBoost/K-Means despite the research report mentioning them). The whole point is staying rule-based.
- No multi-agent DAG — single-strategy function, single LLM call per day (synthesis path replaces, not adds).
- No changes to luigi/smart_rule/runner/evaluator.
- **No deep per-regime tuning.** Rule sets start as thin stubs (luigi-style defaults). The user's existing tuned playbooks (e.g., tourist-season) get folded in during Phase 2.
