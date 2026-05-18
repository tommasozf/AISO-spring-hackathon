# Hybrid Agent Decision Log

### Architecture
- **Approach**: Hybrid Agent combining Rule-Based Logistics and LLM-driven Executive Brain.
- **Logistics Core**: Pure Python arithmetic calculating inventory lookahead based on Exponential Moving Average (EMA) of ingredient burn rates.
- **Executive Brain**: Uses `claude-3-5-sonnet` via `litellm`. Responsible for pricing, menu changes, marketing, and promotions.
- **State Persistence**: Uses the `save_notes` tool to store a stringified JSON containing EMA burn rates, strategy state, and other persistent data between turns.

### Parameters
- **LLM Model**: `claude-3-5-sonnet-20241022`
- **Initial Staff Level**: 9
- **Safety Stock Threshold**: Target 3 days of stock for expected burn.
- **Menu Size**: Start with all active dishes, allow LLM to prune.

### Pivot Log
- *Initial creation based on user plan and hackathon cheatsheet.*
