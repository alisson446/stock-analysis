---
description: "Use when modifying valuation logic, changing financial model parameters, adding new valuation methods, or debugging fair value calculations. Covers DCF 2-stage, Excess Returns, DDM fallback, and Graham valuation rules."
---
# Valuation Models

## Constants (src/valuation.py)

Update these when economic conditions change — never hardcode these values elsewhere:

| Constant | Current Value | Meaning |
|---|---|---|
| `SELIC` | 0.1425 | Discount rate / cost of equity |
| `TERMINAL_GROWTH` | 0.035 | Long-term growth (Brazilian inflation target) |
| `PROJECTION_YEARS` | 10 | DCF projection horizon |
| `MAX_PROJECTABLE_GROWTH` | 0.20 | Threshold above which a growth rate is not projectable — returns `NaN` (no DCF), never replaces the rate |
| `FORWARD_GROWTH_DRIVER` | `revenue` | Which forward growth feeds DCF stage 1: `revenue` or `earnings` (env) |
| `MIN_SAFETY_MARGIN_PCT` | 20.0 | Threshold for `forte_desconto` flag |

Always import constants from `src/valuation.py`. Never redefine `SELIC` or `TERMINAL_GROWTH` in other modules.

## Model Selection by Company Type

| Type | Primary Model | Fallback | Secondary |
|------|--------------|---------|-----------|
| Non-bank stocks | DCF 2-stage | DDM (when FCF unavailable) | Graham |
| Banks | Excess Returns | — (no DCF for banks) | Graham |

## DCF 2-Stage (Non-bank stocks)

**Requirements before computing DCF:**
- FCF base (most recent) must be **positive** — negative FCF → use DDM fallback
- `shares_outstanding` must be > 0

**Stage 1 — Linear decay:**
- Growth rate decays linearly from historical FCF CAGR → `TERMINAL_GROWTH` over `PROJECTION_YEARS` years
- Year 1 uses `initial_growth`; Year `PROJECTION_YEARS` uses `TERMINAL_GROWTH`

**Stage 2 — Terminal value (Gordon Growth Model):**
```
terminal_value = FCF_final × (1 + TERMINAL_GROWTH) / (SELIC - TERMINAL_GROWTH)
```

**FCF CAGR rules:**
- Only use positive FCF data points to compute CAGR
- Above `MAX_PROJECTABLE_GROWTH` → `NaN` (not projectable; caller falls back to DDM). Negative CAGR passes through unchanged — there is no floor.
- Fewer than 2 data points → CAGR = 0.0

**Output column:** `preco_justo_dcf`

## DDM Fallback (Non-bank stocks, FCF unavailable)

Only use DDM when FCF data is unavailable or negative. Formula (Gordon Growth on dividends):

```
preco_justo_ddm = dividend_rate / (SELIC - TERMINAL_GROWTH)
```

Requires `dividend_rate > 0`. If no dividends, output is `NaN` — do not force a value.

**Output column:** `preco_justo_primario` (shared with DCF; DDM fills in when DCF is `NaN`)

## Excess Returns Model (Banks only)

Only produces a fair value when **ROE > SELIC** (bank generates excess returns). Formula:

```
excess_return = (ROE - SELIC) × VPA
terminal_value = excess_return / (SELIC - TERMINAL_GROWTH)
preco_justo_er = VPA + terminal_value
```

When ROE ≤ SELIC → output is `NaN`. Never force a value for banks that don't generate excess returns.

**Input:** `roe_decimal` (e.g., `0.18` for 18%) and `vpa` (Value per Share in R$)

**Output column:** `preco_justo_primario`

## Graham Valuation (Both types)

Formula uses **sector averages** instead of Graham's fixed 22.5:

```
V = √(sector_avg_PE × sector_avg_PB × LPA × VPA)
```

- Requires `LPA > 0` and `VPA > 0`
- Requires valid `sector_avg_PE` and `sector_avg_PB` (not NaN, not 0)
- Banks use Financial Services sector averages

**Output column:** `preco_justo_graham`

## Undervalued Classification

A stock is `undervalued = True` when **both** conditions hold:
```
preco_atual < preco_justo_primario  AND  preco_atual < preco_justo_graham
```

A stock has `forte_desconto = True` when the **average** safety margin across both models is ≥ `MIN_SAFETY_MARGIN_PCT` (20%):
```
safety_margin_avg = mean(margem_seg_primario_pct, margem_seg_graham_pct)
forte_desconto = undervalued AND safety_margin_avg >= MIN_SAFETY_MARGIN_PCT
```

Safety margin is computed as:
```
margem_seg_pct = (preco_justo - preco_atual) / preco_justo × 100
```

## Output Columns

Every row produced by `calculate_valuations()` must contain:
- `preco_justo_primario` — DCF (stocks) or Excess Returns (banks), with DDM as stock fallback
- `preco_justo_graham`
- `margem_seg_primario_pct`
- `margem_seg_graham_pct`
- `undervalued` (bool)
- `forte_desconto` (bool)

Missing fair values must be `NaN`, never `0` or negative placeholders.
