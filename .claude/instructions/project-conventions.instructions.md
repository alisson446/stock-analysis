---
description: "Use when adding or modifying code in src/ files, creating new functions, naming variables, or working with yfinance data. Enforces Portuguese naming, DataFrame conventions, and safe yfinance data extraction patterns."
---
# Project Conventions

## Language

All code, variable names, column names, comments, and docstrings must be in **Portuguese**.

```python
# Correto
margem_ebit_pct = (ebit / receita_total) * 100
divida_liquida = total_debt - total_cash

# Errado
ebit_margin_pct = ...
net_debt = ...
```

## Ticker Format

- Raw ticker: uppercase string — `PETR4`
- yfinance ticker: with `.SA` suffix — `PETR4.SA`
- DataFrames always have both columns: `ticker` (raw) and `ticker_sa` (with `.SA`)

## DataFrame-Centric Flow

All module functions must accept a `pd.DataFrame` and return a `pd.DataFrame`. Never return raw dicts or lists as the final output of a public function.

## Column Naming Conventions

| Column | Meaning |
|--------|---------|
| `pl` | P/L (Preço/Lucro) |
| `pvp` | P/VP (Preço/Valor Patrimonial) |
| `lpa` | Lucro Por Ação |
| `vpa` | Valor Patrimonial por Ação |
| `roe_pct` | ROE em % |
| `dy_pct` | Dividend Yield em % |
| `margem_ebit_pct` | Margem EBIT em % |
| `margem_liquida_pct` | Margem Líquida em % |
| `dl_ebit` | Dívida Líquida / EBIT |
| `dl_pl` | Dívida Líquida / Patrimônio Líquido |
| `liq_media_diaria` | Liquidez Média Diária (R$) |
| `liquidez_corrente` | Liquidez Corrente |
| `passivos_ativos` | Passivos / Ativos |

Percentage columns always use the `_pct` suffix and are stored as actual percentage values (e.g., `10.5` means 10.5%, not 0.105).

## Safe yfinance Extraction

### For `.info` dicts — use `_safe_get()`

```python
# Correto
current_price = _safe_get(info, 'currentPrice')
roe = _safe_get(info, 'returnOnEquity')

# Errado — crashes on None
current_price = info['currentPrice']
```

### For financial statement DataFrames — use `_extract_financial_value()` / `_extract_financial_series()`

Always provide **multiple label alternatives** because yfinance row labels vary between tickers:

```python
ebit = _extract_financial_value(financials, ['EBIT', 'Ebit'])
total_revenue = _extract_financial_value(financials, [
    'Total Revenue', 'TotalRevenue', 'Operating Revenue'
])
```

### Rate Limiting

Always add `time.sleep(delay)` between per-ticker yfinance API calls. Default delay is `0.5s`. Never call yfinance in a tight loop without sleeping.

## NaN Guarding

Always check for `NaN` and zero before division. Never propagate unchecked `NaN` through calculations:

```python
# Correto
dl_ebit = (
    divida_liquida / ebit
    if pd.notna(divida_liquida) and pd.notna(ebit) and ebit != 0
    else np.nan
)

# Errado
dl_ebit = divida_liquida / ebit
```

## Bank Classification

Banks are identified by:
1. `KNOWN_BANK_TICKERS` hard-coded set (fast path) — update this set when new bank tickers appear on Ibovespa
2. yfinance `sector` / `industry` lookup (fallback)

Do not assume all financial companies are banks. Use the classification logic in `src/scraper.py`.

## Module Structure

| Module | Responsibility |
|--------|---------------|
| `scraper.py` | Scrape tickers, classify banks vs non-banks |
| `fundamentals.py` | Fetch fundamentals via yfinance, cache to CSV |
| `filters.py` | Apply screening criteria using thresholds from `config/filters.json` |
| `valuation.py` | Calculate DCF, Excess Returns, DDM, Graham valuations |

Each module does exactly one thing. Do not mix scraping with valuation logic, or filtering with data fetching.

## Filter Thresholds

Screening thresholds are defined in `config/filters.json`, **not** hardcoded in `filters.py`. Always read from the JSON config via `_load_config()`.
