# Stock Analysis — Copilot Instructions

## Project Overview

Brazilian stock screener that filters Ibovespa stocks using fundamental analysis, calculates fair value (DCF + Graham), and ranks undervalued opportunities. Data is sourced from **dadosdemercado.com.br** (ticker list) and **yfinance** (fundamentals/financials).

## Architecture

```
analysis.ipynb          # Main notebook — orchestrates the full pipeline
src/
  scraper.py            # Scrapes tickers from dadosdemercado.com.br, classifies banks vs non-banks
  fundamentals.py       # Fetches fundamental data via yfinance (.info, .financials, .balance_sheet, .cashflow)
  filters.py            # Applies screening criteria (11 for stocks, 7 for banks)
  valuation.py          # 2-stage DCF, Excess Returns (banks), DDM fallback, Graham valuation
```

**Pipeline flow:** Scrape tickers → Fetch fundamentals → Classify banks/stocks → Apply filters → Calculate valuations (DCF/Excess Returns/DDM + Graham) → Merge & rank top 20 undervalued

## Key Conventions

- **Language:** Code and variable names are in Portuguese (e.g., `margem_ebit_pct`, `preco_justo_dcf`, `divida_liquida`). Keep this convention.
- **Ticker format:** Raw tickers are uppercase strings (`PETR4`); yfinance tickers use `.SA` suffix (`PETR4.SA`). Both are stored in DataFrames as `ticker` and `ticker_sa` columns.
- **DataFrame-centric:** Data flows as pandas DataFrames between all modules. Functions accept and return DataFrames.
- **Defensive extraction:** Use `_safe_get()` for yfinance `.info` dicts and `_extract_financial_value()`/`_extract_financial_series()` for financial statement DataFrames — yfinance labels vary across tickers.

## Valuation Models (inspired by Simply Wall St)

### Non-bank stocks: 2-Stage DCF + DDM fallback + Graham
- **DCF 2-Stage:** Growth rate decays linearly from historical CAGR → `TERMINAL_GROWTH` over `PROJECTION_YEARS` (10). Terminal value via Gordon Growth Model.
- **DDM Fallback:** When FCF data is unavailable, uses `dividend_rate / (SELIC - TERMINAL_GROWTH)` (Gordon Growth on dividends).
- **Graham:** `V = √(sector_avg_PE × sector_avg_PB × LPA × VPA)`

### Banks: Excess Returns Model + Graham
- **Excess Returns:** `FV = VPA + (ROE - SELIC) × VPA / (SELIC - TERMINAL_GROWTH)` — only produces a value when ROE > SELIC (bank generates excess returns).
- **Graham:** Same formula as stocks, using Financial Services sector averages.

### Undervalued criteria
- `undervalued = True` when price < primary fair value AND price < Graham fair value.
- `forte_desconto = True` when average safety margin ≥ `MIN_SAFETY_MARGIN_PCT` (20%).

## Valuation Parameters

Constants in `src/valuation.py` — update these when economic conditions change:

| Constant | Current Value | Meaning |
|---|---|---|
| `SELIC` | 0.1425 | Discount rate / cost of equity |
| `TERMINAL_GROWTH` | 0.035 | Long-term growth (Brazilian inflation target) |
| `PROJECTION_YEARS` | 10 | DCF projection horizon (2-stage) |
| `MAX_GROWTH_RATE` | 0.20 | Cap on FCF growth rate |
| `MIN_GROWTH_RATE` | 0.0 | Floor on FCF growth rate |
| `MIN_SAFETY_MARGIN_PCT` | 20.0 | Threshold for "forte desconto" flag |

## Screening Criteria

**Non-bank stocks** (11 criteria in `apply_stock_filters`):
P/L 0–10, P/PV 0–1.5, EBIT margin > 0, net margin > 10%, DL/EBIT < 3, DL/PL < 2, ROE > 10%, current ratio > 1, liabilities/assets < 1, daily liquidity > R$100k, EPS > 0.

**Banks** (7 criteria in `apply_bank_filters`):
P/L 0–10, P/PV 0–2.0, ROE > 15%, net margin > 10%, EPS > 0, daily liquidity > R$100k, DY > 3%.

## Development

```bash
pip install -r requirements.txt    # Install dependencies
```

Run cells sequentially in `analysis.ipynb`. The fundamentals fetch takes several minutes due to per-ticker yfinance API calls with rate-limiting delays.

## Common Pitfalls

- **yfinance label inconsistency:** Financial statement row labels (e.g., `'Free Cash Flow'` vs `'FreeCashFlow'`) vary between tickers. Always provide multiple label alternatives in extraction functions.
- **Rate limiting:** yfinance calls need `time.sleep()` delays to avoid HTTP 429 errors. Default is 0.5s per ticker.
- **Bank vs stock classification:** Uses both a hard-coded set (`KNOWN_BANK_TICKERS`) and yfinance sector/industry lookup. The hard-coded set is the fast path; update it when new bank tickers appear on Ibovespa.
- **NaN propagation:** Many metrics may be `NaN` for a given ticker. All calculations guard against division by zero and missing data.
