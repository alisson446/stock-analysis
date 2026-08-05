# Filtro de Crescimento Projetado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coletar o crescimento projetado por analistas para receita e lucro, exibi-lo no screener e permitir filtrar por ele antes do valuation.

**Architecture:** A coleta entra no loop existente de `_fetch_fundamentals_from_api`, reaproveitando o objeto `yf.Ticker` já criado — 1 requisição HTTP extra por ticker, porque `revenue_estimate` e `earnings_estimate` saem do mesmo módulo `earningsTrend` do quoteSummary e ficam em cache no objeto. Três colunas novas entram no `data/fundamentals.csv`. A filtragem vira uma função auxiliar `_growth_mask` em `src/filters.py`, compartilhada por `apply_stock_filters` e `apply_bank_filters`, controlada por cinco chaves novas nos dois blocos de `config/filters.json`.

**Tech Stack:** Python 3.14, pandas, numpy, yfinance 1.2.0, pytest, Jupyter.

**Spec:** `docs/superpowers/specs/2026-08-05-filtro-crescimento-projetado-design.md`

## Global Constraints

- Período de estimativa: **`+1y`** (próximo exercício fiscal). É o horizonte mais longo que a Yahoo entrega — `LTG` retorna NaN para a ação em todos os tickers testados.
- **Não usar `growth_estimates`** para os dados novos: dispara uma segunda requisição HTTP para `industryTrend`/`sectorTrend`/`indexTrend`, que o projeto não consome. Usar `revenue_estimate` e `earnings_estimate`, que já trazem a coluna `growth`.
- Valores de crescimento em **pontos percentuais** (ex.: `12.5` para 12,5%), seguindo a convenção de `roe_pct` e `margem_liquida_pct`. A Yahoo devolve decimal; multiplicar por 100.
- Cortes de crescimento usam comparação **estrita** (`>`), como os demais critérios `_min` do projeto. `num_analistas_min` usa **`>=`**, porque é contagem.
- As duas flags são **independentes**. Nenhuma altera o significado da outra. Flag desligada = critério não aplicado.
- Qualquer falha na coleta de estimativas resulta em **NaN**, nunca em exceção que interrompa o loop de 372 tickers.
- Não alterar `src/valuation.py` nem `get_forward_growth`. O valuation sai desta entrega com resultado idêntico.
- Nenhuma dependência nova. Nada além do que já está em `requirements.txt`.
- Rodar testes com `python3 -m pytest` (o comando `python` não existe neste ambiente).

---

### Task 1: Extração das estimativas em `fundamentals.py`

**Files:**
- Modify: `src/fundamentals.py` (adicionar após `compute_ttm_net_income`, que termina na linha 94)
- Test: `tests/test_fundamentals.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: `_extract_growth_estimates(stock) -> tuple[float, float, float]`, devolvendo `(crescimento_receita_pct, crescimento_lucro_pct, num_analistas)` nessa ordem, todos `float` e podendo ser `np.nan`. Usada pela Task 2.

- [ ] **Step 1: Escrever os testes que falham**

Adicionar ao final de `tests/test_fundamentals.py`. O arquivo já tem os imports necessários (`numpy as np`, `pandas as pd`, `pytest`, `from src import fundamentals as f`) — não duplicar.

```python
class _FakeEstimateTicker:
    """Stub de yf.Ticker cujos frames de estimativa são controlados no teste."""

    def __init__(self, revenue=None, earnings=None, raises=False):
        self._revenue = revenue
        self._earnings = earnings
        self._raises = raises

    @property
    def revenue_estimate(self):
        if self._raises:
            raise RuntimeError('falha de rede')
        return self._revenue

    @property
    def earnings_estimate(self):
        if self._raises:
            raise RuntimeError('falha de rede')
        return self._earnings


def _revenue_frame(rows):
    """rows: {periodo: growth_decimal}"""
    return pd.DataFrame(
        {'growth': list(rows.values())},
        index=list(rows.keys()),
    )


def _earnings_frame(rows):
    """rows: {periodo: (growth_decimal, num_analistas)}"""
    return pd.DataFrame(
        {
            'growth': [v[0] for v in rows.values()],
            'numberOfAnalysts': [v[1] for v in rows.values()],
        },
        index=list(rows.keys()),
    )


class TestExtractGrowthEstimates:

    def test_reads_next_year_and_converts_to_percentage_points(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.0256, '+1y': 0.1693}),
            earnings=_earnings_frame({'0y': (0.0084, 11), '+1y': (0.2057, 11)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(16.93)
        assert lucro == pytest.approx(20.57)
        assert analistas == 11

    def test_ignores_current_year_and_quarterly_periods(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0q': 0.9, '+1q': 0.8, '0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0q': (0.9, 6), '+1y': (0.10, 4)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert analistas == 4

    def test_returns_nan_when_next_year_row_missing(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.05}),
            earnings=_earnings_frame({'0y': (0.05, 3)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert np.isnan(receita)
        assert np.isnan(lucro)
        assert np.isnan(analistas)

    def test_returns_nan_for_empty_frames(self):
        stock = _FakeEstimateTicker(
            revenue=pd.DataFrame(),
            earnings=pd.DataFrame(),
        )

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_when_frames_are_none(self):
        stock = _FakeEstimateTicker(revenue=None, earnings=None)

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_on_exception(self):
        stock = _FakeEstimateTicker(raises=True)

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_for_nan_cell(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': np.nan}),
            earnings=_earnings_frame({'+1y': (np.nan, np.nan)}),
        )

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_revenue_available_without_earnings(self):
        """Caso real: CYRE4 e IGTI3 têm receita mas não lucro."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.1594}),
            earnings=pd.DataFrame(),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(15.94)
        assert np.isnan(lucro)
        assert np.isnan(analistas)
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `python3 -m pytest tests/test_fundamentals.py::TestExtractGrowthEstimates -v`
Expected: FAIL — `AttributeError: module 'src.fundamentals' has no attribute '_extract_growth_estimates'`

- [ ] **Step 3: Implementar**

Inserir em `src/fundamentals.py`, logo após a função `compute_ttm_net_income` (que termina na linha 94) e antes de `fetch_betas`:

```python
# Período das estimativas de analistas. A Yahoo só entrega '0q', '+1q', '0y'
# e '+1y' — o '+1y' (próximo exercício) é o horizonte mais longo disponível.
# A linha 'LTG' existe no schema mas vem NaN para a ação em todos os tickers
# testados, BR e US.
_ESTIMATE_PERIOD = '+1y'


def _estimate_cell(frame, column: str) -> float:
    """
    Lê uma célula da linha '+1y' de um frame de estimativas do yfinance.

    Defensiva por design: layout e cobertura variam entre tickers e versões
    do yfinance, e uma exceção aqui interromperia a coleta dos 372 tickers.
    """
    try:
        if frame is None or frame.empty or _ESTIMATE_PERIOD not in frame.index:
            return np.nan
        value = frame.loc[_ESTIMATE_PERIOD, column]
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


def _extract_growth_estimates(stock) -> tuple[float, float, float]:
    """
    Crescimento projetado por analistas para o próximo exercício.

    Usa `revenue_estimate` e `earnings_estimate`, que saem do mesmo módulo
    `earningsTrend` do quoteSummary e ficam em cache no objeto Ticker: acessar
    as duas custa 1 requisição HTTP. Não usar `growth_estimates`, que dispara
    uma segunda requisição para buscar dados de índice que não consumimos.

    O número de analistas vem do `earnings_estimate` — o `revenue_estimate` não
    expõe contagem — e governa as duas métricas de crescimento.

    Args:
        stock: objeto `yf.Ticker` já construído.

    Returns:
        (crescimento_receita_pct, crescimento_lucro_pct, num_analistas), em
        pontos percentuais. Qualquer campo indisponível vem como NaN.
    """
    try:
        revenue = stock.revenue_estimate
        earnings = stock.earnings_estimate
    except Exception:
        return np.nan, np.nan, np.nan

    return (
        _estimate_cell(revenue, 'growth') * 100,
        _estimate_cell(earnings, 'growth') * 100,
        _estimate_cell(earnings, 'numberOfAnalysts'),
    )
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_fundamentals.py -v`
Expected: PASS — os 8 testes novos e todos os que já existiam.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals.py tests/test_fundamentals.py
git commit -m "feat: extrai crescimento projetado de receita e lucro do yfinance"
```

---

### Task 2: Coleta no loop de `_fetch_fundamentals_from_api`

**Files:**
- Modify: `src/fundamentals.py:301-327` (dict de sucesso), `src/fundamentals.py:331-344` (dict de fallback do `except`)
- Test: nenhum. A função faz rede em loop sobre 372 tickers e não é testável sem reescrevê-la; a lógica testável já está isolada na Task 1.

**Interfaces:**
- Consumes: `_extract_growth_estimates(stock)` da Task 1.
- Produces: as colunas `crescimento_receita_pct`, `crescimento_lucro_pct` e `num_analistas` em `data/fundamentals.csv`. Consumidas pelas Tasks 4 e 5.

- [ ] **Step 1: Chamar a extração dentro do loop**

Em `src/fundamentals.py`, localizar o bloco que calcula `pe_ratio` (linhas 297-299) e inserir logo depois dele, antes do `records.append({`:

```python
            # Estimativas de analistas para o próximo exercício. Reaproveita o
            # objeto `stock` já criado: 1 requisição HTTP a mais por ticker.
            crescimento_receita_pct, crescimento_lucro_pct, num_analistas = (
                _extract_growth_estimates(stock)
            )
```

- [ ] **Step 2: Adicionar as três chaves ao registro de sucesso**

No dict de `records.append({...})` das linhas 301-327, adicionar após `'dividend_rate': dividend_rate,`:

```python
                'crescimento_receita_pct': crescimento_receita_pct,
                'crescimento_lucro_pct': crescimento_lucro_pct,
                'num_analistas': num_analistas,
```

- [ ] **Step 3: Adicionar as três chaves ao registro de fallback**

No `except Exception as e:` das linhas 329-344, a lista de chaves preenchidas com NaN termina em `'dividend_rate',`. Adicionar as três logo após:

```python
                    'dividend_rate', 'crescimento_receita_pct',
                    'crescimento_lucro_pct', 'num_analistas',
```

- [ ] **Step 4: Verificar a coleta ponta a ponta em 3 tickers**

Run:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.fundamentals import _fetch_fundamentals_from_api
df = _fetch_fundamentals_from_api(['WEGE3.SA', 'ITUB4.SA', 'GRND3.SA'], delay=0.3)
print(df[['ticker', 'crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas']])
"
```

Expected: três linhas. WEGE3 e ITUB4 com valores em pontos percentuais (WEGE3 na ordem de 16,9% de receita e 20,6% de lucro, com ~11 analistas; os valores mudam conforme o consenso é revisado). GRND3 com NaN nas três colunas — não tem cobertura de analistas.

⚠️ Este comando **sobrescreve `data/fundamentals.csv` com apenas 3 tickers**. Antes de rodar, salvar o cache atual: `cp data/fundamentals.csv data/fundamentals.csv.bak`. Depois de conferir a saída, restaurar: `mv data/fundamentals.csv.bak data/fundamentals.csv`.

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals.py
git commit -m "feat: coleta crescimento projetado no fetch de fundamentals"
```

---

### Task 3: Parâmetros em `config/filters.json`

**Files:**
- Modify: `config/filters.json`
- Test: nenhum. É configuração de dados; a leitura é exercitada pelos testes da Task 4.

**Interfaces:**
- Consumes: nada.
- Produces: as chaves `crescimento_receita_pct_min`, `crescimento_lucro_pct_min`, `num_analistas_min`, `exigir_num_analistas` e `exigir_estimativa` em `stock_filters` e em `bank_filters`. Lidas pela Task 4.

- [ ] **Step 1: Substituir o arquivo inteiro**

```json
{
  "stock_filters": {
    "pl_min": 0,
    "pl_max": 10,
    "pvp_min": 0,
    "pvp_max": 1.5,
    "margem_ebit_pct_min": 0,
    "margem_liquida_pct_min": 10,
    "dl_ebit_max": 3,
    "dl_pl_max": 2,
    "roe_pct_min": 10,
    "liquidez_corrente_min": 1,
    "passivos_ativos_max": 1,
    "liq_media_diaria_min": 100000,
    "lpa_min": 0,
    "crescimento_receita_pct_min": 0,
    "crescimento_lucro_pct_min": 0,
    "num_analistas_min": 2,
    "exigir_num_analistas": false,
    "exigir_estimativa": false
  },
  "bank_filters": {
    "pl_min": 0,
    "pl_max": 10,
    "pvp_min": 0,
    "pvp_max": 2.0,
    "roe_pct_min": 15,
    "margem_liquida_pct_min": 10,
    "lpa_min": 0,
    "liq_media_diaria_min": 100000,
    "dy_pct_min": 3,
    "crescimento_receita_pct_min": 0,
    "crescimento_lucro_pct_min": 0,
    "num_analistas_min": 2,
    "exigir_num_analistas": false,
    "exigir_estimativa": false
  }
}
```

As duas flags nascem desligadas: o screening sai desta entrega com o mesmo resultado de hoje, e os cortes entram quando o usuário quiser. Os limiares ficam configurados ao lado, preservados para quando a flag for religada.

- [ ] **Step 2: Verificar que o JSON é válido e carrega**

Run: `python3 -c "import sys; sys.path.insert(0,'.'); from src.filters import _load_config; c=_load_config(); print(c['stock_filters']['exigir_estimativa'], c['bank_filters']['num_analistas_min'])"`
Expected: `False 2`

- [ ] **Step 3: Commit**

```bash
git add config/filters.json
git commit -m "feat: adiciona parametros de crescimento projetado aos filtros"
```

---

### Task 4: `_growth_mask` e integração nos dois filtros

**Files:**
- Modify: `src/filters.py`
- Create: `tests/test_filters.py`

**Interfaces:**
- Consumes: as colunas produzidas na Task 2 e as chaves de config da Task 3.
- Produces: `_growth_mask(df: pd.DataFrame, cfg: dict) -> pd.Series` (máscara booleana indexada como `df`), e o comportamento novo de `apply_stock_filters` / `apply_bank_filters`.

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/test_filters.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import filters


def _cfg(exigir_estimativa=False, exigir_num_analistas=False,
         receita_min=0, lucro_min=0, analistas_min=2):
    return {
        'crescimento_receita_pct_min': receita_min,
        'crescimento_lucro_pct_min': lucro_min,
        'num_analistas_min': analistas_min,
        'exigir_num_analistas': exigir_num_analistas,
        'exigir_estimativa': exigir_estimativa,
    }


def _df(rows):
    """rows: lista de (crescimento_receita_pct, crescimento_lucro_pct, num_analistas)"""
    return pd.DataFrame(
        rows,
        columns=['crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas'],
    )


class TestGrowthMaskBothFlagsOn:
    """Com as duas flags ligadas, ambos os critérios valem em conjunto."""

    def _mask(self, rows):
        return filters._growth_mask(
            _df(rows),
            _cfg(exigir_estimativa=True, exigir_num_analistas=True),
        )

    def test_growth_above_cuts_with_enough_analysts_passes(self):
        assert self._mask([(15.9, 21.9, 10)]).tolist() == [True]

    def test_negative_earnings_growth_fails(self):
        assert self._mask([(14.8, -2.4, 7)]).tolist() == [False]

    def test_negative_revenue_growth_fails_despite_positive_earnings(self):
        assert self._mask([(-4.6, 12.0, 7)]).tolist() == [False]

    def test_zero_growth_fails_strict_comparison(self):
        assert self._mask([(0.0, 5.0, 7)]).tolist() == [False]

    def test_missing_earnings_estimate_fails(self):
        assert self._mask([(9.5, np.nan, np.nan)]).tolist() == [False]

    def test_missing_revenue_estimate_fails(self):
        assert self._mask([(np.nan, 12.0, 5)]).tolist() == [False]

    def test_analysts_equal_to_minimum_passes(self):
        assert self._mask([(11.0, 30.0, 2)]).tolist() == [True]

    def test_analysts_below_minimum_fails_despite_good_growth(self):
        assert self._mask([(6.0, 12.8, 1)]).tolist() == [False]

    def test_nan_analysts_fails(self):
        assert self._mask([(11.4, 14.5, np.nan)]).tolist() == [False]


class TestGrowthMaskFlagsIndependent:
    """Cada flag liga somente o seu critério; nenhuma altera a outra."""

    def test_both_off_passes_everything(self):
        rows = [(-50.0, -80.0, 1), (np.nan, np.nan, np.nan)]
        mask = filters._growth_mask(_df(rows), _cfg())
        assert mask.tolist() == [True, True]

    def test_estimativa_off_ignores_negative_growth(self):
        mask = filters._growth_mask(
            _df([(14.8, -2.4, 7)]), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True]

    def test_analistas_off_ignores_analyst_count(self):
        mask = filters._growth_mask(
            _df([(6.0, 12.8, np.nan)]), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [True]

    def test_only_analysts_decide_when_estimativa_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._growth_mask(_df(rows), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True, False]

    def test_only_growth_decides_when_analistas_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._growth_mask(_df(rows), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [False, True]


class TestGrowthMaskThresholds:

    def test_custom_thresholds_are_respected(self):
        mask = filters._growth_mask(
            _df([(9.0, 9.0, 5), (11.0, 11.0, 5)]),
            _cfg(exigir_estimativa=True, receita_min=10, lucro_min=10),
        )
        assert mask.tolist() == [False, True]

    def test_mask_preserves_dataframe_index(self):
        df = _df([(15.0, 15.0, 5), (15.0, 15.0, 5)])
        df.index = [7, 42]
        mask = filters._growth_mask(df, _cfg(exigir_estimativa=True))
        assert mask.index.tolist() == [7, 42]
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `python3 -m pytest tests/test_filters.py -v`
Expected: FAIL — `AttributeError: module 'src.filters' has no attribute '_growth_mask'`

- [ ] **Step 3: Implementar `_growth_mask`**

Em `src/filters.py`, inserir após `_load_config` (que termina na linha 11) e antes de `apply_stock_filters`:

```python
def _growth_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Máscara de crescimento projetado por analistas.

    Duas flags independentes, cada uma ligando o seu próprio critério:
    `exigir_estimativa` aplica os cortes de crescimento de receita e lucro;
    `exigir_num_analistas` aplica o mínimo de analistas. Nenhuma altera o
    significado da outra, e flag desligada significa critério não aplicado.

    Com a flag ligada, valor NaN reprova: sem dado não há como atestar o
    critério, e é justamente isso que a flag exige. Comparações do pandas com
    NaN já retornam False, então esse comportamento sai de graça.

    Args:
        df: DataFrame com as colunas `crescimento_receita_pct`,
            `crescimento_lucro_pct` e `num_analistas`.
        cfg: bloco de configuração (`stock_filters` ou `bank_filters`).

    Returns:
        Série booleana indexada como `df`.
    """
    mask = pd.Series(True, index=df.index)

    if cfg.get('exigir_estimativa'):
        mask &= (
            (df['crescimento_receita_pct'] > cfg['crescimento_receita_pct_min']) &
            (df['crescimento_lucro_pct'] > cfg['crescimento_lucro_pct_min'])
        )

    # '>=' porque é contagem: num_analistas_min = 2 significa "pelo menos dois".
    if cfg.get('exigir_num_analistas'):
        mask &= df['num_analistas'] >= cfg['num_analistas_min']

    return mask
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_filters.py -v`
Expected: PASS — 16 testes.

- [ ] **Step 5: Integrar em `apply_stock_filters`**

Substituir o corpo de `apply_stock_filters` (linhas 14-37) por:

```python
def apply_stock_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios fundamentalistas para ações não-bancárias.
    Os limites são lidos de config/filters.json (chave 'stock_filters').
    """
    cfg = _load_config()['stock_filters']

    mask = (
        (df['pl'] > cfg['pl_min']) & (df['pl'] <= cfg['pl_max']) &
        (df['pvp'] > cfg['pvp_min']) & (df['pvp'] <= cfg['pvp_max']) &
        (df['margem_ebit_pct'] > cfg['margem_ebit_pct_min']) &
        (df['margem_liquida_pct'] > cfg['margem_liquida_pct_min']) &
        (df['dl_ebit'] < cfg['dl_ebit_max']) &
        (df['dl_pl'] < cfg['dl_pl_max']) &
        (df['roe_pct'] > cfg['roe_pct_min']) &
        (df['liquidez_corrente'] > cfg['liquidez_corrente_min']) &
        (df['passivos_ativos'] < cfg['passivos_ativos_max']) &
        (df['liq_media_diaria'] > cfg['liq_media_diaria_min']) &
        (df['lpa'] > cfg['lpa_min'])
    )

    growth = _growth_mask(df, cfg)
    filtered = df[mask & growth].copy().reset_index(drop=True)

    # Quantas passaram nos fundamentos e caíram só pelo crescimento projetado
    por_crescimento = int((mask & ~growth).sum())
    print(f"[filters] Ações: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_crescimento} reprovadas por crescimento projetado)")
    return filtered
```

- [ ] **Step 6: Integrar em `apply_bank_filters`**

Substituir o corpo de `apply_bank_filters` (linhas 40-59) por:

```python
def apply_bank_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios de screening adaptados para bancos.
    Os limites são lidos de config/filters.json (chave 'bank_filters').
    """
    cfg = _load_config()['bank_filters']

    mask = (
        (df['pl'] > cfg['pl_min']) & (df['pl'] <= cfg['pl_max']) &
        (df['pvp'] > cfg['pvp_min']) & (df['pvp'] <= cfg['pvp_max']) &
        (df['roe_pct'] > cfg['roe_pct_min']) &
        (df['margem_liquida_pct'] > cfg['margem_liquida_pct_min']) &
        (df['lpa'] > cfg['lpa_min']) &
        (df['liq_media_diaria'] > cfg['liq_media_diaria_min']) &
        (df['dy_pct'] > cfg['dy_pct_min'])
    )

    growth = _growth_mask(df, cfg)
    filtered = df[mask & growth].copy().reset_index(drop=True)

    por_crescimento = int((mask & ~growth).sum())
    print(f"[filters] Bancos: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_crescimento} reprovados por crescimento projetado)")
    return filtered
```

- [ ] **Step 7: Verificar contra o cache real**

Este passo depende de `data/fundamentals.csv` já ter as colunas novas. Se o cache ainda for o antigo, apagá-lo e recoletar (~24 min) ou pular a verificação até a Task 5.

Run:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src import filters
df = pd.read_csv('data/fundamentals.csv')
for c in df.columns:
    if c not in ('ticker','ticker_sa','nome','setor','industria'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
filters.apply_stock_filters(df)
"
```

Expected: com as flags desligadas no config, `28/372 passaram nos critérios (0 reprovadas por crescimento projetado)` — mesmo total de antes desta entrega.

- [ ] **Step 8: Rodar a suíte inteira**

Run: `python3 -m pytest tests/ -v`
Expected: PASS em tudo, incluindo `test_valuation.py`, que não foi tocado.

- [ ] **Step 9: Commit**

```bash
git add src/filters.py tests/test_filters.py
git commit -m "feat: filtra screening por crescimento projetado de receita e lucro"
```

---

### Task 5: Colunas no `analysis.ipynb`

**Files:**
- Modify: `analysis.ipynb` — células `69223fb5` (screening de ações), `6cb63ddc` (bancos), `96a4302e` (markdown de critérios)
- Test: execução do notebook.

**Interfaces:**
- Consumes: as colunas da Task 2 e o comportamento de filtro da Task 4.
- Produces: nada consumido por tasks posteriores.

Usar a ferramenta NotebookEdit para editar as células, não Edit — o `.ipynb` é JSON e edição textual corrompe o arquivo.

- [ ] **Step 1: Sanitizar as colunas novas na célula `69223fb5`**

Na lista `numeric_cols`, a última linha é `'dividend_rate'`. Substituir por:

```python
    'dividend_rate', 'crescimento_receita_pct', 'crescimento_lucro_pct',
    'num_analistas'
```

- [ ] **Step 2: Exibir as colunas novas na mesma célula `69223fb5`**

Em `display_cols`, substituir `'liquidez_corrente', 'passivos_ativos', 'lpa']` por:

```python
                    'liquidez_corrente', 'passivos_ativos', 'lpa',
                    'crescimento_receita_pct', 'crescimento_lucro_pct']
```

E no dict de `.style.format(...)`, adicionar após `'lpa': 'R$ {:.2f}',`:

```python
        'crescimento_receita_pct': '{:.1f}%', 'crescimento_lucro_pct': '{:.1f}%',
```

`num_analistas` não é exibida: é insumo do filtro, não métrica de decisão.

- [ ] **Step 3: Exibir as colunas na tabela de bancos, célula `6cb63ddc`**

Em `val_cols`, substituir `'undervalued', 'forte_desconto']` por:

```python
                'undervalued', 'forte_desconto',
                'crescimento_receita_pct', 'crescimento_lucro_pct']
```

E no `.style.format(...)`, adicionar após `'margem_seg_graham_pct': '{:.1f}%',`:

```python
        'crescimento_receita_pct': '{:.1f}%',
        'crescimento_lucro_pct': '{:.1f}%',
```

`valued_banks` vem de `valuation.apply_valuation(filtered_banks, ...)`, que faz `df = df.copy()` em `src/valuation.py:493` e só acrescenta colunas — as colunas de entrada chegam intactas ao resultado, então `val_cols` funciona.

- [ ] **Step 4: Atualizar a célula markdown `96a4302e`**

Na tabela "Critérios de Screening (Ações)", adicionar duas linhas ao final:

```markdown
| 12 | Cresc. Receita proj. (`+1y`) | > 0% (desligado por padrão) |
| 13 | Cresc. Lucro proj. (`+1y`) | > 0% (desligado por padrão) |
```

E trocar o cabeçalho `**Critérios de Screening (Ações — 11 critérios):**` por `**Critérios de Screening (Ações — 13 critérios, 2 opcionais):**`.

Adicionar um parágrafo após a tabela de bancos:

```markdown
**Crescimento projetado (opcional):** consenso de analistas via yfinance para o próximo exercício
(`+1y`) — o horizonte mais longo que a Yahoo entrega. Controlado em `config/filters.json` por
`exigir_estimativa` (aplica os cortes de crescimento) e `exigir_num_analistas` (aplica o mínimo de
analistas). Ambos nascem desligados. A cobertura é irregular em ações BR: 20 das 28 ações filtradas
têm estimativa de receita e 17 têm de lucro.
```

- [ ] **Step 5: Recoletar os fundamentals com as colunas novas**

O `data/fundamentals.csv` é cache descartável e precisa ser regenerado para conter as três colunas.

```bash
rm data/fundamentals.csv
```

Depois executar as células do notebook de cima até a de screening. A coleta leva ~24 min para 372 tickers (22 min de base mais ~2 min das estimativas).

- [ ] **Step 6: Conferir a saída**

Expected:
- A tabela de ações mostra `crescimento_receita_pct` e `crescimento_lucro_pct` formatadas como percentual, com NaN visível nas ações sem cobertura (GRND3, SHUL4, TECN3, VLID3, EUCA4, LPSB3, CSUD3, RSUL4 no cache anterior).
- O print de filtros termina com `(0 reprovadas por crescimento projetado)`, já que as flags estão desligadas.
- O total de ações filtradas é o mesmo de antes desta entrega.

- [ ] **Step 7: Sanity check ligando as flags**

Editar `config/filters.json`, pôr `"exigir_estimativa": true` e `"exigir_num_analistas": true` em `stock_filters`, e reexecutar apenas a célula de screening.

Expected: o total cai e o print reporta as reprovadas por crescimento. Sobre o cache medido em 2026-08-05 o resultado era 14 ações; os números mudam conforme o consenso é revisado, então o que se verifica é que o filtro morde e que o contador bate com a diferença.

Reverter as duas flags para `false` depois do teste.

- [ ] **Step 8: Commit**

```bash
git add analysis.ipynb config/filters.json
git commit -m "feat: exibe crescimento projetado no screener"
```

---

## Verificação final

- [ ] `python3 -m pytest tests/ -v` passa por inteiro
- [ ] `data/fundamentals.csv` tem as três colunas novas
- [ ] Com as flags desligadas, o screening devolve o mesmo total de antes da entrega
- [ ] Com as flags ligadas, o total cai e o contador de reprovadas bate com a diferença
- [ ] `git diff main --stat` não mostra alteração em `src/valuation.py`
