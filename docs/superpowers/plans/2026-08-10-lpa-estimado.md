# LPA projetado no screener — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar a coluna `lpa_estimado` (nível de lucro por ação projetado para o próximo exercício) ao screener, e um critério de filtro sobre ela que barra papéis com prejuízo projetado.

**Architecture:** Nada novo é buscado na rede. O frame `earnings_estimate` já é lido por ticker para extrair `growth` e `numberOfAnalysts`; `lpa_estimado` é uma terceira leitura de célula (`avg`) do mesmo objeto em memória. A coluna viaja pelo `fundamentals.csv` até o filtro (`_growth_mask`, terceira flag independente), a exibição do notebook e o snapshot histórico. Nenhum modelo de valuation muda.

**Tech Stack:** Python 3.14, pandas, numpy, yfinance 1.2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-lpa-estimado-design.md`

## Global Constraints

- **Rodar pytest sempre com `rtk proxy`:** `rtk proxy python3 -m pytest tests/ -q`. O hook do RTK reescreve invocações diretas de `pytest` e a execução falha com `No such file or directory (os error 2)` — que parece dependência faltando, mas é o hook.
- **Baseline de testes antes de começar: 116 passando.** Cada task adiciona testes; nenhuma pode reduzir esse número.
- **Docstrings e comentários em português, registrando o *porquê*** — Guideline 5 de `docs/GUIDELINES.md`. Quando a decisão contraria o óbvio, o comentário nomeia o caso concreto (o código já faz isso com RSUL4, PETR4, RIAA3).
- **`lpa_estimado` é nível em R$ por ação, não percentual.** Nunca multiplicar por 100, nunca sufixo `_pct`.
- **Ordem da coluna, idêntica em todos os lugares:** imediatamente após `crescimento_lucro_pct` e antes de `num_analistas`.
- **O dado exibido é sempre o bruto** — Guideline 1. Nenhum clamp, arredondamento ou substituição na coleta ou na exibição.

## File Structure

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `src/fundamentals.py` | extrair `avg` do frame já carregado; gravar a coluna no CSV | Modificar |
| `tests/test_fundamentals.py` | cobrir a extração e os caminhos de ausência | Modificar |
| `config/filters.json` | limiar e flag, nos dois blocos | Modificar |
| `src/filters.py` | aplicar o critério em `_growth_mask` | Modificar |
| `tests/test_filters.py` | cobrir o critério e a independência entre flags | Modificar |
| `src/valuation.py` | carregar a coluna para o snapshot histórico | Modificar |
| `tests/test_valuation.py` | cobrir a coluna no snapshot | Modificar |
| `analysis.ipynb` | exibir a coluna nos 5 pontos; corrigir markdown desatualizado | Modificar |

Sem arquivos novos. A mudança segue o caminho que `crescimento_lucro_pct` já percorre.

---

### Task 1: Coleta do LPA projetado

**Files:**
- Modify: `src/fundamentals.py:119-148` (função), `:356-360` (chamada), `:388-390` (dict do registro), `:401-408` (lista do caminho de erro)
- Test: `tests/test_fundamentals.py:91-99` (helper), `:102-177` (classe)

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: `_extract_analyst_estimates(stock) -> tuple[float, float, float, float]` devolvendo `(crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado, num_analistas)`, todos `float` e podendo ser `np.nan`. A coluna `lpa_estimado` em `data/fundamentals.csv`. Consumidas pelas Tasks 2, 3 e 4.

- [ ] **Step 1: Atualizar o helper de fixture para carregar a coluna `avg`**

Em `tests/test_fundamentals.py`, substituir `_earnings_frame` (linhas 91-99):

```python
def _earnings_frame(rows):
    """rows: {periodo: (growth_decimal, num_analistas, lpa_avg)}"""
    return pd.DataFrame(
        {
            'growth': [v[0] for v in rows.values()],
            'numberOfAnalysts': [v[1] for v in rows.values()],
            'avg': [v[2] for v in rows.values()],
        },
        index=list(rows.keys()),
    )
```

- [ ] **Step 2: Reescrever a classe de teste inteira**

Substituir `class TestExtractGrowthEstimates` (linhas 102-177) por:

```python
class TestExtractAnalystEstimates:

    def test_reads_next_year_and_converts_growth_to_percentage_points(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.0256, '+1y': 0.1693}),
            earnings=_earnings_frame({'0y': (0.0084, 11, 1.20),
                                      '+1y': (0.2057, 11, 1.45)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(16.93)
        assert lucro == pytest.approx(20.57)
        assert lpa_est == pytest.approx(1.45)
        assert analistas == 11

    def test_lpa_estimado_is_not_scaled_to_percentage_points(self):
        """É nível em R$ por ação, não variação: nada de multiplicar por 100."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.05}),
            earnings=_earnings_frame({'+1y': (0.10, 4, 2.49874)}),
        )

        _, _, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lpa_est == pytest.approx(2.49874)

    def test_reads_next_year_not_current_year(self):
        """Caso SEER3: 0y = 2,17 e +1y = 2,50. A coluna carrega o +1y."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0y': (0.1530, 4, 2.16767),
                                      '+1y': (0.1527, 4, 2.49874)}),
        )

        _, _, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lpa_est == pytest.approx(2.49874)

    def test_negative_lpa_estimado_survives_extraction(self):
        """Caso AURE3: prejuízo projetado com 'crescimento' positivo.

        O valor negativo PRECISA chegar cru ao CSV — é ele que o filtro usa
        para distinguir lucro crescendo de prejuízo encolhendo.
        """
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.08}),
            earnings=_earnings_frame({'+1y': (0.8864, 3, -0.14196)}),
        )

        _, lucro, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lucro == pytest.approx(88.64)
        assert lpa_est == pytest.approx(-0.14196)

    def test_ignores_current_year_and_quarterly_periods(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0q': 0.9, '+1q': 0.8, '0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0q': (0.9, 6, 0.30), '+1y': (0.10, 4, 1.10)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert lpa_est == pytest.approx(1.10)
        assert analistas == 4

    def test_returns_nan_when_next_year_row_missing(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.05}),
            earnings=_earnings_frame({'0y': (0.05, 3, 0.90)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert np.isnan(receita)
        assert np.isnan(lucro)
        assert np.isnan(lpa_est)
        assert np.isnan(analistas)

    def test_returns_nan_when_avg_column_absent(self):
        """Frame sem a coluna 'avg' não interrompe a coleta dos demais campos."""
        earnings = pd.DataFrame({'growth': [0.10], 'numberOfAnalysts': [4]},
                                index=['+1y'])
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.05}), earnings=earnings)

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert np.isnan(lpa_est)
        assert analistas == 4

    def test_returns_nan_for_empty_frames(self):
        stock = _FakeEstimateTicker(revenue=pd.DataFrame(), earnings=pd.DataFrame())

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_when_frames_are_none(self):
        stock = _FakeEstimateTicker(revenue=None, earnings=None)

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_on_exception(self):
        """Caso real: CEBR3 responde 404 no quoteSummary."""
        stock = _FakeEstimateTicker(raises=True)

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_for_nan_cell(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': np.nan}),
            earnings=_earnings_frame({'+1y': (np.nan, np.nan, np.nan)}),
        )

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_revenue_available_without_earnings(self):
        """Caso real: CYRE4 e IGTI3 têm receita mas não lucro."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.1594}), earnings=pd.DataFrame())

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(15.94)
        assert np.isnan(lucro)
        assert np.isnan(lpa_est)
        assert np.isnan(analistas)

    def test_lpa_estimado_available_without_growth(self):
        """Caso real VALE3: tem 'avg' e não tem 'growth' (falta yearAgoEps)."""
        earnings = pd.DataFrame(
            {'growth': [np.nan], 'numberOfAnalysts': [12], 'avg': [1.60]},
            index=['+1y'],
        )
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.03}), earnings=earnings)

        _, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert np.isnan(lucro)
        assert lpa_est == pytest.approx(1.60)
        assert analistas == 12
```

- [ ] **Step 3: Rodar os testes para vê-los falhar**

Run: `rtk proxy python3 -m pytest tests/test_fundamentals.py -q`
Expected: FAIL com `AttributeError: module 'src.fundamentals' has no attribute '_extract_analyst_estimates'`

- [ ] **Step 4: Renomear a função e acrescentar a extração do `avg`**

Substituir `_extract_growth_estimates` inteira (`src/fundamentals.py:119-148`):

```python
def _extract_analyst_estimates(stock) -> tuple[float, float, float, float]:
    """
    Estimativas de analistas para o próximo exercício.

    Usa `revenue_estimate` e `earnings_estimate`, que saem do mesmo módulo
    `earningsTrend` do quoteSummary e ficam em cache no objeto Ticker: acessar
    as duas custa 1 requisição HTTP. Não usar `growth_estimates`, que dispara
    uma segunda requisição para buscar dados de índice que não consumimos.

    O `lpa_estimado` é o NÍVEL de lucro por ação projetado (coluna `avg`, em R$
    por ação), não uma variação — sai do mesmo frame já em memória, então não
    custa requisição nenhuma.

    Ele existe porque o `crescimento_lucro_pct` ao lado é estimativa sobre
    estimativa: o `yearAgoEps` da linha '+1y' é o `avg` da linha '0y', não o
    lucro realizado. Com as duas estimativas negativas a razão fica POSITIVA, e
    um prejuízo encolhendo vira "crescimento de lucro" — a AURE3 projeta -1,25
    -> -0,14 por ação e isso aparece como +88,6%. Sem o nível ao lado, isso é
    indistinguível de lucro crescendo.

    O número de analistas vem do `earnings_estimate` — o `revenue_estimate` não
    expõe contagem — e governa as duas métricas de crescimento.

    Args:
        stock: objeto `yf.Ticker` já construído.

    Returns:
        (crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado,
        num_analistas). Os dois primeiros em pontos percentuais, o terceiro em
        R$ por ação (sem escalar), o último em contagem. Qualquer campo
        indisponível vem como NaN.
    """
    try:
        revenue = stock.revenue_estimate
        earnings = stock.earnings_estimate
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    return (
        _estimate_cell(revenue, 'growth') * 100,
        _estimate_cell(earnings, 'growth') * 100,
        _estimate_cell(earnings, 'avg'),
        _estimate_cell(earnings, 'numberOfAnalysts'),
    )
```

- [ ] **Step 5: Atualizar a chamada no loop de coleta**

Em `src/fundamentals.py:356-360`, substituir:

```python
            # Estimativas de analistas para o próximo exercício. Reaproveita o
            # objeto `stock` já criado: 1 requisição HTTP a mais por ticker.
            crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado, num_analistas = (
                _extract_analyst_estimates(stock)
            )
```

- [ ] **Step 6: Acrescentar a coluna ao dict do registro**

Em `src/fundamentals.py`, no `records.append({...})`, entre `'crescimento_lucro_pct'` e `'num_analistas'`:

```python
                'crescimento_receita_pct': crescimento_receita_pct,
                'crescimento_lucro_pct': crescimento_lucro_pct,
                'lpa_estimado': lpa_estimado,
                'num_analistas': num_analistas,
```

A ordem das chaves do dict define a ordem das colunas no CSV (`pd.DataFrame(records)` na linha 413), então essa posição é o que coloca a coluna no lugar certo do arquivo.

- [ ] **Step 7: Acrescentar a coluna à lista do caminho de erro**

Em `src/fundamentals.py`, no `except` que monta o registro vazio, na lista de chaves NaN:

```python
                    'dividend_rate', 'crescimento_receita_pct',
                    'crescimento_lucro_pct', 'lpa_estimado', 'num_analistas',
```

- [ ] **Step 8: Rodar os testes**

Run: `rtk proxy python3 -m pytest tests/ -q`
Expected: PASS, 121 testes (116 do baseline − 8 removidos da classe antiga + 13 da classe nova)

- [ ] **Step 9: Confirmar que nenhuma referência ao nome antigo sobrou**

Run: `rtk proxy grep -rn "_extract_growth_estimates" src tests`
Expected: nenhuma saída (só `docs/` pode citar, e é histórico)

- [ ] **Step 10: Commit**

```bash
git add src/fundamentals.py tests/test_fundamentals.py
git commit -m "feat: coleta o LPA projetado do frame de estimativas ja carregado

_extract_growth_estimates vira _extract_analyst_estimates e devolve 4 valores:
o nome antigo passaria a mentir, ja que a funcao deixa de devolver so
crescimento. O lpa_estimado sai da coluna 'avg' da linha '+1y', o mesmo frame
que ja era lido — zero requisicao HTTP nova.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Critério de filtro sobre o LPA projetado

**Files:**
- Modify: `config/filters.json` (blocos `stock_filters` e `bank_filters`), `src/filters.py:14-47`
- Test: `tests/test_filters.py:13-29` (helpers), classe nova

**Interfaces:**
- Consumes: a coluna `lpa_estimado` produzida pela Task 1.
- Produces: as chaves `lpa_estimado_min` e `exigir_lpa_estimado` nos dois blocos de `config/filters.json`, e o terceiro bloco condicional em `_growth_mask(df, cfg) -> pd.Series`. A assinatura de `_growth_mask` **não muda**.

- [ ] **Step 1: Estender os helpers de fixture**

Em `tests/test_filters.py`, substituir `_cfg` e `_df` (linhas 13-29):

```python
def _cfg(exigir_estimativa=False, exigir_num_analistas=False,
         exigir_lpa_estimado=False,
         receita_min=0, lucro_min=0, analistas_min=2, lpa_est_min=0):
    return {
        'crescimento_receita_pct_min': receita_min,
        'crescimento_lucro_pct_min': lucro_min,
        'num_analistas_min': analistas_min,
        'lpa_estimado_min': lpa_est_min,
        'exigir_num_analistas': exigir_num_analistas,
        'exigir_estimativa': exigir_estimativa,
        'exigir_lpa_estimado': exigir_lpa_estimado,
    }


def _df(rows, lpa_estimado=1.0):
    """rows: lista de (crescimento_receita_pct, crescimento_lucro_pct, num_analistas)

    `lpa_estimado` entra como coluna à parte, escalar ou lista. O default
    positivo mantém os testes que não são sobre esse critério indiferentes a
    ele; os testes do critério novo passam os valores explicitamente.
    """
    df = pd.DataFrame(
        rows,
        columns=['crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas'],
    )
    df['lpa_estimado'] = lpa_estimado
    return df
```

- [ ] **Step 2: Escrever os testes do critério novo**

Acrescentar ao final de `tests/test_filters.py`:

```python
class TestGrowthMaskLpaEstimado:
    """Guarda de sinal: o nível projetado, não a variação."""

    def _mask(self, rows, lpa_estimado):
        return filters._growth_mask(
            _df(rows, lpa_estimado=lpa_estimado),
            _cfg(exigir_lpa_estimado=True),
        )

    def test_positive_lpa_estimado_passes(self):
        assert self._mask([(6.0, 12.0, 4)], 2.49874).tolist() == [True]

    def test_negative_lpa_estimado_fails(self):
        assert self._mask([(6.0, 12.0, 4)], -0.14196).tolist() == [False]

    def test_zero_lpa_estimado_fails_strict_comparison(self):
        assert self._mask([(6.0, 12.0, 4)], 0.0).tolist() == [False]

    def test_nan_lpa_estimado_fails(self):
        assert self._mask([(6.0, 12.0, 4)], np.nan).tolist() == [False]

    def test_positive_growth_with_projected_loss_fails(self):
        """O caso que motiva a spec: AURE3 e HBRE3.

        'Crescimento de lucro' de +88,6% que é um prejuízo encolhendo de
        -1,25 para -0,14 por ação. Sem esta guarda a linha passa.
        """
        mask = filters._growth_mask(
            _df([(8.0, 88.64, 3)], lpa_estimado=-0.14196),
            _cfg(exigir_estimativa=True, exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [False]

    def test_same_row_passes_without_the_guard(self):
        """Prova que a guarda é o que reprova, não os cortes de crescimento."""
        mask = filters._growth_mask(
            _df([(8.0, 88.64, 3)], lpa_estimado=-0.14196),
            _cfg(exigir_estimativa=True),
        )
        assert mask.tolist() == [True]

    def test_custom_threshold_is_respected(self):
        mask = filters._growth_mask(
            _df([(6.0, 12.0, 4), (6.0, 12.0, 4)], lpa_estimado=[0.40, 0.60]),
            _cfg(exigir_lpa_estimado=True, lpa_est_min=0.5),
        )
        assert mask.tolist() == [False, True]


class TestGrowthMaskLpaEstimadoIndependent:
    """A terceira flag não altera o significado das outras duas."""

    def test_off_ignores_negative_lpa_estimado(self):
        mask = filters._growth_mask(
            _df([(6.0, 12.0, 4)], lpa_estimado=-1.0), _cfg())
        assert mask.tolist() == [True]

    def test_only_lpa_decides_when_other_flags_off(self):
        mask = filters._growth_mask(
            _df([(-50.0, -80.0, 1), (99.0, 99.0, 9)], lpa_estimado=[2.0, -2.0]),
            _cfg(exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [True, False]

    def test_all_three_flags_on_combine(self):
        mask = filters._growth_mask(
            _df([(6.0, 12.0, 4), (6.0, 12.0, 1), (6.0, -2.0, 4)],
                lpa_estimado=[2.0, 2.0, 2.0]),
            _cfg(exigir_estimativa=True, exigir_num_analistas=True,
                 exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [True, False, False]

    def test_all_three_flags_off_passes_everything(self):
        mask = filters._growth_mask(
            _df([(-50.0, -80.0, 1), (np.nan, np.nan, np.nan)],
                lpa_estimado=[-5.0, np.nan]),
            _cfg(),
        )
        assert mask.tolist() == [True, True]
```

- [ ] **Step 3: Rodar os testes para vê-los falhar**

Run: `rtk proxy python3 -m pytest tests/test_filters.py -q`
Expected: FAIL — os testes de `exigir_lpa_estimado=True` devolvem `[True]` onde esperam `[False]`, porque `_growth_mask` ainda ignora a flag.

- [ ] **Step 4: Acrescentar o bloco em `_growth_mask`**

Em `src/filters.py`, substituir o docstring e o corpo da função (linhas 14-47):

```python
def _growth_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Máscara de estimativas de analistas.

    Três flags independentes, cada uma ligando o seu próprio critério:
    `exigir_estimativa` aplica os cortes de crescimento de receita e lucro;
    `exigir_num_analistas` aplica o mínimo de analistas; `exigir_lpa_estimado`
    aplica o corte sobre o nível de lucro por ação projetado. Nenhuma altera o
    significado da outra, e flag desligada significa critério não aplicado.

    A guarda do `lpa_estimado` existe porque `crescimento_lucro_pct` é
    estimativa sobre estimativa (o `yearAgoEps` da linha '+1y' do yfinance é a
    estimativa do exercício corrente, não o lucro realizado). Com as duas
    negativas a razão fica positiva, então um prejuízo encolhendo passa no
    corte de crescimento: a AURE3 projeta -1,25 -> -0,14 por ação e aparece
    como +88,6%. Comparar o NÍVEL contra zero é o que separa lucro crescendo de
    prejuízo encolhendo — a variação sozinha não separa.

    Com a flag ligada, valor NaN reprova: sem dado não há como atestar o
    critério, e é justamente isso que a flag exige. Comparações do pandas com
    NaN já retornam False, então esse comportamento sai de graça.

    Args:
        df: DataFrame com as colunas `crescimento_receita_pct`,
            `crescimento_lucro_pct`, `lpa_estimado` e `num_analistas`.
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

    # '>' estrito, como os demais cortes `_min` do projeto: LPA projetado
    # exatamente zero não é lucro.
    if cfg.get('exigir_lpa_estimado'):
        mask &= df['lpa_estimado'] > cfg['lpa_estimado_min']

    return mask
```

- [ ] **Step 5: Rodar os testes**

Run: `rtk proxy python3 -m pytest tests/test_filters.py -q`
Expected: PASS

- [ ] **Step 6: Acrescentar as chaves no `config/filters.json`**

Nos **dois** blocos. As quatro últimas linhas de `stock_filters` hoje são:

```json
    "num_analistas_min": 2,
    "exigir_num_analistas": false,
    "exigir_estimativa": true
```

Substituir por:

```json
    "num_analistas_min": 2,
    "lpa_estimado_min": 0,
    "exigir_num_analistas": false,
    "exigir_estimativa": true,
    "exigir_lpa_estimado": true
```

Repetir exatamente a mesma substituição em `bank_filters`, cujas linhas finais são idênticas — `_growth_mask` é compartilhada pelos dois filtros, e o artefato de base negativa não é exclusivo de não-bancos.

- [ ] **Step 7: Rodar a suíte inteira**

Run: `rtk proxy python3 -m pytest tests/ -q`
Expected: PASS, 132 testes

- [ ] **Step 8: Commit**

```bash
git add src/filters.py tests/test_filters.py config/filters.json
git commit -m "feat: filtra por LPA projetado positivo, nao so por crescimento

crescimento_lucro_pct e estimativa sobre estimativa, entao fica positivo com
as duas negativas: dos 7 papeis com LPA projetado <= 0 no universo medido, os
7 exibem crescimento positivo. HBRE3 passa hoje no lpa_min (realizado +0,29) e
no corte de crescimento (+85%) com prejuizo projetado de -0,03 por acao.

Terceira flag independente, ligada por padrao: os papeis sem cobertura ja caem
pelo exigir_estimativa, entao so a guarda de sinal morde.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: LPA projetado no snapshot histórico

**Files:**
- Modify: `src/valuation.py:652-659`
- Test: `tests/test_valuation.py` (classe `TestAppendSnapshot`)

**Interfaces:**
- Consumes: a coluna `lpa_estimado` produzida pela Task 1.
- Produces: `lpa_estimado` em `_SNAPSHOT_RESULT_COLS`, e portanto em `data/valuation_history.csv`. Consumida pela Task 4 (o notebook exibe as mesmas colunas na prévia do snapshot).

- [ ] **Step 1: Escrever o teste falhando**

Acrescentar à classe `TestAppendSnapshot` em `tests/test_valuation.py`:

```python
    def test_snapshots_lpa_estimado(self, tmp_path):
        p = tmp_path / 'hist.csv'
        df = self._valued()
        df['crescimento_lucro_pct'] = [15.27]
        df['lpa_estimado'] = [2.49874]
        v.append_snapshot(df, path=p, snapshot_date='2026-08-10')
        out = pd.read_csv(p)
        assert out.loc[0, 'lpa_estimado'] == pytest.approx(2.49874)

    def test_lpa_estimado_aligns_with_history_written_before_it(self, tmp_path):
        """As 277 linhas já gravadas recebem NaN, não desalinham."""
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-08-06')

        df = self._valued()
        df['lpa_estimado'] = [2.49874]
        v.append_snapshot(df, path=p, snapshot_date='2026-08-10')

        out = pd.read_csv(p)
        assert len(out) == 2
        assert pd.isna(out.loc[0, 'lpa_estimado'])
        assert out.loc[1, 'lpa_estimado'] == pytest.approx(2.49874)
```

- [ ] **Step 2: Rodar o teste para vê-lo falhar**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py::TestAppendSnapshot -q`
Expected: FAIL com `KeyError: 'lpa_estimado'` na leitura do CSV — a coluna é filtrada fora por `_SNAPSHOT_RESULT_COLS`.

- [ ] **Step 3: Acrescentar a coluna à lista de snapshot**

Em `src/valuation.py`, na constante `_SNAPSHOT_RESULT_COLS`, substituir a última linha:

```python
    'crescimento_receita_pct', 'crescimento_lucro_pct', 'lpa_estimado',
    'num_analistas',
```

- [ ] **Step 4: Rodar os testes**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py -q`
Expected: PASS

- [ ] **Step 5: Rodar a suíte inteira**

Nenhuma linha de produção precisou mudar além da constante: o `concat` do `append_snapshot` já alinha por nome.

Run: `rtk proxy python3 -m pytest tests/ -q`
Expected: PASS, 134 testes

- [ ] **Step 6: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: lpa_estimado entra no snapshot historico

Junto das outras tres colunas de estimativa. O concat do append_snapshot ja
alinha por nome, entao as linhas antigas recebem NaN sem quebrar.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Exibição no notebook

**Files:**
- Modify: `analysis.ipynb` — célula `69223fb5` (código: `numeric_cols`, `CRESCIMENTO_COLS`, `CRESCIMENTO_FMT`) e célula `96a4302e` (markdown de critérios)

**Interfaces:**
- Consumes: a coluna `lpa_estimado` (Task 1), as flags de config (Task 2), a coluna de snapshot (Task 3).
- Produces: nada consumido por outra task. É a ponta da cadeia.

`CRESCIMENTO_COLS` é a alavanca única dos 5 pontos de exibição (screening de ações, tabela de bancos, valuation, top 20, prévia do snapshot) — as outras 4 células referenciam `*CRESCIMENTO_COLS` e não precisam ser tocadas.

- [ ] **Step 1: Acrescentar a coluna à sanitização numérica**

Na célula `69223fb5`, na lista `numeric_cols`, substituir a última linha:

```python
    'dividend_rate', 'crescimento_receita_pct', 'crescimento_lucro_pct',
    'lpa_estimado', 'num_analistas'
]
```

- [ ] **Step 2: Acrescentar a coluna à exibição e ao formato**

Na mesma célula, substituir o bloco de `CRESCIMENTO_COLS` / `CRESCIMENTO_FMT`:

```python
# Crescimento projetado acompanha toda listagem: sem os analistas por trás,
# o número não diz se é consenso ou palpite de um analista só. O `lpa_estimado`
# vem logo depois da variação porque é o nível que ela alcança — sem ele,
# "+88,6% de crescimento" não distingue lucro subindo de prejuízo encolhendo.
CRESCIMENTO_COLS = ['crescimento_receita_pct', 'crescimento_lucro_pct',
                    'lpa_estimado', 'num_analistas']
CRESCIMENTO_FMT = {
    'crescimento_receita_pct': '{:.1f}%', 'crescimento_lucro_pct': '{:.1f}%',
    'lpa_estimado': 'R$ {:.2f}', 'num_analistas': '{:.0f}',
}
```

- [ ] **Step 3: Atualizar a tabela de critérios no markdown**

Na célula `96a4302e`, substituir o cabeçalho e as duas últimas linhas da tabela de ações:

```markdown
**Critérios de Screening (Ações — 14 critérios, 3 opcionais):**
```

e, ao final da tabela:

```markdown
| 12 | Cresc. Receita proj. (`+1y`) | > 0% |
| 13 | Cresc. Lucro proj. (`+1y`) | > 0% |
| 14 | LPA proj. (`+1y`) | > 0 |
```

- [ ] **Step 4: Corrigir o parágrafo desatualizado sobre as flags**

Na mesma célula, substituir o parágrafo "Crescimento projetado (opcional)" inteiro:

```markdown
**Estimativas de analistas (opcionais):** consenso via yfinance para o próximo exercício
(`+1y`) — o horizonte mais longo que a Yahoo entrega. Controlado em `config/filters.json` por
`exigir_estimativa` (aplica os cortes de crescimento), `exigir_num_analistas` (aplica o mínimo
de analistas) e `exigir_lpa_estimado` (exige lucro por ação projetado positivo). Hoje
`exigir_estimativa` e `exigir_lpa_estimado` nascem ligadas; `exigir_num_analistas`, desligada.

O `lpa_estimado` é o **nível** projetado, em R$ por ação, e não uma variação. Ele acompanha o
`crescimento_lucro_pct` porque esse percentual é estimativa sobre estimativa — compara o
próximo exercício com o exercício corrente, não com o lucro realizado — e por isso fica
positivo quando as duas são negativas. A AURE3 projeta −1,25 → −0,14 por ação, o que aparece
como "+88,6% de crescimento de lucro".

A cobertura é irregular em ações BR: 170 dos 248 papéis acima do corte de liquidez têm
estimativa de LPA, contra 169 com estimativa de crescimento de lucro.
```

- [ ] **Step 5: Verificar a coerência do critério de bancos no markdown**

Na mesma célula, substituir a linha dos critérios de bancos:

```markdown
**Critérios de Screening (Bancos — 7 critérios + os 3 opcionais acima):**
P/L 0–10, P/PV 0–2, ROE > 15%, Margem Líq. > 10%, LPA > 0, Liq. Diária > R$ 100k, DY > 3%.
```

- [ ] **Step 6: Commit**

```bash
git add analysis.ipynb
git commit -m "docs: notebook exibe o LPA projetado e corrige o texto das flags

CRESCIMENTO_COLS cobre os 5 pontos de exibicao de uma vez. O paragrafo dizia
que as flags de estimativa nascem desligadas — exigir_estimativa esta ligada
desde a spec de 2026-08-05.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Regenerar o cache e validar ponta a ponta

**Files:**
- Modify: `data/fundamentals.csv` (regenerado), `data/valuation_history.csv` (nova linha de snapshot)

**Interfaces:**
- Consumes: tudo das Tasks 1-4.
- Produces: nada de código. É a validação de que a cadeia funciona sobre dado real.

O `fundamentals.csv` atual não tem a coluna `lpa_estimado`, e `_growth_mask` indexa direto — rodar o notebook sobre o cache antigo levanta `KeyError`. É o mesmo procedimento das duas features anteriores que adicionaram colunas.

- [ ] **Step 1: Confirmar que o cache antigo realmente quebra**

Run:
```bash
rtk proxy python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src import filters
df = pd.read_csv('data/fundamentals.csv')
try:
    filters.apply_stock_filters(df)
    print('NAO QUEBROU — investigar antes de seguir')
except KeyError as e:
    print(f'KeyError esperado: {e}')
"
```
Expected: `KeyError esperado: 'lpa_estimado'`

Se **não** quebrar, parar: significa que a Task 2 não ligou a flag ou o config não foi salvo.

- [ ] **Step 2: Regenerar o cache**

Run:
```bash
rtk proxy python3 -c "
import sys; sys.path.insert(0, '.')
from src import scraper, fundamentals
t = scraper.get_tickers()
fundamentals.fetch_fundamentals(t['ticker_sa'].tolist(), delay=0.4, force_refresh=True)
"
```
Expected: ~20-25 minutos para 372 tickers. Ao final, `372 tickers processados`.

- [ ] **Step 3: Conferir a coluna nova no CSV**

Run:
```bash
rtk proxy python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
d = pd.read_csv('data/fundamentals.csv')
print('coluna presente:', 'lpa_estimado' in d.columns)
print('posicao:', list(d.columns).index('lpa_estimado'),
      '| vizinhas:', list(d.columns)[-4:])
print('com dado:', d.lpa_estimado.notna().sum(), '/', len(d))
print('negativos:', (d.lpa_estimado <= 0).sum())
print(d[d.lpa_estimado <= 0][['ticker','lpa','crescimento_lucro_pct','lpa_estimado']].to_string(index=False))
"
```
Expected: coluna presente, imediatamente antes de `num_analistas`, ~170 com dado, e a lista de negativos exibindo `crescimento_lucro_pct` positivo ao lado de `lpa_estimado` negativo — a evidência do artefato que a spec descreve.

- [ ] **Step 4: Rodar o notebook de ponta a ponta**

Run: `rtk proxy python3 -m jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`
Expected: termina sem erro. Qualquer célula que levante exceção aborta o comando com stack trace — é esse o critério de aprovação.

Depois, conferir o resultado no arquivo gravado:

```bash
rtk proxy python3 -c "
import json
nb = json.load(open('analysis.ipynb'))
txt = json.dumps(nb)
print('lpa_estimado aparece nos outputs:', txt.count('lpa_estimado'))
erros = [c for c in nb['cells']
         for o in c.get('outputs', []) if o.get('output_type') == 'error']
print('celulas com erro:', len(erros))
for c in nb['cells']:
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream' and '[filters]' in ''.join(o.get('text', [])):
            print(''.join(o['text']).strip())
"
```
Expected: `celulas com erro: 0`, `lpa_estimado` presente nos outputs, e as duas linhas do `[filters]` com as contagens de aprovadas.

- [ ] **Step 5: Registrar o efeito no resultado**

Run:
```bash
rtk proxy python3 -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
h = pd.read_csv('data/valuation_history.csv')
ult = h[h.data_snapshot == h.data_snapshot.max()]
print('snapshot:', ult.data_snapshot.iloc[0], '| linhas:', len(ult))
print(ult[['ticker','preco','preco_justo_dcf','crescimento_lucro_pct','lpa_estimado']].to_string(index=False))
"
```
Expected: a coluna `lpa_estimado` preenchida nas linhas novas, e NaN nas 277 linhas antigas.

**Não compare contra `valuation_history.csv`.** O arquivo é append-only e o notebook foi rodado várias vezes na mesma data (a data 2026-08-06 tem 179 linhas com 18 tickers duplicados), então "a lista aprovada da rodada anterior" não é recuperável dele. Comparar contra essa base produz uma divergência inventada.

A medição correta é um **contrafactual sobre o dado de hoje**: rodar o filtro com a flag nova ligada e desligada, e ver o que muda.

```bash
rtk proxy python3 -c "
import sys, json; sys.path.insert(0,'.')
import pandas as pd
from src import filters
cfg = json.load(open('config/filters.json'))
d = pd.read_csv('data/fundamentals.csv')
for c in d.columns:
    if c not in ('ticker_sa','ticker','nome','setor','industria'):
        d[c] = pd.to_numeric(d[c], errors='coerce')

def passa(flag_lpa):
    c = dict(cfg['stock_filters']); c['exigir_lpa_estimado'] = flag_lpa
    base = ((d.pl>c['pl_min'])&(d.pl<=c['pl_max'])&(d.pvp>c['pvp_min'])&(d.pvp<=c['pvp_max'])&
            (d.margem_ebit_pct>c['margem_ebit_pct_min'])&(d.margem_liquida_pct>c['margem_liquida_pct_min'])&
            (d.dl_ebit<c['dl_ebit_max'])&(d.dl_pl<c['dl_pl_max'])&(d.roe_pct>c['roe_pct_min'])&
            (d.liquidez_corrente>c['liquidez_corrente_min'])&(d.passivos_ativos<c['passivos_ativos_max'])&
            (d.liq_media_diaria>c['liq_media_diaria_min'])&(d.lpa>c['lpa_min']))
    return set(d[base & filters._growth_mask(d, c)].ticker)

com, sem = passa(True), passa(False)
print('com o criterio novo:', len(com), '| sem:', len(sem))
print('reprovadas pelo criterio novo:', sorted(sem - com) or 'NENHUMA')
"
```

Expected: a lista de reprovadas contém apenas papéis cujo `lpa_estimado` é ≤ 0 **e** que passam em todos os demais critérios. `NENHUMA` é resultado válido e esperado quando os papéis com LPA projetado negativo já reprovam antes, em filtros de fundamento — o critério fica inerte no dado do dia sem estar defeituoso. Os testes unitários da Task 2 é que provam a lógica; este passo só confirma que ela não derruba nada que não devia.

Investigar antes de commitar apenas se a lista de reprovadas contiver algum ticker com `lpa_estimado > 0`.

- [ ] **Step 6: Commit**

`data/` está no `.gitignore` deste repositório — `fundamentals.csv` e `valuation_history.csv` nunca foram versionados. O único arquivo com diff rastreável é o notebook, cujos outputs foram reescritos pela execução do Step 4.

```bash
git add analysis.ipynb
git commit -m "chore: outputs do notebook com a coluna lpa_estimado

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Verificação final

- [ ] `rtk proxy python3 -m pytest tests/ -q` → **134 passando** (116 do baseline − 8 substituídos na classe renomeada + 26 novos: 13 na Task 1, 11 na Task 2, 2 na Task 3)
- [ ] `rtk proxy grep -rn "_extract_growth_estimates" src tests` → sem saída
- [ ] `lpa_estimado` presente em: `data/fundamentals.csv`, `data/valuation_history.csv`, e nas 5 tabelas do notebook
- [ ] Nenhum preço justo mudou por conta desta mudança — as diferenças no `valuation_history` entre a rodada anterior e esta vêm da atualização do cache, não do código. Confirmar comparando `preco_justo_dcf` de um papel cujo `fcf` e `shares_total` não mudaram entre as duas coletas.
