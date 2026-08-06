# Crescimento forward no estágio 1 do DCF — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** O estágio 1 do DCF passa a ser semeado pelo crescimento forward que já está em `data/fundamentals.csv` (driver configurável, default receita), e os limites de crescimento deixam de substituir valores — passam a dizer apenas "esta taxa não é projetável".

**Architecture:** Nada de novo é coletado. `resolve_forward_growth(row)` lê a coluna de crescimento da própria linha do DataFrame (`crescimento_receita_pct` ou `crescimento_lucro_pct`, conforme a env `FORWARD_GROWTH_DRIVER`) e converte para decimal; `apply_valuation` passa esse valor no parâmetro `forward_growth` que `dcf_valuation` já aceita. O piso `MIN_GROWTH_RATE` é removido e `MAX_GROWTH_RATE` vira `MAX_PROJECTABLE_GROWTH`, um limiar que faz o DCF devolver `NaN` (chamador recai no DDM, rotulado) em vez de trocar a taxa por 20%. `get_forward_growth` e suas constantes saem de `src/fundamentals.py` — eram 2 requisições HTTP por ticker para o mesmo dado.

**Tech Stack:** Python 3.14, pandas 3.0, numpy, pytest 9 (`python3 -m pytest`). Sem novas dependências.

**Spec:** `docs/superpowers/specs/2026-08-06-crescimento-forward-dcf-design.md`

## Global Constraints

- `docs/GUIDELINES.md` é vinculante. Em particular: (2) limites só dentro de `src/valuation.py` e só com o significado "não projetável por 10 anos"; (3) nunca calibrar constantes contra `data/fundamentals.csv`; (4) na dúvida, preferir o preço justo **menor**; (5) docstrings e comentários em português acessível, registrando o **porquê**.
- Comentários e docstrings novos em **português**, seguindo o padrão do arquivo (explicam a razão, não o óbvio).
- `MAX_PROJECTABLE_GROWTH = 0.20` — valor mantido exatamente como está hoje. Não redefinir, não calibrar.
- Nenhuma coluna nova em `data/fundamentals.csv`, nenhum refetch, nenhuma requisição HTTP nova.
- `src/filters.py`, `config/filters.json` e a exibição do screener **não são tocados** por este plano.
- Nenhum teste pode bater rede: `get_fcf_series` e `dcf_valuation` são sempre monkeypatchados nos testes que os alcançam.
- Comando de teste: `python3 -m pytest tests/ -q`. **Se o hook do rtk interceptar** e falhar com `rtk: Failed to run pytest`, use a forma que escapa do hook: `PY=python3; $PY -m pytest tests/ -q`.
- Baseline antes de começar: `96 passed`.

## Estrutura de arquivos

| Arquivo | Responsabilidade | O que muda |
|---|---|---|
| `src/valuation.py` | Modelos de valuation e suas premissas | Constantes de crescimento, `resolve_forward_growth` (nova), `_compute_fcf_cagr`, `dcf_valuation`, `apply_valuation` (call site), `append_snapshot` |
| `src/fundamentals.py` | Coleta de dados (yfinance → CSV) | Só remoção: `get_forward_growth`, `_FORWARD_GROWTH_PERIODS`, `_FORWARD_GROWTH_STOCK_COLS` |
| `tests/test_valuation.py` | Testes do valuation | Nova classe `TestResolveForwardGrowth`; ajustes em `TestComputeFcfCagr`, `TestDiscountFcfToEquity`, `TestForwardGrowth`, `TestAppendSnapshot` |
| `tests/test_fundamentals.py` | Testes da coleta | Remoção da classe `TestGetForwardGrowth` e dos helpers `_FakeTicker`/`_est` |
| `.env.example` | Documentação das envs | Nova entrada `FORWARD_GROWTH_DRIVER` |
| `.claude/instructions.md`, `.claude/instructions/valuation-models.instructions.md` | Instruções para agentes | Linhas da tabela de constantes que citam `MIN_GROWTH_RATE`/`MAX_GROWTH_RATE` |

**Ordem das tarefas** foi escolhida para a suíte ficar verde ao final de cada uma:
1. adição pura (`resolve_forward_growth`) — nada quebra;
2. constantes + regra de projetabilidade nos dois pontos que usavam o piso/teto (acoplados pelas mesmas constantes);
3. ligação do call site + remoção do código morto em `fundamentals.py`;
4. snapshot da premissa + docs.

---

### Task 1: `resolve_forward_growth` e a env `FORWARD_GROWTH_DRIVER`

Adição pura: nada existente muda de comportamento. A função lê da linha do DataFrame a coluna de crescimento correspondente ao driver, converte pontos percentuais para decimal e barra valores inválidos (≤ −100%, que só ocorrem quando o lucro cruza zero e a razão deixa de significar uma taxa).

Mora em `valuation.py`, não em `fundamentals.py`: é decisão de modelagem do DCF e lê uma flag de valuation. `fundamentals.py` apenas coleta.

**Files:**
- Modify: `src/valuation.py` (constantes, após a linha 47; nova função imediatamente antes de `dcf_valuation`, hoje linha 204)
- Modify: `.env.example`
- Test: `tests/test_valuation.py` (nova classe, inserir depois de `TestEnvHelpers`, hoje linha 297)

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces:
  - `FORWARD_GROWTH_DRIVER: str` — sempre `'revenue'` ou `'earnings'`.
  - `resolve_forward_growth(row) -> float` — `row` é uma linha de DataFrame (`pd.Series`) ou qualquer objeto com `.get(coluna, default)`. Retorna crescimento **decimal** (0.148 para 14,8%) ou `np.nan`. Não avalia projetabilidade.

- [ ] **Step 1: Escrever os testes que falham**

Em `tests/test_valuation.py`, adicionar `import importlib` no topo (junto dos imports existentes, antes de `import numpy as np`):

```python
import importlib
import sys
from pathlib import Path
```

E inserir a classe depois de `TestEnvHelpers`:

```python
class TestResolveForwardGrowth:
    """
    O crescimento forward vem da linha do CSV (já coletado pelo screener), não
    de uma requisição nova por ticker. O driver escolhe a coluna: receita
    (default) ou lucro.

    Receita é o default porque o DCF projeta fluxo de caixa livre — receita
    menos custos caixa e capex — e não lucro contábil, que oscila muito mais
    para a mesma variação de receita (alavancagem operacional, itens não
    recorrentes, efeitos fiscais).
    """

    @staticmethod
    def _row(receita=14.8, lucro=9.2):
        return pd.Series({
            'crescimento_receita_pct': receita,
            'crescimento_lucro_pct': lucro,
        })

    def test_reads_revenue_column_by_default(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row()) == pytest.approx(0.148)

    def test_reads_earnings_column_when_driver_is_earnings(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert v.resolve_forward_growth(self._row()) == pytest.approx(0.092)

    def test_converts_percentage_points_to_decimal(self, monkeypatch):
        # O CSV guarda pontos percentuais; o DCF trabalha em decimal.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row(receita=100.0)) == pytest.approx(1.0)

    def test_keeps_negative_growth(self, monkeypatch):
        # PETR4 com -4,61%: declínio é dado válido, não valor a ser corrigido.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row(receita=-4.61)) == pytest.approx(-0.0461)

    def test_returns_nan_when_column_is_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert np.isnan(v.resolve_forward_growth(self._row(receita=np.nan)))

    def test_returns_nan_when_column_is_absent(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert np.isnan(v.resolve_forward_growth(pd.Series({'ticker': 'X'})))

    def test_returns_nan_at_minus_one_hundred_percent(self, monkeypatch):
        # Lucro que vira prejuízo: o denominador |realizado| cruza zero e a
        # razão deixa de significar uma taxa. Dado inválido, não valor extremo.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert np.isnan(v.resolve_forward_growth(self._row(lucro=-100.0)))

    def test_returns_nan_below_minus_one_hundred_percent(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert np.isnan(v.resolve_forward_growth(self._row(lucro=-250.0)))

    def test_invalid_driver_in_env_falls_back_to_revenue(self, monkeypatch):
        # A validação acontece na leitura da env (import do módulo), por isso o
        # reload: é o único jeito de reexecutar essa linha no teste.
        monkeypatch.setenv('FORWARD_GROWTH_DRIVER', 'ebitda')
        importlib.reload(v)
        assert v.FORWARD_GROWTH_DRIVER == 'revenue'
        monkeypatch.delenv('FORWARD_GROWTH_DRIVER')
        importlib.reload(v)  # devolve o módulo ao estado normal p/ os outros testes

    def test_earnings_driver_is_read_from_env(self, monkeypatch):
        monkeypatch.setenv('FORWARD_GROWTH_DRIVER', 'earnings')
        importlib.reload(v)
        assert v.FORWARD_GROWTH_DRIVER == 'earnings'
        monkeypatch.delenv('FORWARD_GROWTH_DRIVER')
        importlib.reload(v)
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `python3 -m pytest tests/test_valuation.py::TestResolveForwardGrowth -q`
Expected: FAIL com `AttributeError: module 'src.valuation' has no attribute 'FORWARD_GROWTH_DRIVER'` (e `resolve_forward_growth`).

- [ ] **Step 3: Adicionar a constante**

Em `src/valuation.py`, logo depois da linha `USE_FORWARD_ESTIMATES = _env_bool('USE_FORWARD_ESTIMATES', False)` (linha 47), inserir:

```python
# Qual crescimento alimenta o estágio 1 do DCF: 'revenue' (receita, default) ou
# 'earnings' (lucro). Receita é o default por razão estrutural, não amostral: o
# DCF projeta fluxo de caixa livre (receita menos custos caixa e capex), e o
# lucro contábil amplifica a mesma variação de receita via alavancagem
# operacional, itens não recorrentes e efeitos fiscais.
# Validado aqui em vez de num helper _env_str: é o único uso de string na env.
FORWARD_GROWTH_DRIVER = os.getenv('FORWARD_GROWTH_DRIVER', 'revenue').strip().lower()
if FORWARD_GROWTH_DRIVER not in ('revenue', 'earnings'):
    print(f"[valuation] FORWARD_GROWTH_DRIVER={FORWARD_GROWTH_DRIVER!r} inválido, "
          f"usando 'revenue'")
    FORWARD_GROWTH_DRIVER = 'revenue'
```

- [ ] **Step 4: Adicionar `resolve_forward_growth`**

Em `src/valuation.py`, inserir imediatamente antes de `def dcf_valuation(` (hoje linha 204), separada por duas linhas em branco:

```python
def resolve_forward_growth(row) -> float:
    """
    Crescimento forward da linha do DataFrame, em decimal.

    O dado já está em `data/fundamentals.csv` (colunas `crescimento_receita_pct`
    e `crescimento_lucro_pct`, coletadas pelo screener), então não há requisição
    nova: lê-se a coluna do driver configurado e converte de pontos percentuais
    para decimal (14,8 -> 0,148).

    Crescimento <= -100% é tratado como dado AUSENTE, não como valor extremo: só
    acontece com o driver de lucro, quando o lucro vira prejuízo e o denominador
    da razão (estimativa - realizado)/|realizado| cruza zero — a partir daí o
    número não significa mais uma taxa. Recai no CAGR histórico como qualquer
    estimativa faltante.

    Não decide projetabilidade: essa avaliação fica concentrada em
    `dcf_valuation`, com o limiar MAX_PROJECTABLE_GROWTH.

    Args:
        row: linha do DataFrame de fundamentos (pd.Series).

    Returns:
        Crescimento anual como decimal, ou NaN se ausente/inválido.
    """
    col = ('crescimento_receita_pct' if FORWARD_GROWTH_DRIVER == 'revenue'
           else 'crescimento_lucro_pct')
    value = row.get(col, np.nan)
    if pd.isna(value):
        return np.nan
    growth = float(value) / 100.0
    return growth if growth > -1.0 else np.nan
```

- [ ] **Step 5: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_valuation.py -q`
Expected: PASS (as 10 novas mais as existentes; nenhuma regressão).

- [ ] **Step 6: Documentar a env em `.env.example`**

Acrescentar ao final do bloco `# --- Estimativas forward (analistas via yfinance) ---`, depois da linha `USE_FORWARD_ESTIMATES=0`:

```
# Qual crescimento alimenta o estágio 1 do DCF quando o forward está ligado:
# 'revenue' (receita, default) ou 'earnings' (lucro). O DCF projeta fluxo de
# caixa livre, que é receita menos custos e investimentos — mais próximo de
# receita que de lucro contábil, e o lucro oscila muito mais para a mesma
# variação de receita. Valor inválido cai no default com aviso impresso.
FORWARD_GROWTH_DRIVER=revenue
```

- [ ] **Step 7: Rodar a suíte inteira**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, 106 passed.

- [ ] **Step 8: Commit**

```bash
git add src/valuation.py tests/test_valuation.py .env.example
git commit -m "feat: resolve crescimento forward a partir da linha do CSV

Nova FORWARD_GROWTH_DRIVER (revenue|earnings) e resolve_forward_growth(row),
que le crescimento_receita_pct/crescimento_lucro_pct do proprio DataFrame em
vez de buscar por ticker. Crescimento <= -100% conta como dado ausente."
```

---

### Task 2: Limites de crescimento param de substituir valores

Duas mudanças acopladas pelas mesmas constantes, por isso na mesma tarefa:

- `MIN_GROWTH_RATE` (piso em 0,0) é **removido**. Um piso seleciona por magnitude, e magnitude não indica falta de confiabilidade — o contrário é mais comum (uma empresa que cai todo ano por quatro anos produz um número grande *e* confiável). Remover o piso é monotonicamente conservador: todo preço justo ou fica igual ou cai.
- `MAX_GROWTH_RATE` → `MAX_PROJECTABLE_GROWTH` (mesmo valor 0,20), e deixa de substituir o valor: acima do limiar, o CAGR histórico devolve `NaN` (DCF inaplicável, chamador recai no DDM rotulado) e o forward é ignorado em favor do histórico.

**Files:**
- Modify: `src/valuation.py:6` (import), `:49-50` (constantes), `:134-159` (`_compute_fcf_cagr`), `:204-288` (`dcf_valuation`)
- Test: `tests/test_valuation.py` (`TestComputeFcfCagr` linhas 40-65, `TestDiscountFcfToEquity` linhas 110-158, `TestForwardGrowth` linhas 299-337)

**Interfaces:**
- Consumes: `MAX_PROJECTABLE_GROWTH` passa a ser o único limite de crescimento do módulo (`MIN_GROWTH_RATE` deixa de existir).
- Produces:
  - `_compute_fcf_cagr(fcf_series) -> float` — pode devolver `NaN` (não projetável), negativo (declínio) ou `0.0` (série com ano ≤ 0, regra preservada).
  - `dcf_valuation(ticker_sa, shares_total=None, beta=None, forward_growth=None) -> dict` — assinatura inalterada, mas `forward_growth` nunca mais é buscado internamente; sem crescimento válido, devolve o dict com `preco_justo_dcf = NaN`.

- [ ] **Step 1: Ajustar os testes existentes que assumem clamp**

Em `tests/test_valuation.py`, substituir os dois testes de clamp de `TestComputeFcfCagr` (linhas 56-62):

```python
    def test_caps_growth_at_max_rate(self):
        serie = pd.Series([1000.0, 100.0])
        assert v._compute_fcf_cagr(serie) == v.MAX_GROWTH_RATE

    def test_floors_growth_at_min_rate_when_declining(self):
        serie = pd.Series([50.0, 100.0])
        assert v._compute_fcf_cagr(serie) == v.MIN_GROWTH_RATE
```

por:

```python
    def test_returns_nan_when_cagr_is_above_projectable_threshold(self):
        # 100 -> 1000 em 1 ano = +900%. Antes virava seed de 20%, o que produz
        # o MAIOR preço justo que o modelo consegue emitir; agora o DCF se
        # declara inaplicável e o chamador recai no DDM, rotulado.
        serie = pd.Series([1000.0, 100.0])
        assert np.isnan(v._compute_fcf_cagr(serie))

    def test_lets_negative_cagr_through_unchanged(self):
        # 100 -> 50 em 1 ano = -50%. Sem piso: declínio é projetado como
        # declínio, e isso só reduz o preço justo (erra excluindo).
        serie = pd.Series([50.0, 100.0])
        assert v._compute_fcf_cagr(serie) == pytest.approx(-0.50, abs=1e-9)

    def test_accepts_cagr_exactly_at_the_threshold(self):
        # 100 -> 120 em 1 ano = +20%: no limiar, ainda projetável.
        serie = pd.Series([120.0, 100.0])
        assert v._compute_fcf_cagr(serie) == pytest.approx(v.MAX_PROJECTABLE_GROWTH, abs=1e-9)
```

Manter intocado `test_returns_zero_when_series_contains_negative_year` (linhas 46-49): a regra "qualquer ano ≤ 0 zera o CAGR" está **fora do escopo** desta spec e continua devolvendo `0.0`.

- [ ] **Step 2: Reescrever `TestForwardGrowth` para passar o forward direto**

Substituir a classe inteira `TestForwardGrowth` (linhas 299-337) por:

```python
class TestForwardGrowth:
    """
    Com USE_FORWARD_ESTIMATES ligado, o estágio 1 usa o crescimento forward que
    o chamador passa (vindo de resolve_forward_growth); senão, o CAGR histórico.
    O forward não é mais buscado dentro do DCF — o valor já vem do CSV.

    O limiar de projetabilidade NÃO substitui a taxa: forward acima dele é
    descartado em favor do histórico.
    """

    FCF = pd.Series([121e6, 110e6, 100e6])  # CAGR histórico = 10%

    def test_uses_forward_growth_when_enabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.05)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_ignores_forward_when_disabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', False)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.05)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_falls_back_to_historical_when_forward_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=float('nan'))
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_falls_back_to_historical_when_forward_is_not_projectable(self, monkeypatch):
        # ONCO3 com lucro +1410%: antes virava seed de 20% (preço justo máximo).
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=14.107)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_uses_negative_forward_growth_as_is(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=-0.10)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(-0.10)

    def test_negative_growth_yields_lower_price_than_zero_growth(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        declining = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                                    forward_growth=-0.10)
        flat = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.0)
        assert declining['preco_justo_dcf'] < flat['preco_justo_dcf']

    def test_no_price_when_historical_is_not_projectable_and_no_forward(self, monkeypatch):
        # CAGR de +900% -> NaN. Sem forward utilizável, o DCF não sai: o
        # chamador recai no DDM e metodo_valuation registra a substituição.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: pd.Series([1000e6, 100e6]))
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', False)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert np.isnan(res['preco_justo_dcf'])
        assert np.isnan(res['growth_rate'])
```

- [ ] **Step 3: Acrescentar os testes de crescimento negativo em `TestDiscountFcfToEquity`**

Adicionar ao final da classe `TestDiscountFcfToEquity` (depois de `test_higher_discount_rate_lowers_fair_value`, linha 158):

```python
    def test_negative_growth_yields_a_positive_finite_price(self):
        # Sem piso, o estágio 1 pode começar em queda: o fluxo encolhe nos
        # primeiros anos e converge para o crescimento terminal. O resultado
        # precisa continuar sendo um número, só menor.
        got = v.discount_fcf_to_equity(10e6, -0.10, 0.20, 1e6)
        assert got > 0 and np.isfinite(got)
        assert got < v.discount_fcf_to_equity(10e6, 0.0, 0.20, 1e6)

    def test_returns_nan_at_minus_one_hundred_percent_growth(self):
        # -100% zera o fluxo do primeiro ano: não há empresa a avaliar.
        assert np.isnan(v.discount_fcf_to_equity(10e6, -1.0, 0.20, 1e6))

    def test_returns_nan_below_minus_one_hundred_percent_growth(self):
        assert np.isnan(v.discount_fcf_to_equity(10e6, -1.5, 0.20, 1e6))
```

- [ ] **Step 4: Rodar os testes e confirmar que falham**

Run: `python3 -m pytest tests/test_valuation.py -q`
Expected: FAIL — `AttributeError: module 'src.valuation' has no attribute 'MAX_PROJECTABLE_GROWTH'` nos testes de CAGR, e `TestForwardGrowth::test_falls_back_to_historical_when_forward_is_not_projectable` devolvendo `growth_source == 'forward'`.

- [ ] **Step 5: Trocar as constantes**

Em `src/valuation.py`, substituir as linhas 49-50:

```python
MAX_GROWTH_RATE = 0.20     # Cap de crescimento anual
MIN_GROWTH_RATE = 0.0      # Floor de crescimento
```

por:

```python
# Limiar de PROJETABILIDADE, não teto de crescimento: responde "consigo projetar
# essa taxa por 10 anos?". Quando a resposta é não, o modelo recua para outra
# fonte ou se declara inaplicável (NaN) — nunca troca a taxa por 0,20. Substituir
# silenciosamente produz um preço justo que aparenta ter modelado a empresa
# quando modelou outra, mais saudável, e isso chega ao usuário como recomendação.
# Não existe piso: crescimento negativo é projetado como negativo. Um piso
# seleciona por MAGNITUDE, e magnitude não indica falta de confiabilidade — uma
# empresa que cai todo ano por quatro anos produz um número grande e confiável.
MAX_PROJECTABLE_GROWTH = 0.20
```

- [ ] **Step 6: Aplicar o limiar em `_compute_fcf_cagr`**

Substituir o final da função (linhas 156-159):

```python
    cagr = (last / first) ** (1 / n_years) - 1

    # Aplicar limites
    return max(MIN_GROWTH_RATE, min(MAX_GROWTH_RATE, cagr))
```

por:

```python
    cagr = (last / first) ** (1 / n_years) - 1

    # Acima do limiar não é "20%": é uma taxa que não se projeta por 10 anos.
    # Devolver NaN faz o DCF se declarar inaplicável em vez de emitir o maior
    # preço justo possível. Abaixo de zero passa direto: declínio é dado válido.
    if cagr > MAX_PROJECTABLE_GROWTH:
        return np.nan

    return cagr
```

E acrescentar ao final da docstring de `_compute_fcf_cagr` (depois do parágrafo sobre a RSUL4, linha 142-143):

```python
    Não há piso: um CAGR negativo atravessa e a projeção começa em queda. E o
    limiar de cima não substitui o valor — devolve NaN, ou seja "não projetável".
```

- [ ] **Step 7: Reescrever a resolução do crescimento em `dcf_valuation`**

Substituir o bloco das linhas 261-272:

```python
        # Crescimento inicial: forward (analistas) se ligado e disponível,
        # senão CAGR histórico.
        initial_growth = _compute_fcf_cagr(fcf_series)
        growth_source = 'historical'
        if USE_FORWARD_ESTIMATES:
            if forward_growth is None:
                forward_growth = get_forward_growth(ticker_sa)
            if forward_growth is not None and pd.notna(forward_growth):
                initial_growth = max(
                    MIN_GROWTH_RATE, min(MAX_GROWTH_RATE, float(forward_growth))
                )
                growth_source = 'forward'
```

por:

```python
        # Crescimento inicial: CAGR histórico (pode ser NaN), substituído pela
        # estimativa forward quando ela está ligada, existe e é projetável.
        # Forward acima do limiar NÃO é capado: recua para o histórico.
        initial_growth = _compute_fcf_cagr(fcf_series)
        growth_source = 'historical'
        if (USE_FORWARD_ESTIMATES
                and forward_growth is not None and pd.notna(forward_growth)
                and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
            initial_growth = float(forward_growth)
            growth_source = 'forward'

        # Nenhuma taxa projetável (histórico acima do limiar e sem forward
        # utilizável): sai sem preço. O chamador recai no DDM e a coluna
        # metodo_valuation registra a substituição, em vez de o número aparecer
        # como se tivesse saído de um DCF.
        if pd.isna(initial_growth):
            return result
```

- [ ] **Step 8: Remover o import de `get_forward_growth`**

Linha 6 de `src/valuation.py`:

```python
from src.fundamentals import get_fcf_series, resolve_share_count, get_forward_growth
```

passa a:

```python
from src.fundamentals import get_fcf_series, resolve_share_count
```

- [ ] **Step 9: Atualizar a docstring de `dcf_valuation`**

Substituir, na docstring (linhas 213-215 e 224-225):

```python
    Crescimento inicial: por padrão é o CAGR histórico do FCF. Com
    USE_FORWARD_ESTIMATES ligado, usa a estimativa forward de analistas
    (yfinance) quando disponível, caindo de volta no CAGR histórico se não for.
```

por:

```python
    Crescimento inicial: por padrão é o CAGR histórico do FCF. Com
    USE_FORWARD_ESTIMATES ligado, usa a estimativa forward passada pelo chamador
    quando ela existe e é projetável (<= MAX_PROJECTABLE_GROWTH), caindo de volta
    no CAGR histórico caso contrário. Se nem uma nem outra for projetável, sai
    sem preço e o chamador recai no DDM.
```

e:

```python
        forward_growth: Crescimento forward já resolvido (decimal). Se None e
            USE_FORWARD_ESTIMATES estiver ligado, busca via get_forward_growth.
```

por:

```python
        forward_growth: Crescimento forward já resolvido em decimal, tipicamente
            de resolve_forward_growth(row). Nunca é buscado aqui dentro: o dado
            já está no CSV de fundamentos.
```

- [ ] **Step 10: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_valuation.py -q`
Expected: PASS. `TestRsul4Regression` continua verde (série com ano negativo → CAGR 0,0, preço justo abaixo do preço de mercado).

- [ ] **Step 11: Rodar a suíte inteira**

Run: `python3 -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: limites de crescimento param de substituir a taxa no DCF

Remove MIN_GROWTH_RATE (declinio passa a ser projetado como declinio) e
renomeia MAX_GROWTH_RATE para MAX_PROJECTABLE_GROWTH, que agora devolve NaN
em vez de capar: acima do limiar o DCF se declara inaplicavel e o chamador
recai no DDM, rotulado. Forward acima do limiar recua para o historico."
```

---

### Task 3: Ligar o call site e remover `get_forward_growth`

`apply_valuation` passa a alimentar o DCF com o crescimento da própria linha. Com isso `get_forward_growth` fica sem nenhum chamador e sai — junto com `_FORWARD_GROWTH_PERIODS`, cuja preferência por `'LTG'` era código morto (NaN para a ação em 100% dos tickers testados), e a economia é de 2 requisições HTTP por ticker.

**Files:**
- Modify: `src/valuation.py:457-461` (chamada dentro de `apply_valuation`)
- Modify: `src/fundamentals.py:437-489` (remoção)
- Test: `tests/test_valuation.py` (`TestMetodoValuation`), `tests/test_fundamentals.py:13-75` (remoção)

**Interfaces:**
- Consumes: `resolve_forward_growth(row)` da Task 1; `dcf_valuation(..., forward_growth=...)` da Task 2.
- Produces: `src.fundamentals` deixa de exportar `get_forward_growth`.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_valuation.py`, adicionar ao final da classe `TestMetodoValuation` (depois de `test_stock_falls_back_to_ddm_when_dcf_nan`):

```python
    def test_passes_row_forward_growth_to_dcf(self, monkeypatch):
        # O crescimento vem da LINHA (já no CSV), não de uma busca por ticker.
        captured = {}

        def fake_dcf(ticker_sa, shares_total=None, beta=None, forward_growth=None):
            captured['forward_growth'] = forward_growth
            return {'preco_justo_dcf': 20.0, 'growth_rate': 0.148, 'fcf_base': 1.0,
                    'cost_of_equity': 0.18, 'growth_source': 'forward'}

        monkeypatch.setattr(v, 'dcf_valuation', fake_dcf)
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        df = pd.DataFrame({
            'ticker': ['X'], 'ticker_sa': ['X.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
            'crescimento_receita_pct': [14.8], 'crescimento_lucro_pct': [9.2],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert captured['forward_growth'] == pytest.approx(0.148)
        assert out.loc[0, 'growth_source'] == 'forward'
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `python3 -m pytest "tests/test_valuation.py::TestMetodoValuation::test_passes_row_forward_growth_to_dcf" -q`
Expected: FAIL com `assert None == 0.148` (o chamador ainda não passa o valor).

- [ ] **Step 3: Passar o crescimento da linha no call site**

Em `src/valuation.py`, dentro de `apply_valuation`, substituir as linhas 456-461:

```python
            # DCF 2-estágios para ações
            dcf_result = dcf_valuation(
                row['ticker_sa'],
                row.get('shares_total'),
                beta,
            )
```

por:

```python
            # DCF 2-estágios para ações. O crescimento forward sai da própria
            # linha (colunas já coletadas no CSV): nenhuma requisição por ticker.
            dcf_result = dcf_valuation(
                row['ticker_sa'],
                row.get('shares_total'),
                beta,
                forward_growth=resolve_forward_growth(row),
            )
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_valuation.py -q`
Expected: PASS.

- [ ] **Step 5: Remover `get_forward_growth` de `src/fundamentals.py`**

Apagar o bloco inteiro das linhas 437-489 — os comentários de `_FORWARD_GROWTH_PERIODS`, as duas constantes e a função `get_forward_growth` — deixando o arquivo terminar em `get_fcf_series` (linha 434). Não tocar em `_estimate_cell` nem em `_extract_growth_estimates`: são elas que coletam o `+1y` de receita, lucro e número de analistas para o CSV.

Confirmar que nada mais referencia o nome:

Run: `grep -rn "get_forward_growth" src/ tests/ analysis.ipynb`
Expected: só as ocorrências em `tests/test_fundamentals.py` (removidas no próximo passo).

- [ ] **Step 6: Remover os testes órfãos de `tests/test_fundamentals.py`**

Apagar as linhas 13-75: a classe `TestGetForwardGrowth` e os helpers `_FakeTicker` e `_est`, que só ela usa. Manter os imports do topo (`numpy`, `pytest`, `pandas` continuam usados pelas outras classes) e o helper `_quarterly` (linha 78).

- [ ] **Step 7: Rodar a suíte inteira**

Run: `python3 -m pytest tests/ -q`
Expected: PASS. A contagem cai (6 testes removidos) e nenhuma falha aparece.

- [ ] **Step 8: Commit**

```bash
git add src/valuation.py src/fundamentals.py tests/test_valuation.py tests/test_fundamentals.py
git commit -m "refactor: DCF le crescimento forward do CSV e get_forward_growth sai

apply_valuation passa resolve_forward_growth(row) ao DCF. Com isso
get_forward_growth fica sem chamador: sao 2 requisicoes HTTP por ticker para
um dado que ja esta em data/fundamentals.csv, e a preferencia por 'LTG' era
codigo morto (NaN para a acao em todos os tickers testados)."
```

---

### Task 4: Registrar o driver no snapshot e alinhar as instruções

Sem o driver no histórico append-only, uma rodada com `earnings` fica indistinguível de uma com `revenue` — e o preço justo muda entre elas. As instruções para agentes em `.claude/` ainda listam `MIN_GROWTH_RATE`, que deixou de existir.

**Files:**
- Modify: `src/valuation.py:576-580` (premissas em `append_snapshot`)
- Modify: `.claude/instructions.md:51-52`
- Modify: `.claude/instructions/valuation-models.instructions.md:15-16,45`
- Test: `tests/test_valuation.py` (`TestAppendSnapshot`)

**Interfaces:**
- Consumes: `FORWARD_GROWTH_DRIVER` da Task 1.
- Produces: coluna `forward_growth_driver` em `data/valuation_history.csv`.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_valuation.py`, na classe `TestAppendSnapshot`, incluir a coluna na verificação de premissas de `test_writes_header_row_and_assumptions`:

```python
        for col in ('risk_free_rate', 'equity_risk_premium',
                    'terminal_growth', 'use_forward_estimates',
                    'forward_growth_driver'):
            assert col in out.columns
```

E adicionar um teste dedicado ao final da classe (depois de `test_empty_df_is_a_noop`):

```python
    def test_records_the_forward_growth_driver(self, monkeypatch, tmp_path):
        # Sem isso, uma rodada com 'earnings' fica indistinguível de uma com
        # 'revenue' no histórico — e o preço justo difere entre as duas.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-08-06')
        out = pd.read_csv(p)
        assert out.loc[0, 'forward_growth_driver'] == 'earnings'
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `python3 -m pytest tests/test_valuation.py::TestAppendSnapshot -q`
Expected: FAIL — `KeyError: 'forward_growth_driver'` no teste dedicado e `assert 'forward_growth_driver' in out.columns` falhando.

- [ ] **Step 3: Gravar a premissa no snapshot**

Em `src/valuation.py`, dentro de `append_snapshot`, depois da linha `snap['use_forward_estimates'] = USE_FORWARD_ESTIMATES`:

```python
    snap['forward_growth_driver'] = FORWARD_GROWTH_DRIVER
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

Run: `python3 -m pytest tests/test_valuation.py::TestAppendSnapshot -q`
Expected: PASS. `test_new_columns_align_with_older_history` continua verde — o `concat` alinha por nome e preenche o histórico antigo com NaN.

- [ ] **Step 5: Alinhar as instruções para agentes**

Em `.claude/instructions.md`, substituir as linhas 51-52:

```
| `MAX_GROWTH_RATE` | 0.20 | Cap on FCF growth rate |
| `MIN_GROWTH_RATE` | 0.0 | Floor on FCF growth rate |
```

por:

```
| `MAX_PROJECTABLE_GROWTH` | 0.20 | Threshold above which a growth rate is not projectable — returns `NaN` (no DCF), never replaces the rate |
| `FORWARD_GROWTH_DRIVER` | `revenue` | Which forward growth feeds DCF stage 1: `revenue` or `earnings` (env) |
```

Em `.claude/instructions/valuation-models.instructions.md`, substituir as linhas 15-16 pelas mesmas duas linhas acima, e substituir na seção "FCF CAGR rules" (linha 45):

```
- Cap between `MIN_GROWTH_RATE` and `MAX_GROWTH_RATE`
```

por:

```
- Above `MAX_PROJECTABLE_GROWTH` → `NaN` (not projectable; caller falls back to DDM). Negative CAGR passes through unchanged — there is no floor.
```

Não corrigir as demais linhas obsoletas dessas tabelas (`SELIC`, `TERMINAL_GROWTH = 0.035`): já estavam erradas antes desta spec e não fazem parte dela.

- [ ] **Step 6: Rodar a suíte inteira**

Run: `python3 -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 7: Confirmar que nada quebrou por nome antigo**

Run: `grep -rn "MIN_GROWTH_RATE\|MAX_GROWTH_RATE\|get_forward_growth" src/ tests/ .claude/instructions.md .claude/instructions/ analysis.ipynb`
Expected: nenhuma ocorrência (o diretório `.claude/worktrees/` é uma cópia antiga e pode ser ignorado).

- [ ] **Step 8: Commit**

```bash
git add src/valuation.py tests/test_valuation.py .claude/instructions.md .claude/instructions/valuation-models.instructions.md
git commit -m "feat: snapshot registra o driver de crescimento forward

Sem forward_growth_driver no historico append-only, uma rodada com earnings
fica indistinguivel de uma com revenue e o drift do preco justo nao pode ser
atribuido. Instrucoes de agente atualizadas para o novo nome do limiar."
```

---

## Fora de escopo (registrado na spec, não implementar aqui)

- Acoplamento `TERMINAL_GROWTH = RISK_FREE_RATE`.
- Regra "qualquer ano negativo zera o CAGR" (`src/valuation.py:150-151`) — fica a inconsistência conhecida: série com ano negativo devolve `0.0` (um valor), CAGR alto devolve `NaN` (inaplicável).
- CAGR calculado só pelas pontas da série (prompt pronto em `docs/superpowers/specs/2026-08-06-cagr-melhorado-prompt.md`).
- Revisão do valor `0.20` de `MAX_PROJECTABLE_GROWTH`.
- Exibição e filtragem do screener: seguem com o valor bruto, sem alteração.
