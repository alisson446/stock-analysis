# Crescimento histórico do FCF por tendência log-linear — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar o cálculo de crescimento do FCF pelas pontas da série por uma regressão log-linear sobre todos os pontos, com a qualidade do ajuste (R²) como critério de projetabilidade.

**Architecture:** A mudança inteira cabe numa função privada e numa constante de `src/valuation.py`. `_compute_fcf_cagr` vira `_compute_fcf_growth`: ajusta uma reta sobre o logaritmo dos FCFs, usa `exp(inclinação) − 1` como taxa e devolve `NaN` quando o R² fica abaixo de `MIN_TREND_R2` (série sem tendência), quando algum ano é ≤ 0, quando a série é constante ou curta demais, e quando a taxa passa de `MAX_PROJECTABLE_GROWTH`. `0.0` deixa de ser retorno possível. `dcf_valuation` já trata `NaN` corretamente (sai sem preço, chamador recai no DDM rotulado), então nenhum outro ponto do módulo muda.

**Tech Stack:** Python 3, pandas, numpy (`np.polyfit`, `np.corrcoef`, `np.log`, `np.ptp`), pytest. **Nenhuma dependência nova** — numpy já está no `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-08-06-cagr-tendencia-log-linear-design.md`

## Global Constraints

- **Nenhuma dependência nova.** Nada de scipy. `numpy` e `pandas` já cobrem tudo.
- **Comando de teste:** `rtk proxy python3 -m pytest ...`. O hook do rtk intercepta `pytest` e `python3 -m pytest` diretos e falha com `Failed to spawn process`; `rtk proxy` executa o comando cru. Baseline atual: **79 testes passando**.
- **Guideline 1** — nada aqui toca coleta, filtro ou exibição. `MIN_TREND_R2` existe apenas dentro de `src/valuation.py`.
- **Guideline 2** — todo limite tem o significado "esta taxa não é projetável", nunca "o crescimento é outro". Na prática: devolver `NaN`, nunca substituir o valor.
- **Guideline 3** — nenhum limiar é derivado de `data/fundamentals.csv`. As séries reais nos testes ilustram mecanismo; não calibram nada. **Não ajuste `MIN_TREND_R2` para fazer alguma ação passar.**
- **Guideline 5** — docstrings e comentários explicam o *porquê*, em linguagem que não pressupõe formação em finanças. Registrar a decisão contraintuitiva, não só o que o código faz.
- **Idioma:** código, comentários, docstrings, nomes de teste e mensagens de commit seguem o padrão do repositório (comentários e docstrings em português; nomes de teste e mensagens de commit em inglês/português como já praticado no arquivo).

---

## Estrutura de Arquivos

| Arquivo | Responsabilidade | O que muda |
|---|---|---|
| `src/valuation.py` | Modelos de valuation e suas premissas | Constante `MIN_TREND_R2` (nova, perto de `MAX_PROJECTABLE_GROWTH` em `:68`), `_compute_fcf_cagr` → `_compute_fcf_growth` (`:152-185`), call site em `dcf_valuation` (`:326`) |
| `tests/test_valuation.py` | Testes do módulo | `TestComputeFcfCagr` → `TestComputeFcfGrowth` (`:41-76`), `TestRsul4Regression` reescrito (`:624-641`) |
| `.claude/instructions.md` | Instruções para agentes | Linhas 30 e 51 |
| `.claude/instructions/valuation-models.instructions.md` | Instruções para agentes | Linhas 15, 34-35, 43-46 |
| `analysis.ipynb` | Notebook de análise | Célula markdown na linha 887 do JSON |

Duas tarefas: a Tarefa 1 é a mudança de comportamento inteira (código + testes, suíte verde ao final); a Tarefa 2 é documentação. Elas foram separadas porque um revisor pode aprovar uma e rejeitar a outra de forma independente.

---

### Task 1: Estimador de tendência log-linear com gate de R²

**Files:**
- Modify: `src/valuation.py:68` (constante nova logo abaixo de `MAX_PROJECTABLE_GROWTH`), `src/valuation.py:152-185` (a função), `src/valuation.py:326` (call site)
- Test: `tests/test_valuation.py:41-76` (classe substituída), `tests/test_valuation.py:624-641` (regressão reescrita)

**Interfaces:**
- Consumes: `MAX_PROJECTABLE_GROWTH` (float, `0.20`, já existe em `src/valuation.py:68`); `compute_fcf_base(fcf_series) -> float`; `discount_fcf_to_equity(fcf_base, growth, discount_rate, shares, terminal_growth=..., years=...) -> float`; `cost_of_equity(beta=None) -> float`.
- Produces:
  - `MIN_TREND_R2: float = 0.5` — módulo `src/valuation.py`.
  - `_compute_fcf_growth(fcf_series: pd.Series) -> float` — recebe a série do yfinance (**mais recente primeiro**) e devolve a taxa anual em decimal (pode ser negativa) ou `np.nan` quando a série não é modelável. **`0.0` não é mais um retorno possível.**
  - `_compute_fcf_cagr` **deixa de existir**. Nenhum código novo deve chamá-lo.

- [ ] **Step 1: Substituir a classe de teste `TestComputeFcfCagr` (linhas 41-76) pela nova**

Apague as linhas 41 a 76 de `tests/test_valuation.py` inteiras (da linha `class TestComputeFcfCagr:` até a linha `assert v._compute_fcf_cagr(pd.Series([100.0])) == 0.0`, inclusive) e ponha no lugar:

```python
class TestComputeFcfGrowth:
    """
    O crescimento sai da TENDÊNCIA da série inteira (uma reta sobre o log dos
    valores), não da comparação entre o primeiro e o último ponto. E só sai
    quando existe tendência: se a reta não explica ao menos metade da variação
    da série, a função devolve NaN e o DCF se declara inaplicável.

    As séries reais abaixo ilustram comportamentos que o código precisa
    distinguir. Nenhum limiar foi escolhido a partir delas.
    """

    def test_reads_clean_compound_growth_exactly(self):
        # 100 -> 110 -> 121 é 10% a.a. exato: a reta passa pelos três pontos.
        serie = pd.Series([121.0, 110.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(0.10, abs=1e-6)

    def test_accepts_consistent_decline(self):
        # KEPL3: 292 -> 207 -> 153 -> 51. Cai em todos os anos, então o número
        # é grande E confiável. Passa sem piso: declínio é dado válido.
        serie = pd.Series([51.0, 153.0, 207.0, 292.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(-0.4252, abs=1e-3)

    def test_rejects_cyclical_series(self):
        # RIAA3: 519 -> 951 -> 1.087 -> 351. Sobe, sobe, despenca. O cálculo
        # pelas pontas dizia -12,2%, um número que não descreve nenhum ano.
        # A regressão sozinha também não salva (-9,9%); quem barra é o R².
        serie = pd.Series([351.0, 1087.0, 951.0, 519.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_rejects_series_that_returns_to_its_start(self):
        # BLAU3: 134 -> 106 -> 366 -> 134. Termina onde começou. A regressão
        # sozinha leria +13,2%, puxada pelo pico no penúltimo ponto.
        serie = pd.Series([134.0, 366.0, 106.0, 134.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_returns_nan_when_series_contains_negative_year(self):
        # RSUL4: 21,2M -> 35,4M -> -9,8M -> 41,8M. Antes devolvia 0.0, que o
        # estágio 1 do DCF transforma em ACELERAÇÃO até TERMINAL_GROWTH --
        # ou seja, zerar inflava o preço justo em vez de ser conservador.
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_returns_nan_for_single_year(self):
        # Antes devolvia 0.0, pelo mesmo motivo e com o mesmo efeito.
        assert np.isnan(v._compute_fcf_growth(pd.Series([100.0])))

    def test_returns_nan_for_constant_series(self):
        # Sem variação não há R² (divisão por zero). É o caso extremo da
        # empresa muito estável, rejeitada de propósito: ela sai da lista em
        # vez de aparecer como barata.
        assert np.isnan(v._compute_fcf_growth(pd.Series([100.0] * 4)))

    def test_returns_nan_when_growth_is_above_projectable_threshold(self):
        # 100 -> 1000 em 1 ano = +900%. Dois pontos formam uma reta perfeita,
        # então o R² deixa passar; quem barra é o limiar de projetabilidade.
        serie = pd.Series([1000.0, 100.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_accepts_growth_exactly_at_the_threshold(self):
        # 100 -> 120 em 1 ano = +20%: no limiar, ainda projetável.
        serie = pd.Series([120.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(
            v.MAX_PROJECTABLE_GROWTH, abs=1e-9)

    def test_lets_negative_growth_through_unchanged(self):
        # 100 -> 50 em 1 ano = -50%. Sem piso: só reduz o preço justo.
        serie = pd.Series([50.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(-0.50, abs=1e-9)
```

- [ ] **Step 2: Rodar a nova classe e confirmar que falha**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py::TestComputeFcfGrowth -v`

Expected: os 10 testes falham com `AttributeError: module 'src.valuation' has no attribute '_compute_fcf_growth'`.

- [ ] **Step 3: Adicionar a constante `MIN_TREND_R2`**

Em `src/valuation.py`, logo depois da linha `MAX_PROJECTABLE_GROWTH = 0.20` (linha 68), insira:

```python
# Segundo limiar de PROJETABILIDADE, e a outra metade da pergunta. O
# MAX_PROJECTABLE_GROWTH acima responde "essa taxa se sustenta por 10 anos?".
# Este responde "essa série descreve uma trajetória, ou são quatro números
# soltos?" — via R², a fatia da variação da série que a reta de tendência
# explica. Duas perguntas independentes, por isso duas constantes.
# 0,5 é premissa, não calibração: a tendência precisa explicar MAIS DA METADE
# da variação; abaixo disso o que a série tem é mais ruído do que trajetória.
# Qualquer 0,45 ou 0,6 só se justificaria olhando quais ações passam, que é
# exatamente o que a Guideline 3 proíbe.
MIN_TREND_R2 = 0.5
```

- [ ] **Step 4: Substituir `_compute_fcf_cagr` por `_compute_fcf_growth`**

Apague a função inteira em `src/valuation.py:152-185` (de `def _compute_fcf_cagr(fcf_series: pd.Series) -> float:` até `return cagr`, inclusive) e ponha no lugar:

```python
def _compute_fcf_growth(fcf_series: pd.Series) -> float:
    """
    Crescimento anual do FCF a partir da TENDÊNCIA da série inteira.

    Ajusta uma reta sobre o logaritmo dos FCFs e usa a inclinação dela como
    taxa. O logaritmo entra porque crescimento é multiplicativo: crescer 10%
    todo ano vira uma reta perfeita no log, e a inclinação se lê direto como
    taxa (exp(inclinação) - 1).

    A versão anterior comparava só o primeiro e o último ponto e ignorava o
    caminho entre eles. A RIAA3 (519 -> 951 -> 1.087 -> 351) saía como -12,2%
    a.a., um número que não descreve nenhum dos anos: o cálculo não enxergava
    que a empresa passou por 1.087 no meio.

    Mas trocar o estimador não basta -- a regressão lê a RIAA3 como -9,9%, que
    continua sem sentido. O defeito nunca foi o VALOR da tendência: é que não
    existe tendência ali. Por isso o R² decide. Ele é a fatia da variação da
    série que a reta explica, ou seja, responde "essa série conta uma história
    ou são números soltos?". Abaixo de MIN_TREND_R2 a resposta é não e a função
    devolve NaN: o DCF se declara inaplicável, o chamador recai no DDM e
    metodo_valuation registra a troca. A taxa nunca é substituída por outra.

    Uma empresa muito estável é rejeitada de propósito: quase toda a variação
    dela é ruído -- porque quase não há variação -- e o R² fica baixo. É um erro
    conhecido, e ele erra na direção barata: tira a ação da lista em vez de
    fazê-la parecer barata.

    Devolve NaN (não modelável) ou uma taxa que pode ser negativa. Nunca 0,0:
    zerar seria substituir o número por outro e, pior, não é conservador --
    o estágio 1 do DCF faz a taxa SUBIR de 0 até TERMINAL_GROWTH (12,4%), de
    modo que a série com prejuízo acabava projetada acelerando.
    """
    if len(fcf_series) < 2:
        return np.nan

    # Ordenar do mais antigo ao mais recente: o yfinance entrega ao contrário.
    values = fcf_series.values[::-1].astype(float)

    # Não existe logaritmo de número negativo -- e uma série que passa pelo
    # prejuízo não descreve trajetória de crescimento composto. Caso RSUL4:
    # 21,2M -> 35,4M -> -9,8M -> 41,8M.
    if any(v <= 0 for v in values):
        return np.nan

    log_values = np.log(values)
    years = np.arange(len(values))

    # Série constante: o R² seria uma divisão por zero (numpy devolveria NaN
    # com RuntimeWarning). Mesmo destino da empresa muito estável descrito na
    # docstring -- não modelável por DCF.
    if np.ptp(log_values) == 0:
        return np.nan

    slope = np.polyfit(years, log_values, 1)[0]
    growth = float(np.exp(slope) - 1)

    r2 = float(np.corrcoef(years, log_values)[0, 1] ** 2)
    if r2 < MIN_TREND_R2:
        return np.nan

    # Acima do limiar não é "20%": é uma taxa que não se projeta por 10 anos.
    # Abaixo de zero passa direto: declínio é dado válido.
    if growth > MAX_PROJECTABLE_GROWTH:
        return np.nan

    return growth
```

- [ ] **Step 5: Atualizar o call site em `dcf_valuation`**

Em `src/valuation.py:326`, troque:

```python
        initial_growth = _compute_fcf_cagr(fcf_series)
```

por:

```python
        initial_growth = _compute_fcf_growth(fcf_series)
```

Depois confirme que não sobrou nenhuma referência ao nome antigo no código:

Run: `grep -rn "_compute_fcf_cagr" src/ tests/`
Expected: nenhuma saída.

- [ ] **Step 6: Rodar a nova classe e confirmar que passa**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py::TestComputeFcfGrowth -v`
Expected: 10 passed.

- [ ] **Step 7: Rodar a suíte inteira e observar a falha esperada**

Run: `rtk proxy python3 -m pytest tests/ -q`

Expected: **1 falha**, em `TestRsul4Regression::test_fair_value_is_below_market_price`. Ela é esperada e o Step 8 a resolve: o teste afirmava `fv < 47.36`, e agora `discount_fcf_to_equity` recebe `growth=NaN` e devolve `NaN` — qualquer comparação com `NaN` é falsa. Se aparecer **qualquer outra** falha, pare: alguma coisa fora do previsto quebrou.

- [ ] **Step 8: Reescrever `TestRsul4Regression`**

O teste não estava errado; ele agora pode afirmar algo mais forte. Substitua a classe inteira (`tests/test_valuation.py:624-641`) por:

```python
class TestRsul4Regression:
    """
    Regressão do caso que originou a correção: com os dados reais da RSUL4 o
    modelo dizia undervalued (R$ 309,53 vs preço R$ 47,36) enquanto a Simply
    Wall St dizia overvalued.

    A série passa pelo prejuízo (-9,8M), então não há trajetória de crescimento
    composto para projetar: o DCF não se aplica e o chamador recai no DDM. É
    uma afirmação mais forte que a anterior (o preço justo ficar abaixo do
    mercado), porque nenhum preço de DCF chega a ser emitido.
    """

    FCF_SERIES = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
    PRICE = 47.36
    TOTAL_SHARES = 6_072_128
    BETA = 1.09

    def test_growth_is_not_projectable(self):
        assert np.isnan(v._compute_fcf_growth(self.FCF_SERIES))

    def test_dcf_emits_no_price(self):
        base = v.compute_fcf_base(self.FCF_SERIES)
        growth = v._compute_fcf_growth(self.FCF_SERIES)
        coe = v.cost_of_equity(beta=self.BETA)
        fv = v.discount_fcf_to_equity(base, growth, coe, self.TOTAL_SHARES)
        assert np.isnan(fv), f"esperado NaN (DCF inaplicável), obtido R$ {fv:.2f}"
```

**Não adicione** um teste de `apply_valuation` para a RSUL4: o rótulo `ddm` no fallback já está coberto por `TestMetodoValuation::test_stock_falls_back_to_ddm_when_dcf_nan` (`tests/test_valuation.py:514`), que mocka o DCF devolvendo `NaN`. Os dois testes acima mais aquele compõem a garantia ponta a ponta.

- [ ] **Step 9: Rodar a suíte inteira e confirmar tudo verde**

Run: `rtk proxy python3 -m pytest tests/ -q`
Expected: **84 passed, 0 failed** — 79 do baseline, menos os 6 testes da `TestComputeFcfCagr` antiga e 1 do `TestRsul4Regression` antigo, mais os 10 da classe nova e os 2 do regression novo. Nenhum `RuntimeWarning` de `invalid value encountered in divide` na saída.

Se aparecer o `RuntimeWarning`, a guarda de série constante do Step 4 (`np.ptp`) não está sendo alcançada — investigue antes de commitar.

- [ ] **Step 10: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: crescimento do FCF sai da tendencia da serie, nao das pontas

_compute_fcf_cagr vira _compute_fcf_growth: regressao log-linear sobre
todos os pontos, com o R2 (MIN_TREND_R2) barrando serie sem tendencia.
RIAA3 (519 -> 951 -> 1.087 -> 351) deixa de virar -12,2% a.a.

Serie com ano <= 0 e serie curta passam a devolver NaN em vez de 0.0:
zerar nao era conservador -- o estagio 1 do DCF faz a taxa subir de 0
ate TERMINAL_GROWTH, entao a serie com prejuizo era projetada
acelerando ate 12,4% a.a.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Documentação

**Files:**
- Modify: `.claude/instructions.md:30,51`
- Modify: `.claude/instructions/valuation-models.instructions.md:15,31,34-35,43-46`
- Modify: `analysis.ipynb` (célula markdown, linha 887 do JSON)

**Interfaces:**
- Consumes: `MIN_TREND_R2` e `_compute_fcf_growth` da Task 1 — os nomes precisam bater exatamente com o que foi implementado lá.
- Produces: nada consumido por código.

- [ ] **Step 1: Atualizar `.claude/instructions.md`**

Linha 30, troque:

```markdown
- **DCF 2-Stage:** Growth rate decays linearly from historical CAGR → `TERMINAL_GROWTH` over `PROJECTION_YEARS` (10). Terminal value via Gordon Growth Model.
```

por:

```markdown
- **DCF 2-Stage:** Growth rate decays linearly from the historical FCF trend → `TERMINAL_GROWTH` over `PROJECTION_YEARS` (10). Terminal value via Gordon Growth Model.
```

Logo abaixo da linha 51 (a linha de `MAX_PROJECTABLE_GROWTH` na tabela de constantes), acrescente a linha:

```markdown
| `MIN_TREND_R2` | 0.5 | Minimum share of the series' variation the trend line must explain — below it there is no trend to project, so `_compute_fcf_growth` returns `NaN` (no DCF) |
```

- [ ] **Step 2: Atualizar `.claude/instructions/valuation-models.instructions.md`**

Abaixo da linha 15 (`MAX_PROJECTABLE_GROWTH` na tabela), acrescente:

```markdown
| `MIN_TREND_R2` | 0.5 | Minimum share of the series' variation the trend line must explain — below it there is no trend to project, so `_compute_fcf_growth` returns `NaN` (no DCF) |
```

Linha 31, troque:

```markdown
- FCF base (most recent) must be **positive** — negative FCF → use DDM fallback
```

por:

```markdown
- FCF base is the **median** of the historical series (not the most recent year, which anchors on the cycle peak) and must be **positive** — otherwise use DDM fallback
```

Linha 35, troque:

```markdown
- Growth rate decays linearly from historical FCF CAGR → `TERMINAL_GROWTH` over `PROJECTION_YEARS` years
```

por:

```markdown
- Growth rate decays linearly from the historical FCF trend → `TERMINAL_GROWTH` over `PROJECTION_YEARS` years
```

Linhas 43-46, troque o bloco inteiro:

```markdown
**FCF CAGR rules:**
- Only use positive FCF data points to compute CAGR
- Above `MAX_PROJECTABLE_GROWTH` → `NaN` (not projectable; caller falls back to DDM). Negative CAGR passes through unchanged — there is no floor.
- Fewer than 2 data points → CAGR = 0.0
```

por:

```markdown
**Historical FCF growth rules (`_compute_fcf_growth`):**
- Fit a line over `log(FCF)` across **all** points; the growth rate is `exp(slope) - 1`. Never compare only the first and last point.
- Any data point ≤ 0 → `NaN`. There is no logarithm of a negative number, and a series that passes through a loss does not describe compound growth.
- R² below `MIN_TREND_R2` → `NaN`. The series has no trend to project (a cyclical series and a consistent decline can share the same average slope).
- Constant series → `NaN` (R² would be a division by zero). A very stable company is rejected on purpose; it errs by excluding.
- Above `MAX_PROJECTABLE_GROWTH` → `NaN` (not projectable; caller falls back to DDM). Negative growth passes through unchanged — there is no floor.
- Fewer than 2 data points → `NaN`.
- **`0.0` is never returned.** Zeroing is not conservative: stage 1 raises the rate from 0 up to `TERMINAL_GROWTH`, so it inflates the fair price.
```

- [ ] **Step 3: Atualizar `analysis.ipynb`**

Numa célula markdown (linha 887 do JSON) há a string:

```
"1. **DCF 2-Estágios** — sobre **FCF alavancado** (FCFE). Base = *mediana* do FCF histórico (não o último ano, que ancora no pico do ciclo). Crescimento decai linearmente de CAGR histórico → crescimento terminal ao longo de 10 anos. Fallback: DDM quando FCF indisponível.\n",
```

Troque `de CAGR histórico` por `da tendência histórica do FCF`, deixando o resto da linha idêntico (inclusive o `\n",` final). Edite só esse trecho: `analysis.ipynb` já tem alterações não commitadas no working tree e elas não são desta mudança.

- [ ] **Step 4: Verificar que nada mais cita o nome ou o método antigo**

Run: `grep -rn "_compute_fcf_cagr\|CAGR histórico\|historical CAGR" --exclude-dir=.git --exclude-dir=worktrees .claude/ src/ tests/ analysis.ipynb docs/GUIDELINES.md`

Expected: nenhuma saída. Ocorrências em `docs/superpowers/specs/` e `docs/superpowers/plans/` são **registro histórico e não devem ser alteradas** — por isso não estão na busca.

- [ ] **Step 5: Commit**

```bash
git add .claude/instructions.md .claude/instructions/valuation-models.instructions.md analysis.ipynb
git commit -m "docs: instrucoes descrevem a tendencia log-linear e MIN_TREND_R2

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

⚠️ Se `git status` mostrar outras mudanças em `analysis.ipynb` além da linha 887 (o arquivo já estava modificado antes desta tarefa), use `git add -p analysis.ipynb` e selecione apenas o hunk da linha 887.

---

## Observado, fora de escopo

Não conserte estes itens dentro deste plano — registre e siga.

1. **`.claude/instructions/valuation-models.instructions.md` tem outras linhas obsoletas** que não têm relação com esta mudança: a tabela lista `SELIC | 0.1425 | Discount rate / cost of equity` (o código usa CAPM via `cost_of_equity`, com `RISK_FREE_RATE` e `EQUITY_RISK_PREMIUM`) e `TERMINAL_GROWTH | 0.035` (o código usa `TERMINAL_GROWTH = RISK_FREE_RATE = 0.124`). A linha 40 também referencia `SELIC` na fórmula do valor terminal. Vale uma passada de sincronização própria.

2. **A interpolação do estágio 1 converge para `TERMINAL_GROWTH` nos dois sentidos** (`src/valuation.py:213`). A KEPL3, a −42,5%, é projetada convergindo para +12,4% de crescimento em dez anos. É a mesma mecânica que tornava o `0.0` de hoje generoso, e a spec a deixou explicitamente fora de escopo.
