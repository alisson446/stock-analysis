# Base do FCF pelo nível da tendência — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer `compute_fcf_base` devolver o nível da reta de tendência no último ano quando a série de FCF descreve uma trajetória, mantendo a mediana em todos os outros casos, e registrar qual das duas foi usada no histórico de valuation.

**Architecture:** Uma função privada nova (`_fcf_trend_base`) responde "esta série tem trajetória? se sim, qual o nível dela hoje?" e devolve `NaN` quando a resposta é não. `compute_fcf_base` tenta essa via primeiro e cai na mediana quando ela recusa — assinatura inalterada, nenhum call site quebrado. `dcf_valuation` chama o mesmo helper só para rotular a origem (`'trend'`/`'median'`/`''`), e a coluna viaja até o CSV de histórico pelo mesmo caminho que `growth_source` já percorre.

**Tech Stack:** Python 3.14, pandas, numpy, pytest. Nenhuma dependência nova — `np.polyfit` e `np.corrcoef` já são usados na função vizinha.

**Spec:** `docs/superpowers/specs/2026-08-17-base-fcf-tendencia-design.md`

## Global Constraints

- **Rodar pytest via `rtk proxy`** (o hook do RTK quebra a invocação direta): `rtk proxy python3 -m pytest tests/test_valuation.py -q`
- **Guideline 2 (`docs/GUIDELINES.md`):** limites só dentro do valuation e só com o significado "não projetável". Nenhum valor é substituído por um teto; ou sai o número medido, ou sai a outra via declarada.
- **Guideline 3:** constantes saem de premissa explícita, nunca de ajuste à amostra. **Nenhum teste lê `data/*.csv`** — todas as séries são sintéticas e construídas a partir da regra.
- **Guideline 4:** na dúvida, manter a mediana e deixar a ação de fora.
- **Guideline 5:** docstrings e comentários explicam o **porquê**, sem jargão financeiro.
- **Nenhuma constante nova.** O limiar de trajetória é o `MIN_TREND_R2 = 0.5` que já existe.
- **Mínimo de pontos para usar a tendência: `4`**, literal na função. Vem da distribuição nula do R² (com n=3, 50,1% do ruído puro passa; com n=4, 29,3%), não de olhar quais ações passam.
- **`_compute_fcf_growth` não é tocada.** O mínimo de 4 pontos vale só para a base.

## Baseline vermelho conhecido (NÃO é regressão sua)

Rodando `rtk proxy python3 -m pytest tests/ -q` **antes** de qualquer mudança: **2 failed, 285 passed**. As duas falhas são pré-existentes e alheias a este plano:

| teste | causa |
|---|---|
| `test_valuation.py::TestCostOfEquity::test_uses_beta_from_info` | o teste fixa `RF = 0.124`, mas o `.env` local define `RISK_FREE_RATE=0.1235` (→ 0,20525 em vez de 0,20575). Teste dependente de ambiente. |
| `test_filters.py::TestConfigDaRegiaoUS::test_limiares_de_barato_sao_iguais_aos_do_br` | `config/us/filters.json` está modificado no working tree (`pl_max` 10→15, `exigir_*` true→false), divergindo do `br`. Alteração do usuário, não commitada. |

**Não corrija nenhuma das duas neste plano.** O alvo ao final é **2 failed, 301 passed** — os 285 do baseline mais os 16 testes novos (6 na Task 1, 4 na Task 2, 3 na Task 4, 3 na Task 5). Se aparecer uma terceira falha, é sua.

`tests/test_valuation.py` sozinho sai de **117** testes coletados para **133**.

## Estrutura de arquivos

| arquivo | responsabilidade |
|---|---|
| `src/valuation.py` | `_fcf_trend_base` (nova); `compute_fcf_base` passa a tentá-la primeiro; `dcf_valuation` rotula a origem; `apply_valuation` propaga a coluna; `_SNAPSHOT_RESULT_COLS` a inclui |
| `tests/test_valuation.py` | novos casos em `TestFcfBase`; novo caso de origem; `TestDcfValuationPorMoeda._esperado` deixa de replicar a fórmula da base; 4 mocks de `dcf_valuation` ganham a chave nova |
| `.claude/instructions/valuation-models.instructions.md` | linha 32 descreve a base como "the **median**"; passa a descrever as duas vias |
| `analysis.ipynb` | célula markdown da seção 4 descreve a base como "*mediana* do FCF histórico" |

Nenhum arquivo novo. A mudança inteira cabe em uma função privada e três call sites.

---

### Task 1: `_fcf_trend_base` — o nível da tendência, ou `NaN`

**Files:**
- Modify: `src/valuation.py` (inserir imediatamente **acima** de `compute_fcf_base`, hoje na linha 203)
- Test: `tests/test_valuation.py` (nova classe, imediatamente **acima** de `class TestFcfBase`, hoje na linha 126)

**Interfaces:**
- Consumes: `MIN_TREND_R2` (constante já existente, `= 0.5`), `np`, `pd`
- Produces: `_fcf_trend_base(fcf_series: pd.Series) -> float` — devolve o nível da reta ajustada sobre `log(FCF)` avaliada no ano mais recente, ou `np.nan` quando a série não tem trajetória projetável. **Nunca** devolve `0.0`.

- [ ] **Step 1: Write the failing test**

Inserir em `tests/test_valuation.py`, logo acima de `class TestFcfBase`:

```python
class TestFcfTrendBase:
    """
    O nível da tendência responde "onde a empresa está hoje?" -- pergunta
    diferente de "ela continua nesse ritmo?", que é do _compute_fcf_growth.
    Série sem trajetória devolve NaN e o chamador fica com a mediana.
    """

    # Todas as séries abaixo vêm do mais RECENTE ao mais antigo, como o
    # yfinance entrega. Nenhuma foi copiada de data/ -- são construídas a
    # partir da regra (Guideline 3).

    def test_serie_subindo_devolve_o_nivel_da_reta(self):
        # 100 -> 120 -> 144 -> 172,8: crescimento composto exato de 20%.
        # A reta passa pelos quatro pontos, então o nível em t=3 é 172,8.
        serie = pd.Series([172.8, 144.0, 120.0, 100.0])
        assert v._fcf_trend_base(serie) == pytest.approx(172.8)

    def test_serie_caindo_devolve_o_nivel_da_reta(self):
        # A mesma série invertida: 172,8 -> 100. A tendência se aplica nas
        # duas direções, e aqui o nível fica ABAIXO da mediana (132,0).
        serie = pd.Series([100.0, 120.0, 144.0, 172.8])
        assert v._fcf_trend_base(serie) == pytest.approx(100.0)

    def test_serie_erratica_devolve_nan(self):
        # 100 -> 163 -> 130 -> 160. R² = 0,4498 (medido), abaixo de
        # MIN_TREND_R2 -- sobe e desce sem padrão, não há trajetória.
        serie = pd.Series([160.0, 130.0, 163.0, 100.0])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_menos_de_quatro_pontos_devolve_nan(self):
        # 100 -> 120 -> 144: ajuste perfeito (R² = 1), e ainda assim recusado.
        # Com 3 pontos, metade das séries sem tendência nenhuma passam no R².
        serie = pd.Series([144.0, 120.0, 100.0])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_ano_negativo_devolve_nan(self):
        # Não existe log de número negativo, e uma série que atravessou o
        # prejuízo é justamente aquela em que extrapolar o nível é menos
        # confiável. A guarda coincide com a prudência.
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_serie_constante_devolve_nan(self):
        # Variação nula no log: o R² seria divisão por zero. Devolve NaN por
        # guarda explícita, não por acidente de ponto flutuante. Aqui tanto
        # faz para o resultado -- numa série constante o nível da reta É a
        # mediana --, mas o caminho precisa ser uma decisão, não um acidente.
        assert np.isnan(v._fcf_trend_base(pd.Series([100.0] * 4)))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfTrendBase -q
```

Esperado: 6 errors/failures com `AttributeError: module 'src.valuation' has no attribute '_fcf_trend_base'`.

- [ ] **Step 3: Write minimal implementation**

Inserir em `src/valuation.py` imediatamente **acima** de `def compute_fcf_base`:

```python
def _fcf_trend_base(fcf_series: pd.Series) -> float:
    """
    Nível da tendência no último ano, ou NaN se a série não tem trajetória.

    Responde "onde a empresa está hoje?". É pergunta diferente da que
    `_compute_fcf_growth` faz ("ela continua nesse ritmo?"), e por isso as duas
    podem discordar sobre a mesma reta -- ver a nota em `compute_fcf_base`.

    Ajusta a mesma reta sobre o logaritmo dos FCFs e a avalia no ano mais
    recente. Não é o último valor observado: a reta amortece um ano isolado de
    pico, que é a preocupação que fez a base ser a mediana em primeiro lugar.

    Devolve NaN em quatro situações, todas com o mesmo significado -- "não há
    trajetória aqui, use a mediana":

    - Menos de 4 pontos. Com 3, metade das séries sem tendência nenhuma passam
      no teste de R² (é a distribuição do R² sob ruído, não uma observação
      sobre estas ações); com 4, 29%. O yfinance entrega 4 anos, às vezes 3.
    - Qualquer ano com FCF <= 0. Não existe logaritmo de número negativo, e uma
      série que atravessou o prejuízo é justamente aquela em que extrapolar o
      nível é menos confiável -- a limitação técnica coincide com a prudência.
    - Série constante. A variação nula no log tornaria o R² uma divisão por
      zero; a guarda explícita evita decidir por acidente de ponto flutuante.
      (O resultado seria o mesmo: numa série constante o nível da reta é a
      mediana.)
    - R² abaixo de MIN_TREND_R2. A reta explica menos da metade da variação da
      série, ou seja, o que existe ali é mais ruído do que trajetória.
    """
    if len(fcf_series) < 4:
        return np.nan

    # Ordenar do mais antigo ao mais recente: o yfinance entrega ao contrário.
    values = fcf_series.values[::-1].astype(float)

    if any(v <= 0 for v in values):
        return np.nan

    log_values = np.log(values)
    years = np.arange(len(values))

    if np.ptp(log_values) == 0:
        return np.nan

    slope, intercept = np.polyfit(years, log_values, 1)

    r2 = float(np.corrcoef(years, log_values)[0, 1] ** 2)
    if r2 < MIN_TREND_R2:
        return np.nan

    return float(np.exp(intercept + slope * (len(values) - 1)))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfTrendBase -q
```

Esperado: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: _fcf_trend_base devolve o nivel da tendencia ou NaN"
```

---

### Task 2: `compute_fcf_base` tenta a tendência antes da mediana

**Files:**
- Modify: `src/valuation.py:203-214` (`compute_fcf_base`)
- Test: `tests/test_valuation.py` (adicionar a `class TestFcfBase`, hoje na linha 126)

**Interfaces:**
- Consumes: `_fcf_trend_base(fcf_series) -> float` da Task 1
- Produces: `compute_fcf_base(fcf_series: pd.Series) -> float` — **assinatura inalterada**. Devolve o nível da tendência quando ela existe, a mediana caso contrário, e `NaN` quando o resultado não é positivo ou a série é vazia.

- [ ] **Step 1: Write the failing test**

Adicionar dentro de `class TestFcfBase`, depois dos três testes existentes (que **não** devem ser alterados):

```python
    def test_serie_com_trajetoria_usa_a_tendencia_nao_a_mediana(self):
        # 100 -> 120 -> 144 -> 172,8. A mediana (132,0) é o nível de dois anos
        # atrás: numa série que sobe todo ano ela não resiste a pico nenhum,
        # ela mede o ano errado. Esperado é o nível de hoje.
        #
        # ESTE TESTE TRAVA A DECISÃO BIDIRECIONAL: uma variante
        # min(mediana, tendência) devolveria 132,0 e quebraria aqui.
        serie = pd.Series([172.8, 144.0, 120.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(172.8)

    def test_trajetoria_de_queda_tambem_usa_a_tendencia(self):
        # Mesma série invertida. O nível fica ABAIXO da mediana (132,0) -- a
        # mediana estava inflando uma empresa em declínio.
        serie = pd.Series([100.0, 120.0, 144.0, 172.8])
        assert v.compute_fcf_base(serie) == pytest.approx(100.0)

    def test_serie_sem_trajetoria_continua_na_mediana(self):
        # R² = 0,4498: sem trajetória, a mediana segue valendo.
        serie = pd.Series([160.0, 130.0, 163.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(145.0)

    def test_tres_pontos_continuam_na_mediana(self):
        serie = pd.Series([144.0, 120.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(120.0)
```

**Sobre a guarda `base > 0`:** ela permanece no código e continua valendo, mas na prática só a via da mediana pode acioná-la — `_fcf_trend_base` termina em `np.exp(...)`, que é sempre positivo. Não há teste para "tendência não-positiva" porque o caso é inalcançável, e um teste de caso inalcançável dá falsa sensação de cobertura.

- [ ] **Step 2: Run test to verify it fails**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfBase -q
```

Esperado: `2 failed, 5 passed` — falham `test_serie_com_trajetoria_usa_a_tendencia_nao_a_mediana` e `test_trajetoria_de_queda_tambem_usa_a_tendencia`, ambos recebendo 132,0 (a mediana). Os outros dois testes novos já passam, porque neles a mediana continua sendo a resposta certa.

- [ ] **Step 3: Write minimal implementation**

Substituir o corpo de `compute_fcf_base` em `src/valuation.py`:

```python
def compute_fcf_base(fcf_series: pd.Series) -> float:
    """
    Base de FCF para o DCF: nível da tendência quando existe, mediana quando não.

    A mediana foi a regra única por um tempo, para não ancorar a projeção num
    ano de pico (RSUL4: 41,8M no último ano vs. mediana de 28,3M). O argumento
    vale para série ERRÁTICA e falha para série que sobe (ou desce) todo ano:
    ali a mediana não resiste a pico nenhum -- ela É, por construção, um valor
    do meio da série, ou seja, o nível de dois anos atrás.

    O efeito era grande. A SEER3 (38,1 -> 67,7 -> 116,3 -> 289,0 em R$ mi) tinha
    base 92,0 e o DCF projetava o ano 1 em 96,9 -- abaixo do que a empresa já
    havia entregue. Como o preço justo é linear na base, a subestimação passava
    inteira: R$ 10,98 contra R$ 31,08 partindo do nível atual.

    E errava nos dois sentidos: na CMIG4, caindo todo ano, a mediana ficava
    ACIMA do nível atual e inflava uma empresa em declínio.

    Quem decide qual via vale é `_fcf_trend_base`, que devolve NaN quando a
    série não tem trajetória. Na dúvida, sempre a mediana (Guideline 4).

    Nota sobre uma aparente contradição: na SEER3 esta função aceita o nível da
    reta (260,4M) enquanto `_compute_fcf_growth` rejeita a INCLINAÇÃO da mesma
    reta (+93,8% ao ano, acima do projetável). Não é incoerência -- são duas
    afirmações separáveis. "Onde a empresa está hoje" é sobre o passado
    observado, e a reta responde bem. "Se ela mantém esse ritmo" é sobre dez
    anos de futuro, e a reta responde mal. O contrário é que seria estranho:
    projetar a taxa e desprezar o nível.
    """
    if fcf_series.empty:
        return np.nan

    base = _fcf_trend_base(fcf_series)
    if pd.isna(base):
        base = float(np.median(fcf_series.values))

    return base if base > 0 else np.nan
```

- [ ] **Step 4: Run test to verify it passes**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfBase tests/test_valuation.py::TestFcfTrendBase -q
```

Esperado: `13 passed` (6 da Task 1 + 3 existentes + 4 novos).

- [ ] **Step 5: Rodar o arquivo inteiro para ver o estrago previsto**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py -q
```

Esperado: `3 failed` — o `test_uses_beta_from_info` do baseline, mais `TestDcfValuationPorMoeda::test_dolar_usa_juro_e_perpetuidade_do_dolar` e `::test_default_continua_em_reais`. As duas últimas são **previstas** e resolvidas na Task 3. Não conserte nada agora.

- [ ] **Step 6: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "fix: base do FCF usa o nivel da tendencia em serie com trajetoria"
```

---

### Task 3: destravar `TestDcfValuationPorMoeda`

**Files:**
- Modify: `tests/test_valuation.py:980-983` (`TestDcfValuationPorMoeda._esperado`)

**Interfaces:**
- Consumes: `compute_fcf_base` da Task 2
- Produces: nada — só devolve dois testes existentes ao verde.

**Contexto:** `_esperado` replica a fórmula da base (`np.median(...)`) para calcular o preço justo que espera. A série da classe — `[133.1e6, 121e6, 110e6, 100e6]`, crescimento composto exato de 10% — tem 4 pontos, é toda positiva e ajusta com R² = 1, então a base passou de 115,5M (mediana) para 133,1M (tendência) e a expectativa ficou desatualizada.

Esses dois testes existem para verificar **qual moeda decide o juro de desconto e a perpetuidade**, não qual é a base. Replicar a regra da base ali foi o que os tornou frágeis. Delegar ao módulo devolve a eles o escopo que deveriam ter.

- [ ] **Step 1: Confirmar que as duas falhas são exatamente estas**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestDcfValuationPorMoeda -q
```

Esperado: `2 failed, 1 passed`. `test_moeda_sem_premissas_sai_sem_preco` passa (não usa `_esperado`).

- [ ] **Step 2: Trocar a réplica da fórmula por uma chamada ao módulo**

Substituir `_esperado` em `tests/test_valuation.py`:

```python
    def _esperado(self, discount_rate, terminal_growth):
        # A base sai de compute_fcf_base, não de uma cópia da regra aqui: o
        # que estes testes verificam é qual MOEDA decide o juro de desconto e
        # a perpetuidade. Duplicar a fórmula da base fazia eles quebrarem a
        # cada mudança nela, por um motivo que não é o deles.
        return v.discount_fcf_to_equity(
            fcf_base=v.compute_fcf_base(self._serie_crescendo()),
            growth=0.10, discount_rate=discount_rate, shares=1e6,
            terminal_growth=terminal_growth)
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestDcfValuationPorMoeda -q
```

Esperado: `3 passed`.

- [ ] **Step 4: Confirmar que só sobra o vermelho do baseline**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py -q
```

Esperado: `1 failed, 126 passed` — só o `test_uses_beta_from_info` do baseline.

- [ ] **Step 5: Commit**

```bash
git add tests/test_valuation.py
git commit -m "test: _esperado usa compute_fcf_base em vez de replicar a mediana"
```

---

### Task 4: `fcf_base_source` — origem da base no resultado do DCF

**Files:**
- Modify: `src/valuation.py:410-416` (dict `result` de `dcf_valuation`), `src/valuation.py:423` (logo após `fcf_base`), `src/valuation.py:406-408` (docstring `Returns:`)
- Test: `tests/test_valuation.py` (nova classe, imediatamente **acima** de `class TestDcfValuationPorMoeda`, hoje na linha 972)

**Interfaces:**
- Consumes: `_fcf_trend_base` da Task 1
- Produces: `dcf_valuation(...)` passa a incluir a chave `'fcf_base_source'` no dict de retorno, com três valores possíveis: `'trend'`, `'median'` e `''`.

**Contrato dos três valores** — a distinção existe porque sem ela o histórico registraria `'median'` para linhas em que o DCF nem começou, e a coluna passaria a mentir sobre o que aconteceu:

| valor | significado |
|---|---|
| `'trend'` | a série tinha trajetória; a base é o nível da reta |
| `'median'` | a série não tinha trajetória; a base é a mediana |
| `''` | não se chegou a escolher uma base — série de FCF vazia, mediana não-positiva, ou contagem de ações ausente |

- [ ] **Step 1: Write the failing test**

Inserir em `tests/test_valuation.py`, logo acima de `class TestDcfValuationPorMoeda`:

```python
class TestFcfBaseSource:
    """
    A origem da base viaja junto com o preço justo, pelo mesmo motivo que
    growth_source: sem ela, o preço justo de uma ação salta entre duas rodadas
    do histórico sem nada no arquivo explicando por quê, e o salto fica
    indistinguível de uma mudança de fundamento.
    """

    def test_serie_com_trajetoria_e_rotulada_trend(self, monkeypatch):
        serie = pd.Series([172.8e6, 144e6, 120e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == 'trend'

    def test_serie_erratica_e_rotulada_median(self, monkeypatch):
        # R² = 0,4498: sem trajetória, base pela mediana.
        serie = pd.Series([160e6, 130e6, 163e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == 'median'

    def test_sem_base_escolhida_fica_vazio(self, monkeypatch):
        # Série vazia: o DCF sai antes de escolher qualquer base. String vazia
        # é diferente de 'median' -- aqui não houve escolha nenhuma.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: pd.Series(dtype=float))
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == ''
        assert pd.isna(res['preco_justo_dcf'])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfBaseSource -q
```

Esperado: `3 failed` com `KeyError: 'fcf_base_source'`.

- [ ] **Step 3: Write minimal implementation**

Em `src/valuation.py`, no dict `result` de `dcf_valuation`, adicionar a chave logo depois de `'growth_source'`:

```python
    result = {
        'preco_justo_dcf': np.nan,
        'growth_rate': np.nan,
        'fcf_base': np.nan,
        'cost_of_equity': np.nan,
        'growth_source': 'historical',
        # String vazia = não se chegou a escolher uma base (série vazia,
        # mediana não-positiva, sem contagem de ações). É diferente de
        # 'median', que significa "escolheu-se a mediana".
        'fcf_base_source': '',
    }
```

Logo **depois** da guarda `if pd.isna(fcf_base): return result` (hoje nas linhas 423–425), gravar a origem:

```python
        # Mesmo helper que compute_fcf_base consultou. O ajuste é refeito de
        # propósito: são 4 pontos de numpy, e o custo é nulo perto de manter
        # compute_fcf_base com assinatura estável (devolver uma tupla
        # (base, origem) quebraria os testes e o call site sem ganho).
        result['fcf_base_source'] = (
            'trend' if pd.notna(_fcf_trend_base(fcf_series)) else 'median')
```

E na docstring, na seção `Returns:`, trocar a última linha por:

```
        'growth_source' ('forward' | 'historical'),
        'fcf_base_source' ('trend' | 'median' | '' quando não houve base).
```

- [ ] **Step 4: Run test to verify it passes**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestFcfBaseSource -q
```

Esperado: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: dcf_valuation rotula a origem da base do FCF"
```

---

### Task 5: propagar `fcf_base_source` até o CSV de histórico

**Files:**
- Modify: `src/valuation.py:634` (listas de acumulação), `:656` (reset por linha), `:682` (leitura do dict), `:694` (append), `:713` (coluna), `:752-760` (`_SNAPSHOT_RESULT_COLS`)
- Modify: `tests/test_valuation.py:549`, `:563`, `:581`, `:965` (os quatro mocks de `dcf_valuation`)
- Test: `tests/test_valuation.py` (adicionar a `class TestAppendSnapshot`, hoje na linha 596)

**Interfaces:**
- Consumes: `dcf_valuation(...)['fcf_base_source']` da Task 4
- Produces: `apply_valuation(...)` devolve a coluna `fcf_base_source`; `append_snapshot` a grava no CSV logo após `growth_source`.

**Atenção aos mocks:** `apply_valuation` lê `dcf_result['growth_source']` por indexação direta, e a chave nova segue o mesmo padrão. Os quatro `monkeypatch.setattr(v, 'dcf_valuation', ...)` do arquivo devolvem dicts montados à mão e **vão estourar `KeyError`** até ganharem a chave. Isso é desejável: um mock que se afasta do contrato real é exatamente o que um `KeyError` deve pegar. Não troque a indexação por `.get()` para contornar.

- [ ] **Step 1: Write the failing test**

Adicionar dentro de `class TestAppendSnapshot`:

```python
    def test_snapshots_fcf_base_source(self, tmp_path):
        # A origem da base viaja até o CSV pelo mesmo caminho que a origem do
        # crescimento: sem ela, um salto no preço justo entre duas rodadas
        # fica indistinguível de uma mudança de fundamento.
        p = tmp_path / 'hist.csv'
        df = self._valued()
        df['fcf_base_source'] = ['trend']
        v.append_snapshot(df, path=p, snapshot_date='2026-08-17')
        out = pd.read_csv(p)
        assert out.loc[0, 'fcf_base_source'] == 'trend'
```

E adicionar dentro de `class TestApplyValuation`, junto dos outros testes de rótulo:

```python
    def test_stock_propaga_fcf_base_source(self, monkeypatch):
        monkeypatch.setattr(v, 'dcf_valuation', lambda *a, **k: {
            'preco_justo_dcf': 20.0, 'growth_rate': 0.1, 'fcf_base': 1.0,
            'cost_of_equity': 0.18, 'growth_source': 'historical',
            'fcf_base_source': 'trend',
        })
        df = pd.DataFrame({
            'ticker': ['X'], 'ticker_sa': ['X.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert out.loc[0, 'fcf_base_source'] == 'trend'

    def test_banco_nao_tem_fcf_base_source(self, monkeypatch):
        # Banco não passa por DCF; a coluna existe e fica vazia, como
        # growth_source já faz.
        df = pd.DataFrame({
            'ticker': ['B'], 'ticker_sa': ['B.SA'], 'setor': ['Banks'],
            'roe_pct': [25.0], 'vpa': [10.0], 'lpa': [2.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Banks'), model='bank')
        assert out.loc[0, 'fcf_base_source'] == ''
```

- [ ] **Step 2: Run test to verify it fails**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestApplyValuation tests/test_valuation.py::TestAppendSnapshot -q
```

Esperado: falhas por `KeyError: 'fcf_base_source'` nos testes novos.

- [ ] **Step 3: Atualizar os quatro mocks de `dcf_valuation`**

Nos quatro pontos abaixo, acrescentar `'fcf_base_source': 'median',` ao dict devolvido, ao lado de `'growth_source'`:

- `tests/test_valuation.py:549` — `test_stock_with_valid_dcf_is_labeled_dcf`
- `tests/test_valuation.py:563` — `test_stock_falls_back_to_ddm_when_dcf_nan`
- `tests/test_valuation.py:581` — `test_passes_row_forward_growth_to_dcf` (dentro de `fake_dcf`)
- `tests/test_valuation.py:965` — `test_dcf_recebe_a_moeda_da_linha` (dentro de `fake_dcf`)

- [ ] **Step 4: Write minimal implementation**

Em `apply_valuation`, quatro edições espelhando o que `growth_source` já faz:

```python
    # junto das outras listas de acumulação (hoje linha 634)
    fcf_base_sources = []
```

```python
    # junto do reset por linha (hoje linha 656)
    fcf_base_source = ''
```

```python
    # no ramo de sucesso do DCF (hoje linha 682), junto de growth_source
    growth_source = dcf_result['growth_source']
    fcf_base_source = dcf_result['fcf_base_source']
```

```python
    # junto do append de growth_sources (hoje linha 694)
    fcf_base_sources.append(fcf_base_source)
```

```python
    # junto da atribuição de colunas (hoje linha 713)
    df['fcf_base_source'] = fcf_base_sources
```

E em `_SNAPSHOT_RESULT_COLS`, logo depois de `'growth_source'`:

```python
    'preco_justo_dcf', 'metodo_valuation', 'growth_source', 'fcf_base_source',
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py -q
```

Esperado: `1 failed, 132 passed` — só o `test_uses_beta_from_info` do baseline.

- [ ] **Step 6: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: fcf_base_source viaja de apply_valuation ate o historico"
```

---

### Task 6: documentação e verificação de ponta a ponta

**Files:**
- Modify: `.claude/instructions/valuation-models.instructions.md:32`
- Modify: `analysis.ipynb` (célula markdown `4f45fd7b`, seção "## 4. Valuation — Preço Justo")

**Interfaces:**
- Consumes: tudo das Tasks 1–5
- Produces: nada de código.

- [ ] **Step 1: Atualizar `valuation-models.instructions.md`**

A linha 32 hoje diz:

```markdown
- FCF base is the **median** of the historical series (not the most recent year, which anchors on the cycle peak) and must be **positive** — otherwise use DDM fallback
```

Substituir por:

```markdown
- FCF base is the **trend level at the most recent year** when the series has a trajectory — at least 4 points, all positive, and R² ≥ `MIN_TREND_R2`. Otherwise it is the **median** of the historical series. The median alone anchors on the wrong year in a monotonic series: it is by construction a mid-series value, i.e. the level from two years ago. The base must be **positive** either way — otherwise use DDM fallback
- The base is labeled in `fcf_base_source` (`trend` / `median` / empty when no base was chosen) and travels to `valuation_history.csv`, same as `growth_source`
- Note the deliberate asymmetry: `_fcf_trend_base` may accept a line's **level** while `_compute_fcf_growth` rejects the same line's **slope**. "Where the company is today" and "does it keep this pace for 10 years" are different questions
```

- [ ] **Step 2: Atualizar a célula markdown do notebook**

Em `analysis.ipynb`, na célula da seção 4, o item 1 hoje diz:

```
1. **DCF 2-Estágios** — sobre **FCF alavancado** (FCFE). Base = *mediana* do FCF histórico (não o último ano, que ancora no pico do ciclo). Crescimento decai linearmente da tendência histórica do FCF → crescimento terminal ao longo de 10 anos. Fallback: DDM quando FCF indisponível.
```

Substituir a segunda frase por:

```
1. **DCF 2-Estágios** — sobre **FCF alavancado** (FCFE). Base = *nível da tendência* no último ano quando a série tem trajetória (≥ 4 anos, todos positivos, R² ≥ 0,5); *mediana* do histórico caso contrário. A mediana sozinha mede o ano errado numa série que sobe ou desce todo ano — ela é, por construção, o nível de dois anos atrás. Qual das duas foi usada fica em `fcf_base_source`. Crescimento decai linearmente da tendência histórica do FCF → crescimento terminal ao longo de 10 anos. Fallback: DDM quando FCF indisponível.
```

Use a ferramenta de edição de notebook, não edição de texto no JSON cru.

- [ ] **Step 3: Rodar a suíte inteira**

```bash
rtk proxy python3 -m pytest tests/ -q
```

Esperado: **`2 failed, 301 passed`** — exatamente as duas falhas do baseline documentadas no topo deste plano (`test_uses_beta_from_info` e `test_limiares_de_barato_sao_iguais_aos_do_br`), e nenhuma outra.

- [ ] **Step 4: Verificar o efeito real nas duas ações previstas pela spec**

```bash
rtk proxy python3 -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src import valuation as v, fundamentals as f
for t, beta in [('SEER3', 0.74), ('CMIG4', 0.54)]:
    fu = pd.read_csv('data/br/fundamentals.csv')
    r = fu[fu.ticker == t].iloc[0]
    res = v.dcf_valuation(t + '.SA', r.shares_total, beta,
                          forward_growth=r.crescimento_receita_pct / 100)
    print(f\"{t}: R\$ {res['preco_justo_dcf']:.2f} | origem={res['fcf_base_source']}\")
"
```

Esperado, conforme a seção "Consequências esperadas" da spec:

```
SEER3: R$ 31.08 | origem=trend
CMIG4: R$ 22.56 | origem=trend
```

Se algum dos dois divergir em mais de um centavo, **pare e reporte** — o modelo não está fazendo o que a spec previu.

- [ ] **Step 5: Commit**

```bash
git add .claude/instructions/valuation-models.instructions.md analysis.ipynb
git commit -m "docs: base do FCF pelo nivel da tendencia nas instrucoes e no notebook"
```

---

## Fora de escopo (não implemente)

Registrados na spec, cada um com spec própria pela frente:

1. **O crescimento forward sobrescreve um histórico bem ajustado sem olhar a direção.** CMIG4 cai 21,1% ao ano com R² = 0,93 e o código usa +4,26% de receita. Maior impacto isolado da investigação, mas mexe no fluxo forward, não na base.
2. **Beta setorial grosso demais.** SEER3 em *Consumer Defensive* (beta 0,74) contra 1,48 individual. É spec de custo de capital.
3. **A rampa do estágio 1 acelera empresas lentas** até `TERMINAL_GROWTH`. Já registrado como fora de escopo desde a spec de 2026-08-06; a Simply Wall St faz igual.
4. **As duas falhas do baseline.** `test_uses_beta_from_info` depende do `.env` local; `test_limiares_de_barato_sao_iguais_aos_do_br` depende de uma edição não commitada do usuário em `config/us/filters.json`.
