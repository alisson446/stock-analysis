# Direção do crescimento no estágio 1 do DCF — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o estágio 1 do DCF usar o crescimento histórico do FCF, em vez da estimativa
forward, quando as duas fontes discordam sobre a **direção** — uma diz que o caixa encolhe, a
outra que cresce.

**Architecture:** Uma função privada nova (`_forward_contradicts_history`) responde a pergunta
"as duas fontes discordam de direção?", e o bloco que hoje sobrescreve o crescimento em
`dcf_valuation` ganha um `else`. Quando a sobreposição ocorre, `initial_growth` simplesmente
não é trocado e `growth_source` recebe um terceiro valor, `'historical_override'`. Nenhuma
assinatura pública muda, nenhum call site quebra, nenhuma migração de CSV.

**Tech Stack:** Python 3, pandas, numpy, pytest. **Nenhuma dependência nova.**

**Spec:** [`docs/superpowers/specs/2026-08-18-direcao-crescimento-dcf-design.md`](../specs/2026-08-18-direcao-crescimento-dcf-design.md)

## Global Constraints

- **Nenhuma dependência nova.** `numpy` e `pandas` já cobrem tudo o que é preciso.
- **Nenhuma constante nova.** O mínimo de pontos é `4`, literal na função, com a premissa na
  docstring — é o mesmo número já derivado na spec de 2026-08-17. `MIN_TREND_R2` e
  `MAX_PROJECTABLE_GROWTH` continuam sendo aplicados por `_compute_fcf_growth`, não reabertos.
- **Rodar pytest via `rtk proxy`:** `rtk proxy python3 -m pytest tests/ -q`. A invocação direta
  quebra por causa de um hook do RTK, e o erro se parece com dependência faltando.
- **A suíte tem 2 falhas pré-existentes e dependentes de ambiente.** Não são regressão e não
  são para consertar neste plano. Confira pelos **nomes**, não pela contagem:
  - `tests/test_valuation.py::TestCostOfEquity::test_uses_beta_from_info`
  - `tests/test_filters.py::TestConfigDaRegiaoUS::test_limiares_de_barato_sao_iguais_aos_do_br`

  Estado de partida: `2 failed, 304 passed`.
- **NUNCA commitar `analysis.ipynb` nem `config/us/filters.json`.** Os dois estão modificados
  no working tree do autor de propósito. Um `git add analysis.ipynb` arrasta ~1.900 linhas de
  saída de execução para dentro do commit. Sempre `git add` por caminho explícito, nunca
  `git add -A` nem `git add .`.
- **Guideline 3:** as séries de teste são sintéticas, construídas a partir da regra. Nunca
  copiar séries do cache `data/br/fundamentals.csv`.
- **Guideline 5:** docstrings e comentários em português acessível, registrando o **porquê**.
- **Ordem da série de FCF:** `get_fcf_series` devolve do **mais recente para o mais antigo**
  (ordem do yfinance). `_compute_fcf_growth` inverte internamente. Portanto, uma série *em
  queda* escrita como `pd.Series` fica em ordem **crescente**:
  `pd.Series([100e6, 120e6, 144e6, 172.8e6])` é a série que cai de 172,8 para 100.

## File Structure

| arquivo | responsabilidade | mudança |
|---|---|---|
| `src/valuation.py` | modelos de valuation | **Modificar**: função privada nova + um `else` em `dcf_valuation` + docstring |
| `tests/test_valuation.py` | suíte de valuation | **Modificar**: 7 casos novos em `TestForwardGrowth` |
| `.claude/instructions/valuation-models.instructions.md` | referência das regras de modelagem | **Modificar**: seção "Stage 1 — Linear decay" |

Nenhum arquivo novo. A mudança é local a uma função e ao seu helper.

### Fora deste plano, de propósito

`analysis.ipynb` (célula de markdown da seção 4, linha ~1040) descreve o estágio 1 sem
mencionar o forward e ficaria desatualizado. A spec registra isso como **pendência
operacional**: o notebook está modificado no working tree do autor por outro motivo e não pode
entrar neste commit. A edição é combinada com o autor separadamente. **Não editar o notebook
neste plano.**

---

### Task 1: A regra de direção em `dcf_valuation`

**Files:**
- Modify: `src/valuation.py` (inserir função entre `resolve_forward_growth`, que termina na
  linha 448, e `dcf_valuation`, que começa na linha 451; alterar o bloco de crescimento em
  `dcf_valuation`, linhas 533-543; alterar a docstring de `dcf_valuation`, linhas 461-487)
- Modify: `.claude/instructions/valuation-models.instructions.md` (seção "Stage 1 — Linear decay")
- Test: `tests/test_valuation.py` (classe `TestForwardGrowth`, que começa na linha 531)

**Interfaces:**
- Consumes: `_compute_fcf_growth(fcf_series) -> float` (já existe, `src/valuation.py:294`),
  `MAX_PROJECTABLE_GROWTH` e `USE_FORWARD_ESTIMATES` (já existem).
- Produces: `_forward_contradicts_history(fcf_series: pd.Series, historical_growth: float,
  forward_growth: float) -> bool`. E o terceiro valor possível de
  `dcf_valuation(...)['growth_source']`: a string `'historical_override'`, ao lado das já
  existentes `'forward'`, `'historical'` e `''`.

- [ ] **Step 1: Escrever os 7 testes que falham**

Em `tests/test_valuation.py`, classe `TestForwardGrowth` (começa na linha 531), em **dois
lugares**:

- **As quatro séries** (`QUEDA_4`, `QUEDA_3`, `ALTA_4`, `ANO_NEGATIVO_4`) vão logo **abaixo do
  atributo `FCF` que já existe** na linha 541, antes do primeiro método. É onde a classe já
  guarda a sua série, e atributo de classe declarado depois dos métodos é confuso de ler.
- **Os sete métodos de teste** vão ao **final da classe**, depois do último método atual,
  `test_no_price_when_historical_is_not_projectable_and_no_forward`.

Manter a indentação de 4 espaços dos membros da classe.

As quatro séries abaixo foram conferidas numericamente: `QUEDA_4` e `QUEDA_3` dão histórico de
**−16,667%**, `ALTA_4` dá **+10,0%**, e `ANO_NEGATIVO_4` dá **NaN**.

```python
    # Séries sintéticas, na ordem do yfinance (mais recente primeiro).
    # QUEDA_4 lida do mais antigo ao mais recente é 172,8 -> 144 -> 120 -> 100:
    # cai 16,667% ao ano, R² = 1. É o formato da CMIG4, sem os números dela
    # (Guideline 3: nada copiado do cache).
    QUEDA_4 = pd.Series([100e6, 120e6, 144e6, 172.8e6])
    QUEDA_3 = pd.Series([100e6, 120e6, 144e6])          # mesma queda, 3 pontos
    ALTA_4 = pd.Series([133.1e6, 121e6, 110e6, 100e6])  # sobe 10% ao ano
    ANO_NEGATIVO_4 = pd.Series([100e6, -20e6, 120e6, 144e6])  # sem log -> NaN

    def test_historico_em_queda_derruba_forward_positivo(self, monkeypatch):
        # O caso CMIG4. O histórico diz que o caixa encolhe, com a reta
        # explicando quase toda a variação; o forward de RECEITA diz que
        # cresce. São afirmações contraditórias sobre a mesma empresa, e nada
        # na série de caixa sustenta a reversão -> projeta-se a queda.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.QUEDA_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=0.05)
        assert res['growth_source'] == 'historical_override'
        assert res['growth_rate'] == pytest.approx(-0.16667, abs=1e-5)

    def test_direcoes_concordam_na_queda_mantem_forward(self, monkeypatch):
        # Os dois dizem "encolhe". Não há discordância de DIREÇÃO para
        # arbitrar, e comparar as MAGNITUDES seria comparar crescimento de
        # receita com crescimento de caixa. O forward segue valendo.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.QUEDA_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=-0.03)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(-0.03)

    def test_direcoes_concordam_na_alta_mantem_forward(self, monkeypatch):
        # Comportamento de sempre, preservado: os dois dizem "cresce".
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.ALTA_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=0.05)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_forward_negativo_com_historico_positivo_mantem_forward(self, monkeypatch):
        # O ramo simétrico da regra: também aqui as direções discordam, e
        # também aqui vence quem projeta queda -- só que esse já é o forward,
        # então nenhum código novo participa. O teste existe para provar que
        # _forward_contradicts_history não dispara com o sinal invertido.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.ALTA_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=-0.03)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(-0.03)

    def test_serie_de_tres_pontos_nao_derruba_forward(self, monkeypatch):
        # Mesma queda de 16,667%, um ponto a menos. Com 3 pontos, metade das
        # séries SEM tendência nenhuma passam no teste de R², então a direção
        # medida não tem autoridade para derrubar uma estimativa de analista.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.QUEDA_3)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=0.05)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_historico_inutilizavel_mantem_forward(self, monkeypatch):
        # Série que passou pelo prejuízo: não existe log de negativo, o
        # histórico é NaN e não há direção para comparar. É o caso da maioria
        # dos papéis da base (10 de 15 em 2026-08-18).
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.ANO_NEGATIVO_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=0.05)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_forward_zero_e_derrubado_por_historico_em_queda(self, monkeypatch):
        # "Estagnado" ainda contradiz "encolhendo 16,667% ao ano". O divisor
        # da regra é o zero -- crescer contra encolher --, e zero fica do lado
        # de quem não encolhe.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.QUEDA_4)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=0.0)
        assert res['growth_source'] == 'historical_override'
        assert res['growth_rate'] == pytest.approx(-0.16667, abs=1e-5)
```

- [ ] **Step 2: Rodar os testes novos e confirmar que falham pelo motivo certo**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestForwardGrowth -q
```

Esperado: **2 failed** — `test_historico_em_queda_derruba_forward_positivo` e
`test_forward_zero_e_derrubado_por_historico_em_queda`, ambos com
`AssertionError: assert 'forward' == 'historical_override'`.

Os outros 5 casos novos **passam desde já**, porque descrevem comportamento que já existe.
Isso é esperado e correto: eles são testes de regressão do que a mudança **não** pode quebrar.
Se algum dos 5 falhar aqui, pare — a série de teste está errada, não o código.

- [ ] **Step 3: Escrever a função privada**

Inserir em `src/valuation.py` **entre** o fim de `resolve_forward_growth` (linha 448) e o
início de `dcf_valuation` (linha 451), separada por duas linhas em branco de cada lado:

```python
def _forward_contradicts_history(fcf_series: pd.Series, historical_growth: float,
                                 forward_growth: float) -> bool:
    """
    As duas fontes de crescimento discordam sobre a DIREÇÃO?

    O forward é crescimento de RECEITA (ver FORWARD_GROWTH_DRIVER) e o
    histórico é crescimento de CAIXA LIVRE. São grandezas diferentes, então
    comparar as magnitudes não significa nada: uma empresa pode ter receita
    subindo 4% e caixa caindo 21% sem contradição nenhuma -- basta a margem
    comprimir ou o capex subir. É o retrato normal de uma elétrica em ciclo de
    investimento.

    A DIREÇÃO, essa sim, significa a mesma coisa nas duas medidas. "O dinheiro
    que sobra para o acionista está encolhendo, com a reta explicando 93% da
    variação" e "vai crescer" são afirmações que não podem ser verdadeiras ao
    mesmo tempo sobre a mesma empresa. Quando isso acontece, o DCF projeta a
    queda: a taxa vem do histórico e o forward é descartado.

    Não é clamp e não é teto (Guideline 2): a taxa continua saindo de uma
    fonte real, apenas da outra. O divisor é o zero -- crescer contra
    encolher --, que é premissa, não um limiar ajustado à amostra
    (Guideline 3).

    Exige 4 pontos. Com 3, metade das séries SEM tendência nenhuma passam no
    teste de R² (é a distribuição do R² sob ruído, medida na spec de
    2026-08-17); com exatamente 2, o R² é sempre 1 porque a reta passa pelos
    dois pontos, e o critério de qualidade não filtra nada. Sem esse mínimo,
    dois anos de queda por acaso derrubariam uma estimativa de analista.

    O mínimo vale SÓ aqui. `_compute_fcf_growth` continua aceitando 2 pontos e
    continua sendo a taxa usada quando não há forward: este mínimo não decide
    quem recebe DCF, decide quem tem autoridade para derrubar o forward.

    A ordem da série é irrelevante -- só o tamanho dela é lido.

    Defeito declarado e aceito: uma queda real medida em 3 anos NÃO derruba o
    forward, e nesse caso o preço justo fica mais alto. Contraria a Guideline
    4 nesse caso específico. A Guideline 4 arbitra ignorância simétrica, e um
    R² de 3 pontos não é ignorância -- é uma medida que já se sabe quebrada.
    """
    return (len(fcf_series) >= 4
            and pd.notna(historical_growth) and historical_growth < 0
            and forward_growth >= 0)
```

- [ ] **Step 4: Ligar a função ao `dcf_valuation`**

Em `src/valuation.py`, substituir o bloco atual (linhas 533-543):

```python
        # Crescimento inicial: crescimento histórico do FCF (pode ser NaN),
        # substituído pela estimativa forward quando ela está ligada, existe e
        # é projetável. Forward acima do limiar NÃO é capado: recua para o
        # histórico.
        initial_growth = _compute_fcf_growth(fcf_series)
        growth_source = 'historical'
        if (USE_FORWARD_ESTIMATES
                and forward_growth is not None and pd.notna(forward_growth)
                and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
            initial_growth = float(forward_growth)
            growth_source = 'forward'
```

por:

```python
        # Crescimento inicial: crescimento histórico do FCF (pode ser NaN),
        # substituído pela estimativa forward quando ela está ligada, existe e
        # é projetável. Forward acima do limiar NÃO é capado: recua para o
        # histórico.
        initial_growth = _compute_fcf_growth(fcf_series)
        growth_source = 'historical'
        if (USE_FORWARD_ESTIMATES
                and forward_growth is not None and pd.notna(forward_growth)
                and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
            if _forward_contradicts_history(fcf_series, initial_growth,
                                            float(forward_growth)):
                # As duas fontes discordam sobre a direção: o histórico fica
                # como está e o forward é descartado. Rótulo próprio porque
                # 'historical' já significa outra coisa -- "não havia forward
                # utilizável" --, e confundir os dois faria o histórico de
                # valuation atribuir a dado o que foi mudança de premissa.
                growth_source = 'historical_override'
            else:
                initial_growth = float(forward_growth)
                growth_source = 'forward'
```

A guarda seguinte, `if pd.isna(initial_growth): return result`, fica **intacta** e continua
correta: a sobreposição só dispara com `historical_growth` não-NaN.

- [ ] **Step 5: Atualizar a docstring de `dcf_valuation`**

Em `src/valuation.py`, no parágrafo "Crescimento inicial" da docstring (linhas 461-466),
substituir:

```
    Crescimento inicial: por padrão é o crescimento histórico do FCF (tendência
    log-linear). Com USE_FORWARD_ESTIMATES ligado, usa a estimativa forward
    passada pelo chamador quando ela existe e é projetável (<=
    MAX_PROJECTABLE_GROWTH), caindo de volta no crescimento histórico caso
    contrário. Se nem uma nem outra for projetável, sai sem preço e o chamador
    recai no DDM.
```

por:

```
    Crescimento inicial: por padrão é o crescimento histórico do FCF (tendência
    log-linear). Com USE_FORWARD_ESTIMATES ligado, usa a estimativa forward
    passada pelo chamador quando ela existe e é projetável (<=
    MAX_PROJECTABLE_GROWTH), caindo de volta no crescimento histórico caso
    contrário. Se nem uma nem outra for projetável, sai sem preço e o chamador
    recai no DDM.

    Exceção: quando as duas fontes discordam sobre a DIREÇÃO -- o histórico
    bem ajustado diz que o caixa encolhe e o forward diz que cresce -- o
    forward é descartado e o histórico prevalece (ver
    _forward_contradicts_history). O forward é crescimento de receita usado
    como proxy do de caixa, e nenhuma das duas medidas sustenta a reversão que
    a substituição afirmaria.
```

E, na seção `Returns:` (linhas 483-487), substituir:

```
        dict com 'preco_justo_dcf', 'growth_rate', 'fcf_base', 'cost_of_equity',
        'growth_source' ('forward' | 'historical'),
```

por:

```
        dict com 'preco_justo_dcf', 'growth_rate', 'fcf_base', 'cost_of_equity',
        'growth_source' ('forward' = a estimativa de analista foi usada |
        'historical' = não havia forward utilizável | 'historical_override' =
        havia forward projetável, mas ele contradizia a direção do histórico e
        foi descartado),
```

- [ ] **Step 6: Rodar a classe de testes e confirmar que passa**

```bash
rtk proxy python3 -m pytest tests/test_valuation.py::TestForwardGrowth -q
```

Esperado: **14 passed** (os 7 casos que já existiam + os 7 novos), 0 failed.

- [ ] **Step 7: Rodar a suíte inteira e conferir pelos nomes**

```bash
rtk proxy python3 -m pytest tests/ -q
```

Esperado: `2 failed, 311 passed` — 304 + 7 casos novos. **Conferir os nomes das 2 falhas**,
que têm de ser exatamente as duas pré-existentes listadas em "Global Constraints"
(`test_uses_beta_from_info` e `test_limiares_de_barato_sao_iguais_aos_do_br`). Qualquer outro
nome na lista de falhas é regressão e precisa ser investigado antes de seguir.

- [ ] **Step 8: Atualizar `.claude/instructions/valuation-models.instructions.md`**

Na seção **"Stage 1 — Linear decay"**, que hoje tem duas linhas, acrescentar ao final:

```markdown
- The seed rate comes from the forward estimate when `USE_FORWARD_ESTIMATES` is on, it exists, and it is `<= MAX_PROJECTABLE_GROWTH`; otherwise from `_compute_fcf_growth`
- **Direction rule** (`_forward_contradicts_history`): when the historical FCF growth is usable **and negative** and the forward is **>= 0**, the two sources disagree about direction and the historical wins — the forward is discarded. Requires at least 4 data points; with 3 points half of trendless series pass the R² gate, and with 2 the R² is always 1. The rule is one-sided in code only: the mirror case (historical positive, forward negative) already resolves to the forward, which is the declining one
- The forward is **revenue** growth used as a proxy for **cash** growth, so only the sign is comparable across the two — never the magnitude. No threshold in percentage points; the divider is zero
- `growth_source` records which happened: `forward` / `historical` (no usable forward) / `historical_override` (there was a projectable forward and it was discarded) / empty (the DCF never reached a rate). It travels to `valuation_history.csv`
```

- [ ] **Step 9: Commit**

Usar caminhos explícitos. **Nunca** `git add -A` ou `git add .` — `analysis.ipynb` e
`config/us/filters.json` estão modificados no working tree e não podem entrar.

```bash
git add src/valuation.py tests/test_valuation.py .claude/instructions/valuation-models.instructions.md
git status --short   # confirmar que analysis.ipynb e config/us/filters.json seguem como ' M' (não staged)
git commit -m "feat: historico bem ajustado derruba o forward que contradiz a direcao

Quando o crescimento historico do FCF e utilizavel e negativo e o forward
de receita e nao-negativo, as duas fontes discordam sobre a direcao do
caixa. O DCF passa a projetar a queda: a taxa vem do historico e o forward
e descartado, com rotulo proprio em growth_source.

Recuo para fonte alternativa (Guideline 2, opcao 1), nunca clamp. Exige 4
pontos na serie -- com 3, metade das series sem tendencia passam no R2.

Spec: docs/superpowers/specs/2026-08-18-direcao-crescimento-dcf-design.md

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Verificar as consequências medidas na spec

A spec afirma números concretos sobre a base real. Esta task confere se eles se realizam com o
código escrito, em vez de presumir. Ela **não altera código** — se algum número divergir, o
resultado é uma correção na spec ou um defeito encontrado, e a decisão é do autor.

**Files:**
- Nenhum arquivo do projeto é modificado.
- Script temporário no scratchpad, descartado ao final.

**Interfaces:**
- Consumes: `dcf_valuation` e `_forward_contradicts_history` da Task 1; `apply_stock_filters`
  de `src/filters.py`; o cache `data/br/fundamentals.csv`.
- Produces: nada em código. Um relatório para o autor.

**Pré-requisito:** este passo lê `data/br/fundamentals.csv` e o `.env` (para `RISK_FREE_RATE`),
**ambos gitignored**. Se estiver rodando num worktree, ele não herda nenhum dos dois — semeie
os dois à mão antes, ou rode esta task no diretório principal.

- [ ] **Step 1: Rodar a verificação de ponta a ponta sobre a base real**

Rodar **a partir da raiz do repositório** — o script usa caminhos relativos e `from src import
...`, que dependem disso.

```bash
python3 - <<'PY'
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from src import filters, valuation as v

df = pd.read_csv('data/br/fundamentals.csv')
f = filters.apply_stock_filters(df, 'br')
betas = v.compute_sector_betas(df)

print(f"RF={v.RISK_FREE_RATE} USE_FORWARD={v.USE_FORWARD_ESTIMATES} "
      f"DRIVER={v.FORWARD_GROWTH_DRIVER}")
for _, r in f.iterrows():
    res = v.dcf_valuation(r['ticker_sa'], r.get('shares_total'),
                          betas.get(r.get('setor', ''), np.nan),
                          forward_growth=v.resolve_forward_growth(r),
                          moeda=r.get('moeda', 'BRL'))
    print(f"{r['ticker']:7} {res['growth_source']:20} "
          f"g={res['growth_rate']:+.4f} justo={res['preco_justo_dcf']:8.2f} "
          f"preco={r['preco']:.2f}")
PY
```

**Esperado**, conforme a seção "Consequências esperadas" da spec (medido em 2026-08-18 com
`RF = 0,1235`):

- **CMIG4**: `growth_source == 'historical_override'`, `g ≈ −0.2107`, preço justo
  **≈ R$ 7,35** contra preço de R$ 10,07 — ou seja, ela deixa de estar abaixo do preço justo.
- **Exatamente 1 papel** com `historical_override`. Os outros **14** saem com `forward` e com
  o mesmo preço justo de antes.
- **SEER3** continua com `forward` e continua com preço justo pelo DCF (não cai no DDM). Era o
  risco principal do desenho.

Se `RISK_FREE_RATE` no `.env` for diferente de 0,1235, os preços justos mudam de forma
proporcional e **os valores absolutos acima não se aplicam** — o que precisa valer em qualquer
caso é a contagem (1 override, 14 forward), o ticker (CMIG4), e a CMIG4 ficando **abaixo** do
preço de mercado.

- [ ] **Step 2: Relatar ao autor**

Reportar os três pontos acima com a saída real colada, e sinalizar qualquer divergência em vez
de ajustar o código para bater com a spec. Uma divergência aqui significa que a spec mediu
errado ou que a implementação diverge do desenho, e as duas são decisão do autor.

Nenhum commit nesta task.

---

## Pós-plano: o que continua pendente

- **`analysis.ipynb`** — a célula de markdown da seção 4 segue descrevendo o estágio 1 sem
  mencionar o forward nem a regra de direção. Combinar a edição com o autor fora deste commit
  (ver "Fora deste plano, de propósito").
- **Spec 2 (beta setorial)** — só começa depois desta implementação estar commitada, para que
  a tabela de consequências dela seja medida sobre o código já corrigido. A seção "Fora de
  escopo" da spec 1 já carrega as medições levantadas para ela.
