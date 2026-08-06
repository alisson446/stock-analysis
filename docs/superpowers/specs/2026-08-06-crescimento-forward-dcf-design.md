# Crescimento forward no estágio 1 do DCF

Spec de design — 2026-08-06

## Problema

O estágio 1 do DCF é semeado por um único número de crescimento, e a forma como esse
número é obtido e tratado tem quatro defeitos independentes:

1. `get_forward_growth` (src/fundamentals.py:447) busca `growth_estimates` por ticker —
   2 requisições HTTP, das quais uma serve só para trazer `industryTrend`/`sectorTrend`/
   `indexTrend`, que o projeto não consome. O dado equivalente já está em
   `data/fundamentals.csv` desde a spec do filtro de crescimento projetado.
2. A função prioriza o período `'LTG'` em `_FORWARD_GROWTH_PERIODS`, que retorna NaN para
   a ação em 100% dos tickers testados (BR e US). A preferência é código morto: sempre cai
   no `'+1y'`.
3. Ela lê `growth_estimates.stockTrend`, que é idêntico a `earnings_estimate.growth`. Um
   DCF de fluxo de caixa livre está sendo alimentado por crescimento de **lucro**.
4. `MIN_GROWTH_RATE = 0.0` e `MAX_GROWTH_RATE = 0.20` substituem silenciosamente o valor
   de entrada, produzindo um preço justo que aparenta ter modelado a empresa quando
   modelou outra.

## Decisões

### 1. Ancoragem: um único período (`+1y`)

A Yahoo entrega dois períodos anuais — `0y` (exercício corrente) e `+1y` (próximo).
Considerou-se usá-los como anos 1 e 2 da projeção, ou semear o fade com a média dos dois.
**Ambas foram descartadas.**

Motivo determinante: o filtro de screening corta por `+1y > 0`
(src/filters.py:37-41, `exigir_estimativa`), e o DCF só roda sobre o DataFrame já filtrado
(src/valuation.py:420). Semear o DCF com uma média que inclui o `0y` cria uma incoerência
entre as duas partes do mesmo pipeline: uma certifica a empresa como crescente e a outra a
projeta com uma taxa que pode ser negativa, sobre o mesmo dado da mesma empresa, sem razão
principiada para divergirem.

Motivo secundário: a base do DCF é a **mediana** da série histórica de FCF
(src/valuation.py:120-131), escolhida de propósito para não ancorar no pico do ciclo. O
"ano 1" da projeção não corresponde a exercício fiscal nenhum. Tratar o `0y` como "ano 1"
afirma um alinhamento temporal que o modelo não sustenta.

Consequência: **nenhuma coluna nova em `data/fundamentals.csv` e nenhum refetch.**

### 2. Driver: configurável, default receita

Nova variável de ambiente `FORWARD_GROWTH_DRIVER`, valores `revenue` (default) ou
`earnings`. As duas colunas já existem no CSV, então o custo é uma leitura condicional.

Receita é o default por razão estrutural, não amostral: alavancagem operacional, itens não
recorrentes e efeitos fiscais amplificam a oscilação do lucro sobre a mesma variação de
receita. Além disso o DCF projeta fluxo de caixa livre, que é receita menos custos caixa e
capex — não lucro contábil.

O cache atual ilustra o efeito (receita: mediana 7,9%, desvio 22,7; lucro: mediana 22,8%,
desvio 152,9, máximo 1410,7% em ONCO3), mas a justificativa **não** se apoia nesses
números: os fundamentos são atualizados trimestralmente e a composição da base muda.

### 3. Piso removido

`MIN_GROWTH_RATE` é eliminado, nos dois usos (src/valuation.py:159 e :269-271).

Um piso seleciona por **magnitude**, e magnitude não indica falta de confiabilidade. O
contrário é mais comum: uma empresa que cai todo ano por quatro anos produz um número
grande *e* confiável. Nos dados de 2026-08-06 — ilustração, não prova — um piso em −0,20
afetaria apenas KEPL3 (FCF 292 → 207 → 153 → 51 R$ mi, queda monotônica, CAGR −44,1%),
justamente a série mais limpa, e não encostaria em RIAA3 (519 → 951 → 1.087 → 351, CAGR
−12,2%), cujo número é um resumo ruim de uma série volátil.

Remover o piso é **monotonicamente conservador**: todo preço justo ou fica igual ou cai,
nunca sobe. O erro resultante é falso negativo (perder uma oportunidade), nunca falso
positivo (comprar uma empresa em declínio achando que está barata). Para um screener de
valor, essa é a direção segura de errar.

### 4. Teto vira critério de projetabilidade

`MAX_GROWTH_RATE` é renomeado para `MAX_PROJECTABLE_GROWTH` e **deixa de substituir o
valor**. Passa a responder a pergunta "consigo projetar essa taxa por 10 anos?" — e quando
a resposta é não, o modelo recua ou se declara inaplicável, em vez de inventar um número.

O teto atual não protege; disfarça. Com `earnings` e o lucro `+1y` de ONCO3 (+1410,7%), o
teto produz seed de 20% e preço justo de 1.990 numa escala onde uma empresa boa e normal
vale ~1.200 — ou seja, o **maior preço justo que o modelo consegue emitir**, o que
maximiza a chance de a ação aparecer como `undervalued`. Sem teto o resultado é
35.548.661.631 e escandalosamente óbvio. O erro discreto é pior que o escandaloso, porque
chega ao usuário como recomendação.

O valor 0,20 é mantido nesta spec por não haver base para trocá-lo; sua revisão fica
registrada como item adiado.

### 5. Guarda de validade em −100%

Crescimento ≤ −100% só ocorre com driver `earnings`, quando o lucro vira prejuízo: a razão
`(estimativa − realizado)/|realizado|` tem o denominador cruzando zero e deixa de
significar uma taxa. Não é um valor extremo, é um **número inválido**.

Tratado como dado ausente (`NaN`), recaindo no CAGR histórico como qualquer estimativa
faltante. Esta guarda pertence à validação de dado, não à premissa econômica — a distinção
que estava conflada no piso.

### 6. Leitura do CSV no call site

`dcf_valuation` já aceita o parâmetro `forward_growth` (src/valuation.py:205); o chamador
em lote simplesmente não o passa (src/valuation.py:457-461). Passar o valor da linha
elimina a busca por ticker sem reescrever nada.

## O que o screener exibe e filtra: inalterado

Nenhuma decisão desta spec toca a exibição ou a filtragem. As colunas
`crescimento_receita_pct`, `crescimento_lucro_pct` e `num_analistas` continuam com o
**valor bruto** em `data/fundamentals.csv`, em `src/filters.py` e no snapshot. PETR4
aparece com −4,61%; ONCO3 aparece com +1410,7%.

Limites existem apenas dentro da projeção do DCF, e apenas com o significado "esta taxa não
é projetável por 10 anos" — nunca "o crescimento é outro".

## Implementação

### `src/fundamentals.py` — só remoção

Excluir `get_forward_growth`, `_FORWARD_GROWTH_PERIODS` e `_FORWARD_GROWTH_STOCK_COLS`
(linhas 437-489). A prioridade morta por `'LTG'` sai junto.

`_estimate_cell` e `_extract_growth_estimates` ficam inalteradas — já coletam o `+1y` de
receita, lucro e número de analistas para o CSV, que é tudo de que a decisão 1 precisa.

### `src/valuation.py`

**Constantes**

- Remover `MIN_GROWTH_RATE`.
- Renomear `MAX_GROWTH_RATE` → `MAX_PROJECTABLE_GROWTH` (valor 0.20 mantido), com
  docstring explicando que é limiar de projetabilidade, não teto de crescimento.
- Adicionar `FORWARD_GROWTH_DRIVER`, lido com `os.getenv('FORWARD_GROWTH_DRIVER',
  'revenue')`. Valor fora de `{'revenue', 'earnings'}` cai no default com aviso impresso,
  seguindo o padrão de `_env_float`. Não criar helper `_env_str` para um único uso.

**`resolve_forward_growth(row) -> float`** (nova)

Lê da linha do DataFrame a coluna correspondente ao driver configurado
(`crescimento_receita_pct` ou `crescimento_lucro_pct`), converte de pontos percentuais para
decimal e retorna `NaN` quando o valor é ausente ou ≤ −1,0. Não avalia projetabilidade —
essa decisão fica concentrada em `dcf_valuation`.

Mora em `valuation.py`, não em `fundamentals.py`: é decisão de modelagem do DCF e lê a flag
de valuation. `fundamentals.py` apenas coleta.

**`_compute_fcf_cagr`**

- Remover o `max(MIN_GROWTH_RATE, ...)`: CAGR negativo passa adiante sem alteração.
- Trocar `min(MAX_GROWTH_RATE, cagr)` por: se `cagr > MAX_PROJECTABLE_GROWTH`, retornar
  `np.nan` (não projetável).
- A regra "qualquer ano ≤ 0 na série zera o CAGR" (linhas 150-151) permanece — está fora
  do escopo desta spec. Fica registrada a inconsistência resultante: série com ano negativo
  devolve `0.0` (um valor), CAGR alto devolve `NaN` (inaplicável). Resolver junto com o
  item adiado correspondente.

**`dcf_valuation`**

Ordem de resolução do crescimento inicial:

1. `initial_growth = _compute_fcf_cagr(fcf_series)`, `growth_source = 'historical'`.
   Pode ser `NaN`.
2. Se `USE_FORWARD_ESTIMATES` e `forward_growth` for válido (não `None`, não `NaN`) e
   `forward_growth <= MAX_PROJECTABLE_GROWTH`: usar, `growth_source = 'forward'`.
   Caso contrário, manter o histórico — inclusive quando o forward existe mas não é
   projetável.
3. Se `initial_growth` for `NaN` após os passos acima, retornar `result` sem preço. O
   chamador recai no DDM e o rótulo `metodo_valuation` registra a substituição.

Remover o ramo `if forward_growth is None: forward_growth = get_forward_growth(ticker_sa)`
e o import correspondente na linha 6.

**`valuate_dataframe`** (linha 457)

```python
dcf_result = dcf_valuation(
    row['ticker_sa'],
    row.get('shares_total'),
    beta,
    forward_growth=resolve_forward_growth(row),
)
```

**`append_snapshot`**

Adicionar `snap['forward_growth_driver'] = FORWARD_GROWTH_DRIVER` junto das demais
premissas da rodada. Sem isso, um snapshot rodado com `earnings` fica indistinguível de um
rodado com `revenue` no histórico append-only.

## Testes

`tests/test_fundamentals.py`
- Excluir a classe `TestGetForwardGrowth` (linhas 30-75).

`tests/test_valuation.py`
- `resolve_forward_growth`: driver `revenue`; driver `earnings`; conversão de pontos
  percentuais para decimal; coluna com `NaN` → `NaN`; valor ≤ −1,0 → `NaN`; driver
  inválido na env → usa `revenue`.
- `_compute_fcf_cagr`: CAGR negativo atravessa sem alteração; CAGR acima do limiar →
  `NaN`; série com ano ≤ 0 continua devolvendo `0.0`.
- `dcf_valuation`: forward acima do limiar → `growth_source == 'historical'`; forward
  válido → `'forward'`; histórico `NaN` e sem forward → sem `preco_justo_dcf`; crescimento
  negativo produz preço menor que crescimento zero.
- `discount_fcf_to_equity`: crescimento negativo produz preço positivo e finito;
  crescimento ≤ −100% produz `NaN`.
- Ajustar os testes existentes que fazem `monkeypatch` de `get_forward_growth`
  (linhas 308-336) para passar `forward_growth` diretamente.

## Consequências esperadas

Medidas sobre `data/fundamentals.csv` de 2026-08-06 — **ilustrativas**. Os fundamentos são
atualizados trimestralmente e a composição da base muda; nenhuma decisão acima depende
destes números.

Das 14 ações que hoje passam no filtro, com `USE_FORWARD_ESTIMATES` desligado (o default):

- 3 tinham CAGR negativo achatado para zero e passam a ser projetadas em queda: BLAU3
  (−0,1%), RIAA3 (−12,2%), KEPL3 (−44,1%). Preço justo cai até 84% no caso da KEPL3.
- 2 tinham CAGR acima do limiar e recebiam seed de 20%: SEER3 (+96,4%) e VTRU3 (+42,4%).
  Passam a não ter DCF e recaem no DDM, rotuladas como tal.
- 9 são zeradas pela regra do ano negativo e não mudam de comportamento.

Com `USE_FORWARD_ESTIMATES` ligado e driver de receita, as 14 têm estimativa entre +4,3% e
+30,4%; 4 delas (EVEN3, EZTC3, JHSF3, LAVV3) ficam acima do limiar de 20% e recaem no CAGR
histórico.

## Fora de escopo

Registrados aqui com os números medidos, para não se perderem.

**Acoplamento `TERMINAL_GROWTH = RISK_FREE_RATE`.** Como `coe = RF + beta × ERP`, o spread
da perpetuidade é `coe − terminal = beta × ERP` e o RF **cancela**: o múltiplo terminal é
15,0x com RF a 12,4% e 13,9x com RF a 4%. O nível do RF é praticamente inócuo enquanto
acoplado. O acoplamento em si é que é questionável — o comentário em valuation.py:44 invoca
"perpetuidade não pode exceder a economia", e o RF brasileiro provavelmente excede o
crescimento nominal do PIB. Desacoplar para 6% cortaria ~34% de todo preço justo da base.
Decisão grande, merece spec própria.

**Regra "qualquer ano negativo zera o CAGR"** (valuation.py:150-151). Foi o mecanismo que
mais zerou seeds na medição: 9 das 14, contra 3 do piso. Tem justificativa documentada
(caso RSUL4) que precisaria ser reaberta. Pelo mesmo argumento estrutural usado na decisão
3, ela também impede o modelo de expressar declínio.

**Cálculo do CAGR pelas pontas.** `_compute_fcf_cagr` compara apenas o primeiro e o último
ponto da série, ignorando o caminho. RIAA3 (519 → 951 → 1.087 → 351) vira "−12,2% ao ano",
resumo pobre de uma série que oscila. Prompt pronto em
`2026-08-06-cagr-melhorado-prompt.md`.

**Valor de `MAX_PROJECTABLE_GROWTH`.** Os 0,20 nunca foram derivados de nada. Agora que o
piso não existe, o limiar ficou sozinho e sua revisão faz sentido junto com a do terminal.
