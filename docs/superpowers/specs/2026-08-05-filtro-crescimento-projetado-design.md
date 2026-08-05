# Filtro de crescimento projetado no screening

**Data:** 2026-08-05
**Escopo:** coleta, exibição e filtragem de crescimento projetado de receita e lucro, antes do valuation.

## Objetivo

Exibir no screener a taxa de crescimento projetada por analistas para receita e lucro, e permitir
filtrar por elas antes de rodar o valuation — hoje o DCF roda sobre ações que o consenso projeta
encolhendo, e isso só aparece no preço justo, não no screening.

## Descoberta: o horizonte disponível é de 1 ano, não 3

A intenção original era projeção de 3 anos. **A Yahoo Finance não fornece isso.** Verificado ao vivo
com yfinance 1.2.0 em 12 tickers BR e US:

Os quatro endpoints de estimativa (`growth_estimates`, `earnings_estimate`, `revenue_estimate`,
`eps_trend`) expõem sempre os mesmos quatro períodos: `0q`, `+1q`, `0y`, `+1y`. O horizonte máximo é
`+1y` — o próximo exercício fiscal.

A linha `LTG` (long-term growth, que seria o proxy de 3–5 anos) existe no schema mas retorna **NaN
para a ação em 100% dos tickers testados** — VALE3, ITUB4, BBAS3, MGLU3, ABEV3, WEGE3, PETR4, AAPL,
MSFT, NVDA, KO, TSLA. Apenas a coluna `indexTrend` tem valor, e ela é o crescimento do índice, não da
empresa. A Yahoo aparentemente descontinuou o LTG por ação.

Consequência para o código atual: `_FORWARD_GROWTH_PERIODS` em `src/fundamentals.py` prioriza `'LTG'`,
que nunca resolve. Na prática `get_forward_growth` sempre cai no `'+1y'`. O fallback funciona; a
preferência por LTG é código morto. Fora do escopo desta spec (ver "Trabalho adiado").

Esta spec adota `+1y` como o horizonte, por ser o único disponível.

## Cobertura medida

Medida sobre o conjunto que efetivamente passa nos filtros hoje, não sobre o universo.

**Ações (28 que passam em `apply_stock_filters`):**

| | Tem dado | % |
|---|---|---|
| Crescimento de receita `+1y` | 20/28 | 71% |
| Crescimento de lucro `+1y` | 17/28 | 61% |
| Lucro com ≥3 analistas | 13/28 | 46% |

Sem estimativa alguma: GRND3, SHUL4, TECN3, VLID3, EUCA4, LPSB3, CSUD3, RSUL4.

**Bancos (6 que passam em `apply_bank_filters`):** 6/6 receita, 4/6 lucro. ITSA3 e PINE3 não têm
estimativa de lucro embora ITSA4 e PINE4 (mesma empresa, outra classe) tenham.

**Qualidade do dado:** o número de analistas é baixo e muito variável. Mediana de 4 no universo,
mínimo de 1. KEPL3 tem 1 analista; LAVV3 e TRIS3 têm 2. No universo completo, LJQQ3 tem 1 analista
projetando +143% de crescimento de lucro e VVEO3 tem 4 projetando +175% — ruído, não previsão. Daí a
existência do parâmetro `num_analistas_min`.

## Decisão de desenho: ausência de dado é configurável

Ausência de estimativa não é informação sobre a empresa, é informação sobre a cobertura do sell-side —
que em small caps BR correlaciona com liquidez, não com qualidade. Reprovar por ausência descarta
ações boas por obscuridade; um filtro estrito derruba as 28 ações atuais para 11.

O comportamento é controlado por `exigir_estimativa`, com `false` como default.

`num_analistas_min` age **antes** dos cortes de crescimento, como porta de qualidade do dado, e não
como um critério paralelo. `exigir_num_analistas` liga e desliga essa porta:

```
exigir_num_analistas: true  e  analistas < num_analistas_min  →  estimativa tratada como ausente (NaN)
(analistas NaN conta como abaixo do mínimo)                        ↓
                                             exigir_estimativa decide o que fazer com NaN
                                                true  → reprova
                                                false → passa para o valuation
                                                                   ↓
                                       tendo dado válido, aplica os cortes de crescimento
```

Poucos analistas significa "não sei", não "crescimento ruim". Colapsar esse caso em NaN dá um único
caminho de decisão, em vez de uma tabela-verdade de quatro casos.

`num_analistas` ausente (NaN) conta como abaixo do mínimo: sem saber a cobertura, não dá para atestar
a qualidade do dado.

**A porta de analistas pode ser mais permissiva, não mais restritiva.** Consequência do desenho que vale
registrar: com `exigir_estimativa: false`, ligar `exigir_num_analistas` faz uma ação com poucos analistas
e crescimento projetado *negativo* virar NaN e **passar**, quando sem a porta ela seria reprovada pelo
corte de crescimento. É coerente — a tese é que estimativa de 1 analista não é informação suficiente nem
para aprovar nem para reprovar — mas é o oposto do que "exigir mais analistas" sugere à primeira vista.
Para que a porta restrinja, é preciso `exigir_estimativa: true` junto.

Nenhuma das 28 ações atuais cai nesse caso (a única com menos de 2 analistas, KEPL3, projeta crescimento
positivo), então a combinação não muda o resultado hoje.

## Componentes

### 1. Coleta — `src/fundamentals.py`

Dentro do loop de `_fetch_fundamentals_from_api`, reaproveitando o objeto `stock = yf.Ticker(ticker_sa)`
já criado, ler `stock.revenue_estimate` e `stock.earnings_estimate` e extrair a linha `+1y`.

**Custo: 1 requisição HTTP extra por ticker.** Ambas as propriedades vêm do mesmo módulo `earningsTrend`
do quoteSummary e ficam em cache no objeto `Ticker` (`yfinance/scrapers/analysis.py:191-194`), então o
segundo acesso não faz requisição. Medido em ~0.35s/ticker, ou cerca de +2 min sobre os ~22 min atuais
da coleta completa de 372 tickers.

Não usar `growth_estimates`: ela dispara uma segunda requisição para `industryTrend`/`sectorTrend`/
`indexTrend`, dados que o projeto não consome. `earnings_estimate` já traz a coluna `growth`.

A extração é defensiva no mesmo padrão de `get_forward_growth`: qualquer exceção ou frame vazio resulta
em NaN, sem interromper a coleta.

### 2. Colunas novas em `data/fundamentals.csv`

Três colunas. O `num_analistas` precisa viajar junto para o filtro poder aplicar `num_analistas_min`.

| Coluna | Fonte | Unidade |
|---|---|---|
| `crescimento_receita_pct` | `revenue_estimate.loc['+1y','growth'] × 100` | pontos percentuais |
| `crescimento_lucro_pct` | `earnings_estimate.loc['+1y','growth'] × 100` | pontos percentuais |
| `num_analistas` | `earnings_estimate.loc['+1y','numberOfAnalysts']` | contagem |

Sufixo `_pct` e valores em pontos percentuais seguem a convenção existente (`roe_pct`,
`margem_liquida_pct`). As três entram na lista `numeric_cols` de sanitização do notebook.

`num_analistas` vem do `earnings_estimate` e governa as duas colunas de crescimento — o
`revenue_estimate` não expõe contagem de analistas. É uma aproximação deliberada: a cobertura de
receita e de lucro vem do mesmo conjunto de analistas.

Sem tratamento para coluna ausente. O `fundamentals.csv` é cache descartável: apagá-lo regenera o
arquivo com as colunas novas. Rodar sobre um cache antigo levanta `KeyError` no filtro, e a correção
é apagar o arquivo.

### 3. Configuração — `config/filters.json`

Os mesmos cinco parâmetros nos dois blocos, `stock_filters` e `bank_filters`:

```json
"crescimento_receita_pct_min": 0,
"crescimento_lucro_pct_min": 0,
"num_analistas_min": 2,
"exigir_num_analistas": true,
"exigir_estimativa": false
```

`exigir_num_analistas: false` ignora `num_analistas_min` por completo, preservando o valor configurado
para quando for religado.

### 4. Filtragem — `src/filters.py`

`apply_stock_filters` e `apply_bank_filters` ganham a mesma lógica de crescimento. Como é idêntica nos
dois, extrair uma função auxiliar `_growth_mask(df, cfg)` que recebe o bloco de config e devolve a
máscara booleana, chamada pelas duas.

A máscara implementa o fluxo da seção de desenho:

1. Se `exigir_num_analistas` for `true`, estimativas com `num_analistas < num_analistas_min` ou
   `num_analistas` NaN são tratadas como NaN. Se for `false`, esta etapa é pulada e
   `num_analistas_min` não é lido.
2. Linhas com crescimento de receita ou de lucro NaN passam se `exigir_estimativa` for `false`,
   reprovam se for `true`.
3. Linhas com ambos os valores válidos passam se `crescimento_receita_pct > crescimento_receita_pct_min`
   e `crescimento_lucro_pct > crescimento_lucro_pct_min`.

Comparação estrita (`>`), seguindo a convenção dos demais critérios `_min` do projeto
(`df['roe_pct'] > cfg['roe_pct_min']`, `df['lpa'] > cfg['lpa_min']`). Com o default `0`, crescimento
projetado exatamente nulo reprova.

Note que a regra 2 reprova, sob `exigir_estimativa: true`, a linha que tem apenas uma das duas
estimativas — POMO3 tem crescimento de receita mas não de lucro, por exemplo. É a leitura literal de
"exigir estimativa" e está explícita aqui para não virar surpresa.

O `print` de resultado de cada função passa a informar quantas reprovaram por crescimento, separando do
total, para o efeito do filtro novo ficar visível na saída do notebook.

### 5. Exibição — `analysis.ipynb`

`crescimento_receita_pct` e `crescimento_lucro_pct` entram em `display_cols` na célula de screening de
ações e na tabela de bancos, formatadas como `'{:.1f}%'`. O `num_analistas` não é exibido por padrão —
é um insumo do filtro, não uma métrica de decisão.

A célula markdown de critérios é atualizada com os critérios novos.

## Efeito esperado

Com os defaults (`exigir_estimativa: false`, `exigir_num_analistas: true`, `num_analistas_min: 2`,
cortes em 0), sobre as 28 ações atuais: **reprova 2** — SUZB3 (lucro projetado -2,4%, 7 analistas) e
RANI3 (-11,7%, 3 analistas). KEPL3 tem 1 analista, cai para NaN e passa.

Desligar `exigir_num_analistas` não muda nada hoje: KEPL3 passa a ser avaliada pelos cortes e, com
+6,0% de receita e +12,8% de lucro, passa de qualquer forma.

Com `exigir_estimativa: true` e `num_analistas_min: 3`: **28 → 11**. Sobram CYRE3, RECV3, EZTC3, SEER3,
INTB3, EVEN3, MELK3, MDNE3, RIAA3, VTRU3, BLAU3.

Os dois extremos ficam a uma edição de JSON de distância, que é o objetivo do desenho configurável.

## Testes

Em `tests/`, seguindo o padrão existente. Sobre `_growth_mask`, com DataFrames construídos à mão — sem
chamadas de rede:

- Crescimento acima dos cortes com analistas suficientes → passa.
- Crescimento de lucro negativo com analistas suficientes → reprova.
- Crescimento NaN com `exigir_estimativa: false` → passa.
- Crescimento NaN com `exigir_estimativa: true` → reprova.
- Crescimento válido mas `num_analistas` abaixo do mínimo, com `exigir_num_analistas: true` → tratado
  como NaN, seguindo `exigir_estimativa` (um caso para cada valor da flag).
- `num_analistas` NaN com crescimento presente e `exigir_num_analistas: true` → tratado como NaN.
- `exigir_num_analistas: false` com `num_analistas` abaixo do mínimo → `num_analistas_min` ignorado,
  crescimento avaliado normalmente pelos cortes.
- O caso permissivo documentado no desenho: `num_analistas` abaixo do mínimo, crescimento negativo,
  `exigir_num_analistas: true` e `exigir_estimativa: false` → **passa**. Com
  `exigir_num_analistas: false`, a mesma linha reprova.

Para a extração em `fundamentals.py`, um teste com frame vazio e um com exceção, verificando que o
resultado é NaN e a coleta não interrompe.

## Trabalho adiado

Fora do escopo, deliberadamente:

- **Estágios de crescimento no DCF.** Hoje o estágio 1 semeia o fade linear de 10 anos com um único
  número (`+1y` de lucro) e descarta o `0y`. Discutimos ancorar os anos 1 e 2 nas duas estimativas
  reais e iniciar o fade no ano 3, e trocar o driver de lucro para receita. Nada disso foi decidido.
  Spec separada — ver o prompt em `docs/superpowers/specs/2026-08-05-proximo-passo-dcf-prompt.md`.
  Motivo da separação: esta spec muda **quais ações entram** no valuation; aquela muda **quanto cada
  uma vale**. Juntas, uma mudança no top 20 fica sem causa atribuível.
- **`get_forward_growth` refazendo trabalho.** Depois desta spec, o crescimento já estará no
  `fundamentals.csv`, mas `get_forward_growth` continuará fazendo sua própria chamada de API por ticker
  durante o valuation — e via `growth_estimates`, que custa 2 requisições em vez de 1. Limpeza óbvia,
  mas muda resultado de valuation. Vai junto com a spec do DCF.
- **Prioridade morta por `LTG`** em `_FORWARD_GROWTH_PERIODS`. Mesma spec.
