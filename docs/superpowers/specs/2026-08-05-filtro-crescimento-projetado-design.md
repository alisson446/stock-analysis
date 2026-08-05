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

## Decisão de desenho: dois critérios independentes, cada um com sua flag

Ausência de estimativa não é informação sobre a empresa, é informação sobre a cobertura do sell-side —
que em small caps BR correlaciona com liquidez, não com qualidade. Reprovar por ausência descarta ações
boas por obscuridade: exigir estimativa completa com pelo menos 3 analistas derruba as 28 ações atuais
para 11. Por outro lado, rodar DCF sobre empresa que o consenso projeta encolhendo é exatamente o que
esta spec quer evitar. Nenhum dos dois lados é certo o tempo todo, então a escolha fica no JSON.

São **duas flags independentes**, cada uma ligando e desligando o seu próprio critério. Não há cascata
entre elas: nenhuma altera o significado da outra.

| Flag | `true` | `false` |
|---|---|---|
| `exigir_estimativa` | aplica `crescimento_receita_pct_min` e `crescimento_lucro_pct_min`; valor NaN reprova | os dois cortes não são aplicados |
| `exigir_num_analistas` | aplica `num_analistas_min`; `num_analistas` NaN reprova | `num_analistas_min` não é aplicado |

Em ambos os casos, NaN reprova quando a flag está ligada: sem dado não há como atestar o critério, e o
propósito da flag é justamente exigi-lo. Quem quiser o comportamento permissivo desliga a flag.

Uma versão anterior deste desenho encadeava as duas — poucos analistas rebaixava a estimativa a NaN, que
então caía na regra de `exigir_estimativa`. Foi descartado: produzia o resultado invertido de a porta de
analistas tornar o filtro *mais permissivo* em vez de mais restritivo, e exigia uma tabela-verdade de
quatro casos para descrever duas flags. Independentes, cada flag faz exatamente o que o nome diz.

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
"exigir_num_analistas": false,
"exigir_estimativa": false
```

As duas flags nascem desligadas: o screening sai desta spec com o mesmo resultado de hoje, e os cortes
entram quando você quiser. Desligar uma flag preserva o limiar configurado ao lado dela para quando for
religada — é essa a razão de existirem, já que `num_analistas_min: 0` e cortes em `-inf` desligariam os
critérios ao custo de perder o valor ajustado.

### 4. Filtragem — `src/filters.py`

`apply_stock_filters` e `apply_bank_filters` ganham a mesma lógica de crescimento. Como é idêntica nos
dois, extrair uma função auxiliar `_growth_mask(df, cfg)` que recebe o bloco de config e devolve a
máscara booleana, chamada pelas duas.

A máscara começa como "tudo passa" e cada flag ligada acrescenta a sua condição:

1. Se `exigir_estimativa` for `true`, exige `crescimento_receita_pct > crescimento_receita_pct_min` **e**
   `crescimento_lucro_pct > crescimento_lucro_pct_min`. NaN em qualquer uma das duas reprova. Se for
   `false`, nenhum dos dois cortes é lido.
2. Se `exigir_num_analistas` for `true`, exige `num_analistas >= num_analistas_min`. NaN reprova. Se for
   `false`, `num_analistas_min` não é lido.

Comparação estrita (`>`) nos cortes de crescimento, seguindo a convenção dos demais critérios `_min` do
projeto (`df['roe_pct'] > cfg['roe_pct_min']`, `df['lpa'] > cfg['lpa_min']`). Com o default `0`,
crescimento projetado exatamente nulo reprova.

`num_analistas_min` usa `>=` porque é uma contagem: `num_analistas_min: 2` significa "pelo menos dois
analistas", não "mais de dois".

Sob `exigir_estimativa: true`, a linha que tem apenas uma das duas estimativas reprova — POMO3 tem
crescimento de receita mas não de lucro, por exemplo. É a leitura literal de "exigir estimativa" e está
explícita aqui para não virar surpresa.

O `print` de resultado de cada função passa a informar quantas reprovaram por crescimento, separando do
total, para o efeito do filtro novo ficar visível na saída do notebook.

### 5. Exibição — `analysis.ipynb`

`crescimento_receita_pct` e `crescimento_lucro_pct` entram em `display_cols` na célula de screening de
ações e na tabela de bancos, formatadas como `'{:.1f}%'`. O `num_analistas` não é exibido por padrão —
é um insumo do filtro, não uma métrica de decisão.

A célula markdown de critérios é atualizada com os critérios novos.

## Efeito esperado

Sobre as 28 ações que passam nos filtros hoje, com `num_analistas_min: 2` e cortes em 0:

| `exigir_estimativa` | `exigir_num_analistas` | Passam |
|---|---|---|
| `false` | `false` | 28 — filtro inerte, é o default |
| `false` | `true` | 16 |
| `true` | `false` | 15 |
| `true` | `true` | 14 |

Com as duas ligadas, reprovam: SUZB3 e RANI3 por lucro projetado negativo (-2,4% e -11,7%); KEPL3 por ter
1 analista; e as 11 sem estimativa completa — POMO3, GRND3, SHUL4, TECN3, VLID3, CYRE4, EUCA4, IGTI3,
LPSB3, CSUD3, RSUL4.

Subindo `num_analistas_min` para 3 com as duas flags ligadas: **28 → 11**. Sobram CYRE3, RECV3, EZTC3,
SEER3, INTB3, EVEN3, MELK3, MDNE3, RIAA3, VTRU3, BLAU3.

Os dois extremos ficam a uma edição de JSON de distância, que é o objetivo do desenho configurável.

## Testes

Em `tests/`, seguindo o padrão existente. Sobre `_growth_mask`, com DataFrames construídos à mão — sem
chamadas de rede:

Com as duas flags ligadas:

- Crescimento acima dos cortes e analistas suficientes → passa.
- Crescimento de lucro negativo → reprova.
- Crescimento de receita negativo, lucro positivo → reprova (as duas condições valem em conjunto).
- Apenas uma das duas estimativas presente, outra NaN → reprova.
- `num_analistas` exatamente igual a `num_analistas_min` → passa (comparação `>=`).
- `num_analistas` abaixo do mínimo, crescimento bom → reprova.
- `num_analistas` NaN → reprova.

Com as flags desligadas, confirmando a independência entre elas:

- `exigir_estimativa: false` e crescimento de lucro negativo → passa.
- `exigir_num_analistas: false` e `num_analistas` NaN → passa.
- `exigir_estimativa: false` com `exigir_num_analistas: true` → só a contagem de analistas decide.
- `exigir_estimativa: true` com `exigir_num_analistas: false` → só os cortes de crescimento decidem.
- Ambas `false` → máscara toda verdadeira, nenhuma linha reprovada.

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
