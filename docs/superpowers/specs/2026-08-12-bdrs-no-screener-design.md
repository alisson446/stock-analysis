# BDRs no screener: fundamento estrangeiro, na moeda de origem

**Data:** 2026-08-12
**Escopo:** nova região de ativos, alcançada via BDR, percorrendo o pipeline inteiro — universo, coleta, filtros,
valuation e snapshot —, mais a reorganização de `data/` e `config/` em pastas por região. Não altera
o comportamento das ações e bancos brasileiros, exceto por dois pontos: a refatoração das premissas
macro (ver "Premissas por moeda"), necessária para o histórico não mentir, e a mudança de caminho
dos arquivos que eles já usam.

**Decisão estruturante: nenhuma conversão de moeda acontece em lugar nenhum.** Cada ativo é
coletado, avaliado e exibido na moeda do seu próprio balanço. É o que torna o desenho válido para
qualquer país sem multiplicar caminhos de erro.

## Objetivo

Fazer BDRs aparecerem no screener com a mesma qualidade de dado das ações brasileiras: filtrados
pelos mesmos critérios fundamentalistas, com preço justo calculado, e comparáveis no ranking.

A pergunta que originou a spec era se dá para "apontar o screener para os tickers de BDR". A
resposta medida é **não** — e a razão define todo o desenho.

## Descoberta 1: o ticker do BDR não sustenta o pipeline

Medição de 2026-08-12, via `yf.Ticker` em `AAPL34.SA`, `MSFT34.SA`, `GOGL34.SA`, `JPMC34.SA`.

**As estimativas de analistas não existem.** `earnings_estimate` volta vazio em 100% dos BDRs
testados. Como `_estimates_mask` roda com `exigir_estimativa` e `exigir_lpa_estimado` ligados, e
NaN reprova por construção, **todo BDR seria eliminado** independentemente da qualidade da empresa.
No ticker americano correspondente o frame vem completo (AAPL: 40 analistas na linha `+1y`).

**Preço e lucro vêm em moedas diferentes.** O `.info` do BDR mistura as duas pontas:

| campo | `AAPL34.SA` | `AAPL` | unidade |
|---|---|---|---|
| `currentPrice` | 78,64 | 304,755 | BDR em R$, ação em US$ |
| `netIncomeToCommon` | 128.929.996.800 | 128.929.996.800 | US$ nos dois |
| `sharesOutstanding` | 291.883.600.000 | 14.594.180.000 | BDR em unidades de BDR |
| `financialCurrency` | USD | USD | |
| `currency` | BRL | USD | |

O repo recalcula P/L e LPA de propósito (`compute_ttm_net_income`, `resolve_share_count`), porque o
`trailingEps` do Yahoo erra a contagem de ações em ações brasileiras. Essa mesma conta, aplicada a
um BDR, faz `preço em R$ ÷ lucro em US$`:

```
LPA  = 128,93e9 US$ / 291,88e9 BDRs = 0,4417 US$ por BDR
P/L  = 78,64 R$    / 0,4417 US$     = 178
```

O P/L real da Apple é ~33 (178 ÷ 5,16 do câmbio). Nenhum BDR passaria em `pl_max: 10`, e por um
motivo falso. Note que o Yahoo *acerta* seus próprios `trailingPE` e `priceToBook` para o BDR
(35,4 e 42,0), porque converte internamente — mas são justamente os campos que o repo não usa, com
razão documentada.

**Corolário:** os indicadores adimensionais do BDR (P/VP, ROE, margens, DL/EBIT, DL/PL,
passivos/ativos, liquidez corrente) estão corretos, porque são US$ ÷ US$ e o câmbio se cancela. O
que quebra é tudo que cruza preço com valor absoluto em outra moeda: P/L, LPA, dívida líquida,
EBIT, FCF — e, por consequência, o DCF inteiro.

## Descoberta 2: `yf.screen(region='us')` não enumera o mercado americano

Medição de 2026-08-12. A consulta `region='us'` com `intradaymarketcap > 500M` reporta
`total=7511`, e paginando os 7511 registros (todos únicos) **faltam** `TGT`, `HD`, `LOW`, `MDT`,
`NUE`, `WDC`, `XRX`, `SE`, `TX`, `SPG`.

Não é teto de paginação: a faixa estreita `30bi < mcap < 100bi` devolve 731 resultados coletados
integralmente e continua sem esses papéis. O endpoint omite de forma sistemática, sem critério que
eu tenha conseguido identificar.

**Consequência de desenho:** o universo não pode sair do lado americano. Sai do lado brasileiro,
onde a mesma API se comporta bem.

## Descoberta 3: `yf.Search` sozinho não é confiável, mas com portão é

Medição de 2026-08-12. Buscando pelo nome da empresa e pegando o resultado em bolsa americana, uma
amostra aleatória de 22 BDRs (semente 7) resolveu 22/22 com candidato — mas um deles errado:
`FMXB34` (Fomento Económico Mexicano) casou com `VIST` (Vista Energy).

Na direção inversa a busca é pior ainda: procurando o BDR a partir do nome, ela achou `AAPL34.SA`,
`JPMC34.SA`, `MELI34.SA` e `C1TA34.SA`, mas **perdeu** `NFLX34.SA` e `ROXO34.SA`, que existem.

O que torna a resolução utilizável é a razão do BDR ser mensurável por dois caminhos que não
compartilham nenhum campo. Pelas ações, ela sai direto:

```
razao_acoes = sharesOutstanding(BDR) / sharesOutstanding(subjacente)
```

Pelos preços, o que sai é a cotação que o par **implica**:

```
fx_implicito = razao_acoes × preco(BDR) / preco(subjacente)
```

Num par correto, esse número é o câmbio de mercado. Num par errado, é um valor sem sentido — e o
teste é que todos os pares do universo têm que implicar aproximadamente a mesma cotação.

Medição de 2026-08-12 (dólar de mercado no mesmo instante: 5,1678):

| BDR | subjacente | razão por ações | cotação implícita | desvio da mediana |
|---|---|---|---|---|
| AAPL34 | AAPL | 20,000 | 5,1848 | 0,03% |
| GOGL34 | GOOGL | 12,000 | 5,1820 | 0,03% |
| XPBR31 | XP | 1,000 | 5,1814 | 0,04% |
| NFLX34 | NFLX | 50,000 | 5,1688 | 0,28% |
| JPMC34 | JPM | 10,000 | 5,1510 | 0,63% |
| MELI34 | MELI | 120,000 | 5,2690 | 1,65% |
| **FMXB34** | **VIST** | **0,615** | **5,9635** | **15,05%** |

A mediana dos implícitos ficou em 5,1834, a 0,3% do dólar de mercado — diferença compatível com os
15 minutos de atraso da cotação do BDR. Os pares corretos ficam todos abaixo de 1,7%. O par errado
`FMXB34 → VIST` (Fomento Económico Mexicano casado com Vista Energy) falha duas vezes: razão de
ações não-inteira e cotação implícita 15% fora.

Com o portão aplicado, a amostra de 22 aprovou 21 e rejeitou exatamente o par errado.

**Alcance dessa evidência:** a medição restringiu os candidatos a bolsas **americanas**
(`NYQ`, `NMS`, `NGM`, `NCM`, `ASE`, `PCX`, `BTS`) e descartou símbolos com `-` (preferenciais). Os
21/22 valem para essa configuração. O desenho aceita candidatos de qualquer bolsa estrangeira, e
para as demais praças a taxa de resolução é **desconhecida** — o portão continua protegendo contra
par errado, mas a fração que resolve não foi medida. A implementação deve imprimir a taxa de
resolução por bolsa, para que isso deixe de ser suposição na primeira rodada real.

**O portão não converte nada e não lê cotação nenhuma.** Ele deriva a cotação implícita e a compara
com a mediana do próprio universo. Por isso continua válido sob a decisão de não haver conversão de
moeda, e funciona com qualquer moeda de pregão do subjacente — desde que a mediana seja calculada
**por moeda de pregão**, não sobre o universo inteiro (ver "Portão de qualidade").

## Descoberta 4: moeda do pregão e moeda do balanço divergem com frequência

Medição de 2026-08-12 sobre 10 papéis estrangeiros:

| ticker | `currency` | `financialCurrency` | estimativas |
|---|---|---|---|
| `UBS`, `YPF` (ADR) | USD | USD | 4 linhas |
| **`UL`** (ADR Unilever) | **USD** | **EUR** | 4 linhas |
| **`TSM`** (ADR TSMC) | **USD** | **TWD** | 4 linhas |
| `NESN.SW` | CHF | CHF | 4 linhas |
| `SAP.DE`, `MC.PA` | EUR | EUR | 4 linhas |
| `7203.T` | JPY | JPY | 4 linhas |
| **`AZN.L`** | **GBp** | **USD** | 4 linhas |
| **`BHP.AX`** | **AUD** | **USD** | 4 linhas |

**Cobertura de analistas não é privilégio americano.** Os 10 papéis trouxeram `earnings_estimate`
completo, incluindo Tóquio, Paris, Zurique e Sydney. O buraco da Descoberta 1 era do ticker do BDR,
não de papel estrangeiro.

**"ADR em dólar" não garante balanço em dólar.** `UL` negocia em USD e reporta em EUR; `TSM`
negocia em USD e reporta em TWD. O P/L recalculado desses dois quebraria exatamente como o do
`AAPL34.SA`.

**`AZN.L` cota em `GBp`, não em `GBP`** — pence, centésimo de libra. Tratar como libra erra por
100x.

## Arquitetura

**Em uma frase: o BDR é o ticker negociável; o subjacente é a empresa, na moeda dela.**

```
yf.screen(region=BDR_REGION)          953 papéis (mcap > 500M, medição de 2026-08-12)
        │  filtra shortName ~ /DR[N123]\b/
        ▼
   625 BDRs, 594 com longName          (universo intermediário, não vira arquivo)
        │  o payload do screener já traz preço, volume e ações do BDR:
        │  nenhuma requisição extra para o lado brasileiro
        │
        │  yf.Search(longName) → candidato em bolsa estrangeira
        ▼
   ┌─── PORTÃO DE QUALIDADE ─────────────────────┐
   │  razao_acoes inteira (tolerância ±0,02)     │
   │  cotação implícita a menos de 3% da mediana │
   │  da sua moeda de pregão (auto-calibrado)    │
   └─────────────────────────────────────────────┘
        │ aprovado                      │ rejeitado
        ▼                               ▼
   ┌─── ELEGIBILIDADE POR MOEDA ─────────────────┐
   │  currency == financialCurrency              │
   │  premissas macro definidas para essa moeda  │
   └─────────────────────────────────────────────┘
        │ aprovado                      │ rejeitado
        ▼                               ▼
  ┌──────────────────┬───────────────┐   fora da lista,
  │ AAPL      (US$)  │ AAPL34  (R$)  │   com motivo no log
  │ fundamentos      │ preço         │
  │ estimativas      │ liquidez      │
  └──────────────────┴───────────────┘
        │  par validado → data/us/tickers.csv
        │  fundamentos  → data/us/fundamentals.csv
        │
        ▼  filtros de config/us/filters.json
        │
        ▼  valuation na moeda do balanço (RF e ERP daquela moeda)
        │
        ▼  preco_justo na mesma moeda — nenhuma conversão
        │
        ▼  snapshot → data/us/valuation_history.csv
```

### Identificação do BDR

O `shortName` do screener carrega o marcador do tipo de recibo: `DRN` para não-patrocinado, `DR1`,
`DR2` e `DR3` para os patrocinados. **O regex é `DR[N123]\b` — sem `\b` na frente.**

A ausência da borda inicial não é descuido. O `shortName` é um campo de largura fixa, e quando o
nome da empresa ocupa a largura inteira o marcador fica **colado** nele, sem espaço:

| symbol | shortName | longName | é BDR |
|---|---|---|---|
| `Z1TS34.SA` | `ZOETIS INC  DRN` | Zoetis Inc. | sim |
| `Z2LL34.SA` | `ZILLOW GROUPDRN` | Zillow Group, Inc. | sim — **sem espaço** |
| `W1HR34.SA` | `WHIRLPOOL CODRN` | Whirlpool Corporation | sim — **sem espaço** |
| `WFCO34.SA` | `WELLS FARGO DRN ED` | Wells Fargo & Company | sim — sufixo `ED` |
| `STZB34.SA` | `CONSTELLATIODRN ED` | Constellation Brands | sim — **os dois casos** |
| `JBSS32.SA` | `JBS N.V.    DR2` | JBS N.V. | sim |
| `XPBR31.SA` | `XP INC      DR1` | XP Inc. | sim |
| `YDUQ3.SA` | `YDUQS PART  ON      NM` | Yduqs Participações S.A. | não |
| `ZIFI11.SA` | `FII ZION    CI` | Zion Capital FI | não |

Um `\b` inicial exigiria o espaço e **descartaria 305 dos 625 BDRs — 49% do universo — em
silêncio.** A borda final é necessária porque `ED` (ex-dividendo) aparece depois do marcador em 36
papéis, então ancorar no fim da string também falharia.

Validado sobre o universo de 2026-08-12: `DR[N123]\b` marca 625 papéis, com **zero** falsos
positivos (nenhum ticker no formato de ação ou FII brasileiro) e **zero** falsos negativos (nenhum
ticker no formato `XXXX3[1-9]` ficou de fora). Os únicos sufixos observados após o marcador são
vazio (589) e `ED` (36).

O `longName` do BDR é o nome legal da empresa estrangeira, sem adaptação — é ele que alimenta a
busca pelo ticker do subjacente.

`NUBR33.SA` não existe mais no Yahoo (404 no `.info`, ausente do screener); o Nubank aparece como
`ROXO34.SA → NU`. Nenhum tratamento especial: o papel simplesmente não entra no universo.

### As colunas do lado brasileiro saem do próprio screener

O payload do `yf.screen` já carrega tudo que o lado BDR precisa, então **nenhuma requisição
adicional é feita ao ticker do BDR**:

| coluna | origem no payload |
|---|---|
| `preco_bdr` (R$) | `regularMarketPrice` |
| `liq_media_diaria_bdr` (R$) | `averageDailyVolume10Day` × `regularMarketPrice` |
| ações do BDR (para a razão) | `sharesOutstanding` |
| `ticker_bdr`, `longName` | `symbol`, `longName` |

Isso importa para o custo da coleta: são duas idas por BDR — a busca do candidato e o `.info` do
subjacente —, nunca uma terceira ao ticker do BDR.

### Portão de qualidade

Um BDR entra na lista somente se as duas medidas concordarem. Sem par resolvido, ou com par
reprovado, o papel **não entra nem com dado parcial**.

É a Guideline 4 aplicada ao mapeamento. Um par errado não produz um erro visível: produz um preço
justo com aparência de calculado, sobre outra empresa. `FMXB34` avaliado como Vista Energy sairia
com número, unidade e formatação corretos, e chegaria ao usuário como recomendação.

A tolerância de 3% e a de ±0,02 no inteiro vêm de premissa, não de ajuste à amostra: o desvio da
cotação implícita é dominado por defasagem — o BDR tem 15 minutos de atraso e o subjacente é outro
instante — e 3% é folga confortável para descasamento intradiário sem acomodar erro de identidade.
O par errado medido deu 15%, cinco vezes o limite.

Duas condições sobre a mediana:

1. É calculada **por moeda de pregão do subjacente**. Pares que implicam BRL/USD e pares que
   implicam BRL/EUR são populações diferentes, e misturá-las produziria uma mediana que não é
   cotação de nada.
2. Entram no cálculo apenas os pares que já passaram no teste da razão inteira, para que um punhado
   de pares errados não desloque a referência.

### Elegibilidade por moeda

Substitui a conversão que existiria num desenho multi-moeda. Duas condições, ambas eliminatórias:

**`currency == financialCurrency`.** É a única circunstância em que `preco ÷ LPA` é um P/L. Sem
conversão, papel com as duas divergindo não tem como ser avaliado corretamente, e sai: `UL`, `TSM`,
`AZN.L`, `BHP.AX` da Descoberta 4.

Isso é deliberadamente um critério de elegibilidade, não uma correção aritmética. Uma comparação
substitui uma família inteira de conversões, cada uma delas um lugar onde uma moeda errada
produziria um número plausível. A armadilha do `GBp` desaparece junto: `AZN.L` sai pela mesma
condição, sem precisar de tratamento de subunidade em lugar nenhum.

**Premissas macro definidas para a moeda do balanço.** Papel que reporta numa moeda sem
`RISK_FREE_RATE_<MOEDA>` e `EQUITY_RISK_PREMIUM_<MOEDA>` no `.env` é excluído, com log nomeando as
variáveis que faltam. Habilitar um país é editar o `.env`; não há código a mudar.

### Estrutura de arquivos por região

`data/` e `config/` passam a ser organizados em subpastas por região, e cada região é um pipeline
autocontido:

```
data/
  br/  tickers.csv   fundamentals.csv   valuation_history.csv    ← os 3 arquivos de hoje, movidos
  us/  tickers.csv   fundamentals.csv   valuation_history.csv    ← novo

config/
  br/  filters.json                                              ← o arquivo de hoje, movido
  us/  filters.json                                              ← novo
```

**A região de uma pasta é onde o ticker é negociado**, não onde a busca começou. `data/us/` guarda
`AAPL` e `MSFT` — os subjacentes —, descobertos a partir dos BDRs de B3. `BDR_REGION` continua sendo
apenas o universo varrido pelo `yf.screen` (`br`), e a pasta de destino sai da bolsa do subjacente
resolvido.

Essa distinção é o que mantém o desenho viável: a Descoberta 2 mostrou que `yf.screen(region='us')`
não enumera o mercado americano, então uma pasta de região **não** pode ser populada varrendo aquela
região diretamente. Ela é populada pelo caminho que funciona — no caso de `us`, via BDR.

**Adicionar uma região é criar as duas pastas**, não escrever código: as funções recebem a região e
montam o caminho.

**Região pedida e região descoberta se comportam de formas diferentes** quando falta o
`config/<regiao>/filters.json`, e a distinção é necessária:

| origem da região | falta o `filters.json` | por quê |
|---|---|---|
| **pedida** — você chamou o pipeline com ela | erro, nomeando o caminho | você pediu por aquela região; filtrar por defaults que ninguém escolheu seria pior que parar |
| **descoberta** — um subjacente resolveu para lá | os papéis daquela região são excluídos, com log | não foi uma escolha sua, e derrubar a rodada inteira por causa dela seria desproporcional |

Sem essa separação, **um único BDR que resolvesse para uma bolsa de Londres derrubaria a coleta
toda**, porque `config/gb/filters.json` não existe. A maioria dos subjacentes resolve para listagem
americana, inclusive os europeus e asiáticos via ADR (`UL`, `TSM`, `UBS`, `YPF` na Descoberta 4),
mas "a maioria" não é garantia — e o modo de falha seria a rodada inteira, não o papel.

O nome das pastas segue o código de região do `yf.screen` (`br`, `us`), em minúsculas, para não
inventar um segundo vocabulário.

**Nota sobre `config/` e não `configs/`:** a pasta existente chama-se `config/`, e renomeá-la junto
misturaria duas mudanças sem ganho. Se preferir `configs/`, é um `git mv` a mais no plano.

### O que cada arquivo carrega

| arquivo | conteúdo | muda a cada rodada? |
|---|---|---|
| `data/<r>/tickers.csv` | identidade estável dos papéis da região | raramente |
| `data/<r>/fundamentals.csv` | o retrato do trimestre | sim |
| `data/<r>/valuation_history.csv` | snapshots append-only | sim |

**O conteúdo difere entre regiões, de propósito.** `data/br/tickers.csv` tem `ticker` e `ticker_sa`.
`data/us/tickers.csv` tem `ticker`, `ticker_bdr`, `razao` e `moeda` — o par validado pelo portão. A
resolução e o portão rodam na construção desse arquivo e ficam em cache, exatamente como
`get_tickers(force_refresh)` já faz hoje: recoletar fundamentos não revalida os pares.

`data/us/fundamentals.csv` ganha `preco_bdr` e `liq_media_diaria_bdr` além das colunas comuns —
valores voláteis, coletados a cada rodada, e por isso fora do `tickers.csv`.

**Um `valuation_history.csv` por região**, e não um único arquivo com todas as linhas. Dentro de um
arquivo, **cada coluna tem uma moeda só, igual em todas as linhas**: em `data/us/`, `preco` e
`preco_justo_dcf` são dólar da primeira à última linha, enquanto `preco_bdr` e
`liq_media_diaria_bdr` são reais da primeira à última. Nenhuma coluna muda de moeda conforme a
linha, que é a propriedade que importa — e é ela que se perderia num arquivo único, onde `preco`
seria R$ em algumas linhas e US$ em outras.

O ranking unificado continua possível, concatenando na leitura do notebook. É só lá que `preco`
passa a carregar moedas diferentes por linha, e é por isso que a coluna `moeda` existe.

`append_snapshot` já grava as premissas da rodada linha a linha e já alinha colunas por `concat`;
passa a receber a região para montar o caminho.

### `tipo` continua sendo ação ou banco; a região é outra coluna

A região `us` tem ações **e** bancos — `JPMC34`, `USBC34` e `WFCO34` existem, e a seção "Filtros"
mantém a divisão banco/não-banco em toda região. Carimbar `tipo = 'bdr'` destruiria justamente essa
distinção: um banco americano ficaria indistinguível de uma varejista americana na coluna que existe
para separar os dois.

Então `tipo` mantém os valores `'ação'` e `'banco'`, e uma coluna nova `regiao` diz de onde a linha
veio. As duas são ortogonais, e é isso que permite perguntar "bancos americanos" ou "tudo do Brasil"
sem cruzar as colunas de cabeça.

Para as 372 linhas brasileiras já gravadas, `regiao` ausente significa `br` — mesma regra de
compatibilidade da coluna `moeda`.

**Compatibilidade com o cache existente.** Os três arquivos de hoje vão para `data/br/` por `mv`
simples — `data/` está no `.gitignore` e nenhum deles é rastreado, então não há histórico de git a
preservar e `git mv` falharia com "not under version control". `data/br/fundamentals.csv` tem 372 linhas gravadas antes desta spec e
**não** possui a coluna `moeda`; código que a exija quebra com `KeyError` numa rodada que só queria
reaproveitar dado já coletado. A leitura assume `BRL` quando a coluna está ausente — verdade para
todas as linhas já gravadas. Coletas novas gravam a coluna a partir de `financialCurrency`.

### Medianas setoriais são por região

`graham_valuation` multiplica o P/L e o P/VP medianos do setor. Se as medianas forem calculadas
sobre ações brasileiras e estrangeiras juntas, o P/L mediano de "Technology" no Brasil entra na
fórmula de Graham de uma empresa americana e produz preço justo sistematicamente baixo.

Não é erro de unidade — é comparar a empresa com um mercado que não é o dela. Pela Guideline 4 o
erro cai no lado seguro (preço justo menor esconde a ação em vez de recomendá-la), mas esconder
justamente os papéis que a funcionalidade existe para mostrar anula a funcionalidade.

A separação por pastas resolve isso por construção: `apply_valuation(df, all_fundamentals, ...)` já
recebe o frame das medianas como argumento separado, e o frame de uma região é o `fundamentals.csv`
daquela região. Não há código novo, e não há como misturar sem escrever o caminho errado de
propósito.

### Onde a mistura de moedas pode aparecer

Dentro de um arquivo, nunca: nenhuma coluna muda de moeda conforme a linha. A mistura existe apenas
na visão concatenada do notebook, onde `preco` e `preco_justo_dcf` passam a carregar moedas
diferentes por linha e a coluna `moeda` rotula cada uma.

O que torna a mistura da Descoberta 1 perigosa é ela ser **implícita e dentro da mesma conta**: nada
no dado avisa que o numerador está em R$ e o denominador em US$, e o resultado sai com cara de P/L.
Na visão concatenada a moeda está rotulada e **nenhuma operação cruza linhas**:

| operação | atravessa moedas? |
|---|---|
| `pl`, `pvp`, `roe_pct`, margens, `dl_ebit` | não — adimensionais |
| `margem_seg_*` = `justo / preco − 1` | não — as duas pontas na mesma moeda |
| ordenação do ranking | não — ordena `margem_seg_media_pct`, adimensional |
| `compute_sector_averages` | mediana de `pl` e `pvp` — adimensionais, e agora por região |
| `compute_sector_betas` | mediana de `beta_raw` — adimensional, e agora por região |

A verificação das duas últimas foi feita lendo o código: ambas agregam apenas grandezas
adimensionais, então nem a mistura as corromperia.

## Filtros

A separação por pasta elimina a necessidade de blocos novos. `config/us/filters.json` reusa as
mesmas chaves `stock_filters` e `bank_filters` de `config/br/filters.json`, e o que muda é a pasta
de onde o arquivo é lido:

```python
apply_stock_filters(df, region='us')   # lê config/us/filters.json
apply_bank_filters(df, region='us')
```

Não existe `apply_bdr_filters`, nem bloco `bdr_filters`. A divisão banco/não-banco continua valendo
por região — para `JPMC34`, `USBC34` e `WFCO34`, `dl_ebit` e `liquidez_corrente` não significam
nada, pelo mesmo motivo de sempre.

Uma região sem `config/<regiao>/filters.json` levanta erro nomeando o caminho ausente. Silenciar
com defaults faria uma região nova ser filtrada por critérios que ninguém escolheu.

**Limiares idênticos aos das ações brasileiras** (`pl_max: 10`, `pvp_max: 1.5`, `roe_pct_min: 10`,
`margem_liquida_pct_min: 10`). A premissa é que o critério de "barato" é do investidor, não do
mercado: se a bolsa americana quase não produz empresa a 10x lucro, a lista vem curta, e lista
curta é informação. Escolher limiares porque produzem uma lista de tamanho agradável é exatamente o
ajuste à amostra que a Guideline 3 proíbe.

**Liquidez é do BDR, em reais, nunca do subjacente.** Em `config/us/filters.json` o corte chama-se
`liq_media_diaria_bdr_min` e lê a coluna `liq_media_diaria_bdr`. Isso resolve dois problemas de uma
vez. O prático: um BDR não-patrocinado de uma empresa excelente pode negociar poucos milhares de
reais por dia enquanto o subjacente movimenta dezenas de milhões em Nova York, e filtrar pela
liquidez de lá aprovaria um papel que não se compra nem se vende. O de unidade: 100000 é um valor em
reais, e só faz sentido contra uma grandeza em reais — comparar com a liquidez do subjacente em USD,
TWD ou JPY seria comparar número com número, sem significado. A liquidez do subjacente não entra em
critério nenhum.

O nome com sufixo `_bdr` é deliberado: é o único critério do arquivo `us` cujo valor está em reais,
e o sufixo impede que ele seja lido como "cem mil dólares" por quem editar o arquivo depois.

**E nenhuma cotação é necessária para aplicá-lo.** `liq_media_diaria_bdr` é o mesmo cálculo que
`fundamentals.py` já faz para ações brasileiras — `averageDailyVolume10Day × preço` —, só que sobre
os campos do BDR: o `regularMarketPrice` do payload do screener já vem em reais (o `currency` do
BDR é `BRL`) e o volume é contagem, sem moeda. `TSMC34.SA`: 278,75 × 165.690 = R$ 46.186.088
(medição de 2026-08-12). A comparação é reais contra reais. Câmbio só faria falta no critério que
esta spec descarta — filtrar pela liquidez do subjacente, que está em USD, TWD ou JPY.

**A coluna `liq_media_diaria` do subjacente não é gravada na região `us`.** `fetch_fundamentals`
a calcularia com o preço da `AAPL`, em dólar, e o arquivo ficaria com duas colunas quase homônimas
em moedas diferentes — sendo a **sem** sufixo a estrangeira:

```
liq_media_diaria      ← US$, da AAPL em Nova York   (nenhum critério usa)
liq_media_diaria_bdr  ← R$, do AAPL34 em B3         (a que filtra)
```

É o par exato que esta spec existe para evitar, e agravado pelo fato de a coluna sem sufixo ser a
que um leitor assumiria ser a principal. Como ela não alimenta critério, valuation nem exibição,
não é gravada. Apagar é mais seguro que renomear: um nome novo ainda pede que alguém o leia
direito, a ausência não pede nada.

Vale registrar por que apenas essa chave precisou de sufixo. Levantando as unidades de todos os
critérios: `pl`, `pvp`, `dl_ebit`, `dl_pl`, `passivos_ativos` e `liquidez_corrente` são
adimensionais; `margem_*_pct`, `roe_pct`, `crescimento_*_pct` e `dy_pct` são percentuais;
`num_analistas_min` é contagem. Sobram `lpa_min` e `lpa_estimado_min`, que **são** em moeda — mas
valem `0`, e zero é o mesmo em qualquer moeda. A liquidez é o único limite com magnitude monetária
no arquivo inteiro.

**`dy_pct` é bruto e não filtra.** O dividendo de empresa estrangeira sofre retenção na fonte antes
de chegar ao detentor do BDR (30% no caso americano), então o `dividendYield` do yfinance é o
rendimento do acionista local. A coluna é exibida com o rótulo dizendo que é bruta, e o
`bank_filters` de `config/us/` omite `dy_pct_min` — diferença deliberada em relação ao de
`config/br/`, que o exige. Modelar tributação por país está fora do escopo, e filtrar por um número
inflado seria pior que não filtrar.

Os critérios adimensionais (margem EBIT, margem líquida, ROE, P/VP, DL/EBIT, DL/PL,
passivos/ativos, liquidez corrente) atravessam sem alteração de semântica.

## Valuation

### Premissas por moeda

As constantes macro passam a ser resolvidas pelo sufixo da moeda do balanço:

```bash
# --- Macro por moeda ---
# Uma dupla por moeda habilitada. Papel cujo balanço esteja numa moeda sem
# premissas definidas é excluído, com log nomeando o que falta.
RISK_FREE_RATE_USD=0.042        # Treasury longo. Default: 0.042
EQUITY_RISK_PREMIUM_USD=0.045   # Prêmio de risco EUA. Default: 0.045

# Região varrida pelo yf.screen para montar o universo de BDRs. Default: br
BDR_REGION=br
```

`RISK_FREE_RATE` e `EQUITY_RISK_PREMIUM` sem sufixo continuam existindo e continuam significando
BRL — nenhum `.env` existente quebra. A resolução procura o sufixo primeiro e cai no nome sem
sufixo quando a moeda é BRL.

`TERMINAL_GROWTH` segue a regra existente (`= RISK_FREE_RATE` da moeda), ficando em ~4,2% para USD.
A regra "perpetuidade não pode exceder a economia" fica mais defensável aí do que em reais, porque
~4% é próximo do crescimento nominal da economia americana.

Sobe apenas com USD configurado. Habilitar Europa é acrescentar `RISK_FREE_RATE_EUR` e
`EQUITY_RISK_PREMIUM_EUR`; até lá, papel que reporta em EUR é excluído com log.

### Beta contra o índice da região, sobre os retornos do subjacente

`fetch_betas` **já recebe** o índice como parâmetro (`index_symbol='^BVSP'`); o que falta é
`_fetch_fundamentals_from_api` repassá-lo em vez de aceitar o default. Com `^GSPC` para papel
americano, a regressão roda sobre os retornos do ticker do subjacente — **não** sobre os do BDR.

O retorno do BDR em reais embute a variação do câmbio. Usá-lo colocaria risco cambial dentro do
beta, enquanto o fluxo descontado está na moeda do balanço e nunca é convertido. Seriam duas
grandezas diferentes carimbadas com o mesmo nome.

### Nenhuma conversão, e o que isso custa

O preço justo sai na moeda do balanço e fica nela. `AAPL` é avaliada em dólar e exibida em dólar.

O que se perde é ver o preço justo do `AAPL34` em reais. O que se ganha é que **a margem de
segurança liga as duas pontas de graça**: `justo / preco − 1` calculada em dólar sobre `AAPL` é a
margem do `AAPL34`, porque uma conversão apareceria nos dois termos e se cancelaria. O portão já
garantiu que o BDR acompanha o subjacente dentro de 3%.

Na prática o BDR aparece com preço em R$ (o que se paga), liquidez em R$ (o que filtra) e a margem
de segurança do subjacente — que é o número que decide a compra, e é comparável com ação
brasileira, europeia ou japonesa, porque margem é adimensional.

Os limiares de projetabilidade (`MAX_PROJECTABLE_GROWTH`, `MIN_TREND_R2`) não mudam: respondem
"consigo projetar essa taxa por 10 anos?", pergunta que não tem moeda.

### Premissas gravadas por linha

`RISK_FREE_RATE`, `EQUITY_RISK_PREMIUM` e `TERMINAL_GROWTH` são hoje constantes de módulo, lidas
direto por `cost_of_equity` e por `append_snapshot`. Passam a ser resolvidas por moeda, com BRL como
default — de modo que toda chamada existente continue funcionando sem alteração.

**Isso não é refatoração cosmética.** `append_snapshot` grava as premissas linha a linha a partir
das constantes do módulo, e seu docstring declara que elas existem para "uma divergência futura ser
atribuída a mudança de dado ou de premissa". Com BDR avaliado a 4,2% em dólar, a linha do BDR
gravaria 12,4% em reais: uma premissa que não foi usada, registrada de forma indistinguível de uma
verdadeira. O histórico passaria a mentir exatamente sobre o que foi feito para não mentir.

### Consequência conhecida: entre os BDRs que passarem, a margem tende a ser maior

Descontar a 4,2% em vez de 12,4% produz preço justo estruturalmente mais alto em relação ao preço.
Como o top-20 ordena por `margem_seg_media_pct`, **os BDRs que chegarem ao ranking tendem a chegar
por cima**.

Isso não é defeito do modelo. Reflete um fato: ativo brasileiro precisa render mais porque o juro em
reais é maior, então uma ação daqui tem que estar genuinamente mais barata para empatar com uma de
lá. A comparação entre regiões é legítima, e é justamente o que a ausência de conversão preserva —
margem de segurança é adimensional e não depende de nenhuma cotação.

Quantos BDRs chegam ao ranking é outra questão, e as duas decisões desta spec puxam em sentidos
opostos: os limiares idênticos (`pl_max: 10`) devem deixar passar poucos papéis americanos, enquanto
o custo de capital menor eleva a margem de quem passa. O resultado combinado — lista curta e
concentrada no topo, ou lista curta e irrelevante — **não é previsível sem rodar**, e nenhuma das
duas decisões foi tomada olhando para esse resultado, o que a Guideline 3 proíbe. A primeira
execução responde.

**Nenhuma normalização será aplicada.** Qualquer ajuste para "equilibrar" a lista seria escolher o
resultado antes de calculá-lo. As colunas `tipo` e `moeda` ficam visíveis no ranking, que é o
suficiente para o leitor ver a origem de cada linha.

## Componentes

| arquivo | responsabilidade |
|---|---|
| `src/bdrs.py` (novo) | universo via screener, marcador `DR[N123]\b`, resolução do par, portão de qualidade, elegibilidade por moeda; escreve `data/us/tickers.csv` |
| `src/paths.py` (novo) | resolve `data/<regiao>/<arquivo>` e `config/<regiao>/filters.json`; único lugar que sabe o formato dos caminhos |
| `src/scraper.py` | `get_tickers` recebe a região e lê/escreve `data/<r>/tickers.csv` |
| `src/fundamentals.py` | `fetch_betas` recebe o índice como parâmetro; `fetch_fundamentals` recebe a região (hoje `FUNDAMENTALS_CACHE` é fixo no módulo) e passa a gravar `moeda` a partir de `financialCurrency`; o corpo da coleta é reaproveitado sem alteração |
| `src/valuation.py` | premissas macro resolvidas por moeda; `append_snapshot` recebe a região; snapshot gravando as premissas efetivamente usadas em cada linha |
| `src/filters.py` | `apply_stock_filters` e `apply_bank_filters` recebem a região; nenhuma função nova |
| `config/br/filters.json` | o arquivo de hoje, movido por `git mv` |
| `config/us/filters.json` | novo: mesmas chaves, `liq_media_diaria_bdr_min` no lugar de `liq_media_diaria_min`, `bank_filters` sem `dy_pct_min` |
| `data/br/*` | os 3 arquivos de hoje, movidos por `git mv` |
| `.env.example` | `RISK_FREE_RATE_USD`, `EQUITY_RISK_PREMIUM_USD`, `BDR_REGION` |
| `analysis.ipynb` | pipeline da região `us` (ações e bancos), leitura por região, coluna `regiao`, concatenação para o ranking unificado |

`src/bdrs.py` é o único módulo com lógica nova. Recebe uma região de varredura e devolve **dois
frames**, porque os dados que ele produz têm tempos de vida diferentes:

| frame | colunas | destino | validade |
|---|---|---|---|
| pares | `ticker`, `ticker_bdr`, `razao`, `moeda` | `data/<r>/tickers.csv` | estável, fica em cache |
| cotações | `ticker`, `preco_bdr`, `liq_media_diaria_bdr` | mesclado em `fundamentals.csv` na coleta | volátil, recoletado a cada rodada |

A separação não é burocracia: o portão precisa de preço para validar o par, mas o preço que ele usou
não pode ser gravado no `tickers.csv`, senão o arquivo "estável" carregaria uma cotação de meses
atrás que ninguém sabe que está velha. O `fetch_fundamentals` da região traz as cotações frescas e
as junta por `ticker`.

Quem consome não precisa saber nada sobre screener, marcadores, tolerâncias ou elegibilidade.

**Custo de rede, declarado:** `bdrs.py` faz duas requisições por BDR candidato (a busca e o `.info`
do subjacente, este último necessário para o `financialCurrency` da elegibilidade), e
`fetch_fundamentals` busca o `.info` do mesmo subjacente de novo. São ~594 requisições duplicadas
numa reconstrução completa do `tickers.csv`. Aceito de propósito: eliminar a duplicata exigiria
`bdrs.py` devolver o `.info` inteiro e `fetch_fundamentals` aceitar dado pré-buscado, acoplando os
dois módulos por causa de um custo que só aparece quando o cache de pares é reconstruído — o que é
raro, como no `get_tickers` de hoje.

## Tratamento de erro

Nenhuma falha de rede ou de dado pode derrubar a coleta, seguindo o padrão já estabelecido em
`_estimate_cell` e `_fetch_fundamentals_from_api`.

| situação | comportamento |
|---|---|
| `yf.screen` falha | erro propagado — sem universo não há o que fazer, e falhar em silêncio produziria lista vazia indistinguível de "nada passou" |
| `yf.Search` falha ou não devolve candidato | BDR descartado, motivo no log |
| razão não-inteira, ou cotação implícita fora da tolerância | BDR descartado, motivo e os dois valores no log |
| menos de 3 pares na mesma moeda de pregão | mediana não é referência confiável com amostra minúscula; nenhum desses pares entra, com aviso explícito distinguindo isso de "nada passou nos filtros" |
| `currency != financialCurrency` | BDR descartado, com as duas moedas no log |
| moeda do balanço sem premissas no `.env` | BDR descartado, log nomeando as variáveis que faltam |
| `.info` do subjacente vazio | linha com NaN, como já acontece hoje |

A contagem de descartes por motivo é impressa ao fim da resolução, no mesmo formato das mensagens
`[filters]` existentes. Os motivos são distinguíveis entre si: "não resolveu", "reprovou no portão",
"moeda divergente" e "moeda sem premissas" pedem ações diferentes do usuário.

## Testes

Seguindo o padrão de `tests/test_filters.py` e `tests/test_fundamentals.py`, sobre dados
construídos à mão — nunca sobre o cache, pela Guideline 3.

**`src/bdrs.py`**
- marcador `DR[N123]\b` aceita `DRN`, `DR1`, `DR2`, `DR3` e rejeita `ON NM`, `PN`, `CI`
- **marcador colado ao nome** (`'ZILLOW GROUPDRN'`, `'WHIRLPOOL CODRN'`) é aceito — é a regressão do
  `\b` inicial, que descartava 49% do universo em silêncio
- **marcador seguido de `ED`** (`'WELLS FARGO DRN ED'`) é aceito, e o caso com os dois problemas
  juntos (`'CONSTELLATIODRN ED'`) também
- portão aprova pares concordantes; rejeita razão não-inteira; rejeita cotação implícita fora da
  tolerância
- fronteira da tolerância: desvio exatamente no limite, logo abaixo e logo acima
- a mediana é por moeda de pregão: um grupo em EUR não desloca a referência do grupo em USD
- a mediana ignora os pares já reprovados pela razão inteira
- grupo de moeda com menos de 3 pares válidos não aprova ninguém, e o aviso é distinguível de "nada
  passou nos filtros"
- `currency != financialCurrency` exclui, mesmo com o portão aprovado
- moeda sem premissas no `.env` exclui, e a mensagem nomeia as variáveis faltantes
- região **descoberta** sem `config/<r>/filters.json` exclui só aqueles papéis; região **pedida** sem
  o arquivo levanta erro — um BDR resolvido para Londres não pode derrubar a rodada
- os dois frames devolvidos são disjuntos: `tickers.csv` não recebe `preco_bdr` nem
  `liq_media_diaria_bdr`, para o arquivo estável não guardar cotação velha
- BDR sem candidato não aparece na saída
- reprovado não aparece na saída **com dado parcial** — a asserção é sobre ausência da linha

**`src/valuation.py`**
- resolução das premissas por moeda: `USD` usa `RISK_FREE_RATE_USD`; `BRL` usa `RISK_FREE_RATE`
- default preservado: chamada sem moeda continua usando as premissas em reais
- snapshot de linha `bdr` grava RF/ERP em dólar e linha `ação` grava em reais, no mesmo
  `append_snapshot`
- margem de segurança de um par BDR/subjacente é a mesma calculada em qualquer das duas moedas —
  trava a invariância que substitui a conversão
- medianas setoriais vindas de frames distintos produzem preços de Graham distintos para a mesma
  empresa: trava a separação por região
- `tipo` de um banco americano é `'banco'`, não `'bdr'`, e `regiao` é `'us'` — trava a
  ortogonalidade das duas colunas
- linha antiga sem `regiao` é lida como `br`, como a linha sem `moeda` é lida como `BRL`

**`src/filters.py`**
- `apply_stock_filters(df, region='us')` lê `config/us/filters.json`, e `region='br'` lê o de `br`
- região sem `config/<regiao>/filters.json` levanta erro nomeando o caminho, em vez de usar defaults
- o corte de liquidez da região `us` usa `liq_media_diaria_bdr`
- `data/us/fundamentals.csv` **não tem** a coluna `liq_media_diaria` — a asserção é sobre ausência,
  para que a coluna em dólar não reapareça por descuido ao lado da homônima em reais
- `bank_filters` de `us` não aplica `dy_pct_min`; o de `br` aplica
- NaN reprova nos critérios exigidos, como já vale para `_estimates_mask`

**`src/paths.py`**
- monta `data/<r>/<arquivo>` e `config/<r>/filters.json` para regiões arbitrárias
- região com nome vazio ou com separador de caminho é rejeitada, para o nome não escapar da pasta

## Fora de escopo

- **Conversão de moeda em qualquer ponto do pipeline.** Papel cujo pregão e balanço divergem é
  excluído em vez de convertido
- Preço justo do BDR exibido em reais — consequência direta do item acima
- Modelagem de tributação (retenção na fonte sobre dividendo, IOF, ganho de capital)
- Tratamento de subunidades monetárias (`GBp`, `ILA`, `ZAc`): os papéis afetados já saem pela
  divergência entre pregão e balanço
- Normalização do ranking entre regiões
- Reclassificação dos BDRs de empresa brasileira (`JBSS32`, `XPBR31`, `INBR32`, `ROXO34`,
  `AURA33`): passam pelo mesmo caminho dos demais, sem tratamento próprio, ainda que sua operação
  seja no Brasil
