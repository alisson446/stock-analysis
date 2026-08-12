# BDRs no screener: fundamento estrangeiro, na moeda de origem

**Data:** 2026-08-12
**Escopo:** novo balde de ativos (BDR) percorrendo o pipeline inteiro — universo, coleta, filtros,
valuation e snapshot. Não altera o comportamento das ações e bancos brasileiros, exceto por uma
refatoração das premissas macro (ver "Premissas por moeda"), necessária para o histórico não mentir.

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
   625 BDRs, 594 com longName          data/bdrs.csv (cache, padrão do data/tickers.csv)
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
        │
        ▼  bdr_filters / bdr_bank_filters
        │
        ▼  valuation na moeda do balanço (RF e ERP daquela moeda)
        │
        ▼  preco_justo na mesma moeda — nenhuma conversão
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

Isso importa para o custo da coleta: o "dois tickers por BDR" citado em "Onde os dados ficam" é o
subjacente mais a busca, não uma terceira ida ao BDR.

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

### Onde os dados ficam

**`data/fundamentals_bdr.csv`, separado de `data/fundamentals.csv`.** Duas razões:

1. **O portão de cache é único.** `fetch_fundamentals` decide por `FUNDAMENTALS_CACHE.exists()` com
   um só `force_refresh`. Unificados, atualizar as ações brasileiras obrigaria a recoletar todos os
   BDRs — que custam dois tickers cada.
2. Colunas exclusivas de BDR (`ticker_subjacente`, `razao`, `preco_bdr`, `liq_media_diaria_bdr`)
   ficariam NaN nas 372 linhas brasileiras.

**`data/valuation_history.csv`, o mesmo arquivo.** Aqui unir é o certo e o mecanismo já existe: a
célula 13 do notebook já carimba `tipo` (`'ação'` / `'banco'`), `_SNAPSHOT_RESULT_COLS` já inclui a
coluna, e `append_snapshot` já alinha conjuntos de colunas diferentes por `concat` de propósito. O
terceiro balde entra com `tipo = 'bdr'`.

### A coluna `moeda`, e por que a mistura é aceitável aqui

`preco` e `preco_justo_dcf` passam a conter valores em moedas diferentes conforme a linha, e uma
coluna `moeda` rotula cada uma. Isso contraria a regra que o resto desta spec aplica com rigor, e a
diferença precisa estar explícita.

O que torna a mistura da Descoberta 1 perigosa é ela ser **implícita e dentro da mesma conta**:
nada no dado avisa que o numerador está em R$ e o denominador em US$, e o resultado sai com cara de
P/L. Aqui a moeda está rotulada na própria linha, e **nenhuma operação cruza linhas de moedas
diferentes**:

| operação | atravessa moedas? |
|---|---|
| `pl`, `pvp`, `roe_pct`, margens, `dl_ebit` | não — adimensionais |
| `margem_seg_*` = `justo / preco − 1` | não — as duas pontas na mesma moeda |
| ordenação do ranking | não — ordena `margem_seg_media_pct`, adimensional |
| `compute_sector_averages` | mediana de `pl` e `pvp` — adimensionais |
| `compute_sector_betas` | mediana de `beta_raw` — adimensional |

A verificação das duas últimas foi feita lendo o código: ambas agregam apenas grandezas
adimensionais, então a mistura de moedas não as corrompe.

Elas têm, ainda assim, um problema de **modelagem** — ver a seção seguinte.

**Compatibilidade com o cache existente.** `data/fundamentals.csv` tem 372 linhas gravadas antes
desta spec e **não** possui a coluna `moeda`. Código que a exija quebra ao ler o cache atual, e o
sintoma seria um `KeyError` numa rodada que só queria reaproveitar dado já coletado. A leitura
assume `BRL` quando a coluna está ausente — o que é verdade para todas as linhas já gravadas, já
que o universo até aqui é inteiramente brasileiro. Coletas novas gravam a coluna a partir de
`financialCurrency`.

### Medianas setoriais são por balde, não do universo inteiro

`graham_valuation` multiplica o P/L e o P/VP medianos do setor. Se as medianas forem calculadas
sobre ações brasileiras e estrangeiras juntas, o P/L mediano de "Technology" no Brasil entra na
fórmula de Graham de uma empresa americana e produz preço justo sistematicamente baixo.

Não é erro de unidade — é comparar a empresa com um mercado que não é o dela. Pela Guideline 4 o
erro cai no lado seguro (preço justo menor esconde a ação em vez de recomendá-la), mas esconder
justamente os papéis que a funcionalidade existe para mostrar anula a funcionalidade.

`apply_valuation(df, all_fundamentals, model=...)` já recebe o frame das medianas como argumento
separado. Basta o notebook passar `fundamentals_bdr` no balde de BDR, em vez do frame brasileiro. É
uma linha, e nenhum código novo.

## Filtros

Blocos novos em `config/filters.json`: `bdr_filters` e `bdr_bank_filters`, espelhando a divisão
entre `stock_filters` e `bank_filters` que já existe — para `JPMC34`, `USBC34` e `WFCO34`,
`dl_ebit` e `liquidez_corrente` não significam nada, pelo mesmo motivo de sempre.

**Limiares idênticos aos das ações brasileiras** (`pl_max: 10`, `pvp_max: 1.5`, `roe_pct_min: 10`,
`margem_liquida_pct_min: 10`). A premissa é que o critério de "barato" é do investidor, não do
mercado: se a bolsa americana quase não produz empresa a 10x lucro, a lista vem curta, e lista
curta é informação. Escolher limiares porque produzem uma lista de tamanho agradável é exatamente o
ajuste à amostra que a Guideline 3 proíbe.

**Liquidez é do BDR, em reais, nunca do subjacente.** O corte lê `liq_media_diaria_bdr`. Isso
resolve dois problemas de uma vez. O prático: um BDR não-patrocinado de uma empresa excelente pode
negociar poucos milhares de reais por dia enquanto o subjacente movimenta dezenas de milhões em
Nova York, e filtrar pela liquidez de lá aprovaria um papel que não se compra nem se vende. O de
unidade: `liq_media_diaria_min: 100000` é um valor em reais, e só faz sentido contra uma grandeza
em reais — comparar com a liquidez do subjacente em USD, TWD ou JPY seria comparar número com
número, sem significado. A liquidez do subjacente não entra em critério nenhum.

**`dy_pct` é bruto e não filtra.** O dividendo de empresa estrangeira sofre retenção na fonte antes
de chegar ao detentor do BDR (30% no caso americano), então o `dividendYield` do yfinance é o
rendimento do acionista local. A coluna é exibida com o rótulo dizendo que é bruta, e
`bdr_bank_filters` não usa `dy_pct_min`. Modelar tributação por país está fora do escopo, e filtrar
por um número inflado seria pior que não filtrar.

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

`fetch_betas` ganha o índice como parâmetro (`^GSPC` para papel americano), e a regressão roda sobre
os retornos do ticker do subjacente — **não** sobre os do BDR.

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

Quantos chegam é outra questão, e as duas decisões desta spec puxam em sentidos opostos: os
limiares idênticos (`pl_max: 10`) devem deixar passar poucos papéis americanos, enquanto o custo de
capital menor eleva a margem de quem passa. O resultado combinado — lista curta e concentrada no
topo, ou lista curta e irrelevante — **não é previsível sem rodar**, e nenhuma das duas decisões
foi tomada olhando para esse resultado, o que a Guideline 3 proíbe. A primeira execução responde.

Não é defeito do modelo. Reflete um fato: ativo brasileiro precisa render mais porque o juro em
reais é maior, então uma ação daqui tem que estar genuinamente mais barata para empatar com uma de
lá. A comparação entre baldes é legítima, e é justamente o que a ausência de conversão preserva —
margem de segurança é adimensional e não depende de nenhuma cotação.

**Nenhuma normalização será aplicada.** Qualquer ajuste para "equilibrar" a lista seria escolher o
resultado antes de calculá-lo. As colunas `tipo` e `moeda` ficam visíveis no ranking, que é o
suficiente para o leitor ver a origem de cada linha.

## Componentes

| arquivo | responsabilidade |
|---|---|
| `src/bdrs.py` (novo) | universo via screener, marcador `DR[N123]`, resolução do par, portão de qualidade, elegibilidade por moeda, cache `data/bdrs.csv` |
| `src/fundamentals.py` | `fetch_betas` recebe o índice como parâmetro; `fetch_fundamentals` recebe o caminho do cache como parâmetro (hoje `FUNDAMENTALS_CACHE` é fixo no módulo) e passa a gravar `moeda` a partir de `financialCurrency`; o corpo da coleta é reaproveitado sem alteração |
| `src/valuation.py` | premissas macro resolvidas por moeda; snapshot gravando as premissas efetivamente usadas em cada linha |
| `src/filters.py` | `apply_bdr_filters`, lendo `bdr_filters` / `bdr_bank_filters` |
| `config/filters.json` | dois blocos novos |
| `.env.example` | `RISK_FREE_RATE_USD`, `EQUITY_RISK_PREMIUM_USD`, `BDR_REGION` |
| `analysis.ipynb` | terceiro balde com `tipo = 'bdr'`; medianas setoriais do balde de BDR vindas de `fundamentals_bdr` |

`src/bdrs.py` é o único módulo com lógica nova. Sua fronteira é estreita: recebe uma região e
devolve um DataFrame já validado, com uma linha por par aprovado e as colunas
`ticker_bdr`, `ticker_subjacente`, `razao`, `moeda`, `preco_bdr` e `liq_media_diaria_bdr` — as duas
últimas vindas do payload do screener, como descrito em "As colunas do lado brasileiro saem do
próprio screener". Quem consome não precisa saber nada sobre screener, marcadores, tolerâncias ou
elegibilidade.

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
  empresa: trava a separação por balde

**`src/filters.py`**
- `apply_bdr_filters` corta por `liq_media_diaria_bdr` e ignora `liq_media_diaria`
- `bdr_bank_filters` não aplica `dl_ebit` nem `liquidez_corrente`
- NaN reprova nos critérios exigidos, como já vale para `_estimates_mask`

## Fora de escopo

- **Conversão de moeda em qualquer ponto do pipeline.** Papel cujo pregão e balanço divergem é
  excluído em vez de convertido
- Preço justo do BDR exibido em reais — consequência direta do item acima
- Modelagem de tributação (retenção na fonte sobre dividendo, IOF, ganho de capital)
- Tratamento de subunidades monetárias (`GBp`, `ILA`, `ZAc`): os papéis afetados já saem pela
  divergência entre pregão e balanço
- Normalização do ranking entre baldes
- Reclassificação dos BDRs de empresa brasileira (`JBSS32`, `XPBR31`, `INBR32`, `ROXO34`,
  `AURA33`): passam pelo mesmo caminho dos demais, sem tratamento próprio, ainda que sua operação
  seja no Brasil
