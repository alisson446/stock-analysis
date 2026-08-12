# BDRs no screener: fundamento americano, preço brasileiro

**Data:** 2026-08-12
**Escopo:** novo balde de ativos (BDR) percorrendo o pipeline inteiro — universo, coleta, filtros,
valuation e snapshot. Não altera o comportamento das ações e bancos brasileiros, exceto por uma
refatoração das premissas macro (ver "Premissas por balde"), necessária para o histórico não mentir.

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
que quebra é tudo que cruza preço em R$ com valor absoluto em US$: P/L, LPA, dívida líquida, EBIT,
FCF — e, por consequência, o DCF inteiro.

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
razao_acoes = sharesOutstanding(BDR) / sharesOutstanding(US)
```

Pelos preços, o que sai é a cotação do dólar que o par **implica**:

```
fx_implicito = razao_acoes × preco(BDR) / preco(US)
```

Num par correto, esse número é o câmbio de mercado. Num par errado, é um valor sem sentido — e o
teste é que todos os pares do universo têm que implicar aproximadamente o mesmo dólar.

Medição de 2026-08-12 (dólar de mercado no mesmo instante: 5,1678):

| BDR | US | razão por ações | dólar implícito | desvio da mediana |
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
ações não-inteira e dólar implícito 15% fora.

Com o portão aplicado, a amostra de 22 aprovou 21 e rejeitou exatamente o par errado.

**A mediana é calibrada pelo próprio universo, não por uma cotação externa.** Isso é deliberado:
o portão continua funcionando com o dólar em qualquer patamar, sem depender de nenhum valor
configurado estar atualizado (ver "Cotação do dólar").

## Arquitetura

**Em uma frase: o BDR é o preço; a ação americana é a empresa.**

```
yf.screen(region=BDR_REGION)          944 papéis (mcap > 500M, medição de 2026-08-12)
        │  filtra shortName ~ /DR[N123]/
        ▼
   312 BDRs                            data/bdrs.csv (cache, padrão do data/tickers.csv)
        │
        │  yf.Search(longName) → candidato em bolsa americana
        ▼
   ┌─── PORTÃO DE QUALIDADE ─────────────────────┐
   │  razao_acoes inteira (tolerância ±0,02)     │
   │  fx_implicito a menos de 3% da mediana      │
   │  do universo (auto-calibrado)               │
   └─────────────────────────────────────────────┘
        │ aprovado                      │ rejeitado
        ▼                               ▼
  ┌──────────────┬───────────────┐   fora da lista,
  │ AAPL   (US$) │ AAPL34  (R$)  │   com motivo no log
  │ fundamentos  │ preço         │
  │ estimativas  │ liquidez      │
  └──────────────┴───────────────┘
        │
        ▼  bdr_filters / bdr_bank_filters
        │
        ▼  valuation em US$ (RF e ERP americanos, beta vs ^GSPC)
        │
        ▼  preco_justo_bdr = preco_justo(US$) × USD_BRL_RATE ÷ razao
```

### Identificação do BDR

O `shortName` do screener carrega o marcador do tipo de recibo: `DRN` para não-patrocinado, `DR1`,
`DR2` e `DR3` para os patrocinados. O regex é `\bDR[N123]\b`.

Isso separa BDR de ação brasileira e de FII sem lista hard-coded:

| symbol | shortName | longName | é BDR |
|---|---|---|---|
| `Z1TS34.SA` | `ZOETIS INC  DRN` | Zoetis Inc. | sim |
| `JBSS32.SA` | `JBS N.V.    DR2` | JBS N.V. | sim |
| `XPBR31.SA` | `XP INC      DR1` | XP Inc. | sim |
| `YDUQ3.SA` | `YDUQS PART  ON      NM` | Yduqs Participações S.A. | não |
| `ZIFI11.SA` | `FII ZION    CI` | Zion Capital FI | não |

O `longName` do BDR é o nome legal da empresa estrangeira, sem adaptação — é ele que alimenta a
busca pelo ticker americano.

`NUBR33.SA` não existe mais no Yahoo (404 no `.info`, ausente do screener); o Nubank aparece como
`ROXO34.SA → NU`. Nenhum tratamento especial: o papel simplesmente não entra no universo.

### Portão de qualidade

Um BDR entra na lista somente se as duas medidas concordarem. Sem par resolvido, ou com par
reprovado, o papel **não entra nem com dado parcial**.

É a Guideline 4 aplicada ao mapeamento. Um par errado não produz um erro visível: produz um preço
justo com aparência de calculado, sobre outra empresa. `FMXB34` avaliado como Vista Energy sairia
com número, unidade e formatação corretos, e chegaria ao usuário como recomendação.

A tolerância de 3% e a de ±0,02 no inteiro vêm de premissa, não de ajuste à amostra: o desvio do
dólar implícito é dominado por defasagem de cotação — o BDR tem 15 minutos de atraso e a ação
americana é outro instante — e 3% é folga confortável para descasamento intradiário sem acomodar
erro de identidade. O par errado medido deu 15%, cinco vezes o limite.

A mediana é calculada sobre os pares que já passaram no teste da razão inteira, para que um punhado
de pares errados não desloque a referência.

### Onde os dados ficam

**`data/fundamentals_bdr.csv`, separado de `data/fundamentals.csv`.** Três razões:

1. **Moeda na mesma coluna.** Num arquivo único, `preco`, `lpa`, `ebit` e `divida_liquida` seriam
   R$ nas linhas brasileiras e US$ nas de BDR. É a mesma mistura silenciosa que produz o P/L de 178
   da Descoberta 1, agora dentro do cache, onde qualquer código futuro assumiria homogeneidade.
2. **O portão de cache é único.** `fetch_fundamentals` decide por `FUNDAMENTALS_CACHE.exists()` com
   um só `force_refresh`. Unificados, atualizar as ações brasileiras obrigaria a recoletar todos os
   BDRs — que custam dois tickers cada.
3. Colunas exclusivas de BDR (`ticker_us`, `razao`, `preco_bdr`, `liq_media_diaria_bdr`,
   `fx_usdbrl`) ficariam NaN nas 372 linhas brasileiras.

**`data/valuation_history.csv`, o mesmo arquivo.** Aqui unir é o certo e o mecanismo já existe: a
célula 13 do notebook já carimba `tipo` (`'ação'` / `'banco'`), `_SNAPSHOT_RESULT_COLS` já inclui a
coluna, e `append_snapshot` já alinha conjuntos de colunas diferentes por `concat` de propósito. O
terceiro balde entra com `tipo = 'bdr'`.

A condição é a linha do BDR ser gravada **em R$ por BDR**: `preco` é o que se paga na B3 e
`preco_justo_dcf` é o justo convertido e dividido pela razão. Assim a coluna tem uma unidade só no
arquivo inteiro. `fx_usdbrl` e `razao` entram no snapshot para a linha ser reproduzível depois.

Atenção a uma troca de nome entre os dois arquivos, que é deliberada e precisa estar explícita: em
`fundamentals_bdr.csv` a coluna `preco` é o preço da ação americana em US$ e `preco_bdr` é o preço
negociado em R$; ao montar o snapshot, é o `preco_bdr` que alimenta a coluna `preco` do histórico.
Cada arquivo mantém uma unidade coerente por coluna; a conversão entre as duas convenções acontece
uma única vez, na montagem do snapshot.

### Nomenclatura de colunas

As colunas do lado americano mantêm os nomes atuais — são a empresa, em US$. As do lado negociado
levam sufixo `_bdr` e estão em R$: `preco_bdr`, `liq_media_diaria_bdr`. **Nenhuma coluna muda de
moeda dependendo da linha.**

## Filtros

Blocos novos em `config/filters.json`: `bdr_filters` e `bdr_bank_filters`, espelhando a divisão
entre `stock_filters` e `bank_filters` que já existe — para `JPMC34`, `USBC34` e `WFCO34`,
`dl_ebit` e `liquidez_corrente` não significam nada, pelo mesmo motivo de sempre.

**Limiares idênticos aos das ações brasileiras** (`pl_max: 10`, `pvp_max: 1.5`, `roe_pct_min: 10`,
`margem_liquida_pct_min: 10`). A premissa é que o critério de "barato" é do investidor, não do
mercado: se a bolsa americana quase não produz empresa a 10x lucro, a lista vem curta, e lista
curta é informação. Escolher limiares porque produzem uma lista de tamanho agradável é exatamente o
ajuste à amostra que a Guideline 3 proíbe.

**Liquidez é do BDR, nunca da ação americana.** O corte lê `liq_media_diaria_bdr`, em R$. Um BDR
não-patrocinado de uma empresa excelente pode negociar poucos milhares de reais por dia enquanto a
ação movimenta dezenas de milhões de dólares em Nova York; filtrar pela liquidez americana
aprovaria um papel que não se compra nem se vende. A liquidez americana não entra em critério
nenhum.

**`dy_pct` é bruto e não filtra.** O dividendo de empresa americana sofre 30% de retenção na fonte
antes de chegar ao detentor do BDR, então o `dividendYield` do yfinance é o rendimento do acionista
americano — o do BDR recebe cerca de 70% disso. A coluna é exibida com o rótulo dizendo que é
bruta, e `bdr_bank_filters` não usa `dy_pct_min`. Modelar a tributação está fora do escopo, e
filtrar por um número inflado seria pior que não filtrar.

Os critérios adimensionais (margem EBIT, margem líquida, ROE, P/VP, DL/EBIT, DL/PL,
passivos/ativos, liquidez corrente) atravessam sem alteração de semântica.

## Valuation

### Premissas macro em US$

Novas variáveis no `.env`, seguindo o padrão das existentes:

```bash
# --- Macro em US$ (BDRs) ---
RISK_FREE_RATE_USD=0.042        # Treasury longo. Default: 0.042
EQUITY_RISK_PREMIUM_USD=0.045   # Prêmio de risco EUA. Default: 0.045
USD_BRL_RATE=5.17               # Cotação do dólar. Default: 5.17 (de 2026-08-12)
```

`TERMINAL_GROWTH_USD` segue a regra existente (`= RISK_FREE_RATE`), ficando em ~4,2%. A regra
"perpetuidade não pode exceder a economia" fica mais defensável aqui do que em reais, porque ~4% é
próximo do crescimento nominal da economia americana.

Uma terceira variável controla o universo: `BDR_REGION` (default `br`), a região passada ao
`yf.screen`.

### Beta contra o S&P 500, sobre os retornos da ação americana

`fetch_betas` ganha o índice como parâmetro; para BDRs é `^GSPC`, e a regressão roda sobre os
retornos do ticker americano — **não** sobre os do BDR.

O retorno do BDR em reais embute a variação do dólar. Usá-lo colocaria risco cambial dentro do
beta, enquanto o fluxo descontado é em dólar e a conversão só acontece no fim. Misturar as duas
coisas contaria o câmbio duas vezes.

### Cotação do dólar

`USD_BRL_RATE` é lida do `.env` por `_env_float`, como as demais premissas macro, com **5,17** de
default — a cotação de 2026-08-12, data desta spec.

Vale registrar por que ela fica no `.env` mesmo sendo um dado que o yfinance **fornece**
(`USDBRL=X` responde normalmente), ao contrário da convenção declarada no cabeçalho do
`.env.example`. Três motivos:

1. **Reprodutibilidade do snapshot.** O histórico grava `fx_usdbrl` junto com RF e ERP como
   premissa da rodada. Com a cotação vindo da rede, duas execuções no mesmo dia gravam preços
   justos diferentes sem que nenhuma premissa tenha mudado.
2. **É premissa, não medição.** Converter um preço justo de 10 anos de fluxo pelo dólar de um
   instante já é uma escolha; deixá-la explícita permite rodar cenário ("e se o dólar for a 6?")
   editando uma linha.
3. **Remove um modo de falha.** A cotação era o único ponto do pipeline capaz de derrubar todos os
   BDRs por indisponibilidade de rede.

**O portão de qualidade não usa esse valor.** Ele se calibra pela mediana dos dólares implícitos do
próprio universo. Isso importa: se o portão dependesse de `USD_BRL_RATE` e o dólar de mercado
andasse mais que 3% desde a última vez que você editou o `.env`, **todos** os pares passariam a
desviar acima da tolerância e o screener rejeitaria o universo inteiro — uma falha total disparada
por um valor desatualizado, e que se pareceria com "nenhum BDR passou nos critérios". Separando as
duas coisas, uma cotação velha no `.env` erra apenas o valor em R$ exibido, proporcionalmente e de
forma auditável pela coluna `fx_usdbrl` do snapshot.

### Conversão, num lugar só

```
preco_justo_bdr(R$) = preco_justo(US$) × USD_BRL_RATE ÷ razao
```

A margem de segurança é invariante ao câmbio: `justo / preco − 1` dá o mesmo número calculado em
dólar ou em real, porque o `fx` aparece nos dois termos. Uma cotação desatualizada no `.env` move o
valor exibido em R$, nunca a decisão nem o ranking.

Os limiares de projetabilidade (`MAX_PROJECTABLE_GROWTH`, `MIN_TREND_R2`) não mudam: respondem
"consigo projetar essa taxa por 10 anos?", pergunta que não tem moeda.

### Premissas por balde

`RISK_FREE_RATE`, `EQUITY_RISK_PREMIUM` e `TERMINAL_GROWTH` são hoje constantes de módulo, lidas
direto por `cost_of_equity` e por `append_snapshot`. Passam a ser dois conjuntos selecionados pelo
balde, com o de reais como default — de modo que toda chamada existente continue funcionando sem
alteração.

**Isso não é refatoração cosmética.** `append_snapshot` grava as premissas linha a linha a partir
das constantes do módulo, e seu docstring declara que elas existem para "uma divergência futura ser
atribuída a mudança de dado ou de premissa". Com BDR avaliado a 4,2% em dólar, a linha do BDR
gravaria 12,4% em reais: uma premissa que não foi usada, registrada de forma indistinguível de uma
verdadeira. O histórico passaria a mentir exatamente sobre o que foi feito para não mentir.

### Consequência conhecida: BDRs devem dominar o ranking

Descontar a 4,2% em vez de 12,4% produz preço justo estruturalmente mais alto. Como o top-20 ordena
por `margem_seg_media_pct`, é esperado que os BDRs ocupem a maior parte da lista assim que entrarem.

Não é defeito do modelo. Reflete um fato: ativo brasileiro precisa render mais porque o juro em
reais é maior, então uma ação daqui tem que estar genuinamente mais barata para empatar com uma de
lá. A comparação entre baldes é legítima.

**Nenhuma normalização será aplicada.** Qualquer ajuste para "equilibrar" a lista seria escolher o
resultado antes de calculá-lo. A coluna `tipo` fica visível no ranking, que é o suficiente para o
leitor ver a origem de cada linha. Se depois a preferência for listar os baldes em separado, isso é
mudança de exibição e não de modelo.

## Componentes

| arquivo | responsabilidade |
|---|---|
| `src/bdrs.py` (novo) | universo via screener, marcador `DR[N123]`, resolução do par, portão de qualidade, cache `data/bdrs.csv` |
| `src/fundamentals.py` | `fetch_betas` recebe o índice como parâmetro; `fetch_fundamentals` recebe o caminho do cache como parâmetro (hoje `FUNDAMENTALS_CACHE` é fixo no módulo, e os dois baldes gravam em arquivos diferentes); o corpo da coleta é reaproveitado sem alteração |
| `src/valuation.py` | premissas macro por balde; conversão do preço justo; snapshot com `fx_usdbrl` e `razao` |
| `src/filters.py` | `apply_bdr_filters`, lendo `bdr_filters` / `bdr_bank_filters` |
| `config/filters.json` | dois blocos novos |
| `.env.example` | `RISK_FREE_RATE_USD`, `EQUITY_RISK_PREMIUM_USD`, `USD_BRL_RATE`, `BDR_REGION` |
| `analysis.ipynb` | terceiro balde, `tipo = 'bdr'`, na união da célula 13 |

`src/bdrs.py` é o único módulo com lógica nova. Sua fronteira é estreita: recebe uma região,
devolve um DataFrame de pares `(ticker_bdr, ticker_us, razao)` já validados. Quem consome não
precisa saber nada sobre screener, marcadores ou tolerâncias.

## Tratamento de erro

Nenhuma falha de rede ou de dado pode derrubar a coleta, seguindo o padrão já estabelecido em
`_estimate_cell` e `_fetch_fundamentals_from_api`.

| situação | comportamento |
|---|---|
| `yf.screen` falha | erro propagado — sem universo não há o que fazer, e falhar em silêncio produziria lista vazia indistinguível de "nada passou" |
| `yf.Search` falha ou não devolve candidato | BDR descartado, motivo no log |
| razão não-inteira, ou dólar implícito fora da tolerância | BDR descartado, motivo e os dois valores no log |
| menos de 3 pares sobrevivem ao teste da razão inteira | mediana não é referência confiável com amostra minúscula; nenhum BDR entra, com aviso explícito distinguindo isso de "nada passou nos filtros" |
| `.info` do ticker americano vazio | linha com NaN, como já acontece hoje |
| `USD_BRL_RATE` ausente do `.env` | usa o default de 5,17, como `_env_float` já faz com RF e ERP |

A contagem de descartes por motivo é impressa ao fim da resolução, no mesmo formato das mensagens
`[filters]` existentes.

## Testes

Seguindo o padrão de `tests/test_filters.py` e `tests/test_fundamentals.py`, sobre dados
construídos à mão — nunca sobre o cache, pela Guideline 3.

**`src/bdrs.py`**
- marcador `DR[N123]` aceita `DRN`, `DR1`, `DR2`, `DR3` e rejeita `ON NM`, `PN`, `CI`
- portão aprova pares concordantes; rejeita razão não-inteira; rejeita dólar implícito fora da
  tolerância
- fronteira da tolerância: desvio exatamente no limite, logo abaixo e logo acima
- a mediana ignora os pares já reprovados pela razão inteira — um lote de pares errados não desloca
  a referência
- **o portão não lê `USD_BRL_RATE`**: com a variável fixada num valor absurdo, o mesmo conjunto de
  pares é aprovado. É o teste que trava a separação descrita em "Cotação do dólar"
- universo com menos de 3 pares válidos não aprova ninguém, e o aviso é distinguível de "nada
  passou nos filtros"
- BDR sem candidato não aparece na saída
- reprovado não aparece na saída **com dado parcial** — a asserção é sobre ausência da linha

**`src/valuation.py`**
- conversão: `justo(US$) × USD_BRL_RATE ÷ razao` com valores conhecidos
- `USD_BRL_RATE` ausente cai no default; valor inválido cai no default com aviso, como `_env_float`
- margem de segurança idêntica calculada em US$ e em R$ (invariância ao câmbio)
- premissas por balde: snapshot de linha `bdr` grava RF/ERP em dólar; linha `ação` grava em reais;
  as duas no mesmo `append_snapshot`
- default preservado: chamada sem balde continua usando as premissas em reais

**`src/filters.py`**
- `apply_bdr_filters` corta por `liq_media_diaria_bdr` e ignora `liq_media_diaria`
- `bdr_bank_filters` não aplica `dl_ebit` nem `liquidez_corrente`
- NaN reprova nos critérios exigidos, como já vale para `_estimates_mask`

## Fora de escopo

- Modelagem de tributação (retenção de 30% no dividendo, IOF, imposto sobre ganho de capital)
- BDRs cujo ativo subjacente não negocia em bolsa americana em dólar — o portão os descarta, e
  suportá-los exigiria taxa livre de risco por moeda
- Normalização do ranking entre baldes
- Reclassificação dos BDRs de empresa brasileira (`JBSS32`, `XPBR31`, `INBR32`, `ROXO34`,
  `AURA33`): passam pelo mesmo caminho dos demais, sem tratamento próprio, ainda que sua operação
  seja no Brasil
