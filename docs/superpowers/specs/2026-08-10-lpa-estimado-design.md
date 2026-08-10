# LPA projetado no screener e guarda de sinal no filtro de lucro

**Data:** 2026-08-10
**Escopo:** coleta, exibição, filtragem e snapshot do LPA projetado por analistas. Não toca em nenhum
modelo de valuation.

## Objetivo

Exibir no screener o **nível** de lucro por ação que os analistas projetam, ao lado da **variação**
que já é exibida, e usar esse nível como critério de filtro.

Inspiração: o painel "Earnings per Share Growth Forecasts" da Simply Wall St, que mostra o LPA
estimado por exercício em vez de só a taxa de crescimento.

## Descoberta: `crescimento_lucro_pct` é estimativa sobre estimativa

A coluna `crescimento_lucro_pct` que já existe **não** mede crescimento sobre o lucro realizado. O
campo `growth` da linha `+1y` do `earnings_estimate` compara a estimativa do próximo exercício com a
estimativa do exercício **corrente** — o `yearAgoEps` da linha `+1y` é o `avg` da linha `0y`, não o
LPA entregue.

Verificado na SEER3 (medição de 2026-08-10):

| | Valor | Origem |
|---|---|---|
| FY2025 realizado | R$ 1,93 | coluna `lpa`, calculada pelo repo |
| FY2026 estimado (`0y`) | R$ 2,17 | `earnings_estimate.loc['0y','avg']` |
| FY2027 estimado (`+1y`) | R$ 2,50 | `earnings_estimate.loc['+1y','avg']` |
| `crescimento_lucro_pct` exibido | +15,27% | = 2,49874 / 2,16767 − 1 |

Quem lê "+15,27%" ao lado de um LPA realizado de R$ 1,93 conclui naturalmente que o lucro vai de 1,93
para 2,22. Não vai: vai de 2,17 para 2,50, e nenhum desses dois números está na tela.

### Consequência: crescimento positivo em cima de prejuízo

Quando as duas estimativas são negativas, a razão fica positiva. Um prejuízo encolhendo aparece como
"crescimento de lucro". Dos 7 papéis com LPA projetado ≤ 0 no universo medido, **os 7** exibem
`crescimento_lucro_pct` positivo:

| Ticker | `lpa` realizado | `0y` est. | `+1y` est. | `growth` exibido |
|---|---|---|---|---|
| AURE3 | −1,24 | −1,25 | **−0,14** | **+88,6%** |
| HBRE3 | **+0,29** | −0,20 | **−0,03** | **+85,0%** |
| BRKM5 | −11,46 | −3,14 | **−0,84** | **+73,1%** |
| BHIA3 | −3,59 | −2,20 | **−1,29** | **+41,4%** |
| PCAR3 | −1,38 | −0,86 | **−0,54** | **+37,2%** |
| RAIZ4 | −2,62 | −0,37 | **−0,27** | **+26,3%** |
| CSNA3 | −1,51 | −1,22 | **−1,20** | **+1,9%** |

Os valores acima são de uma leitura ao vivo em 2026-08-10. A coluna `crescimento_lucro_pct` do cache
atual traz números diferentes para os mesmos papéis (AURE3 aparece com +108,81%), porque o cache é de
uma coleta anterior e as estimativas se movem entre coletas. O mecanismo é o mesmo nas duas leituras;
os números divergem só na magnitude.

Seis dos sete são barrados hoje **por acidente**: o `lpa` realizado deles é negativo e o `lpa_min: 0`
pega. Isso é propriedade do retrato atual, não garantia do modelo.

**HBRE3 é o caso vivo:** `lpa` realizado +0,29 passa no `lpa_min`, `crescimento_lucro_pct` de +85%
passa no corte de crescimento, e o consenso projeta prejuízo de −0,03 por ação. Nada no screener
barra isso hoje.

Exibir o nível ao lado da variação resolve as duas metades do problema de uma vez: torna o percentual
legível e dá ao filtro um número com sinal para checar.

## Cobertura medida

Universo: 248 papéis do cache atual com `liq_media_diaria > 100.000` — o mesmo corte de liquidez que o
filtro já aplica. Medição de 2026-08-10, uma chamada de `earnings_estimate` por ticker.

| | Tem dado | % |
|---|---|---|
| `lpa_estimado` (`+1y` `avg`) | 170/248 | 69% |
| `crescimento_lucro_pct` (`+1y` `growth`) | 169/248 | 68% |

**Os 169 com crescimento são subconjunto estrito dos 170 com LPA estimado.** Nenhum papel tem
crescimento sem ter o nível, logo **nenhum papel hoje aprovado é reprovado por falta de cobertura** do
critério novo. A diferença de um é a VALE3, que tem LPA estimado (R$ 1,60) e não tem `growth` porque
falta o `yearAgoEps` — ela continua reprovada pelo `exigir_estimativa`, sem mudança.

Essa relação de subconjunto é propriedade do retrato de hoje, não do schema da Yahoo. O critério novo
mantém "NaN reprova" justamente porque o caso inverso pode aparecer numa coleta futura.

Dispersão: 143 dos 170 têm `low ≠ high`; 27 são ponto único (tipicamente 1 analista). Não é usado
nesta spec — ver "Trabalho adiado".

### Conferência contra a Simply Wall St

TTEN3, o ticker do painel anexado à discussão:

| Exercício | SWS (S&P Global) | yfinance |
|---|---|---|
| 12/2026 (`0y`) | EPS 1,26 · 5 analistas | 1,2642 · 5 analistas |
| 12/2027 (`+1y`) | EPS 2,00 · 6 analistas | 1,8684 · 5 analistas |

Bate no exercício corrente e diverge ~7% no seguinte, com um analista de diferença no pool. É a
divergência de fonte que o próprio artigo de suporte da SWS descreve. Registrado para que uma
comparação futura não seja lida como defeito de coleta.

## Decisão de desenho

### Horizonte: `+1y`, um único exercício

A coluna carrega a média das estimativas para o **exercício seguinte ao corrente**, e só ele.

Razão: é o mesmo período que `crescimento_receita_pct` e `crescimento_lucro_pct` já usam. Uma coluna
de nível ao lado de um percentual que fala de outro ano seria pior que não ter coluna.

É também o horizonte máximo que a fonte entrega — a Yahoo expõe apenas `0q`, `+1q`, `0y` e `+1y`, e a
linha `LTG` vem NaN para a ação (já documentado na spec de 2026-08-05). A SWS mostra três exercícios
porque compra da S&P Global; isso não é replicável com yfinance.

**Limitação aceita:** com apenas o `+1y`, o denominador do percentual (`0y`) continua fora da tela. A
guarda de sinal fica inteira, a legibilidade fica parcial. A alternativa de exibir `0y` e `+1y` lado a
lado foi considerada e descartada — ver "Trabalho adiado".

### Só o `avg`, não `low`/`high`

O frame traz `low` e `high` na mesma linha. Ficam de fora: nenhum filtro e nenhum modelo os leriam, e
cada coluna nova atravessa 5 pontos de exibição, o CSV e o snapshot. Entram quando houver consumidor.

### Terceira flag independente

O critério ganha `exigir_lpa_estimado`, própria, e não é embutido no `exigir_estimativa`.

Segue o padrão que o docstring de `_growth_mask` já estabelece: cada flag liga o seu próprio critério
e nenhuma altera o significado da outra. A alternativa considerada foi embutir o corte no
`exigir_estimativa`, com o argumento de que a guarda "conserta" a leitura do `crescimento_lucro_pct` e
portanto não é critério independente. Descartada: uma flag chamada "exigir estimativa" passaria a
ligar três cortes heterogêneos — dois de variação e um de nível — e o nome deixaria de dizer o que ela
faz. Flag separada custa duas chaves de JSON e mantém cada nome honesto.

| Flag | `true` | `false` |
|---|---|---|
| `exigir_lpa_estimado` | aplica `lpa_estimado_min`; NaN reprova | o corte não é aplicado |

**Nasce ligada (`true`).** Custa quase nada em exclusão adicional, pela relação de subconjunto acima:
os papéis sem cobertura já caem pelo `exigir_estimativa: true`, que também está ligado. Na prática só
a guarda de sinal morde. A Guideline 4 (errar excluindo, não incluindo) decide o empate: deixar a
flag desligada mantém um defeito conhecido em produção.

## Componentes

### 1. Coleta — `src/fundamentals.py`

`_extract_growth_estimates` é renomeada para `_extract_analyst_estimates` e passa a devolver 4
valores. O nome antigo passaria a mentir: a função deixa de devolver apenas crescimento.

```
(crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado, num_analistas)
```

**Custo: zero requisição HTTP nova.** O frame `earnings_estimate` já é lido para extrair `growth` e
`numberOfAnalysts`. `lpa_estimado` é uma terceira leitura de célula do mesmo objeto em memória.

A extração continua passando pelo `_estimate_cell`, que já devolve NaN para frame vazio, frame sem a
linha `+1y` ou exceção. A necessidade é real: um ticker do universo (CEBR3) responde 404 no
`quoteSummary`.

### 2. Coluna nova em `data/fundamentals.csv`

| Coluna | Fonte | Unidade |
|---|---|---|
| `lpa_estimado` | `earnings_estimate.loc['+1y','avg']` | R$ por ação |

**Sem sufixo `_pct` e sem multiplicar por 100.** O `avg` já vem em moeda. O nome precisa deixar óbvio
que é nível e não variação, justamente porque a coluna vizinha é percentual.

Entra na dict do registro e na lista explícita de colunas do CSV **imediatamente após**
`crescimento_lucro_pct`, e em `numeric_cols` no notebook — mesma ordem em todos os lugares, para o CSV
e as tabelas lerem igual.

O `fundamentals.csv` é cache descartável: a primeira rodada depois desta mudança precisa de
`force_refresh=True` (ou apagar o arquivo). Rodar sobre cache antigo levanta `KeyError` no filtro, e a
correção é regenerar — mesmo procedimento das duas features anteriores que adicionaram colunas.

### 3. Configuração — `config/filters.json`

Duas chaves novas, **nos dois blocos** (`stock_filters` e `bank_filters`). O `_growth_mask` é
compartilhado pelos dois filtros, e o artefato de base negativa não é exclusivo de não-bancos.

```json
"lpa_estimado_min": 0,
"exigir_lpa_estimado": true
```

### 4. Filtragem — `src/filters.py`

Terceiro bloco em `_growth_mask`, no formato dos dois existentes:

```python
if cfg.get('exigir_lpa_estimado'):
    mask &= df['lpa_estimado'] > cfg['lpa_estimado_min']
```

Comparação estrita (`>`), seguindo a convenção dos demais `_min` do projeto e o `lpa_min` em
particular: LPA projetado exatamente zero não é lucro. NaN reprova de graça — comparação do pandas
com NaN devolve `False`.

O docstring da função passa a explicar **por que** a guarda existe, não só o que faz: que
`crescimento_lucro_pct` é estimativa sobre estimativa e fica positivo com as duas negativas, com AURE3
(+88,6% projetando −0,14/ação) como o caso concreto. Guideline 5.

### 5. Exibição — `analysis.ipynb`

`CRESCIMENTO_COLS` é a alavanca única dos 5 pontos de exibição (screening de ações, tabela de bancos,
valuation, top 20, snapshot). `lpa_estimado` entra **depois** de `crescimento_lucro_pct`, que é a
posição que torna a linha legível: "cresce 15,3% → chegando a R$ 2,50/ação".

```python
CRESCIMENTO_COLS = ['crescimento_receita_pct', 'crescimento_lucro_pct',
                    'lpa_estimado', 'num_analistas']
CRESCIMENTO_FMT  = {..., 'lpa_estimado': 'R$ {:.2f}'}
```

A célula markdown de critérios ganha a linha do critério novo. Na mesma edição, corrigir a frase
"Ambos nascem desligados" sobre `exigir_estimativa` e `exigir_num_analistas`: o
`exigir_estimativa` está `true` no config, e o texto está desatualizado desde então.

### 6. Snapshot — `src/valuation.py`

`lpa_estimado` entra em `_SNAPSHOT_RESULT_COLS`, junto das outras três colunas de estimativa.

Nada mais muda: `append_snapshot` já faz `concat` alinhando por nome, então as 277 linhas históricas
recebem NaN na coluna nova. Esse comportamento já está implementado e comentado no código.

## Efeito esperado

- **Exibição:** todas as linhas com cobertura ganham o nível ao lado da variação.
- **Filtro:** HBRE3 é reprovada, se estiver passando nos demais critérios. Os outros 6 papéis com LPA
  projetado negativo já caem hoje pelo `lpa_min`.
- **Cobertura:** nenhuma ação hoje aprovada é reprovada por ausência de dado (relação de subconjunto
  verificada acima).
- **Valuation:** inalterado. Nenhum preço justo muda por conta desta spec.

## Testes

Sem rede, com DataFrames construídos à mão, no padrão de `tests/test_filters.py`.

Sobre `_growth_mask`, com `exigir_lpa_estimado: true`:

- `lpa_estimado` positivo, demais critérios bons → passa.
- `lpa_estimado` negativo → reprova.
- `lpa_estimado` exatamente `0` → reprova (comparação estrita).
- `lpa_estimado` NaN → reprova.
- **O caso que motiva a spec:** `crescimento_lucro_pct` positivo com `lpa_estimado` negativo →
  reprova. Sem a guarda, essa linha passaria.

Independência entre as flags:

- `exigir_lpa_estimado: false` com `lpa_estimado` negativo → passa.
- `exigir_lpa_estimado: true` com `exigir_estimativa: false` → só o corte de LPA decide.
- As três flags `false` → máscara toda verdadeira.

Sobre a coleta, no padrão defensivo existente: `_extract_analyst_estimates` devolve NaN na posição do
`lpa_estimado` quando o frame é vazio, quando não tem a linha `+1y`, e quando o acesso levanta
exceção — sem interromper a coleta dos demais tickers.

Sobre o snapshot: `lpa_estimado` sobrevive a `append_snapshot` e um histórico sem a coluna ganha NaN
em vez de quebrar.

## Trabalho adiado

Fora do escopo, deliberadamente:

- **Segunda coluna com o `0y`.** Tornaria a linha inteira auditável (1,93 realizado → 2,17 → 2,50 →
  +15,27%) e fecharia a limitação de legibilidade registrada acima. Descartado nesta rodada por custo
  de superfície: mais uma coluna em 5 pontos de exibição, no CSV e no snapshot. O dado vem da mesma
  requisição, então é barato de acrescentar depois.
- **`low` e `high`** como faixa de estimativa, ao estilo do painel da SWS. Entram quando houver um
  consumidor — por exemplo, uma faixa de preço justo em vez de um ponto.
- **Graham sobre LPA projetado.** Hoje o Graham usa o `lpa` realizado. Trocar (ou acrescentar uma
  variante) muda preço justo e ranking, e por isso não pode viajar junto com uma spec que só muda
  quais ações entram — a mesma separação adotada na spec de 2026-08-05.
- **Base do FCF no DCF.** Assunto independente, levantado na análise comparativa com a SWS de
  2026-08-10: `compute_fcf_base` usa a mediana da série, o que desloca o nível no tempo e infla o
  preço justo das empresas em declínio (10 de 10 séries decrescentes medidas). Spec própria.
