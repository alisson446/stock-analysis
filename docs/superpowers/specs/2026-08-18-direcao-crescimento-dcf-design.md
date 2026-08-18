# Direção do crescimento: o histórico bem ajustado derruba o forward que o contradiz

Spec de design — 2026-08-18

Faz o estágio 1 do DCF usar o crescimento histórico do FCF, em vez da estimativa forward,
quando os dois discordam sobre a **direção** — um diz que o caixa encolhe, o outro que
cresce. Fora desse caso, nada muda.

## Problema

`dcf_valuation` (`src/valuation.py`, perto da linha 537) escolhe a taxa que semeia o estágio
1 assim:

```python
initial_growth = _compute_fcf_growth(fcf_series)   # tendência histórica do FCF
growth_source = 'historical'
if (USE_FORWARD_ESTIMATES
        and forward_growth is not None and pd.notna(forward_growth)
        and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
    initial_growth = float(forward_growth)
    growth_source = 'forward'
```

O forward substitui o histórico **incondicionalmente**, bastando existir e estar abaixo do
limiar de projetabilidade. Não há checagem de que os dois concordam, nem de quão bem ajustado
está o histórico que está sendo descartado.

### A CMIG4, medida em 2026-08-18

A série de FCF cai todo ano (R$ mi, do mais antigo ao mais recente):

| ano | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| FCF | 6.252,0 | 5.382,0 | 4.577,0 | 2.999,0 |

A reta ajustada sobre o log da série dá **−21,07% ao ano com R² = 0,928**. Em português: a
queda não é ruído, é uma trajetória que a reta descreve quase perfeitamente — 93% da variação
da série é explicada por ela. Essa taxa passa nos dois critérios de projetabilidade do próprio
módulo (o R² acima de `MIN_TREND_R2`, e a magnitude, porque `MAX_PROJECTABLE_GROWTH` não tem
piso: uma queda é dado válido).

O código descarta esse número e usa **+4,26%**, que é o crescimento de *receita* projetado
por analistas para o exercício seguinte.

Com RF = 0,1235 (o `.env` local), beta do setor Utilities = 0,5368, CoE = 16,38% e base de
FCF de R$ 3.250,9 mi:

| taxa que semeia o estágio 1 | preço justo | margem sobre R$ 10,07 |
|---|---|---|
| forward +4,26% (comportamento atual) | R$ 22,69 | **+55,6%** |
| histórico −21,07% | **R$ 7,35** | −37,1% |

A CMIG4 aparece hoje como fortemente descontada. Pelo seu próprio histórico de geração de
caixa, ela estaria **acima** do preço justo. É o erro que a Guideline 4 existe para evitar —
preço justo alto demais faz uma empresa ruim parecer barata, e isso chega ao usuário como
recomendação.

### A precedência do forward nunca foi justificada

A spec que criou o caminho forward
(`2026-08-06-crescimento-forward-dcf-design.md`) argumenta com cuidado *qual* driver usar
(receita, não lucro), *por que* o piso saiu e *por que* o teto virou critério de
projetabilidade. A ordem "forward vence, histórico é o recuo" aparece apenas como passo 2 de
uma lista de implementação, sem premissa escrita.

Não se trata, portanto, de reverter uma decisão defendida. Trata-se de escrever a premissa que
faltava.

## Decisões

### 1. Discordância de direção resolve-se pela queda

Quando o crescimento histórico do FCF é **utilizável e negativo** e o forward é **maior ou
igual a zero**, o estágio 1 usa o histórico. O forward é descartado.

Enunciada de forma simétrica, a premissa é uma só:

> **Quando as duas fontes discordam sobre a direção, o DCF projeta a queda.**

O ramo oposto — histórico positivo, forward negativo — **já se comporta assim hoje**, sem
código novo: nesse caso o forward é o número menor e ele já vence. Existe inclusive teste de
regressão para ele desde 2026-08-06 (`test_uses_negative_forward_growth_as_is`). Por isso
apenas um ramo precisa ser escrito.

Uma exceção aparente, que não é caso de direção: quando o forward é **≤ −100%**,
`resolve_forward_growth` já o devolve como `NaN` e o histórico positivo vence. Isso é a guarda
de *validade de dado* documentada na spec de 2026-08-06 — um número que deixou de funcionar
como taxa —, não uma decisão sobre direção. A regra desta spec não a altera.

#### Por que só a direção, e não a magnitude

`FORWARD_GROWTH_DRIVER` está em `revenue`. O forward é crescimento de **receita**; o histórico
é crescimento de **caixa livre**. São grandezas diferentes, e a diferença não é um detalhe:
uma empresa pode ter receita subindo 4% e caixa caindo 21% sem nenhuma contradição — basta a
margem comprimir, o capex subir, ou as duas coisas. É o retrato normal de uma elétrica em
ciclo de investimento.

Logo, uma frase como *"forward de +4,26% contra histórico de −21,07%, portanto divergem em 25
pontos"* é uma conta sem significado: está subtraindo percentuais de duas coisas distintas.
Qualquer limiar construído sobre essa subtração seria um número escolhido a dedo, que é
exatamente o que a Guideline 3 proíbe.

A **direção**, ao contrário, significa a mesma coisa nas duas medidas. "O dinheiro que sobra
para o acionista está encolhendo, com R² de 0,93" e "vai crescer" são duas afirmações sobre a
mesma empresa que não podem ser verdadeiras ao mesmo tempo — e nada na série de caixa sustenta
a reversão. É a única dimensão em que a comparação sobrevive à diferença de unidade, e é por
isso que a regra se limita a ela.

#### As Guidelines

**Guideline 2** — esta é a **opção 1** da ordem de preferência: *recuar para fonte
alternativa*. A taxa nunca vira um limite, nunca é zerada, nunca é capada. Sai sempre de uma
das duas fontes reais. Não é preciso chegar à opção 2 (declarar inavaliável) nem à 3
(substituir por limite).

**Guideline 3** — o divisor é **zero**: crescer contra encolher. É uma premissa sobre o
significado do sinal, não um ajuste à amostra. Nenhum "discordam em mais de 15 pontos".

**Guideline 4** — os dois ramos do enunciado simétrico produzem o preço justo **menor**.

#### Considerada e descartada: declarar o DCF inaplicável

A Guideline 2 lista, como opção 2, marcar a ação como não avaliável por aquele modelo com
rótulo visível. No código isso significa cair no DDM.

**Descartada porque não é conservadora aqui.** Medido: a CMIG4 paga R$ 1,28/ação de dividendo,
e o DDM é `DPS / (CoE − g)` = 1,28 / (0,163763 − 0,1235) = **R$ 31,79**. Tirar o DCF dela
faria o preço justo *subir* de R$ 22,69 para R$ 31,79 — o oposto do que a correção pretende, e
uma violação frontal da Guideline 4.

A ordem de preferência da Guideline 2 não é arbitrária: a opção 1 existe justamente para ser
tentada primeiro, e aqui ela está disponível. Recorrer à opção 2 tendo uma fonte alternativa
utilizável em mãos seria pular um degrau.

#### Considerada e descartada: usar sempre a menor das duas taxas

`min(histórico, forward)` corrigiria a CMIG4 e teria um argumento direto na Guideline 4.

**Descartada** pelo motivo do bloco anterior: `min` é uma comparação de **magnitude** entre
crescimento de receita e crescimento de caixa. Ela produziria intervenções em casos onde não
há contradição alguma — histórico de +2% contra forward de +15% viraria +2%, embora as duas
afirmações sejam perfeitamente compatíveis (uma empresa pode crescer receita a 15% e caixa a
2%). A regra de sinal é imune a isso.

A Guideline 4 não chega a ser invocada nem para um lado nem para o outro: ela arbitra entre
opções de modelagem admissíveis, e `min` de duas grandezas incomensuráveis não é uma opção de
modelagem — é uma operação aritmética sobre coisas que não se somam nem se subtraem. Um
`min(4,26%; −21,07%)` só produz um número porque o Python aceita comparar dois floats, não
porque os dois floats midam a mesma coisa.

### 2. Mínimo de 4 anos, aplicado só à sobreposição

A regra da decisão 1 exige `len(fcf_series) >= 4`.

A direção só significa alguma coisa se a reta que a produz for confiável, e o R² sozinho não
garante isso em série curta. A spec de ontem já mediu a distribuição do R² sob a hipótese nula
— série sem tendência nenhuma, 200.000 sorteios de ruído gaussiano:

| n | ruído puro que passa em R² ≥ 0,5 |
|---|---|
| 3 | 50,1% |
| 4 | 29,3% |
| 5 | 18,4% |

(Números reproduzidos de forma independente ao escrever esta spec, com outro gerador e outra
semente: 50,1% / 29,3% / 18,4%. Não são citação de memória.)

E `valuation-models.instructions.md` já registra que com exatamente **2 pontos o R² é sempre
1**, porque a reta passa pelos dois: o critério de qualidade não filtra nada. Sem um mínimo,
uma série de dois anos em que o caixa caiu por acaso derrubaria uma estimativa de analista.

**Nenhuma constante nova, nenhuma premissa nova**: é o mesmo `n >= 4` já aprovado ontem, com a
mesma derivação.

A spec de ontem aplicou esse mínimo **só à base**, com a justificativa de que mexer em
`_compute_fcf_growth` mudaria *quem recebe DCF*. Aqui a objeção não vale: o mínimo aplicado à
regra de sobreposição não tira DCF de ninguém — `_compute_fcf_growth` continua aceitando 2
pontos e continua sendo a taxa usada quando não há forward. O mínimo só decide **quem tem
autoridade para derrubar o forward**.

#### Defeito declarado, e ele é assimétrico

Uma queda real e bem descrita medida em 3 anos **não** derruba o forward, e nesse caso o preço
justo fica **mais alto** do que ficaria sem o mínimo. Isso contraria a Guideline 4 nesse caso
específico, e é aceito de propósito.

A Guideline 4 arbitra ignorância simétrica — situações em que não se sabe qual lado está
certo. Um R² de 3 pontos não é ignorância: é uma medida que já se sabe quebrada, num teste que
ruído puro passa em metade das vezes. Invocar a Guideline 4 ali seria deixar ruído decidir o
preço justo, com o sinal escolhido a dedo — o mesmo raciocínio que a spec de ontem usou para
rejeitar `min(mediana, tendência)`.

### 3. Terceiro valor em `growth_source`: `'historical_override'`

`growth_source` passa a ter três valores possíveis quando o DCF chega ao fim:

| valor | significado |
|---|---|
| `'forward'` | o forward existia, era projetável e foi usado |
| `'historical'` | não havia forward utilizável (ausente, ou acima de `MAX_PROJECTABLE_GROWTH`), ou `USE_FORWARD_ESTIMATES` está desligado |
| `'historical_override'` | havia forward projetável, mas ele contradizia a direção do histórico e foi descartado |

(Continua existindo a string vazia `''` para as linhas em que o DCF não chegou ao fim e nenhuma
taxa foi usada — comportamento atual de `apply_valuation`, inalterado.)

Sem o terceiro valor, o CSV registraria `'historical'` tanto para "não havia forward nenhum"
quanto para "havia forward projetável e ele foi derrubado". São dois eventos diferentes, e
confundi-los destrói exatamente a garantia que a coluna existe para dar: atribuir uma mudança
de preço justo a mudança de **dado** ou de **premissa**. Se a CMIG4 cair de R$ 22,69 para
R$ 7,35 entre duas rodadas, o histórico precisa dizer que foi a premissa que mudou.

É o mesmo motivo que criou `growth_source` e, ontem, `fcf_base_source`.

Considerada e descartada: **`'historical_declining'`**. Diz *por que* em vez de *o quê*. O
"declining" já está implícito na regra (a sobreposição só ocorre com histórico negativo),
enquanto "override" registra o fato que distingue este caso de `'historical'` — houve um
forward, e ele perdeu.

### 4. `MIN_TREND_R2` já é o critério de qualidade

Nenhum limiar novo de ajuste. `_compute_fcf_growth` já devolve `NaN` abaixo de
`MIN_TREND_R2`, então "histórico utilizável" já significa "bem ajustado". A regra da decisão 1
lê o resultado dessa função e não reabre a pergunta.

## Implementação

Sem dependência nova. Nenhuma assinatura pública muda, nenhum call site quebra, nenhuma
migração de CSV.

### `src/valuation.py`

**Função privada nova**, imediatamente acima de `dcf_valuation`:

```python
def _forward_contradicts_history(fcf_series: pd.Series, historical_growth: float,
                                 forward_growth: float) -> bool:
    """As duas fontes discordam sobre a DIREÇÃO do crescimento?"""
    return (len(fcf_series) >= 4
            and pd.notna(historical_growth) and historical_growth < 0
            and forward_growth >= 0)
```

A docstring carrega a premissa da decisão 1 em linguagem acessível: por que a comparação é de
sinal e não de magnitude (receita e caixa são grandezas diferentes), e por que o mínimo de 4
anos existe (o R² de série curta passa com ruído).

**`dcf_valuation`** — o bloco atual ganha um `else`:

```python
initial_growth = _compute_fcf_growth(fcf_series)
growth_source = 'historical'
if (USE_FORWARD_ESTIMATES
        and forward_growth is not None and pd.notna(forward_growth)
        and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
    if _forward_contradicts_history(fcf_series, initial_growth,
                                    float(forward_growth)):
        # initial_growth permanece o histórico
        growth_source = 'historical_override'
    else:
        initial_growth = float(forward_growth)
        growth_source = 'forward'
```

A guarda seguinte (`if pd.isna(initial_growth): return result`) fica intacta e continua
correta: a sobreposição só dispara com `historical_growth` não-NaN.

**Docstring de `dcf_valuation`** — o parágrafo "Crescimento inicial" passa a descrever a regra
de direção, e a seção `Returns:` passa a listar os três valores de `growth_source`.

`apply_valuation` e `_SNAPSHOT_RESULT_COLS` **não mudam**: a coluna já existe e já viaja, e o
valor novo é apenas mais uma string dela.

### Documentação

- `.claude/instructions/valuation-models.instructions.md`, seção "Stage 1 — Linear decay":
  descrever a regra de direção e os três valores de `growth_source`.
- `analysis.ipynb`, célula de markdown da seção 4 (linha 1040): hoje diz apenas "Crescimento
  decai linearmente da tendência histórica do FCF", sem mencionar o forward.
  **Pendência operacional:** o notebook está modificado no working tree por outro motivo e não
  pode ser commitado junto (um `git add analysis.ipynb` arrastaria ~1.900 linhas de saída de
  execução). A edição fica combinada com o autor no momento da implementação, fora deste
  commit.

`docs/GUIDELINES.md` não muda: esta spec aplica os princípios existentes, não os revisa.

## Testes

Novos casos, em `TestForwardGrowth` (`tests/test_valuation.py`):

| caso | histórico | forward | esperado |
|---|---|---|---|
| direção discorda, o histórico vence | negativo, R² alto, 4 pontos | positivo | taxa = histórico, `growth_source == 'historical_override'` |
| direções concordam na queda | negativo, 4 pontos | negativo | taxa = forward, `'forward'` |
| direções concordam na alta | positivo, 4 pontos | positivo | taxa = forward, `'forward'` (comportamento atual intacto) |
| forward negativo com histórico positivo | positivo, 4 pontos | negativo | taxa = forward, `'forward'` (o ramo simétrico) |
| série curta não tem autoridade | queda em 3 pontos | positivo | taxa = forward, `'forward'` |
| histórico inutilizável | série com ano ≤ 0 → `NaN` | positivo | taxa = forward, `'forward'` |
| forward exatamente zero | negativo, 4 pontos | `0.0` | `'historical_override'` — "estagnado" contradiz "encolhendo" |

As séries são sintéticas e construídas a partir da regra, não copiadas do cache
(Guideline 3).

O quarto caso duplica em aparência o `test_uses_negative_forward_growth_as_is` já existente,
e entra assim mesmo: aquele usa série de 3 pontos, que o mínimo da decisão 2 bloqueia de
qualquer jeito. A versão de 4 pontos é a que prova que `_forward_contradicts_history` não
dispara com o sinal invertido — ela roda com todas as outras condições satisfeitas e depende
apenas de `historical_growth < 0` estar escrito na direção certa.

**Nenhum teste existente deve quebrar, e isso foi conferido série por série** ao escrever esta
spec — não é presunção. Varridas todas as chamadas a `dcf_valuation` no arquivo de testes,
nenhuma combina uma série de 4 pontos, toda positiva e em queda com um forward não-negativo,
que é a única entrada que aciona o código novo:

| série usada (ordem do yfinance) | histórico | por que não aciona |
|---|---|---|
| `[121, 110, 100]` — `TestForwardGrowth.FCF` e `TestFcfBaseSource` | +10% | positivo **e** só 3 pontos |
| `[160, 130, 163, 100]` — `TestFcfBaseSource`, com `forward_growth=0.05` | `NaN` (R² = 0,4498) | sem histórico utilizável |
| `[172.8, 144, 120, 100]` — `TestFcfBaseSource` | +20% | positivo, e sem forward |
| `[133.1, 121, 110, 100]` — `TestDcfValuationPorMoeda` | +10% | positivo, e sem forward |
| `[1000, 100]` — `TestForwardGrowth` | `NaN` (+900%) | sem histórico utilizável |

A suíte será rodada de qualquer forma. O estado esperado é `2 failed, 304 passed` **mais** os
casos novos, com as duas falhas sendo as pré-existentes e dependentes de ambiente
(`TestCostOfEquity::test_uses_beta_from_info`, que fixa `RF = 0.124` contra o `.env` local com
`0.1235`, e `TestConfigDaRegiaoUS::test_limiares_de_barato_sao_iguais_aos_do_br`, que reflete
uma edição não commitada em `config/us/filters.json`). Confere-se pelos **nomes**, não pela
contagem.

## Consequências esperadas

Medido em 2026-08-18 sobre os 15 papéis que passam nos filtros de `br`, com RF = 0,1235.

**Um único preço justo muda:**

| ação | `growth_source` | taxa do estágio 1 | preço justo | preço |
|---|---|---|---|---|
| CMIG4 | `forward` → **`historical_override`** | +4,26% → **−21,07%** | R$ 22,69 → **R$ 7,35** | 10,07 |

A CMIG4 sai do ranking: de +55,6% de margem pelo DCF para −37,1%.

**Os outros 14 ficam idênticos, e o motivo importa.** A regra só pode disparar quando existe
crescimento histórico utilizável, e ele é raro:

| motivo pelo qual `_compute_fcf_growth` devolve `NaN` | papéis |
|---|---|
| algum ano com FCF ≤ 0 (não existe log de negativo) | 10 |
| taxa acima de `MAX_PROJECTABLE_GROWTH` (SEER3 +93,8%; VTRU3 +42,4%) | 2 |
| R² abaixo de `MIN_TREND_R2` (RIAA3 0,064; BLAU3 0,083) | 2 |
| **utilizável** (CMIG4) | **1** |

Ou seja: em **6** dos 15 papéis o forward continua rodando **sem contraprova nenhuma**. Isso é
escopo, não descuido — ver "Fora de escopo" abaixo.

Os **8 restantes** não chegam nem lá: a mediana do FCF deles é negativa (são incorporadoras e
construtoras — CYRE3 −729,7; EVEN3 −255,9; JHSF3 −166,7; EZTC3 −136,3; LAVV3 −119,8;
MDNE3 −81,7; MELK3 −73,4; TRIS3 −6,5 R$ mi), então `compute_fcf_base` devolve `NaN` e
`dcf_valuation` sai **antes** do bloco de crescimento. Eles não têm DCF nenhum e caem no DDM,
com `metodo_valuation = 'ddm'` e `growth_source` vazio. A regra desta spec nem é alcançada
por eles.

Isso foi medido na verificação de ponta a ponta, depois da implementação: a redação anterior
desta seção dizia "14 sem contraprova", tratando como equivalentes o papel que roda o forward
sem cross-check e o papel que não roda DCF nenhum. São coisas diferentes, e a diferença muda o
tamanho do item adiado logo abaixo — de 14 papéis para 6.

**A SEER3 mantém o DCF.** Era o risco principal do desenho: o histórico dela é +93,8% ao ano,
acima do limiar de projetabilidade, logo `NaN`. Sem histórico utilizável não há discordância
para arbitrar, o forward de +5,30% permanece, e ela não cai no DDM. A regra de direção não
interage com ela.

**A correção é mais branda do que a taxa sugere**, por causa da interpolação do estágio 1: os
−21,07% não permanecem em −21,07%, eles sobem linearmente até +12,35% no ano 10. A projeção
resultante da CMIG4 (R$ mi) é:

```
2.566 → 2.121 → 1.831 → 1.650 → 1.547 → 1.508 → 1.527 → 1.602 → 1.740 → 1.955
```

O modelo projeta a empresa caindo por seis anos e depois se recuperando. Os R$ 7,35 já embutem
isso. É o defeito de interpolação já registrado como fora de escopo em duas specs anteriores,
e aqui ele atenua a correção em vez de exagerá-la.

**Rodadas anteriores do `valuation_history.csv` não são comparáveis com as novas para a
CMIG4.** A coluna `growth_source` torna a quebra legível — as linhas antigas dizem `'forward'`,
as novas dirão `'historical_override'` — mas não a elimina.

**Guideline 3, corolário.** "Só a CMIG4 é afetada hoje" descreve um retrato de um trimestre.
Na próxima atualização dos fundamentos, qualquer papel cuja série de FCF fique inteiramente
positiva e bem ajustada passa a ser candidato. O alcance pequeno de hoje não é argumento para
simplificar a regra.

## Fora de escopo

**Os 6 papéis em que o forward roda sem contraprova.** Onde `_compute_fcf_growth` devolve
`NaN` não há discordância para arbitrar, e dar uma contraprova a eles exigiria um sinal de
direção que funcione sem logaritmo — reta em nível, sinal do último ano, média dos dois
últimos —, cada um com premissa própria a defender. Além disso, a spec de ontem já argumentou
(decisão 4) que uma série que atravessou o prejuízo é precisamente aquela em que extrapolar o
nível é menos confiável; o mesmo vale para extrapolar a direção. Raio de impacto diferente,
spec própria.

**A interpolação do estágio 1 converge para `TERMINAL_GROWTH` nos dois sentidos.** Uma empresa
em queda é projetada desacelerando a queda e voltando a crescer, como a tabela acima mostra.
Já registrado como fora de escopo nas specs de 2026-08-06 e 2026-08-17, e permanece.

**O beta setorial é grosso demais.** `compute_sector_betas` agrupa pelos 11 mega-setores do
Yahoo. A SEER3, uma rede de faculdades, é classificada em *Consumer Defensive* (balde de 24
papéis dominado por bebidas e alimentos, beta mediano 0,739) enquanto o beta individual dela é
1,485; pela `industria` correta, *Education & Training Services*, o balde tem 8 papéis e beta
mediano 1,455. É uma spec de custo de capital, e é a próxima.

Duas medições feitas em 2026-08-18 ao delimitar essa spec, registradas aqui para não se
perderem:

- **O problema é granularidade, não erro de rótulo.** Agrupando por `setor`, os 11 baldes têm
  mediana de 31 papéis e **nenhum** tem menos de 5. Agrupando por `industria`, os 92 baldes
  têm mediana de 2 papéis, e os 68 baldes com menos de 5 papéis contêm **133 dos 366 papéis
  com beta — 36% da base**. É esse recuo que a spec 2 precisa decidir.
- **A `industria` do yfinance está correta.** Conferidos 26 tickers conhecidos, nenhum rótulo
  errado: as quatro faculdades (ANIM3, COGN3, YDUQ3, SEER3) estão todas em *Education &
  Training Services*, CMIG4 em *Utilities - Regulated Electric*, SBSP3 em *Regulated Water*,
  TOTS3 em *Software - Application*. O código lê o **nível** errado da taxonomia, não uma
  fonte errada.

**Trocar a fonte de setor pela Classificação Setorial da B3.** Levantado e descartado para a
spec 2, por três motivos: (a) a taxonomia da B3 tem a mesma estrutura de três níveis — Setor
Econômico (~10) → Subsetor → Segmento (~90) —, logo o mesmo dilema entre balde grosso e balde
de dois papéis; (b) ela cobre só empresas listadas no Brasil, e o repo tem região `us` e BDRs,
o que exigiria manter duas taxonomias; (c) escolher a taxonomia porque ela põe a SEER3 num
balde de beta mais plausível é calibrar contra a amostra (Guideline 3) — a taxonomia tem de
ser escolhida pelos méritos dela.

O que a B3 resolveria de verdade e a spec 2 não resolve: os 4 papéis hoje **sem setor
nenhum** — VSTE3, AZEV11, BIOM11 e PSVM11, units e TPR que o Yahoo não classifica —, que caem
no beta 1,0 do fallback de `cost_of_equity`. Se depois da spec 2 isso ainda incomodar, é uma
spec de fonte de dado, não de custo de capital.

**O valor 0,20 de `MAX_PROJECTABLE_GROWTH`.** Ele é a razão pela qual o histórico da SEER3 é
descartado e o desta spec não interage com ela. Sua revisão segue registrada como item adiado
desde 2026-08-06.
