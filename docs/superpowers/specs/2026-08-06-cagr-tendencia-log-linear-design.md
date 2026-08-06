# Crescimento histórico do FCF: tendência log-linear com critério de ajuste

Substitui o cálculo pelas pontas em `_compute_fcf_cagr` por uma regressão sobre toda a
série, e transforma a qualidade do ajuste num critério de projetabilidade.

## Problema

`_compute_fcf_cagr` compara apenas o primeiro e o último ponto da série de FCF e ignora
todo o caminho entre eles. Com as séries de 4 pontos que o yfinance entrega, isso produz
resumos enganosos. Três séries reais, medidas em 2026-08-06 (R$ mi, do mais antigo ao mais
recente):

| ação | série | cálculo pelas pontas |
|---|---|---|
| KEPL3 | 292 → 207 → 153 → 51 | −44,1% |
| RIAA3 | 519 → 951 → 1.087 → 351 | −12,2% |
| BLAU3 | 134 → 106 → 366 → 134 | −0,1% |

O −44,1% da KEPL3 é confiável: a queda é monotônica e o número descreve o que aconteceu
todo ano. O −12,2% da RIAA3 não descreve nada — a empresa subiu até 1.087 antes de
despencar, e o cálculo não enxerga isso. Duas séries com histórias completamente
diferentes (declínio consistente e volatilidade cíclica) produzem números da mesma ordem,
e o DCF trata as duas igual.

Duas mudanças recentes aumentaram o custo desse defeito:

- **Sem piso.** `MIN_GROWTH_RATE` deixou de existir: um crescimento negativo alimenta o
  DCF sem alteração. Antes, tudo abaixo de zero virava zero e o defeito ficava escondido
  atrás do piso.
- **Base na mediana.** `compute_fcf_base` usa a mediana da série, deliberadamente, para
  não ancorar no pico do ciclo. O crescimento continuava ancorado em dois pontos
  arbitrários — a base já era robusta, a taxa não.

## Decisões

### 1. Estimador: regressão log-linear sobre todos os pontos

Ajusta-se uma reta sobre o logaritmo dos FCFs. A inclinação dessa reta é uma taxa de
crescimento composta que leva em conta todos os anos, não só as pontas.

O logaritmo entra porque crescimento é multiplicativo: crescer 10% ao ano vira uma reta
perfeita no log, e a inclinação lida direto como taxa (`exp(inclinação) − 1`).

Considerada e descartada: **mediana das variações ano a ano**. Com 4 pontos existem 3
variações, e a mediana de três números *é* um deles. O estimador que deveria impedir um
ano de dominar passaria a escolher um ano e ignorar os outros dois.

### 2. O estimador sozinho não resolve o problema

Aplicando a regressão às três séries:

| ação | pontas | regressão | R² |
|---|---|---|---|
| KEPL3 | −44,1% | −42,5% | 0,90 |
| RIAA3 | −12,2% | −9,9% | 0,06 |
| BLAU3 | −0,1% | **+13,2%** | 0,08 |

A RIAA3 continua com um número sem sentido. E a BLAU3 **piora**: uma série que termina onde
começou vira +13,2% ao ano, porque o pico está no penúltimo ponto e puxa a reta.

Isso não é falha do estimador. O defeito da RIAA3 nunca foi o *valor* da tendência — é que
**não existe tendência ali**. Nenhuma medida de tendência central pode consertar uma série
que não tem tendência; o que ela pode fazer é dizer que não tem.

### 3. Qualidade do ajuste como critério de projetabilidade

A regressão entrega de graça o R²: a fração da variação da série que a reta explica. Lido
sem jargão, ele responde *"essa série conta uma história, ou são quatro números soltos?"*.
Na KEPL3 a reta **é** a série (0,90); na RIAA3 e na BLAU3 o que sobra em torno da reta é
ruído (0,06 e 0,08).

Abaixo de `MIN_TREND_R2`, `_compute_fcf_growth` devolve `NaN`. O mecanismo é o mesmo do
`MAX_PROJECTABLE_GROWTH`: a taxa não é substituída, a ação é declarada não-modelável por
DCF, o chamador recai no DDM e `metodo_valuation` registra a troca. É a Guideline 2
aplicada a uma segunda dimensão — e a RIAA3 sai pelo motivo certo, não porque −9,9% seja
muito ou pouco, mas porque não há trajetória para projetar.

São duas constantes porque são duas perguntas independentes:

- `MAX_PROJECTABLE_GROWTH` — "essa taxa se sustenta por 10 anos?"
- `MIN_TREND_R2` — "essa série descreve uma trajetória?"

### 4. O limiar é 0,5, por premissa

`MIN_TREND_R2 = 0.5`, lido literalmente: **a tendência precisa explicar mais da metade da
variação da série; abaixo disso o que a série tem é mais ruído do que trajetória.**

"Mais da metade" é uma afirmação sobre o que significa haver tendência. Qualquer 0,45 ou
0,6 só seria justificável olhando quais ações passam, que é exatamente o que a Guideline 3
proíbe.

**Defeito declarado:** o R² penaliza a série chata. Uma empresa com FCF 100 → 101 → 100 →
101 tem crescimento praticamente nulo e é a coisa mais previsível possível, mas quase toda
a sua variação é ruído — porque quase não há variação — e o R² fica baixo. Ela vai pro
DDM.

A alternativa seria um segundo limiar que liberasse séries de dispersão absoluta pequena,
mas "pequena" quanto é um número que só se escolhe olhando os dados (Guideline 3). Aceita-se
o falso negativo: ele erra na direção barata (Guideline 4), tirando a ação da lista em vez
de fazê-la parecer barata.

O caso extremo — série perfeitamente constante — recebe o mesmo tratamento, e por mecânica:
com variação nula no log, o R² é uma divisão por zero. A função devolve `NaN` por uma
guarda explícita, não por acidente de ponto flutuante.

### 5. A regra do ano negativo devolve `NaN`, não `0.0`

A regra "qualquer ano ≤ 0 na série zera o crescimento" cai por mecânica: não existe
logaritmo de número negativo. Mas o que a substitui não é indiferente, e aqui o argumento
inverte a leitura da regra atual.

`0.0` é lido por todo mundo como "conservador: assumimos crescimento nulo". **Não é o que
acontece.** O estágio 1 do DCF interpola linearmente de `growth` até `TERMINAL_GROWTH`, que
é 12,4%. Com `growth = 0`, a projeção não fica parada em zero — ela *sobe* de 0% a 12,4% ao
longo dos dez anos. Uma empresa que passou por prejuízo no meio da série é hoje projetada
**acelerando** até 12,4% ao ano.

Ou seja: a regra que existe para punir séries acidentadas é a que mais infla preço justo no
modelo. Viola a Guideline 4 na direção cara e é o exemplo mais puro do "substituir o número
por outro" que a spec anterior decidiu abandonar.

A série com menos de 2 pontos recebe o mesmo tratamento, pelo mesmo motivo: hoje também
devolve `0.0`.

A justificativa original da regra (o caso RSUL4: 21,2M → −9,8M → 41,8M virava "25% a.a." na
versão que pulava os negativos) permanece válida e permanece atendida — `NaN` rejeita a
série de forma mais forte que `0.0`, não mais fraca.

**`0.0` deixa de ser um valor de retorno possível.** Nenhum caminho da função substitui mais
o número por outro: ou sai uma taxa medida, ou sai "não modelável".

### 6. O nome da função muda junto

`_compute_fcf_cagr` → `_compute_fcf_growth`. O resultado deixa de ser um CAGR (taxa entre
duas pontas) e passa a ser a inclinação de uma tendência. Manter o nome antigo faria a
próxima pessoa procurar por uma fórmula que não está mais lá.

## O que o screener exibe e filtra: inalterado

Nada aqui toca coleta, filtros ou exibição. As colunas de crescimento em
`data/fundamentals.csv` continuam carregando o dado bruto da fonte, e `MIN_TREND_R2` existe
apenas dentro de `src/valuation.py`, com o significado "não projetável" (Guidelines 1 e 2).

## Implementação

Sem dependência nova: `numpy` já está no `requirements.txt` e cobre regressão (`np.polyfit`)
e R² (`np.corrcoef`). Nada de scipy.

### `src/valuation.py`

**Constante nova**, junto de `MAX_PROJECTABLE_GROWTH`, com comentário no mesmo espírito
(limiar de projetabilidade, não de crescimento):

```python
MIN_TREND_R2 = 0.5
```

**`_compute_fcf_cagr` → `_compute_fcf_growth`**, na ordem:

1. menos de 2 pontos → `NaN`
2. qualquer valor ≤ 0 → `NaN`
3. série constante (variação nula no log) → `NaN`, por guarda explícita
4. `slope = np.polyfit(t, np.log(v), 1)[0]`, com `t = 0..n-1` e `v` do mais antigo ao mais
   recente
5. `growth = np.exp(slope) - 1`
6. R² = `np.corrcoef(t, np.log(v))[0, 1] ** 2`; abaixo de `MIN_TREND_R2` → `NaN`
7. acima de `MAX_PROJECTABLE_GROWTH` → `NaN` (regra atual, preservada)

Retorno: `NaN` (não modelável) ou um float que pode ser negativo.

A docstring registra, em linguagem acessível (Guideline 5): por que o logaritmo, o que o R²
mede, e que a série estável é rejeitada de propósito.

**Call site:** `dcf_valuation` só muda o nome chamado. O tratamento de `NaN` já existe e já
faz a coisa certa — sair sem preço para o chamador recair no DDM.

### O resto do módulo não muda

`compute_fcf_base` continua na mediana. `apply_valuation` já rotula o fallback em
`metodo_valuation`. O fluxo de crescimento forward não é tocado. A mudança inteira cabe numa
função e numa constante, porque a spec anterior já desenhou o módulo para isso.

### Documentação

`.claude/instructions.md` e `.claude/instructions/valuation-models.instructions.md` listam as
constantes e descrevem o cálculo pelas pontas; `analysis.ipynb` também menciona. Atualizar as
três.

`docs/GUIDELINES.md` não muda: esta spec aplica os princípios existentes, não os revisa.

## Testes

Reescrever `TestComputeFcfCagr` → `TestComputeFcfGrowth`, um caso por decisão:

| caso | série | esperado |
|---|---|---|
| tendência limpa | crescimento composto exato de 10% | `0.10` |
| declínio consistente | KEPL3 | negativo, passa |
| ciclo | RIAA3 | `NaN` |
| volta ao início | BLAU3 | `NaN` |
| ano ≤ 0 | RSUL4 | `NaN` (era `0.0`) |
| ponto único | `[100]` | `NaN` (era `0.0`) |
| série constante | `[100] * 4` | `NaN` |
| acima do teto | `100 → 1000` | `NaN` (inalterado) |

As séries reais entram como **ilustração de mecanismo**: cada uma nomeia um comportamento
distinto que o código precisa distinguir. Nenhum limiar é derivado delas (Guideline 3).

**`TestRsul4Regression` precisa ser reescrito.** Hoje ele passa porque `0.0` alimenta um DCF
que produz preço abaixo do mercado, e a asserção é `fv < 47.36`. Com `NaN`,
`discount_fcf_to_equity` devolve `NaN` e a comparação é sempre falsa. A nova asserção é mais
forte e mais fiel à intenção original: a RSUL4 não é modelável por DCF, e `apply_valuation`
marca `metodo_valuation` como `ddm` ou `none` em vez de emitir um preço justo de R$ 309,53.

## Consequências esperadas

- **O DCF se aplica a menos ações.** Três filtros somados (ajuste ruim, ano negativo, taxa
  acima do teto) rejeitam mais do que a regra única de hoje. Na medição de 2026-08-06, só a
  regra do ano negativo atingia 9 das 14 ações filtradas — todas elas passam a cair no DDM
  em vez de receber um DCF com crescimento acelerando até 12,4%.
- **Quem cai no DDM precisa de dividendo.** Sem `dividend_rate`, `metodo_valuation` fica
  `none` e a ação sai sem preço justo primário. O screener fica visivelmente mais vazio —
  isso é a mudança, não um efeito colateral dela.
- **Para quem perde o DCF, o preço justo só cai**: some junto a projeção acelerando até
  12,4%. Para quem mantém o DCF, a taxa pode subir ou descer — a regressão não é
  sistematicamente menor que o cálculo pelas pontas, ela é apenas o resumo de todos os
  pontos em vez de dois.
- **Uma empresa estável pode ser rejeitada** pelo R², como descrito na decisão 4. Erro
  conhecido, na direção barata.

## Fora de escopo

**A interpolação do estágio 1 converge para `TERMINAL_GROWTH` nos dois sentidos.** A KEPL3,
a −42,5%, é projetada convergindo para +12,4% de crescimento ao longo de dez anos. É a mesma
mecânica que torna o `0.0` de hoje generoso em vez de conservador, e ela merece uma revisão
própria — mas é uma decisão do DCF, não do estimador de crescimento, e fica para outra spec.
