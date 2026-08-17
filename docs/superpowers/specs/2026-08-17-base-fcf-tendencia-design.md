# Base do FCF: nível da tendência quando a série tem trajetória

Faz `compute_fcf_base` usar o nível da reta de tendência no último ano, em vez da mediana,
quando a série de FCF descreve uma trajetória. A mediana continua valendo para todo o resto.

## Problema

`compute_fcf_base` devolve a mediana da série histórica de FCF, com a justificativa de não
ancorar a projeção no pico do ciclo. A justificativa é boa para série **errática**. Para
série **monótona** ela não se aplica, e o efeito é o oposto do pretendido.

A mediana de uma série que sobe (ou desce) todo ano não resiste a pico nenhum: ela **é**, por
construção, um valor do meio da série — o nível de dois anos atrás. A SEER3, medida em
2026-08-17 (R$ mi, do mais antigo ao mais recente):

| ano | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| FCF | 38,1 | 67,7 | 116,3 | 289,0 |

Mediana = 92,0 (a média de 67,7 e 116,3, os dois do meio). O DCF pega esses 92,0, aplica o
crescimento de 5,3% e projeta o ano 1 em **R$ 96,9 mi** — quando a empresa entregou R$ 289,0
mi no exercício encerrado. A projeção começa abaixo do que já é fato consumado.

Não é conservadorismo: é **erro de nível**. O preço justo é linear na base, então a
subestimação passa inteira para o resultado. A SEER3 sai a R$ 10,98 contra preço de mercado de
R$ 11,37 — reprovada — quando a mesma fórmula, com a mesma taxa e o mesmo custo de capital,
partindo do nível atual, dá R$ 31,08.

Referência externa: a Simply Wall St avalia a SEER3 a R$ 39,01 usando FCFE 2 estágios,
`g_terminal = juro livre de risco` e CAPM — **a mesma estrutura de modelo deste repositório**.
Reproduzindo o cálculo deles a partir dos números publicados, chega-se a R$ 39,01 exatos. A
divergência inteira está na base: eles partem de R$ 454 mi (previsão de analista para 2027),
o repositório parte de R$ 92,0 mi. Trocando **só** a base pela LTM deles (R$ 327,3 mi), o
código atual produz R$ 39,06.

### O defeito erra nos dois sentidos

O caso simétrico é a CMIG4, hoje no topo do ranking com 71% de margem de segurança pelo DCF:

| ano | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| FCF | 6.252,0 | 5.382,0 | 4.577,0 | 2.999,0 |

Série caindo todo ano, mediana 4.979,5, último ano 2.999,0. Aqui a mediana fica **acima** do
nível atual e infla uma empresa em declínio: preço justo de R$ 34,56 contra R$ 22,56 pelo
nível da tendência. Esse é o erro que a Guideline 4 existe para evitar — preço justo alto
demais faz empresa ruim parecer barata.

### O módulo já sabe distinguir os dois casos, e joga a informação fora

`_compute_fcf_growth` mede o R² da reta ajustada sobre o log da série justamente para
responder *"essa série descreve uma trajetória, ou são números soltos?"*. `compute_fcf_base`
não consulta essa resposta. O módulo tem o discriminador e não o usa na decisão em que ele
mais importa.

## Decisões

### 1. Base = nível da reta no último ano, quando há trajetória

Com a série ordenada do mais antigo ao mais recente e `t = 0..n-1`, ajusta-se a mesma reta
que `_compute_fcf_growth` já ajusta sobre `log(FCF)`, e avalia-se ela em `t = n-1`:

```
base = exp(intercepto + inclinação × (n-1))
```

Não é o último valor observado, é o valor que a reta prevê para o último ano. Para a SEER3
isso dá R$ 260,4 mi contra R$ 289,0 mi observados — a reta amortece o último ponto. É
exatamente a preocupação original do docstring (não ancorar num ano de pico) atendida **sem**
herdar o erro de nível: um ano isolado de pico puxa a reta apenas em parte, enquanto a mediana
o descarta e junto descarta a trajetória inteira.

Considerada e descartada: **usar o último ano observado**. Reintroduz o problema que motivou a
mediana. A reta entrega robustez e nível correto ao mesmo tempo; o último ponto entrega só
nível.

Considerada e descartada: **média dos dois últimos anos**. Não é robusta a pico (o pico é um
dos dois) nem é derivada de premissa — "dois" seria escolhido olhando os dados (Guideline 3).

### 2. O critério de trajetória é `MIN_TREND_R2`, reaproveitado

Abaixo de `MIN_TREND_R2 = 0.5`, a base continua sendo a mediana. Nenhuma constante nova.

Isso não é economia de código, é o requisito da Guideline 3: um limiar próprio para a base
teria que ser escolhido olhando quais ações passam. O `MIN_TREND_R2` já existe com premissa
escrita — *a tendência precisa explicar mais da metade da variação da série* — e essa é
literalmente a mesma pergunta que a base precisa responder.

Medido em 2026-08-17, o gate rejeita de verdade: a BLAU3 (R² = 0,083) e a RIAA3 (R² = 0,064)
continuam na mediana.

### 3. Mínimo de 4 pontos para usar a tendência

A série do yfinance tem 4 pontos, às vezes 3. Com amostra desse tamanho o R² é fraco. Sob a
hipótese nula — série sem tendência nenhuma —, a fração de séries que passam em R² ≥ 0,5
(200.000 sorteios de ruído gaussiano):

| n | ruído puro que passa |
|---|---|
| 3 | 50,1% |
| 4 | 29,3% |
| 5 | 18,4% |

Com 3 pontos, metade das séries sem trajetória alguma são declaradas trajetória. O repositório
já reconhece essa família de defeito: `valuation-models.instructions.md` registra que com
exatamente 2 pontos o R² é sempre 1 e o gate é vazio. Esta decisão estende a mesma observação
para 3.

`n >= 4` vem da distribuição nula do R², **não** de olhar quais ações passam — logo não é
calibração contra o cache (Guideline 3). O custo é conhecido e aceito: a VTRU3 (n = 3,
R² = 0,744) continua na mediana e mantém preço justo de R$ 11,13 em vez de R$ 20,03. Erra na
direção barata (Guideline 4).

O mínimo se aplica **apenas à base**. `_compute_fcf_growth` continua aceitando 2 pontos:
mudá-lo alteraria quais ações recebem DCF, o que é outra correção com outro raio de impacto.

### 4. A exigência de logaritmo é guarda, não limitação

Não existe log de número negativo, então uma série que passou pelo prejuízo não tem reta nem
R², e cai na mediana. Das 15 ações filtradas hoje, 10 caem aqui e sobram 5 candidatas — das
quais o R² elimina 2 e o mínimo de pontos elimina 1, deixando 2 ações efetivamente afetadas.

**Isso é o gate funcionando, não falta de cobertura.** Foi testada a alternativa de ajustar a
reta em nível (sem log), que funciona com valores negativos e alcançaria as 15. O resultado
desaconselha:

- A **INTB3** — inicialmente suspeita de estar subavaliada, por ter mediana de R$ 379,1 mi
  contra último ano de R$ 819,9 mi — dá R² = 0,06 na reta em nível. A série dela é
  genuinamente errática, sem trajetória, e a mediana é a ferramenta **certa** ali.
- **JHSF3, EZTC3 e EVEN3** sairiam do DDM para o DCF. A JHSF3 iria de mediana −166,7 mi para
  uma base extrapolada de **+1.993,3 mi**, apoiada em 4 pontos e num R² de 0,73 — num teste
  que ruído puro passa em 29% das vezes.

Uma série que atravessou o prejuízo é precisamente aquela em que o nível extrapolado é menos
confiável. A restrição do log coincide com a prudência; mantém-se.

### 5. Rejeitar a inclinação e aceitar o intercepto não é contradição

Na SEER3 a mesma reta produz duas saídas com destinos opostos: a inclinação
(+93,8% ao ano) é rejeitada por `MAX_PROJECTABLE_GROWTH` e o DCF passa a usar o crescimento
forward de 5,3%; o intercepto (R$ 260,4 mi) é aceito como base.

Parece incoerente e não é. São duas afirmações separáveis:

- **Onde a empresa está hoje** — a reta responde bem, e é uma afirmação sobre o passado
  observado.
- **Se ela continua nesse ritmo** — a reta responde mal, e é uma afirmação sobre dez anos de
  futuro.

Acreditar na primeira e não na segunda é a leitura natural de uma série que quadruplicou em
quatro anos. O que seria incoerente é o contrário: projetar a taxa e desprezar o nível.

Isso precisa ficar escrito no docstring. Sem isso, a próxima pessoa lê "usa a reta para a base
e descarta a reta para a taxa" e conclui que é bug.

### 6. `fcf_base_source` no histórico

`dcf_valuation` passa a devolver `fcf_base_source` (`'trend'` ou `'median'`),
`apply_valuation` propaga a coluna, e ela entra em `_SNAPSHOT_RESULT_COLS`.

O `valuation_history.csv` grava as premissas da rodada — RF, ERP, crescimento terminal,
`growth_source` — para que uma divergência futura possa ser atribuída a mudança de dado ou de
premissa. Se a base trocar de método sem coluna equivalente, o preço justo da SEER3 salta de
R$ 10,98 para R$ 31,08 entre duas rodadas **sem nada no arquivo explicando por quê**, e o
salto fica indistinguível de uma mudança de fundamento. Isso destruiria exatamente a garantia
que essas colunas existem para dar.

O padrão já existe no módulo: `growth_source` foi criado pelo mesmo motivo. Esta é a versão
dele para a base.

## Guideline 4: por que a correção é bidirecional

A Guideline 4 manda preferir, entre dois erros possíveis, o que produz preço justo **menor**.
A correção aqui sobe a SEER3 e desce a CMIG4, então a pergunta é legítima.

Foi considerada a variante `min(mediana, tendência)`, que só usaria a tendência quando ela
baixasse o preço justo. Ela corrige a CMIG4 e deixa a SEER3 exatamente como está.

**Descartada.** A Guideline 4 arbitra escolhas de modelagem que *admitem erro nos dois
sentidos* — situações de ignorância simétrica, em que não se sabe qual lado está certo. Não é
o caso: numa série monótona a mediana mede demonstravelmente o ano errado. Usar a Guideline 4
para preservar um número sabidamente errado, em uma das direções apenas, seria aplicá-la fora
do escopo — e produziria um modelo que corrige o viés otimista e cultiva o pessimista, o que
não é conservadorismo, é ruído com sinal escolhido a dedo.

A Guideline 4 continua respeitada onde há dúvida real, e há três lugares nesta spec: o gate de
R², o mínimo de 4 pontos e a restrição do log. Todos os três, em caso de dúvida, mantêm a
mediana e deixam a ação de fora.

## Implementação

Sem dependência nova: `np.polyfit` e `np.corrcoef` já estão em uso na mesma função vizinha.

### `src/valuation.py`

**Função privada nova**, imediatamente acima de `compute_fcf_base`:

```python
def _fcf_trend_base(fcf_series: pd.Series) -> float:
    """Nível da tendência no último ano, ou NaN se a série não tem trajetória."""
```

Na ordem:

1. menos de 4 pontos → `NaN`
2. qualquer valor ≤ 0 → `NaN`
3. série constante (variação nula no log) → `NaN`, pela mesma guarda explícita de
   `_compute_fcf_growth`. Aqui a guarda é apenas higiene: numa série constante o nível da
   reta *é* o valor constante, que também é a mediana, então o resultado é o mesmo pelos dois
   caminhos. Sem a guarda, o R² seria uma divisão por zero (`NaN`), a comparação
   `r2 < MIN_TREND_R2` sairia falsa e o caminho da tendência passaria por acidente de ponto
   flutuante em vez de por decisão
4. `slope, intercept = np.polyfit(t, np.log(v), 1)`, com `t = 0..n-1` e `v` do mais antigo ao
   mais recente
5. R² = `np.corrcoef(t, np.log(v))[0, 1] ** 2`; abaixo de `MIN_TREND_R2` → `NaN`
6. `return float(np.exp(intercept + slope * (len(v) - 1)))`

**`compute_fcf_base`** passa a tentar a tendência primeiro:

```python
base = _fcf_trend_base(fcf_series)
if pd.isna(base):
    base = float(np.median(fcf_series.values))
return base if base > 0 else np.nan
```

A guarda de série vazia e a regra `base > 0 → senão NaN` ficam intactas. A assinatura não
muda, então nenhum call site quebra.

**`dcf_valuation`** registra a origem chamando o mesmo helper:

```python
result['fcf_base_source'] = (
    'trend' if pd.notna(_fcf_trend_base(fcf_series)) else 'median')
```

O `result` inicializado no topo da função ganha `'fcf_base_source': ''` junto dos outros
`NaN`. A string vazia significa **"não se chegou a escolher uma base"** — série de FCF vazia,
mediana não-positiva, ação sem contagem de ações — e é diferente de `'median'`, que significa
"escolheu-se a mediana". Sem essa distinção, o histórico registraria `'median'` para linhas
em que o DCF nem começou, e a coluna passaria a mentir sobre o que aconteceu.

O ajuste é recalculado, e isso é de propósito: são 4 pontos de numpy, e o custo é nulo perto
de manter `compute_fcf_base` com assinatura estável. Alternativa descartada: devolver uma
tupla `(base, origem)` — quebraria os três testes existentes e o call site sem ganho.

**`apply_valuation`** acumula `fcf_base_source` numa lista e grava a coluna, no mesmo padrão de
`growth_source`. Linha de banco e linha sem DCF recebem string vazia, como já acontece lá.

**`_SNAPSHOT_RESULT_COLS`** ganha `'fcf_base_source'` logo após `'growth_source'`. Sem
migração: `append_snapshot` já alinha colunas novas por `concat` e preenche o passado com
`NaN`.

### Documentação

- Docstring de `compute_fcf_base`: por que a mediana falha em série com trajetória, e a
  assimetria da decisão 5.
- `.claude/instructions/valuation-models.instructions.md` linha 32: hoje afirma "FCF base is
  the **median** of the historical series". Passa a descrever as duas vias e o critério.
- `analysis.ipynb`, célula de markdown da seção 4: hoje diz "Base = *mediana* do FCF
  histórico (não o último ano, que ancora no pico do ciclo)".

`docs/GUIDELINES.md` não muda: esta spec aplica os princípios existentes, não os revisa.

## Testes

Novos casos em `TestFcfBase`:

| caso | série (antigo → recente) | esperado |
|---|---|---|
| tendência limpa usa o nível da reta | crescimento composto exato de 20%, 4 pontos | valor da reta em `t=3`, não a mediana |
| série errática mantém a mediana | R² abaixo de 0,5 | mediana |
| 3 pontos mantêm a mediana | crescimento composto exato, 3 pontos | mediana, apesar de R² = 1 |
| ano ≤ 0 mantém a mediana | série com um ano negativo | mediana |
| tendência de queda também aplica | série caindo consistentemente | valor da reta, **abaixo** da mediana |

O último caso é o que trava a decisão bidirecional: sem ele, uma futura variante
`min(mediana, tendência)` passaria nos testes sem que ninguém notasse.

Os três testes atuais de `TestFcfBase` **continuam passando sem alteração** — as séries deles
contêm valor negativo e caem na mediana pelo caminho novo, exatamente como antes.
`TestCostOfEquity::test_dcf_emits_no_price` também: a série RSUL4 tem ano negativo.

Caso novo para a origem: `dcf_valuation` devolve `fcf_base_source == 'trend'` para série com
trajetória e `'median'` para série errática.

As séries dos testes são sintéticas e construídas a partir da regra, não copiadas do cache
(Guideline 3).

## Consequências esperadas

Medido em 2026-08-17, sobre as 15 ações filtradas, **2 preços justos mudam**:

| ação | R² (log) | base | preço justo | preço |
|---|---|---|---|---|
| SEER3 | 0,984 | 92,0 → 260,4 mi | R$ 10,98 → **R$ 31,08** | 11,37 |
| CMIG4 | 0,928 | 4.979,5 → 3.250,9 mi | R$ 34,56 → **R$ 22,56** | 10,07 |

As outras 13 ficam idênticas: 10 têm ano de FCF ≤ 0, 2 têm R² abaixo de 0,5 (BLAU3 0,083,
RIAA3 0,064) e 1 tem 3 pontos (VTRU3).

- **A SEER3 entra no ranking.** Hoje ela reprova só pelo DCF (R$ 10,98 < preço R$ 11,37); o
  Graham dela já indica R$ 12,95. Com a base corrigida ela passa nos dois.
- **A CMIG4 continua no ranking, com margem menor** — de 71,0% para cerca de 55% pelo DCF. É
  o efeito pretendido: ela seguia barata, mas menos do que o modelo dizia.
- **O alcance é pequeno de propósito.** Duas ações em quinze parece pouco para o tamanho da
  discussão; é a consequência direta de três guardas que, na dúvida, mantêm a mediana. O
  defeito corrigido é grande (3× na SEER3) mesmo sendo raro.
- **Rodadas anteriores do `valuation_history.csv` não são comparáveis com as novas** para as
  ações afetadas. A coluna `fcf_base_source` torna a quebra legível em vez de silenciosa, mas
  não a elimina: linhas antigas terão `NaN` ali.

## Fora de escopo

**O crescimento forward sobrescreve um histórico bem ajustado sem olhar a direção.** A CMIG4
tem FCF caindo 21,1% ao ano com R² = 0,93 — uma tendência de queda muito bem descrita — e o
código a descarta em favor de +4,26% (crescimento de *receita*), apenas porque
`0,0426 ≤ MAX_PROJECTABLE_GROWTH`. Com o histórico dela, o preço justo seria R$ 11,18 em vez
de R$ 34,56. É o defeito de maior impacto isolado encontrado nesta investigação e merece spec
própria: mexe no fluxo forward, não na base.

**O beta setorial é grosso demais.** A SEER3 é classificada pelo Yahoo em *Consumer Defensive*
— balde dominado por bebidas e alimentos, mediana de beta 0,74 —, enquanto o beta individual
dela é 1,48 e o da Simply Wall St é 1,047. O custo de capital sai baixo demais e o preço justo
alto demais. Corrigir exige decidir entre beta por `industria` (poucos pares por grupo) ou
outro agrupamento, e é uma spec de custo de capital.

**A interpolação do estágio 1 converge para `TERMINAL_GROWTH` nos dois sentidos.** Uma empresa
crescendo abaixo de 12,35% é projetada *acelerando* até lá — a SEER3 sai de 5,3% e chega a
12,3%, média geométrica de 8,8%. Já registrado como fora de escopo na spec de 2026-08-06 e
permanece. Nota: a Simply Wall St faz o mesmo (rampa de 7,36% a 12,18%), então não é
divergência em relação à referência.
