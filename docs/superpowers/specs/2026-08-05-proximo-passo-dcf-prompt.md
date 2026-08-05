# Próximo passo: estágios de crescimento no DCF

Spec adiada em 2026-08-05, ao concluir o filtro de crescimento projetado no screening.
Copie o bloco abaixo e cole numa conversa nova quando quiser retomar.

---

```
/superpowers:brainstorming Quero revisar como o crescimento de analistas alimenta o estágio 1 do DCF
em src/valuation.py. Contexto do que já foi levantado e decidido, para você não refazer a investigação:

VERIFICADO NA API (yfinance 1.2.0, testado ao vivo em 12 tickers BR e US):
- A Yahoo só entrega dois períodos anuais de estimativa: `0y` (exercício corrente) e `+1y` (próximo).
  Não existem 3 anos.
- A linha `LTG` de `growth_estimates` retorna NaN para a ação em 100% dos tickers testados. Só
  `indexTrend` tem valor, e é o índice, não a empresa.
- `earnings_estimate` e `revenue_estimate` saem do mesmo módulo `earningsTrend` e ficam em cache no
  objeto Ticker: acessar os dois no mesmo `yf.Ticker` custa 1 requisição. `growth_estimates` custa 2,
  porque busca `industryTrend`/`sectorTrend`/`indexTrend` por cima — dados que o projeto não usa.

ESTADO ATUAL DO CÓDIGO:
- `get_forward_growth` (src/fundamentals.py) prioriza 'LTG' em `_FORWARD_GROWTH_PERIODS`, que nunca
  resolve, então sempre cai no '+1y'. Preferência por LTG é código morto.
- Ela usa `growth_estimates.stockTrend`, que é idêntico ao `earnings_estimate.growth` — ou seja, um DCF
  de FCF sendo alimentado por crescimento de LUCRO.
- O estágio 1 (src/valuation.py:209-210) já faz fade linear do crescimento inicial até TERMINAL_GROWTH
  ao longo de PROJECTION_YEARS = 10 anos. O fade já existe; o que se discute é o que semeia a curva.
- Hoje a curva é semeada por um único número (o '+1y') e o '0y' é descartado.
- MIN_GROWTH_RATE = 0.0 achata crescimento negativo para zero. PETR4 teve receita '+1y' em -4,61%,
  que vira 0 e semeia 10 anos de projeção.
- Depois da spec do filtro de screening, `crescimento_receita_pct`, `crescimento_lucro_pct` e
  `num_analistas` já estão em data/fundamentals.csv — então `get_forward_growth` está refazendo por
  ticker uma chamada de API que já foi feita na coleta.

DUAS PERGUNTAS EM ABERTO, que ficaram sem resposta na conversa anterior:

1. Ancoragem do estágio 1. Eu havia recomendado a opção B: usar `0y` e `+1y` como os anos 1 e 2 da
   projeção e iniciar o fade linear no ano 3 partindo do `+1y`, em vez de semear a curva inteira com um
   único número. Alternativas eram A (manter como está, só corrigir a ordem de períodos) e C (anos 1-2
   dos analistas, anos 3+ no CAGR histórico), que eu havia descartado por criar degrau no ano 3.
   Quero decidir isso.

2. Driver: o crescimento forward deve passar a vir de RECEITA (revenue_estimate), continuar em LUCRO,
   ou ser configurável? Meu argumento anterior foi que lucro é bem mais volátil que receita e o DCF é
   de FCF, então receita seria o driver mais estável.

Escopo desejado desta spec: (1) decidir as duas perguntas acima; (2) fazer `get_forward_growth` ler do
fundamentals.csv em vez de chamar a API de novo; (3) remover a prioridade morta por LTG. Comece
confirmando que o estado do código ainda bate com o descrito acima antes de propor abordagens — a spec
do filtro de screening foi implementada no meio.
```
