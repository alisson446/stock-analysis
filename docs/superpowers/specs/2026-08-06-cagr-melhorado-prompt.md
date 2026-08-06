# Próximo passo: cálculo do CAGR histórico do FCF

Item adiado em 2026-08-06, ao decidir a spec de crescimento forward no estágio 1 do DCF.
Use **depois** que essa feature estiver implementada — o prompt assume o código já alterado.

Copie o bloco abaixo e cole numa conversa nova quando quiser retomar.

---

```
/superpowers:brainstorming Quero revisar como o CAGR histórico do FCF é calculado em
`_compute_fcf_cagr` (src/valuation.py). Contexto já levantado, para você não refazer a
investigação:

O PROBLEMA
`_compute_fcf_cagr` compara apenas o PRIMEIRO e o ÚLTIMO ponto da série de FCF e ignora
todo o caminho entre eles. Com séries de 4 pontos vindas do yfinance, isso produz resumos
enganosos. Séries reais medidas em 2026-08-06 (R$ mi, do mais antigo ao mais recente):

  KEPL3:  292 → 207 → 153 → 51      CAGR -44,1%   queda monotônica, o número é CONFIÁVEL
  RIAA3:  519 → 951 → 1.087 → 351   CAGR -12,2%   sobe, sobe, despenca — número SEM SENTIDO
  BLAU3:  134 → 106 → 366 → 134     CAGR  -0,1%   oscila e volta ao início — "flat" é justo

O -12,2% da RIAA3 é o caso ruim: o cálculo não enxerga que a empresa passou por 1.087 no
meio. Duas séries com histórias completamente diferentes (declínio consistente vs.
volatilidade cíclica) podem produzir CAGRs parecidos, e o DCF as trata igual.

ESTADO DO CÓDIGO (após a spec 2026-08-06-crescimento-forward-dcf-design.md)
- `MIN_GROWTH_RATE` não existe mais: CAGR negativo alimenta o DCF sem alteração. Isso
  aumentou a importância de o CAGR ser um bom resumo — antes, tudo abaixo de zero virava
  zero e o defeito ficava escondido.
- `MAX_PROJECTABLE_GROWTH` (0.20) não é mais um teto que substitui valor: CAGR acima dele
  faz `_compute_fcf_cagr` retornar NaN e o DCF se declarar inaplicável.
- A regra "qualquer ano ≤ 0 na série zera o CAGR" (src/valuation.py:150-151) CONTINUA
  ativa. Na medição de 2026-08-06 ela zerava 9 das 14 ações filtradas — muito mais que
  qualquer outro mecanismo. Tem justificativa documentada (caso RSUL4: série 21,2M →
  -9,8M → 41,8M virava "25% a.a." na versão anterior, que pulava os negativos).
- A base do DCF é a MEDIANA da série (`compute_fcf_base`), não o último ano — escolha
  deliberada contra ancorar no pico do ciclo. Qualquer alternativa de CAGR precisa ser
  coerente com isso.

PERGUNTAS EM ABERTO
1. Que estimador substitui o cálculo pelas pontas? Regressão log-linear sobre todos os
   pontos, mediana das variações ano a ano, ou outro? O critério é resumir a TENDÊNCIA sem
   deixar um único ano dominar.
2. A série deveria carregar uma medida de dispersão junto? Uma empresa com tendência clara
   e uma volátil não deveriam alimentar o DCF com a mesma confiança, mesmo que a tendência
   central coincida. Isso pode virar um critério de projetabilidade, análogo ao que o
   MAX_PROJECTABLE_GROWTH faz hoje.
3. A regra do ano negativo deve ser revista junto? Ela e o cálculo pelas pontas tratam do
   mesmo problema — séries acidentadas — por caminhos diferentes e sem conversarem. Pelo
   argumento estrutural adotado na spec anterior, zerar o CAGR é substituir o número por
   outro, exatamente o que decidimos parar de fazer.

RESTRIÇÕES DO PROJETO
- O screener exibe e filtra crescimento BRUTO, sem amortecimento, sempre. Limites só podem
  existir dentro da projeção do DCF, e apenas com o significado "esta taxa não é projetável
  por 10 anos", nunca "o crescimento é outro". Prefira marcar a ação como não-modelável a
  substituir o valor silenciosamente.
- Os fundamentos são atualizados trimestralmente. Não calibre constante nenhuma contra
  data/fundamentals.csv: a composição da base muda, e uma empresa ausente hoje pode entrar
  amanhã. Use os dados para ILUSTRAR mecanismo, nunca para derivar valores.
- yfinance entrega tipicamente 4 pontos anuais de FCF. Qualquer método precisa funcionar
  com n pequeno.

Comece confirmando que o estado do código bate com o descrito acima.
```
