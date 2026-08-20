# Guidelines do projeto

Princípios que valem para todo o código, decididos ao longo das specs e que não são
dedutíveis lendo os arquivos. Quando uma decisão de implementação conflitar com um deles, o
princípio ganha — ou o princípio muda explicitamente, por escrito.

## 1. O que o screener exibe e filtra é sempre o dado bruto

As colunas de crescimento em `data/fundamentals.csv`, os filtros em `src/filters.py` e a
saída do screener carregam o valor **como veio da fonte**, por mais extremo que seja.
Uma ação com receita projetada em −4,6% aparece com −4,6%. Uma com lucro projetado em
+1410% aparece com +1410%.

**Por quê:** amortecer o número exibido esconde defeito e esconde oportunidade ao mesmo
tempo. Uma empresa em declínio forte passa a parecer estável, e um crescimento alto real
some junto com os falsos. Quem lê o screener precisa ver o dado e decidir.

## 2. Limites só dentro do valuation, e só como "não projetável"

Qualquer clamp, teto ou piso pode existir apenas dentro dos modelos de valuation
(`src/valuation.py`), nunca na coleta, na filtragem ou na exibição.

E mesmo lá, um limite só é aceitável se o seu significado for **"esta taxa não é projetável
por 10 anos"**. Nunca **"o crescimento é outro"**.

Na prática isso significa preferir, nesta ordem:

1. recuar para uma fonte alternativa (ex.: estimativa forward inviável → CAGR histórico);
2. marcar a ação como não avaliável por aquele modelo, deixando o rótulo visível na saída
   (`metodo_valuation`);
3. substituir o valor por um limite — **último recurso, e só com justificativa escrita.**

**Por quê:** substituir silenciosamente produz um preço justo que aparenta ter modelado a
empresa quando modelou outra, mais saudável. O erro escandaloso é preferível ao erro
discreto, porque o discreto chega ao usuário como recomendação.

## 3. Nunca calibrar constantes contra o cache

`data/fundamentals.csv` é um retrato de um trimestre. Os fundamentos são atualizados
periodicamente e a composição da base muda: uma empresa que hoje nem entra no cálculo pode
entrar na próxima rodada, e vice-versa.

Constantes do modelo (limiares, prêmios, horizontes) devem ser derivadas de **premissa
explícita**, não de ajuste à amostra atual. Os dados servem para **ilustrar** um mecanismo,
nunca para escolher um número.

Corolário para análises: um snapshot pode confirmar que um caminho de código **executa**.
Nunca pode provar que ele **nunca executa**. "Nenhuma ação da base cai nesse caso hoje" não
é argumento para remover uma guarda.

## 4. Errar excluindo, não incluindo

Quando uma escolha de modelagem admitir erro nos dois sentidos, prefira a que produz preço
justo **menor**.

Preço justo baixo demais faz a ação sumir da lista: você perde uma oportunidade. Preço
justo alto demais faz uma empresa ruim aparecer como barata: você compra. Num screener de
valor esses erros não têm o mesmo custo.

### O que "admitir erro nos dois sentidos" quer dizer

Este princípio arbitra entre candidatos que são, cada um por si, **defensáveis** — situações
em que genuinamente não se sabe qual está certo. Ele não é um critério para escolher o menor
número disponível.

Antes de aplicá-lo, elimine pelos méritos deles os candidatos que já se sabe que não servem:
o que mede demonstravelmente a coisa errada, o que se apoia numa medida sabidamente quebrada,
o que nem chega a ser uma afirmação sobre a empresa. A Guideline 4 arbitra o que sobrar.
Aplicá-la antes dessa eliminação preserva um número sabidamente errado só porque ele é o
menor — e isso não é conservadorismo, é ruído com o sinal escolhido a dedo.

**A trava contra o uso oportunista.** A razão para eliminar um candidato tem que valer igual
se ele fosse o número **maior**. Se o argumento "essa medida não é confiável" só aparece
quando ela puxa o preço para baixo, ele não é um argumento, é uma preferência.

*Escrito em 2026-08-18, depois de duas specs (2026-08-17 e 2026-08-18) terem apoiado decisões
numa leitura da §4 que ainda não estava no texto dela.*

## 5. Documentar em linguagem acessível

Docstrings, comentários e specs não devem pressupor formação em finanças. Conceitos como
CAGR, custo de capital, perpetuidade e prêmio de risco aparecem explicados ou com o
raciocínio à vista, não como jargão.

Comentários devem registrar **por que** a decisão foi tomada, especialmente quando ela
contraria o óbvio — o padrão que já existe no código (ver as notas sobre RSUL4, PETR4 e o
beta do yfinance em `src/valuation.py`) e que vale manter.
