"""
Descoberta e validação dos pares BDR ↔ ativo subjacente.

O ticker do BDR não sustenta o pipeline: `earnings_estimate` volta vazio em
100% dos BDRs testados, e o `.info` mistura preço em reais com lucro na moeda
do balanço da empresa estrangeira — o que faria o P/L recalculado da Apple sair
178 em vez de 33. Então o BDR entra como preço e o subjacente entra como
empresa.
"""
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

from src import paths
from src.valuation import macro_for

# Marcador do tipo de recibo no shortName: DRN é não-patrocinado, DR1/DR2/DR3
# são os patrocinados.
#
# SEM \b na frente, de propósito. O shortName é campo de largura fixa e o
# marcador fica colado no nome quando ele ocupa a largura inteira
# ('ZILLOW GROUPDRN'); exigir a borda inicial descartaria 305 dos 625 BDRs sem
# erro nenhum. A borda final é necessária porque 'ED' (ex-dividendo) aparece
# depois do marcador em 36 papéis, o que também derruba uma âncora de fim.
MARCADOR_BDR = re.compile(r'DR[N123]\b')

_TAMANHO_PAGINA = 250


def e_bdr(short_name: str) -> bool:
    """True se o shortName do screener carrega o marcador de recibo."""
    return bool(short_name) and bool(MARCADOR_BDR.search(short_name))


def selecionar_bdrs(quotes: list[dict]) -> list[dict]:
    """
    Filtra o universo do screener para os BDRs resolvíveis.

    Exige `longName` preenchido: é ele que alimenta a busca do subjacente, e
    sem ele o papel não tem como ser ligado a empresa nenhuma.
    """
    return [q for q in quotes
            if e_bdr(q.get('shortName')) and (q.get('longName') or '').strip()]


def buscar_universo(region: str = 'br', mcap_min: int = 500_000_000,
                    delay: float = 0.3) -> list[dict]:
    """
    Pagina o `yf.screen` da região e devolve os quotes crus.

    O universo sai do lado brasileiro porque `yf.screen(region='us')` não
    enumera o mercado americano: com `total=7511` ele omite TGT, HD, LOW e MDT
    mesmo numa faixa estreita de valor de mercado.

    Falha de rede aqui é propagada: sem universo não há o que fazer, e devolver
    lista vazia seria indistinguível de "nenhum papel passou".
    """
    query = EQ('and', [EQ('eq', ['region', region]),
                       EQ('gt', ['intradaymarketcap', mcap_min])])
    quotes, offset = [], 0
    while True:
        resposta = yf.screen(query, size=_TAMANHO_PAGINA, offset=offset)
        pagina = resposta.get('quotes', [])
        if not pagina:
            break
        quotes += pagina
        offset += len(pagina)
        if offset >= (resposta.get('total') or 0):
            break
        time.sleep(delay)

    print(f"[bdrs] {len(quotes)} papéis na região {region}")
    return quotes


# A bolsa do próprio BDR. Qualquer outra praça é candidata a subjacente.
_BOLSA_DO_BDR = 'SAO'


def resolver_subjacente(long_name: str, buscar=None) -> str | None:
    """
    Encontra o ticker do ativo subjacente a partir do nome legal da empresa.

    A busca do Yahoo não é confiável sozinha: ela casou 'Fomento Económico
    Mexicano' com 'VIST' (Vista Energy). Quem torna isso utilizável é o portão
    de qualidade, que rejeita o par por duas medidas independentes da razão do
    BDR. Aqui só descartamos o que nem candidato é.

    Símbolos com '-' são preferenciais e classes especiais (WFC-PY, WFC-PC) —
    queremos a ordinária, que é a que o BDR referencia.

    Args:
        long_name: nome legal vindo do `longName` do screener.
        buscar: injetável nos testes; por padrão `yf.Search`.

    Returns:
        O ticker do subjacente, ou None quando nada serve.
    """
    buscar = buscar if buscar is not None else yf.Search
    try:
        quotes = buscar(long_name, max_results=12).quotes
    except Exception:
        return None

    for q in quotes:
        simbolo = q.get('symbol') or ''
        if (q.get('quoteType') == 'EQUITY'
                and q.get('exchange') != _BOLSA_DO_BDR
                and '-' not in simbolo):
            return simbolo
    return None


# Tolerâncias do portão. Vêm de premissa, não de ajuste à amostra: o desvio da
# cotação implícita é dominado por defasagem de cotação (o BDR tem 15 minutos
# de atraso e o subjacente é outro instante), e 3% é folga confortável para
# descasamento intradiário sem acomodar erro de identidade — o par errado
# medido deu 15%, cinco vezes o limite.
TOLERANCIA_RAZAO = 0.02
DESVIO_MAXIMO = 0.03
# Mediana de dois pares não é referência: qualquer um dos dois a define.
MIN_PARES_POR_MOEDA = 3


def razao_e_inteira(razao: float) -> bool:
    """
    True se a razão é um inteiro positivo dentro da tolerância.

    A razão do BDR é quantos recibos equivalem a uma ação — 20 para AAPL34, 120
    para MELI34. Um valor quebrado significa que o par está errado.
    """
    if razao is None or pd.isna(razao) or razao <= 0:
        return False
    return abs(razao - round(razao)) <= TOLERANCIA_RAZAO


def cotacao_implicita(razao: float, preco_bdr: float,
                      preco_subjacente: float) -> float:
    """
    A cotação de câmbio que o par implica: razão × preço do BDR ÷ preço da ação.

    Num par correto esse número é o câmbio de mercado. Num par errado é um valor
    sem sentido — e é isso que o portão detecta, sem precisar consultar o câmbio
    de verdade em lugar nenhum.
    """
    for valor in (razao, preco_bdr, preco_subjacente):
        if valor is None or pd.isna(valor):
            return np.nan
    if preco_subjacente == 0:
        return np.nan
    return razao * preco_bdr / preco_subjacente


def aprovar_pelo_portao(candidatos: list[dict]) -> tuple[list[dict], dict]:
    """
    Aprova os pares cujas duas medidas da razão do BDR concordam.

    Duas condições sobre a mediana das cotações implícitas:

    1. É por MOEDA DE PREGÃO do subjacente. Pares que implicam BRL/USD e pares
       que implicam BRL/EUR são populações diferentes, e misturá-las produziria
       uma mediana que não é cotação de nada.
    2. Entram nela apenas os pares que já passaram no teste da razão inteira,
       para que um punhado de pares errados não desloque a referência.

    Returns:
        (aprovados, descartes por motivo)
    """
    descartes = {}

    def descartar(motivo, n=1):
        descartes[motivo] = descartes.get(motivo, 0) + n

    com_razao, aprovados = [], []
    for c in candidatos:
        if razao_e_inteira(c['razao']):
            com_razao.append(c)
        else:
            descartar('razao_nao_inteira')

    por_moeda = {}
    for c in com_razao:
        por_moeda.setdefault(c['moeda_pregao'], []).append(c)

    for moeda, grupo in por_moeda.items():
        implicitas = [cotacao_implicita(c['razao'], c['preco_bdr'],
                                        c['preco_subjacente']) for c in grupo]
        validas = [x for x in implicitas if pd.notna(x)]
        if len(validas) < MIN_PARES_POR_MOEDA:
            descartar('moeda_com_poucos_pares', len(grupo))
            print(f"[bdrs] moeda de pregão {moeda}: só {len(validas)} pares "
                  f"válidos (mínimo {MIN_PARES_POR_MOEDA}) — grupo inteiro fora")
            continue

        mediana = float(np.median(validas))
        for c, implicita in zip(grupo, implicitas):
            if pd.isna(implicita) or abs(implicita / mediana - 1) > DESVIO_MAXIMO:
                descartar('cotacao_divergente')
                continue
            aprovados.append({**c, 'cotacao_implicita': implicita})

    return aprovados, descartes


def motivo_inelegibilidade(moeda_pregao: str, moeda_balanco: str, regiao: str,
                           regioes_pedidas: set) -> str | None:
    """
    Diz por que um par aprovado no portão ainda assim não entra, ou None.

    Três condições, todas eliminatórias:

    - `moeda_divergente`: pregão e balanço em moedas diferentes. Sem conversão,
      `preco ÷ LPA` só é um P/L quando as duas são a mesma. Uma comparação
      substitui uma família inteira de conversões, e a armadilha do `GBp`
      (pence, centésimo de libra) desaparece junto — `AZN.L` sai por aqui.
    - `moeda_sem_premissas`: sem juro livre de risco e prêmio de risco daquela
      moeda não existe taxa de desconto.
    - `regiao_sem_config`: o subjacente caiu numa região que não tem
      `config/<r>/filters.json`.

    A última distingue região PEDIDA de região DESCOBERTA. Se você pediu a
    região, a falta do arquivo é erro — filtrar por defaults que ninguém
    escolheu é pior que parar. Se ela apenas apareceu porque um subjacente
    resolveu para lá, derrubar a rodada inteira por causa de um papel seria
    desproporcional, e ele é apenas excluído.
    """
    if (moeda_pregao or '') != (moeda_balanco or ''):
        return 'moeda_divergente'

    if macro_for(moeda_balanco) is None:
        return 'moeda_sem_premissas'

    if not paths.filters_file(regiao).exists():
        if regiao in regioes_pedidas:
            raise FileNotFoundError(
                f"filtros da região {regiao!r} não encontrados em "
                f"{paths.filters_file(regiao)}. Crie o arquivo antes de rodar "
                f"o screener nessa região."
            )
        return 'regiao_sem_config'

    return None


# Sufixo do ticker -> região da bolsa. Ticker sem ponto é americano, que é
# onde a maioria dos subjacentes negocia, inclusive europeus e asiáticos via
# ADR (UL, TSM, UBS, YPF).
_REGIAO_POR_SUFIXO = {
    'SA': 'br', 'L': 'gb', 'DE': 'de', 'PA': 'fr', 'SW': 'ch',
    'T': 'jp', 'AX': 'au', 'TO': 'ca', 'HK': 'hk', 'MX': 'mx',
}


def regiao_do_ticker(ticker: str) -> str:
    """Região da bolsa em que o ticker negocia, pelo sufixo do símbolo."""
    if '.' not in ticker:
        return 'us'
    return _REGIAO_POR_SUFIXO.get(ticker.rsplit('.', 1)[1].upper(), 'desconhecida')


def razao_acoes_do_par(shares_bdr: float, shares_subjacente: float) -> float:
    """
    Quantos BDRs equivalem a uma ação, pelo número de ações de cada lado.

    O yfinance já entrega o `sharesOutstanding` do BDR em unidades de BDR
    (AAPL34: 291,88 bi contra 14,59 bi da AAPL), então a divisão dá a razão
    direto — 20, no caso.
    """
    for valor in (shares_bdr, shares_subjacente):
        if valor is None or pd.isna(valor) or valor == 0:
            return np.nan
    return shares_bdr / shares_subjacente


def resumo_por_regiao(candidatos: list[dict], aprovados: list[dict]) -> pd.DataFrame:
    """
    Taxa de aprovação por região de bolsa do subjacente.

    Existe porque a única medição feita até aqui restringiu os candidatos a
    bolsas americanas, e aprovou 21 de 22. Para as demais praças a taxa é
    DESCONHECIDA — imprimi-la a cada rodada é o que faz isso deixar de ser
    suposição na primeira execução real.
    """
    total = {}
    for c in candidatos:
        total[c['regiao']] = total.get(c['regiao'], 0) + 1
    ok = {}
    for a in aprovados:
        ok[a['regiao']] = ok.get(a['regiao'], 0) + 1

    linhas = [{'regiao': r, 'candidatos': n, 'aprovados': ok.get(r, 0),
               'taxa_pct': round(100 * ok.get(r, 0) / n, 1)}
              for r, n in sorted(total.items())]
    return pd.DataFrame(linhas)


def montar_frames(aprovados: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o resultado em identidade estável e cotações voláteis.

    O portão precisa do preço do BDR para validar o par, mas esse preço não pode
    ser gravado no `tickers.csv`: o arquivo fica em cache por meses e passaria a
    carregar uma cotação velha que ninguém sabe que está velha. As cotações são
    recoletadas a cada rodada e mescladas no `fundamentals.csv` por `ticker`.

    Returns:
        (pares, cotacoes) — o primeiro vai para data/<r>/tickers.csv.
    """
    if not aprovados:
        vazio_pares = pd.DataFrame(columns=['ticker', 'ticker_bdr', 'razao', 'moeda'])
        vazio_cot = pd.DataFrame(columns=['ticker', 'preco_bdr', 'liq_media_diaria_bdr'])
        return vazio_pares, vazio_cot

    df = pd.DataFrame(aprovados)
    pares = df[['ticker', 'ticker_bdr', 'razao', 'moeda']].copy()

    cotacoes = df[['ticker', 'preco_bdr']].copy()
    # Liquidez do BDR em reais: preço e volume vêm os dois do pregão de B3, e
    # é ela que diz se dá para comprar e vender o papel aqui. A liquidez do
    # subjacente em Nova York não é negociável daqui e não entra em critério.
    cotacoes['liq_media_diaria_bdr'] = df['preco_bdr'] * df['volume_bdr']

    return pares, cotacoes
