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
