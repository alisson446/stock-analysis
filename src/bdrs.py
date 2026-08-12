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
