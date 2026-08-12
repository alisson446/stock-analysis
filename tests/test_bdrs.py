import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bdrs


class TestMarcadorBDR:
    """
    O shortName do screener é campo de largura fixa. Quando o nome da empresa
    ocupa a largura inteira, o marcador fica COLADO nele — e um \\b inicial no
    regex descartaria 305 dos 625 BDRs em silêncio. O sufixo ED (ex-dividendo)
    aparece depois do marcador em 36 papéis, então ancorar no fim também falha.
    """

    @pytest.mark.parametrize('short_name', [
        'ZOETIS INC  DRN',
        'JBS N.V.    DR2',
        'XP INC      DR1',
        'AURA 360    DR3',
    ])
    def test_aceita_marcador_separado_por_espaco(self, short_name):
        assert bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', [
        'ZILLOW GROUPDRN',
        'WHIRLPOOL CODRN',
        'ZEBRA TECHNODRN',
    ])
    def test_aceita_marcador_colado_ao_nome(self, short_name):
        # Regressão do \b inicial.
        assert bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', [
        'WELLS FARGO DRN ED',
        'UNILEVER    DRN ED',
    ])
    def test_aceita_marcador_seguido_de_ed(self, short_name):
        assert bdrs.e_bdr(short_name)

    def test_aceita_os_dois_problemas_juntos(self):
        assert bdrs.e_bdr('CONSTELLATIODRN ED')

    @pytest.mark.parametrize('short_name', [
        'YDUQS PART  ON      NM',
        'PETROBRAS   PN      N2',
        'FII ZION    CI',
        'ITAUUNIBANCOPN      N1',
    ])
    def test_rejeita_acao_brasileira_e_fii(self, short_name):
        assert not bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', ['', None])
    def test_rejeita_vazio(self, short_name):
        assert not bdrs.e_bdr(short_name)


class TestSelecionarBDRs:
    def test_exige_longname_preenchido(self):
        # O longName é o que alimenta a busca do subjacente. Sem ele o papel
        # não tem como ser resolvido, então não entra no universo.
        quotes = [
            {'symbol': 'AAPL34.SA', 'shortName': 'APPLE       DRN', 'longName': 'Apple Inc.'},
            {'symbol': 'XXXX34.SA', 'shortName': 'SEM NOME    DRN', 'longName': ''},
            {'symbol': 'YYYY34.SA', 'shortName': 'SEM NOME2   DRN'},
        ]
        assert [q['symbol'] for q in bdrs.selecionar_bdrs(quotes)] == ['AAPL34.SA']

    def test_descarta_nao_bdr(self):
        quotes = [
            {'symbol': 'PETR4.SA', 'shortName': 'PETROBRAS   PN      N2',
             'longName': 'Petróleo Brasileiro S.A.'},
            {'symbol': 'MSFT34.SA', 'shortName': 'MICROSOFT   DRN',
             'longName': 'Microsoft Corporation'},
        ]
        assert [q['symbol'] for q in bdrs.selecionar_bdrs(quotes)] == ['MSFT34.SA']
