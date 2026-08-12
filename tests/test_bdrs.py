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


class _BuscaFake:
    """Substitui yf.Search nos testes: devolve quotes fixos por nome."""

    def __init__(self, por_nome):
        self.por_nome = por_nome

    def __call__(self, nome, max_results=12):
        if nome not in self.por_nome:
            raise RuntimeError('sem resultado')
        return type('R', (), {'quotes': self.por_nome[nome]})()


class TestResolverSubjacente:
    """
    A busca por nome não é confiável sozinha — ela casou Fomento Económico
    Mexicano com Vista Energy. O portão da Task 9 é que torna isso utilizável;
    aqui só garantimos que o candidato escolhido é uma ação estrangeira.
    """

    def test_escolhe_a_acao_em_bolsa_estrangeira(self):
        busca = _BuscaFake({'Apple Inc.': [
            {'symbol': 'AAPL34.SA', 'exchange': 'SAO', 'quoteType': 'EQUITY'},
            {'symbol': 'AAPL', 'exchange': 'NMS', 'quoteType': 'EQUITY'},
        ]})
        assert bdrs.resolver_subjacente('Apple Inc.', buscar=busca) == 'AAPL'

    def test_ignora_o_proprio_bdr(self):
        busca = _BuscaFake({'X': [{'symbol': 'X34.SA', 'exchange': 'SAO',
                                   'quoteType': 'EQUITY'}]})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None

    def test_ignora_preferenciais_com_hifen(self):
        # WFC-PY e WFC-PC são preferenciais; a ordinária é WFC.
        busca = _BuscaFake({'Wells Fargo & Company': [
            {'symbol': 'WFC-PY', 'exchange': 'NYQ', 'quoteType': 'EQUITY'},
            {'symbol': 'WFC', 'exchange': 'NYQ', 'quoteType': 'EQUITY'},
        ]})
        assert bdrs.resolver_subjacente('Wells Fargo & Company', buscar=busca) == 'WFC'

    def test_ignora_o_que_nao_e_acao(self):
        busca = _BuscaFake({'X': [
            {'symbol': 'XOPT', 'exchange': 'NYQ', 'quoteType': 'OPTION'},
            {'symbol': 'XETF', 'exchange': 'NYQ', 'quoteType': 'ETF'},
        ]})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None

    def test_busca_que_falha_devolve_none(self):
        busca = _BuscaFake({})
        assert bdrs.resolver_subjacente('Inexistente', buscar=busca) is None

    def test_sem_candidato_devolve_none(self):
        busca = _BuscaFake({'X': []})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None
