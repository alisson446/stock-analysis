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


def _cand(ticker, razao, preco_bdr, preco_sub, moeda_pregao='USD'):
    return {'ticker_bdr': f'{ticker}34.SA', 'ticker': ticker, 'razao': razao,
            'preco_bdr': preco_bdr, 'preco_subjacente': preco_sub,
            'moeda_pregao': moeda_pregao}


class TestRazaoEInteira:
    """A razão do BDR é quantos BDRs equivalem a uma ação: sempre inteira."""

    @pytest.mark.parametrize('razao', [20.0, 1.0, 120.0, 20.02, 19.98])
    def test_aceita_inteiro_dentro_da_tolerancia(self, razao):
        assert bdrs.razao_e_inteira(razao)

    @pytest.mark.parametrize('razao', [0.615, 20.05, 19.9, 0.0, -20.0])
    def test_rejeita_fora_da_tolerancia_ou_nao_positiva(self, razao):
        assert not bdrs.razao_e_inteira(razao)

    def test_rejeita_nan(self):
        assert not bdrs.razao_e_inteira(np.nan)


class TestCotacaoImplicita:
    def test_e_a_cotacao_que_o_par_implica(self):
        # 20 BDRs a R$ 78,64 equivalem a uma ação de US$ 304,76 -> R$ 5,16/US$
        assert bdrs.cotacao_implicita(20.0, 78.64, 304.76) == pytest.approx(5.16, abs=0.01)

    def test_preco_zero_ou_ausente_devolve_nan(self):
        assert pd.isna(bdrs.cotacao_implicita(20.0, 78.64, 0))
        assert pd.isna(bdrs.cotacao_implicita(20.0, np.nan, 304.76))


class TestAprovarPeloPortao:
    """
    O portão não lê cotação nenhuma: ele compara a cotação que cada par implica
    com a mediana do próprio universo. Um par errado implica um câmbio sem
    sentido — FMXB34 casado com VIST deu 15% de desvio.
    """

    def _cinco_bons(self):
        # Todos implicam ~5,16
        return [_cand('AAPL', 20.0, 78.64, 304.76),
                _cand('MSFT', 24.0, 108.22, 503.0),
                _cand('GOGL', 12.0, 148.06, 344.3),
                _cand('JPMC', 10.0, 187.07, 362.5),
                _cand('NFLX', 50.0, 7.73, 74.9)]

    def test_aprova_pares_concordantes(self):
        aprovados, descartes = bdrs.aprovar_pelo_portao(self._cinco_bons())
        assert len(aprovados) == 5
        assert descartes == {}

    def test_rejeita_razao_nao_inteira(self):
        cands = self._cinco_bons() + [_cand('FMXB', 0.615, 100.0, 30.0)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(cands)
        assert 'FMXB' not in [a['ticker'] for a in aprovados]
        assert descartes['razao_nao_inteira'] == 1

    def test_rejeita_cotacao_implicita_fora_da_tolerancia(self):
        # razão inteira, mas implica ~5,95 contra mediana ~5,16 (15% de desvio)
        cands = self._cinco_bons() + [_cand('FMXB', 1.0, 5.95, 1.0)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(cands)
        assert 'FMXB' not in [a['ticker'] for a in aprovados]
        assert descartes['cotacao_divergente'] == 1

    def test_fronteira_da_tolerancia(self):
        base = self._cinco_bons()
        mediana = 78.64 * 20 / 304.76
        for fator, esperado in [(1.0299, True), (1.0301, False)]:
            cand = _cand('BORD', 1.0, mediana * fator, 1.0)
            aprovados, _ = bdrs.aprovar_pelo_portao(base + [cand])
            assert ('BORD' in [a['ticker'] for a in aprovados]) is esperado

    def test_mediana_e_por_moeda_de_pregao(self):
        # Um grupo em EUR com cotação bem diferente não pode deslocar o de USD.
        usd = self._cinco_bons()
        eur = [_cand(f'E{i}', 1.0, 6.5, 1.0, moeda_pregao='EUR') for i in range(3)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(usd + eur)
        assert len(aprovados) == 8
        assert descartes == {}

    def test_mediana_ignora_os_reprovados_pela_razao(self):
        # Seis pares com razão quebrada e cotação absurda não podem virar a
        # referência do grupo.
        bons = self._cinco_bons()
        ruins = [_cand(f'R{i}', 0.5, 99.0, 1.0) for i in range(6)]
        aprovados, _ = bdrs.aprovar_pelo_portao(bons + ruins)
        assert sorted(a['ticker'] for a in aprovados) == \
            sorted(c['ticker'] for c in bons)

    def test_grupo_pequeno_demais_nao_aprova_ninguem(self):
        dois = self._cinco_bons()[:2]
        aprovados, descartes = bdrs.aprovar_pelo_portao(dois)
        assert aprovados == []
        assert descartes['moeda_com_poucos_pares'] == 2

    def test_nao_le_cotacao_externa(self, monkeypatch):
        # Trava a separação: mesmo com qualquer variável de câmbio absurda no
        # ambiente, o conjunto aprovado é o mesmo.
        monkeypatch.setenv('USD_BRL_RATE', '999')
        aprovados, _ = bdrs.aprovar_pelo_portao(self._cinco_bons())
        assert len(aprovados) == 5


class TestMotivoInelegibilidade:
    """
    Sem conversão de moeda, `preco ÷ LPA` só é um P/L quando pregão e balanço
    estão na mesma moeda. E sem premissas macro não há taxa de desconto.
    """

    def test_elegivel_devolve_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        (tmp_path / 'us' / 'filters.json').write_text('{}')
        assert bdrs.motivo_inelegibilidade('USD', 'USD', 'us', {'us'}) is None

    def test_moeda_divergente_exclui(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        (tmp_path / 'us' / 'filters.json').write_text('{}')
        # UL negocia em USD e reporta em EUR; TSM em USD e reporta em TWD.
        assert bdrs.motivo_inelegibilidade('USD', 'EUR', 'us', {'us'}) == 'moeda_divergente'
        # AZN.L cota em pence com balanço em dólar — sai pela mesma condição,
        # então a armadilha do GBp nunca precisa de tratamento.
        assert bdrs.motivo_inelegibilidade('GBp', 'USD', 'gb', {'gb'}) == 'moeda_divergente'

    def test_moeda_sem_premissas_exclui(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'tw').mkdir()
        (tmp_path / 'tw' / 'filters.json').write_text('{}')
        assert bdrs.motivo_inelegibilidade('TWD', 'TWD', 'tw', {'tw'}) == 'moeda_sem_premissas'

    def test_regiao_descoberta_sem_config_exclui_so_o_papel(self, tmp_path, monkeypatch):
        # Um BDR resolvido para Londres não pode derrubar a rodada inteira.
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        assert bdrs.motivo_inelegibilidade('USD', 'USD', 'gb', {'us'}) == 'regiao_sem_config'

    def test_regiao_pedida_sem_config_levanta_erro(self, tmp_path, monkeypatch):
        # Você pediu por ela: filtrar por defaults que ninguém escolheu é pior.
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        with pytest.raises(FileNotFoundError):
            bdrs.motivo_inelegibilidade('USD', 'USD', 'us', {'us'})


class TestMontarFrames:
    """
    Os dois frames têm tempos de vida diferentes: o par é estável e vai para o
    tickers.csv; preço e liquidez são voláteis e não podem ser gravados lá,
    senão o arquivo "estável" guarda uma cotação velha que ninguém sabe que é
    velha.
    """

    def _aprovado(self):
        return [{'ticker': 'AAPL', 'ticker_bdr': 'AAPL34.SA', 'razao': 20.0,
                 'moeda': 'USD', 'preco_bdr': 78.64, 'volume_bdr': 468071,
                 'preco_subjacente': 304.76, 'moeda_pregao': 'USD'}]

    def test_pares_tem_so_as_colunas_estaveis(self):
        pares, _ = bdrs.montar_frames(self._aprovado())
        assert list(pares.columns) == ['ticker', 'ticker_bdr', 'razao', 'moeda']

    def test_cotacoes_tem_so_as_volateis(self):
        _, cotacoes = bdrs.montar_frames(self._aprovado())
        assert list(cotacoes.columns) == ['ticker', 'preco_bdr', 'liq_media_diaria_bdr']

    def test_liquidez_e_volume_vezes_preco_do_bdr(self):
        _, cotacoes = bdrs.montar_frames(self._aprovado())
        assert cotacoes['liq_media_diaria_bdr'].iloc[0] == pytest.approx(78.64 * 468071)

    def test_os_dois_frames_sao_disjuntos_fora_do_ticker(self):
        pares, cotacoes = bdrs.montar_frames(self._aprovado())
        comuns = set(pares.columns) & set(cotacoes.columns)
        assert comuns == {'ticker'}
