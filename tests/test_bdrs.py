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

    def test_sufixo_sem_regiao_conhecida_exclui_so_o_papel(self, tmp_path,
                                                           monkeypatch):
        # Um ticker de LSE IOB (.IL) negocia e reporta em dólar, então chega
        # até aqui com as duas moedas iguais. O sentinela de região não é um
        # código válido de pasta: sem esta saída ele chegava em `paths` e a
        # exceção derrubava a rodada inteira por causa de um papel.
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        regiao = bdrs.regiao_do_ticker('BP.IL')

        motivo = bdrs.motivo_inelegibilidade('USD', 'USD', regiao, {'us'})

        assert motivo == 'regiao_sem_config'

    def test_papel_de_sufixo_desconhecido_nao_derruba_os_outros(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        (tmp_path / 'us' / 'filters.json').write_text('{}')
        pares = [('AAPL', 'USD'), ('BP.IL', 'USD'), ('JPM', 'USD')]

        elegiveis = [t for t, moeda in pares
                     if bdrs.motivo_inelegibilidade(
                         moeda, moeda, bdrs.regiao_do_ticker(t), {'us'}) is None]

        assert elegiveis == ['AAPL', 'JPM']


class TestRegiaoDoTicker:
    @pytest.mark.parametrize('ticker,esperado', [
        ('AAPL', 'us'), ('JPM', 'us'), ('PETR4.SA', 'br'),
        ('AZN.L', 'gb'), ('SAP.DE', 'de'), ('7203.T', 'jp'),
        ('XXXX.ZZ', bdrs.REGIAO_DESCONHECIDA),
        ('BP.IL', bdrs.REGIAO_DESCONHECIDA),
    ])
    def test_sufixo_decide_a_regiao(self, ticker, esperado):
        assert bdrs.regiao_do_ticker(ticker) == esperado


class TestRazaoAcoesDoPar:
    def test_divide_as_contagens(self):
        assert bdrs.razao_acoes_do_par(291_883_600_000, 14_594_180_000) == pytest.approx(20.0)

    @pytest.mark.parametrize('a,b', [(None, 10), (10, None), (10, 0), (np.nan, 10)])
    def test_faltando_um_lado_devolve_nan(self, a, b):
        assert pd.isna(bdrs.razao_acoes_do_par(a, b))


class TestResumoPorRegiao:
    """A taxa medida valeu só para bolsas americanas; o resumo é o que revela
    a das demais praças na primeira rodada real."""

    def test_conta_candidatos_e_aprovados_por_regiao(self):
        candidatos = [{'regiao': 'us'}, {'regiao': 'us'}, {'regiao': 'gb'}]
        aprovados = [{'regiao': 'us'}]

        resumo = bdrs.resumo_por_regiao(candidatos, aprovados)

        por_regiao = resumo.set_index('regiao').to_dict('index')
        assert por_regiao['us'] == {'candidatos': 2, 'aprovados': 1, 'taxa_pct': 50.0}
        assert por_regiao['gb'] == {'candidatos': 1, 'aprovados': 0, 'taxa_pct': 0.0}


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

    def _dois_recibos_da_alphabet(self):
        # GOGL34 e GOGL35 compartilham o longName 'Alphabet Inc.', resolvem
        # para o mesmo subjacente e podem passar os dois no portão.
        base = {'ticker': 'GOOGL', 'razao': 12.0, 'moeda': 'USD',
                'preco_subjacente': 200.0, 'moeda_pregao': 'USD'}
        return [{**base, 'ticker_bdr': 'GOGL34.SA', 'preco_bdr': 80.0,
                 'volume_bdr': 100_000},
                {**base, 'ticker_bdr': 'GOGL35.SA', 'preco_bdr': 81.0,
                 'volume_bdr': 5_000}]

    def test_subjacente_repetido_sai_uma_vez_so(self):
        # Quem consome junta os frames por `ticker`: dois pares da mesma
        # empresa viram quatro linhas, cada preço casado com o recibo errado,
        # e o ranking conta a empresa duas vezes.
        pares, cotacoes = bdrs.montar_frames(self._dois_recibos_da_alphabet())

        assert list(pares['ticker']) == ['GOOGL']
        assert list(cotacoes['ticker']) == ['GOOGL']

    def test_par_mantido_fica_coerente_entre_os_dois_frames(self):
        pares, cotacoes = bdrs.montar_frames(self._dois_recibos_da_alphabet())

        assert pares['ticker_bdr'].iloc[0] == 'GOGL34.SA'
        assert cotacoes['preco_bdr'].iloc[0] == pytest.approx(80.0)
        assert cotacoes['liq_media_diaria_bdr'].iloc[0] == pytest.approx(
            80.0 * 100_000)

    def test_colisao_descartada_e_impressa(self, capsys):
        # Jogar fora um mapeamento em silêncio é pior que dizer qual foi.
        bdrs.montar_frames(self._dois_recibos_da_alphabet())

        saida = capsys.readouterr().out
        assert 'GOOGL' in saida
        assert 'GOGL35.SA' in saida

    def test_subjacentes_distintos_nao_sao_afetados(self):
        aprovados = self._aprovado() + self._dois_recibos_da_alphabet()

        pares, _ = bdrs.montar_frames(aprovados)

        assert list(pares['ticker']) == ['AAPL', 'GOOGL']
