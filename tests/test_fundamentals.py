import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fundamentals as f


def _quarterly(net_income_by_label: dict, n_cols: int = 5):
    """Monta um quarterly_income_stmt no layout do yfinance (linhas=contas,
    colunas=datas, mais recente à esquerda)."""
    dates = pd.date_range('2026-03-31', periods=n_cols, freq='-3ME')
    return pd.DataFrame(net_income_by_label, index=dates).T


class TestComputeTTMNetIncome:
    """
    O trailingEps/trailingPE do Yahoo erram a contagem de ações em várias ações
    BR, então o lucro TTM dos controladores vem do .info ou da soma de 4
    trimestres — nunca do número pronto do Yahoo.
    """

    def test_prefers_net_income_to_common_from_info(self):
        # netIncomeToCommon já é TTM e dos controladores: usa direto, ignora tri.
        qi = _quarterly({'Net Income': [1, 1, 1, 1, 1]})
        assert f.compute_ttm_net_income(
            {'netIncomeToCommon': 216_292_992}, qi) == 216_292_992

    def test_sums_last_four_quarters_when_info_missing(self):
        # EVEN3: soma dos 4 trimestres mais recentes (ignora o 5º, mais antigo).
        qi = _quarterly({'Net Income Common Stockholders':
                         [32.5, 44.9, 90.0, 48.9, 53.9]})
        assert f.compute_ttm_net_income({}, qi) == pytest.approx(216.3)

    def test_prefers_common_stockholders_over_net_income(self):
        # Holding: usa o lucro dos controladores, nunca o que inclui minoritários.
        qi = _quarterly({
            'Net Income Common Stockholders': [10, 10, 10, 10, 10],
            'Net Income': [99, 99, 99, 99, 99],
        })
        assert f.compute_ttm_net_income({}, qi) == pytest.approx(40)

    def test_returns_nan_when_fewer_than_four_quarters(self):
        qi = _quarterly({'Net Income': [10, 10, 10]}, n_cols=3)
        assert np.isnan(f.compute_ttm_net_income({}, qi))

    def test_never_falls_back_to_trailing_eps(self):
        # Sem lucro confiável -> NaN, mesmo que o .info traga trailingEps.
        assert np.isnan(f.compute_ttm_net_income(
            {'trailingEps': 7.28}, pd.DataFrame()))

    def test_ignores_zero_net_income_to_common(self):
        qi = _quarterly({'Net Income': [5, 5, 5, 5, 5]})
        assert f.compute_ttm_net_income(
            {'netIncomeToCommon': 0}, qi) == pytest.approx(20)


class _FakeEstimateTicker:
    """Stub de yf.Ticker cujos frames de estimativa são controlados no teste."""

    def __init__(self, revenue=None, earnings=None, raises=False):
        self._revenue = revenue
        self._earnings = earnings
        self._raises = raises

    @property
    def revenue_estimate(self):
        if self._raises:
            raise RuntimeError('falha de rede')
        return self._revenue

    @property
    def earnings_estimate(self):
        if self._raises:
            raise RuntimeError('falha de rede')
        return self._earnings


def _revenue_frame(rows):
    """rows: {periodo: growth_decimal}"""
    return pd.DataFrame(
        {'growth': list(rows.values())},
        index=list(rows.keys()),
    )


def _earnings_frame(rows):
    """rows: {periodo: (growth_decimal, num_analistas, lpa_avg)}"""
    return pd.DataFrame(
        {
            'growth': [v[0] for v in rows.values()],
            'numberOfAnalysts': [v[1] for v in rows.values()],
            'avg': [v[2] for v in rows.values()],
        },
        index=list(rows.keys()),
    )


class TestExtractAnalystEstimates:

    def test_reads_next_year_and_converts_growth_to_percentage_points(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.0256, '+1y': 0.1693}),
            earnings=_earnings_frame({'0y': (0.0084, 11, 1.20),
                                      '+1y': (0.2057, 11, 1.45)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(16.93)
        assert lucro == pytest.approx(20.57)
        assert lpa_est == pytest.approx(1.45)
        assert analistas == 11

    def test_lpa_estimado_is_not_scaled_to_percentage_points(self):
        """É nível em R$ por ação, não variação: nada de multiplicar por 100."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.05}),
            earnings=_earnings_frame({'+1y': (0.10, 4, 2.49874)}),
        )

        _, _, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lpa_est == pytest.approx(2.49874)

    def test_reads_next_year_not_current_year(self):
        """Caso SEER3: 0y = 2,17 e +1y = 2,50. A coluna carrega o +1y."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0y': (0.1530, 4, 2.16767),
                                      '+1y': (0.1527, 4, 2.49874)}),
        )

        _, _, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lpa_est == pytest.approx(2.49874)

    def test_negative_lpa_estimado_survives_extraction(self):
        """Caso AURE3: prejuízo projetado com 'crescimento' positivo.

        O valor negativo PRECISA chegar cru ao CSV — é ele que o filtro usa
        para distinguir lucro crescendo de prejuízo encolhendo.
        """
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.08}),
            earnings=_earnings_frame({'+1y': (0.8864, 3, -0.14196)}),
        )

        _, lucro, lpa_est, _ = f._extract_analyst_estimates(stock)

        assert lucro == pytest.approx(88.64)
        assert lpa_est == pytest.approx(-0.14196)

    def test_ignores_current_year_and_quarterly_periods(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0q': 0.9, '+1q': 0.8, '0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0q': (0.9, 6, 0.30), '+1y': (0.10, 4, 1.10)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert lpa_est == pytest.approx(1.10)
        assert analistas == 4

    def test_returns_nan_when_next_year_row_missing(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.05}),
            earnings=_earnings_frame({'0y': (0.05, 3, 0.90)}),
        )

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert np.isnan(receita)
        assert np.isnan(lucro)
        assert np.isnan(lpa_est)
        assert np.isnan(analistas)

    def test_returns_nan_when_avg_column_absent(self):
        """Frame sem a coluna 'avg' não interrompe a coleta dos demais campos."""
        earnings = pd.DataFrame({'growth': [0.10], 'numberOfAnalysts': [4]},
                                index=['+1y'])
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.05}), earnings=earnings)

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert np.isnan(lpa_est)
        assert analistas == 4

    def test_returns_nan_for_empty_frames(self):
        stock = _FakeEstimateTicker(revenue=pd.DataFrame(), earnings=pd.DataFrame())

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_when_frames_are_none(self):
        stock = _FakeEstimateTicker(revenue=None, earnings=None)

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_on_exception(self):
        """Caso real: CEBR3 responde 404 no quoteSummary."""
        stock = _FakeEstimateTicker(raises=True)

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_returns_nan_for_nan_cell(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': np.nan}),
            earnings=_earnings_frame({'+1y': (np.nan, np.nan, np.nan)}),
        )

        assert all(np.isnan(v) for v in f._extract_analyst_estimates(stock))

    def test_revenue_available_without_earnings(self):
        """Caso real: CYRE4 e IGTI3 têm receita mas não lucro."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.1594}), earnings=pd.DataFrame())

        receita, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert receita == pytest.approx(15.94)
        assert np.isnan(lucro)
        assert np.isnan(lpa_est)
        assert np.isnan(analistas)

    def test_lpa_estimado_available_without_growth(self):
        """Caso real VALE3: tem 'avg' e não tem 'growth' (falta yearAgoEps)."""
        earnings = pd.DataFrame(
            {'growth': [np.nan], 'numberOfAnalysts': [12], 'avg': [1.60]},
            index=['+1y'],
        )
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.03}), earnings=earnings)

        _, lucro, lpa_est, analistas = f._extract_analyst_estimates(stock)

        assert np.isnan(lucro)
        assert lpa_est == pytest.approx(1.60)
        assert analistas == 12


class TestLeituraDoCachePorRegiao:
    """
    O cache passa a viver em data/<regiao>/fundamentals.csv, e a coluna `moeda`
    é nova. As 372 linhas gravadas antes desta mudança não a têm, então
    ausência significa BRL — senão uma rodada que só queria reaproveitar dado
    quebraria com KeyError.
    """

    def _grava(self, tmp_path, regiao, df):
        (tmp_path / regiao).mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp_path / regiao / 'fundamentals.csv', index=False)

    def test_le_o_cache_da_regiao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'us', pd.DataFrame({'ticker': ['AAPL'], 'moeda': ['USD']}))

        out = f.fetch_fundamentals([], region='us')

        assert list(out['ticker']) == ['AAPL']
        assert list(out['moeda']) == ['USD']

    def test_cache_antigo_sem_coluna_moeda_e_lido_como_brl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'br', pd.DataFrame({'ticker': ['PETR4'], 'pl': [4.2]}))

        out = f.fetch_fundamentals([], region='br')

        assert list(out['moeda']) == ['BRL']

    def test_moeda_ja_gravada_nao_e_sobrescrita(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'us', pd.DataFrame({'ticker': ['SAP'], 'moeda': ['EUR']}))

        out = f.fetch_fundamentals([], region='us')

        assert list(out['moeda']) == ['EUR']
