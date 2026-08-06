import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fundamentals as f


class _FakeTicker:
    """Stub de yf.Ticker cujo growth_estimates é controlado no teste."""

    def __init__(self, est):
        self._est = est

    @property
    def growth_estimates(self):
        if isinstance(self._est, Exception):
            raise self._est
        return self._est


def _est(rows):
    """Monta o DataFrame no layout do yfinance 1.2 (index=period, cols de trend)."""
    return pd.DataFrame(rows).set_index('period')


class TestGetForwardGrowth:
    """
    Layout real (yfinance 1.2): index 0q/+1q/0y/+1y/LTG, colunas stockTrend e
    indexTrend. Só a coluna DA AÇÃO pode ser usada; a do índice, nunca.
    """

    def _patch(self, monkeypatch, est):
        monkeypatch.setattr(f.yf, 'Ticker', lambda t: _FakeTicker(est))

    def test_prefers_long_term_stock_estimate(self, monkeypatch):
        self._patch(monkeypatch, _est([
            {'period': '+1y', 'stockTrend': 0.05, 'indexTrend': 0.18},
            {'period': 'LTG', 'stockTrend': 0.12, 'indexTrend': 0.12},
        ]))
        assert f.get_forward_growth('X.SA') == pytest.approx(0.12)

    def test_falls_back_to_next_year_when_long_term_missing(self, monkeypatch):
        # LTG sem estimativa da ação (caso PETR4): usa '+1y'.
        self._patch(monkeypatch, _est([
            {'period': '+1y', 'stockTrend': -0.1262, 'indexTrend': 0.1831},
            {'period': 'LTG', 'stockTrend': np.nan, 'indexTrend': 0.1220},
        ]))
        assert f.get_forward_growth('X.SA') == pytest.approx(-0.1262)

    def test_returns_nan_and_never_uses_index_trend(self, monkeypatch):
        # Nenhuma estimativa da ação (caso VLID3): NÃO cair no indexTrend.
        self._patch(monkeypatch, _est([
            {'period': '+1y', 'stockTrend': np.nan, 'indexTrend': 0.1831},
            {'period': 'LTG', 'stockTrend': np.nan, 'indexTrend': 0.1220},
        ]))
        assert np.isnan(f.get_forward_growth('X.SA'))

    def test_converts_percentage_points_to_decimal(self, monkeypatch):
        self._patch(monkeypatch, _est([
            {'period': 'LTG', 'stockTrend': 12.0, 'indexTrend': 12.0},
        ]))
        assert f.get_forward_growth('X.SA') == pytest.approx(0.12)

    def test_returns_nan_for_empty_frame(self, monkeypatch):
        self._patch(monkeypatch, pd.DataFrame())
        assert np.isnan(f.get_forward_growth('X.SA'))

    def test_returns_nan_on_exception(self, monkeypatch):
        self._patch(monkeypatch, RuntimeError('boom'))
        assert np.isnan(f.get_forward_growth('X.SA'))


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
    """rows: {periodo: (growth_decimal, num_analistas)}"""
    return pd.DataFrame(
        {
            'growth': [v[0] for v in rows.values()],
            'numberOfAnalysts': [v[1] for v in rows.values()],
        },
        index=list(rows.keys()),
    )


class TestExtractGrowthEstimates:

    def test_reads_next_year_and_converts_to_percentage_points(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.0256, '+1y': 0.1693}),
            earnings=_earnings_frame({'0y': (0.0084, 11), '+1y': (0.2057, 11)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(16.93)
        assert lucro == pytest.approx(20.57)
        assert analistas == 11

    def test_ignores_current_year_and_quarterly_periods(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0q': 0.9, '+1q': 0.8, '0y': 0.7, '+1y': 0.05}),
            earnings=_earnings_frame({'0q': (0.9, 6), '+1y': (0.10, 4)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(5.0)
        assert lucro == pytest.approx(10.0)
        assert analistas == 4

    def test_returns_nan_when_next_year_row_missing(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'0y': 0.05}),
            earnings=_earnings_frame({'0y': (0.05, 3)}),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert np.isnan(receita)
        assert np.isnan(lucro)
        assert np.isnan(analistas)

    def test_returns_nan_for_empty_frames(self):
        stock = _FakeEstimateTicker(
            revenue=pd.DataFrame(),
            earnings=pd.DataFrame(),
        )

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_when_frames_are_none(self):
        stock = _FakeEstimateTicker(revenue=None, earnings=None)

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_on_exception(self):
        stock = _FakeEstimateTicker(raises=True)

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_returns_nan_for_nan_cell(self):
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': np.nan}),
            earnings=_earnings_frame({'+1y': (np.nan, np.nan)}),
        )

        assert all(np.isnan(v) for v in f._extract_growth_estimates(stock))

    def test_revenue_available_without_earnings(self):
        """Caso real: CYRE4 e IGTI3 têm receita mas não lucro."""
        stock = _FakeEstimateTicker(
            revenue=_revenue_frame({'+1y': 0.1594}),
            earnings=pd.DataFrame(),
        )

        receita, lucro, analistas = f._extract_growth_estimates(stock)

        assert receita == pytest.approx(15.94)
        assert np.isnan(lucro)
        assert np.isnan(analistas)
