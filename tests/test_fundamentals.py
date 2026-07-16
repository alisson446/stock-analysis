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
