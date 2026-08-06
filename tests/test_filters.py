import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import filters


def _cfg(exigir_estimativa=False, exigir_num_analistas=False,
         receita_min=0, lucro_min=0, analistas_min=2):
    return {
        'crescimento_receita_pct_min': receita_min,
        'crescimento_lucro_pct_min': lucro_min,
        'num_analistas_min': analistas_min,
        'exigir_num_analistas': exigir_num_analistas,
        'exigir_estimativa': exigir_estimativa,
    }


def _df(rows):
    """rows: lista de (crescimento_receita_pct, crescimento_lucro_pct, num_analistas)"""
    return pd.DataFrame(
        rows,
        columns=['crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas'],
    )


class TestGrowthMaskBothFlagsOn:
    """Com as duas flags ligadas, ambos os critérios valem em conjunto."""

    def _mask(self, rows):
        return filters._growth_mask(
            _df(rows),
            _cfg(exigir_estimativa=True, exigir_num_analistas=True),
        )

    def test_growth_above_cuts_with_enough_analysts_passes(self):
        assert self._mask([(15.9, 21.9, 10)]).tolist() == [True]

    def test_negative_earnings_growth_fails(self):
        assert self._mask([(14.8, -2.4, 7)]).tolist() == [False]

    def test_negative_revenue_growth_fails_despite_positive_earnings(self):
        assert self._mask([(-4.6, 12.0, 7)]).tolist() == [False]

    def test_zero_growth_fails_strict_comparison(self):
        assert self._mask([(0.0, 5.0, 7)]).tolist() == [False]

    def test_missing_earnings_estimate_fails(self):
        assert self._mask([(9.5, np.nan, np.nan)]).tolist() == [False]

    def test_missing_revenue_estimate_fails(self):
        assert self._mask([(np.nan, 12.0, 5)]).tolist() == [False]

    def test_analysts_equal_to_minimum_passes(self):
        assert self._mask([(11.0, 30.0, 2)]).tolist() == [True]

    def test_analysts_below_minimum_fails_despite_good_growth(self):
        assert self._mask([(6.0, 12.8, 1)]).tolist() == [False]

    def test_nan_analysts_fails(self):
        assert self._mask([(11.4, 14.5, np.nan)]).tolist() == [False]


class TestGrowthMaskFlagsIndependent:
    """Cada flag liga somente o seu critério; nenhuma altera a outra."""

    def test_both_off_passes_everything(self):
        rows = [(-50.0, -80.0, 1), (np.nan, np.nan, np.nan)]
        mask = filters._growth_mask(_df(rows), _cfg())
        assert mask.tolist() == [True, True]

    def test_estimativa_off_ignores_negative_growth(self):
        mask = filters._growth_mask(
            _df([(14.8, -2.4, 7)]), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True]

    def test_analistas_off_ignores_analyst_count(self):
        mask = filters._growth_mask(
            _df([(6.0, 12.8, np.nan)]), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [True]

    def test_only_analysts_decide_when_estimativa_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._growth_mask(_df(rows), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True, False]

    def test_only_growth_decides_when_analistas_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._growth_mask(_df(rows), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [False, True]


class TestGrowthMaskThresholds:

    def test_custom_thresholds_are_respected(self):
        mask = filters._growth_mask(
            _df([(9.0, 9.0, 5), (11.0, 11.0, 5)]),
            _cfg(exigir_estimativa=True, receita_min=10, lucro_min=10),
        )
        assert mask.tolist() == [False, True]

    def test_mask_preserves_dataframe_index(self):
        df = _df([(15.0, 15.0, 5), (15.0, 15.0, 5)])
        df.index = [7, 42]
        mask = filters._growth_mask(df, _cfg(exigir_estimativa=True))
        assert mask.index.tolist() == [7, 42]
