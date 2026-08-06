import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import valuation as v


class TestResolveShareCount:
    """
    yfinance retorna sharesOutstanding apenas da classe cotada (ex.: PN para
    RSUL4 = 2.495.225), enquanto o equity value do DCF é da empresa inteira.
    O total (ON + PN) vem de impliedSharesOutstanding / Ordinary Shares Number.
    """

    def test_prefers_implied_shares_over_shares_outstanding(self):
        info = {'sharesOutstanding': 2_495_225, 'impliedSharesOutstanding': 6_072_128}
        assert v.resolve_share_count(info) == 6_072_128

    def test_falls_back_to_shares_outstanding_when_implied_missing(self):
        info = {'sharesOutstanding': 1_000_000}
        assert v.resolve_share_count(info) == 1_000_000

    def test_falls_back_to_shares_outstanding_when_implied_is_zero(self):
        info = {'sharesOutstanding': 1_000_000, 'impliedSharesOutstanding': 0}
        assert v.resolve_share_count(info) == 1_000_000

    def test_uses_balance_sheet_shares_when_info_has_none(self):
        info = {}
        assert v.resolve_share_count(info, balance_shares=6_072_128) == 6_072_128

    def test_returns_nan_when_no_source_available(self):
        assert np.isnan(v.resolve_share_count({}))


class TestComputeFcfCagr:
    """
    A versão antiga pulava anos negativos: para RSUL4 media 2022 (21,2M) ->
    2025 (41,8M) e ignorava 2024 (-9,8M), chegando a 25,4% (capado em 20%).
    """

    def test_returns_zero_when_series_contains_negative_year(self):
        # RSUL4: mais recente primeiro
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert v._compute_fcf_cagr(serie) == 0.0

    def test_computes_cagr_from_first_and_last_year_when_all_positive(self):
        # 100 -> 121 em 2 anos = 10% a.a.
        serie = pd.Series([121.0, 110.0, 100.0])
        assert v._compute_fcf_cagr(serie) == pytest.approx(0.10, abs=1e-6)

    def test_caps_growth_at_max_rate(self):
        serie = pd.Series([1000.0, 100.0])
        assert v._compute_fcf_cagr(serie) == v.MAX_GROWTH_RATE

    def test_floors_growth_at_min_rate_when_declining(self):
        serie = pd.Series([50.0, 100.0])
        assert v._compute_fcf_cagr(serie) == v.MIN_GROWTH_RATE

    def test_returns_zero_for_single_year(self):
        assert v._compute_fcf_cagr(pd.Series([100.0])) == 0.0


class TestFcfBase:
    """Base do DCF passa a ser a mediana, para não ancorar em ano de pico."""

    def test_uses_median_not_latest_year(self):
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert v.compute_fcf_base(serie) == pytest.approx(28.3055e6, rel=1e-4)

    def test_returns_nan_when_median_is_not_positive(self):
        serie = pd.Series([10.0, -20.0, -30.0])
        assert np.isnan(v.compute_fcf_base(serie))

    def test_returns_nan_for_empty_series(self):
        assert np.isnan(v.compute_fcf_base(pd.Series(dtype=float)))


class TestCostOfEquity:
    """Custo de capital próprio = RF + beta x ERP (estilo Simply Wall St)."""

    def test_uses_beta_from_info(self):
        # RSUL4: 0.124 + 1.09 * 0.075 = 0.20575
        assert v.cost_of_equity(beta=1.09) == pytest.approx(0.20575, abs=1e-6)

    def test_defaults_to_beta_one_when_missing(self):
        assert v.cost_of_equity(beta=None) == pytest.approx(
            v.RISK_FREE_RATE + v.EQUITY_RISK_PREMIUM, abs=1e-6
        )

    def test_defaults_to_beta_one_when_nan(self):
        assert v.cost_of_equity(beta=float('nan')) == pytest.approx(
            v.RISK_FREE_RATE + v.EQUITY_RISK_PREMIUM, abs=1e-6
        )

    def test_clamps_implausibly_low_beta(self):
        assert v.cost_of_equity(beta=-3.0) == v.cost_of_equity(beta=v.MIN_BETA)

    def test_clamps_implausibly_high_beta(self):
        assert v.cost_of_equity(beta=9.0) == v.cost_of_equity(beta=v.MAX_BETA)

    def test_stays_above_terminal_growth(self):
        assert v.cost_of_equity(beta=v.MIN_BETA) > v.TERMINAL_GROWTH


class TestDiscountFcfToEquity:
    """
    O 'Free Cash Flow' do yfinance = OCF - CapEx, e o OCF já é líquido de juros
    pagos: é FCFE (alavancado). Não se subtrai dívida líquida dele.
    """

    def test_result_is_pure_pv_per_share_with_no_debt_adjustment(self):
        # Perpetuidade pura: se houvesse ajuste de dívida, o resultado se
        # afastaria do valor presente puro dos fluxos.
        gt = v.TERMINAL_GROWTH
        r = 0.25
        fcf = 10e6
        shares = 2e6
        pure_pv_per_share = fcf * (1 + gt) / (r - gt) / shares
        got = v.discount_fcf_to_equity(
            fcf_base=fcf, growth=gt, discount_rate=r, shares=shares, terminal_growth=gt
        )
        assert got == pytest.approx(pure_pv_per_share, rel=1e-6)

    def test_gordon_growth_with_flat_fcf_matches_closed_form(self):
        # growth == terminal == gt -> perpetuidade pura: FCF*(1+g)/(r-g)
        gt = v.TERMINAL_GROWTH
        r = 0.25
        fcf = 10e6
        expected = fcf * (1 + gt) / (r - gt) / 1e6
        got = v.discount_fcf_to_equity(
            fcf_base=fcf, growth=gt, discount_rate=r, shares=1e6, terminal_growth=gt
        )
        assert got == pytest.approx(expected, rel=1e-6)

    def test_returns_nan_when_discount_rate_below_terminal_growth(self):
        assert np.isnan(
            v.discount_fcf_to_equity(
                fcf_base=10e6, growth=0.0, discount_rate=0.05,
                shares=1e6, terminal_growth=0.124,
            )
        )

    def test_returns_nan_for_non_positive_shares(self):
        assert np.isnan(
            v.discount_fcf_to_equity(
                fcf_base=10e6, growth=0.0, discount_rate=0.20, shares=0
            )
        )

    def test_higher_discount_rate_lowers_fair_value(self):
        low = v.discount_fcf_to_equity(10e6, 0.0, 0.18, 1e6)
        high = v.discount_fcf_to_equity(10e6, 0.0, 0.25, 1e6)
        assert high < low


class TestComputeBeta:
    """
    O beta do yfinance é inutilizável para ações BR (PETR4 = -0,139), então
    regredimos os retornos contra o IBOV.
    """

    @staticmethod
    def _market(n=60):
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0.002, 0.03, n), index=range(n))

    def test_beta_of_market_against_itself_is_one(self):
        mkt = self._market()
        assert v.compute_beta(mkt, mkt) == pytest.approx(1.0, abs=1e-9)

    def test_beta_of_double_amplitude_series_is_two(self):
        mkt = self._market()
        assert v.compute_beta(mkt * 2, mkt) == pytest.approx(2.0, abs=1e-9)

    def test_returns_nan_for_zero_market_variance(self):
        mkt = pd.Series([0.01] * 60)
        assert np.isnan(v.compute_beta(pd.Series([0.02] * 60), mkt))

    def test_returns_nan_when_below_minimum_observations(self):
        n = v.MIN_BETA_OBSERVATIONS - 1
        mkt = self._market(n)
        assert np.isnan(v.compute_beta(mkt * 2, mkt))

    def test_accepts_exactly_minimum_observations(self):
        mkt = self._market(v.MIN_BETA_OBSERVATIONS)
        assert v.compute_beta(mkt * 2, mkt) == pytest.approx(2.0, abs=1e-9)

    def test_aligns_on_index_and_ignores_unmatched_dates(self):
        mkt = self._market()
        stock_with_gap = (mkt * 2).drop([3, 7, 11])
        assert v.compute_beta(stock_with_gap, mkt) == pytest.approx(2.0, abs=1e-9)


class TestSectorBetas:
    """
    Beta por setor (mediana), não por empresa: small caps ilíquidas medem beta
    artificialmente baixo (RSUL4 = 0,155 negociando R$ 273k/dia).
    """

    def test_uses_median_beta_of_the_sector(self):
        df = pd.DataFrame({
            'setor': ['Auto Parts', 'Auto Parts', 'Auto Parts', 'Banks'],
            'beta_raw': [0.8, 1.0, 1.2, 0.5],
        })
        out = v.compute_sector_betas(df)
        assert out['Auto Parts'] == pytest.approx(1.0)

    def test_ignores_nan_betas_in_the_median(self):
        df = pd.DataFrame({
            'setor': ['Auto Parts'] * 4,
            'beta_raw': [0.8, np.nan, 1.2, np.nan],
        })
        out = v.compute_sector_betas(df)
        assert out['Auto Parts'] == pytest.approx(1.0)

    def test_skips_sector_with_no_valid_betas(self):
        df = pd.DataFrame({'setor': ['X', 'X'], 'beta_raw': [np.nan, np.nan]})
        assert 'X' not in v.compute_sector_betas(df)

    def test_skips_empty_sector_label(self):
        df = pd.DataFrame({'setor': ['', ''], 'beta_raw': [1.0, 1.2]})
        assert '' not in v.compute_sector_betas(df)


class TestExcessReturnsValuation:
    """Bancos: o CoE default passa a vir do CAPM, não da Selic pura."""

    def test_defaults_to_capm_cost_of_equity_not_selic(self):
        # ROE de 30% contra CoE default; se ainda usasse SELIC (14,25%) o
        # excess return seria maior e o preço justo mais alto.
        got = v.excess_returns_valuation(roe_decimal=0.30, vpa=10.0)
        expected = v.excess_returns_valuation(
            roe_decimal=0.30, vpa=10.0, coe=v.cost_of_equity(beta=1.0)
        )
        assert got == pytest.approx(expected)

    def test_accepts_explicit_coe(self):
        # ROE 30%, CoE 20%, g 12.4% -> 10 + (0.30-0.20)*10/(0.20-0.124)
        expected = 10 + (0.30 - 0.20) * 10 / (0.20 - 0.124)
        got = v.excess_returns_valuation(
            roe_decimal=0.30, vpa=10.0, coe=0.20, terminal_growth=0.124
        )
        assert got == pytest.approx(expected, rel=1e-9)

    def test_returns_nan_when_roe_below_coe(self):
        assert np.isnan(v.excess_returns_valuation(0.10, 10.0, coe=0.20))

    def test_returns_nan_for_non_positive_vpa(self):
        assert np.isnan(v.excess_returns_valuation(0.30, 0.0, coe=0.20))


class TestDdmValuation:
    """DDM fallback: idem, desconto pelo CAPM."""

    def test_defaults_to_capm_cost_of_equity(self):
        got = v.ddm_valuation(dps=5.0)
        expected = 5.0 / (v.cost_of_equity(beta=1.0) - v.TERMINAL_GROWTH)
        assert got == pytest.approx(expected)

    def test_returns_nan_when_discount_rate_below_growth(self):
        assert np.isnan(v.ddm_valuation(5.0, discount_rate=0.05, growth_rate=0.124))

    def test_returns_nan_for_non_positive_dps(self):
        assert np.isnan(v.ddm_valuation(0.0))


class TestEnvHelpers:
    """RF/ERP e a flag forward passam a ser editáveis via env."""

    def test_env_float_reads_value(self, monkeypatch):
        monkeypatch.setenv('X_RF', '0.2')
        assert v._env_float('X_RF', 0.1) == 0.2

    def test_env_float_default_when_missing(self, monkeypatch):
        monkeypatch.delenv('X_RF', raising=False)
        assert v._env_float('X_RF', 0.1) == 0.1

    def test_env_float_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv('X_RF', 'abc')
        assert v._env_float('X_RF', 0.1) == 0.1

    def test_env_bool_accepts_truthy_words(self, monkeypatch):
        for val in ('1', 'true', 'YES', 'on'):
            monkeypatch.setenv('X_B', val)
            assert v._env_bool('X_B') is True

    def test_env_bool_false_and_default(self, monkeypatch):
        monkeypatch.setenv('X_B', '0')
        assert v._env_bool('X_B') is False
        monkeypatch.delenv('X_B', raising=False)
        assert v._env_bool('X_B', default=False) is False


class TestForwardGrowth:
    """
    Com USE_FORWARD_ESTIMATES ligado, o estágio 1 usa o crescimento de analistas
    (yfinance); senão, o CAGR histórico. Sem estimativa forward, cai de volta no
    histórico. get_fcf_series/get_forward_growth são mockados para não bater rede.
    """

    FCF = pd.Series([121e6, 110e6, 100e6])  # CAGR histórico = 10%

    def test_uses_forward_growth_when_enabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'get_forward_growth', lambda t: 0.05)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_ignores_forward_when_disabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'get_forward_growth', lambda t: 0.05)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', False)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_falls_back_to_historical_when_forward_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'get_forward_growth', lambda t: float('nan'))
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_forward_growth_is_capped_at_max(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'get_forward_growth', lambda t: 0.90)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['growth_rate'] == v.MAX_GROWTH_RATE


class TestMetodoValuation:
    """
    A coluna metodo_valuation desfaz a ambiguidade do 'preco_justo_dcf': em
    incorporadoras com FCF negativo o valor vinha do DDM, não do DCF.
    """

    @staticmethod
    def _fundamentals(setor):
        # >=2 linhas por setor para as médias de Graham; beta_raw p/ o setor.
        return pd.DataFrame({
            'ticker': ['A', 'B'],
            'setor': [setor, setor],
            'pl': [8.0, 9.0],
            'pvp': [1.0, 1.1],
            'beta_raw': [1.0, 1.0],
        })

    def test_bank_is_labeled_excess_returns(self):
        df = pd.DataFrame({
            'ticker': ['BANK1'], 'ticker_sa': ['BANK1.SA'], 'setor': ['Banks'],
            'roe_pct': [25.0], 'vpa': [10.0], 'lpa': [2.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Banks'), model='bank')
        assert out.loc[0, 'metodo_valuation'] == 'excess_returns'

    def test_bank_is_none_when_excess_returns_undefined(self):
        # ROE < CoE -> excess returns NaN e sem fallback p/ banco.
        df = pd.DataFrame({
            'ticker': ['BANK1'], 'ticker_sa': ['BANK1.SA'], 'setor': ['Banks'],
            'roe_pct': [5.0], 'vpa': [10.0], 'lpa': [2.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Banks'), model='bank')
        assert out.loc[0, 'metodo_valuation'] == 'none'

    def test_stock_with_valid_dcf_is_labeled_dcf(self, monkeypatch):
        monkeypatch.setattr(v, 'dcf_valuation', lambda *a, **k: {
            'preco_justo_dcf': 20.0, 'growth_rate': 0.1, 'fcf_base': 1.0,
            'cost_of_equity': 0.18, 'growth_source': 'historical',
        })
        df = pd.DataFrame({
            'ticker': ['X'], 'ticker_sa': ['X.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert out.loc[0, 'metodo_valuation'] == 'dcf'
        assert out.loc[0, 'growth_source'] == 'historical'

    def test_stock_falls_back_to_ddm_when_dcf_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'dcf_valuation', lambda *a, **k: {
            'preco_justo_dcf': np.nan, 'growth_rate': np.nan, 'fcf_base': np.nan,
            'cost_of_equity': np.nan, 'growth_source': 'historical',
        })
        df = pd.DataFrame({
            'ticker': ['INCORP'], 'ticker_sa': ['INCORP.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert out.loc[0, 'metodo_valuation'] == 'ddm'


class TestAppendSnapshot:
    """Histórico append-only com preço justo, método e as premissas da rodada."""

    @staticmethod
    def _valued():
        return pd.DataFrame({
            'tipo': ['ação'], 'ticker': ['X'], 'nome': ['X SA'], 'setor': ['Retail'],
            'preco': [5.0], 'preco_justo_dcf': [10.0], 'metodo_valuation': ['dcf'],
            'growth_source': ['historical'], 'preco_justo_graham': [8.0],
            'margem_seg_dcf_pct': [50.0], 'margem_seg_graham_pct': [37.5],
            'margem_seg_media_pct': [43.75], 'undervalued': [True],
            'forte_desconto': [True], 'cost_of_equity_pct': [18.0],
        })

    def test_writes_header_row_and_assumptions(self, tmp_path):
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-07-16')
        out = pd.read_csv(p)
        assert len(out) == 1
        assert out.loc[0, 'data_snapshot'] == '2026-07-16'
        assert out.loc[0, 'metodo_valuation'] == 'dcf'
        for col in ('risk_free_rate', 'equity_risk_premium',
                    'terminal_growth', 'use_forward_estimates'):
            assert col in out.columns

    def test_appends_without_duplicating_header(self, tmp_path):
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-07-16')
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-07-17')
        out = pd.read_csv(p)
        assert len(out) == 2
        assert set(out['data_snapshot']) == {'2026-07-16', '2026-07-17'}

    def test_snapshots_growth_columns(self, tmp_path):
        p = tmp_path / 'hist.csv'
        df = self._valued()
        df['crescimento_receita_pct'] = [14.8]
        df['crescimento_lucro_pct'] = [9.2]
        df['num_analistas'] = [5]
        v.append_snapshot(df, path=p, snapshot_date='2026-07-16')
        out = pd.read_csv(p)
        assert out.loc[0, 'crescimento_receita_pct'] == 14.8
        assert out.loc[0, 'num_analistas'] == 5

    def test_new_columns_align_with_older_history(self, tmp_path):
        """Histórico gravado antes das colunas de crescimento não desalinha."""
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-07-16')

        df = self._valued()
        df['num_analistas'] = [5]
        v.append_snapshot(df, path=p, snapshot_date='2026-07-17')

        out = pd.read_csv(p)
        assert len(out) == 2
        assert pd.isna(out.loc[0, 'num_analistas'])
        assert out.loc[1, 'num_analistas'] == 5
        assert out.loc[0, 'metodo_valuation'] == 'dcf'

    def test_empty_df_is_a_noop(self, tmp_path):
        p = tmp_path / 'hist.csv'
        v.append_snapshot(pd.DataFrame(), path=p)
        assert not p.exists()


class TestRsul4Regression:
    """
    Regressão do caso que originou a correção: com os dados reais da RSUL4 o
    modelo dizia undervalued (R$ 309,53 vs preço R$ 47,36) enquanto a Simply
    Wall St dizia overvalued.
    """

    FCF_SERIES = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
    PRICE = 47.36
    TOTAL_SHARES = 6_072_128
    BETA = 1.09

    def test_fair_value_is_below_market_price(self):
        base = v.compute_fcf_base(self.FCF_SERIES)
        growth = v._compute_fcf_cagr(self.FCF_SERIES)
        coe = v.cost_of_equity(beta=self.BETA)
        fv = v.discount_fcf_to_equity(base, growth, coe, self.TOTAL_SHARES)
        assert fv < self.PRICE, f"esperado overvalued, obtido preço justo R$ {fv:.2f}"
