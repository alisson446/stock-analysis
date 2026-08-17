import importlib
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


class TestComputeFcfGrowth:
    """
    O crescimento sai da TENDÊNCIA da série inteira (uma reta sobre o log dos
    valores), não da comparação entre o primeiro e o último ponto. E só sai
    quando existe tendência: se a reta não explica ao menos metade da variação
    da série, a função devolve NaN e o DCF se declara inaplicável.

    As séries reais abaixo ilustram comportamentos que o código precisa
    distinguir. Nenhum limiar foi escolhido a partir delas.
    """

    def test_reads_clean_compound_growth_exactly(self):
        # 100 -> 110 -> 121 é 10% a.a. exato: a reta passa pelos três pontos.
        serie = pd.Series([121.0, 110.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(0.10, abs=1e-6)

    def test_accepts_consistent_decline(self):
        # KEPL3: 292 -> 207 -> 153 -> 51. Cai em todos os anos, então o número
        # é grande E confiável. Passa sem piso: declínio é dado válido.
        serie = pd.Series([51.0, 153.0, 207.0, 292.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(-0.4252, abs=1e-3)

    def test_rejects_cyclical_series(self):
        # RIAA3: 519 -> 951 -> 1.087 -> 351. Sobe, sobe, despenca. O cálculo
        # pelas pontas dizia -12,2%, um número que não descreve nenhum ano.
        # A regressão sozinha também não salva (-9,9%); quem barra é o R².
        serie = pd.Series([351.0, 1087.0, 951.0, 519.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_rejects_series_that_returns_to_its_start(self):
        # BLAU3: 134 -> 106 -> 366 -> 134. Termina onde começou. A regressão
        # sozinha leria +13,2%, puxada pelo pico no penúltimo ponto.
        serie = pd.Series([134.0, 366.0, 106.0, 134.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_returns_nan_when_series_contains_negative_year(self):
        # RSUL4: 21,2M -> 35,4M -> -9,8M -> 41,8M. Antes devolvia 0.0, que o
        # estágio 1 do DCF transforma em ACELERAÇÃO até TERMINAL_GROWTH --
        # ou seja, zerar inflava o preço justo em vez de ser conservador.
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_returns_nan_for_single_year(self):
        # Antes devolvia 0.0, pelo mesmo motivo e com o mesmo efeito.
        assert np.isnan(v._compute_fcf_growth(pd.Series([100.0])))

    def test_returns_nan_for_constant_series(self):
        # Sem variação não há R² (divisão por zero). É o caso extremo da
        # empresa muito estável, rejeitada de propósito: ela sai da lista em
        # vez de aparecer como barata.
        assert np.isnan(v._compute_fcf_growth(pd.Series([100.0] * 4)))

    def test_returns_nan_when_growth_is_above_projectable_threshold(self):
        # 100 -> 1000 em 1 ano = +900%. Dois pontos formam uma reta perfeita,
        # então o R² deixa passar; quem barra é o limiar de projetabilidade.
        serie = pd.Series([1000.0, 100.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_accepts_growth_exactly_at_the_threshold(self):
        # 100 -> 120 em 1 ano = +20%: no limiar, ainda projetável.
        serie = pd.Series([120.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(
            v.MAX_PROJECTABLE_GROWTH, abs=1e-9)

    def test_lets_negative_growth_through_unchanged(self):
        # 100 -> 50 em 1 ano = -50%. Sem piso: só reduz o preço justo.
        serie = pd.Series([50.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(-0.50, abs=1e-9)

    def test_rejects_series_just_below_the_r2_threshold(self):
        # 100 -> 163 -> 130 -> 160. R² = 0,4498 (medido), logo abaixo de
        # MIN_TREND_R2 = 0,5. Série sintética construída para fixar o limiar
        # por baixo -- nenhum dado real foi calibrado para isso.
        serie = pd.Series([160.0, 130.0, 163.0, 100.0])
        assert np.isnan(v._compute_fcf_growth(serie))

    def test_accepts_series_just_above_the_r2_threshold(self):
        # 100 -> 154 -> 130 -> 160. R² = 0,5602 (medido), logo acima de
        # MIN_TREND_R2 = 0,5. Série sintética construída para fixar o limiar
        # por cima -- nenhum dado real foi calibrado para isso.
        serie = pd.Series([160.0, 130.0, 154.0, 100.0])
        assert v._compute_fcf_growth(serie) == pytest.approx(0.1321, abs=1e-3)


class TestFcfTrendBase:
    """
    O nível da tendência responde "onde a empresa está hoje?" -- pergunta
    diferente de "ela continua nesse ritmo?", que é do _compute_fcf_growth.
    Série sem trajetória devolve NaN e o chamador fica com a mediana.
    """

    # Todas as séries abaixo vêm do mais RECENTE ao mais antigo, como o
    # yfinance entrega. Nenhuma foi copiada de data/ -- são construídas a
    # partir da regra (Guideline 3).

    def test_serie_subindo_devolve_o_nivel_da_reta(self):
        # 100 -> 120 -> 144 -> 172,8: crescimento composto exato de 20%.
        # A reta passa pelos quatro pontos, então o nível em t=3 é 172,8.
        serie = pd.Series([172.8, 144.0, 120.0, 100.0])
        assert v._fcf_trend_base(serie) == pytest.approx(172.8)

    def test_serie_caindo_devolve_o_nivel_da_reta(self):
        # A mesma série invertida: 172,8 -> 100. A tendência se aplica nas
        # duas direções, e aqui o nível fica ABAIXO da mediana (132,0).
        serie = pd.Series([100.0, 120.0, 144.0, 172.8])
        assert v._fcf_trend_base(serie) == pytest.approx(100.0)

    def test_serie_erratica_devolve_nan(self):
        # 100 -> 163 -> 130 -> 160. R² = 0,4498 (medido), abaixo de
        # MIN_TREND_R2 -- sobe e desce sem padrão, não há trajetória.
        serie = pd.Series([160.0, 130.0, 163.0, 100.0])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_menos_de_quatro_pontos_devolve_nan(self):
        # 100 -> 120 -> 144: ajuste perfeito (R² = 1), e ainda assim recusado.
        # Com 3 pontos, metade das séries sem tendência nenhuma passam no R².
        serie = pd.Series([144.0, 120.0, 100.0])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_ano_negativo_devolve_nan(self):
        # Não existe log de número negativo, e uma série que atravessou o
        # prejuízo é justamente aquela em que extrapolar o nível é menos
        # confiável. A guarda coincide com a prudência.
        serie = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
        assert np.isnan(v._fcf_trend_base(serie))

    def test_serie_constante_devolve_nan(self):
        # Variação nula no log: o R² seria divisão por zero. Devolve NaN por
        # guarda explícita, não por acidente de ponto flutuante. Aqui tanto
        # faz para o resultado -- numa série constante o nível da reta É a
        # mediana --, mas o caminho precisa ser uma decisão, não um acidente.
        assert np.isnan(v._fcf_trend_base(pd.Series([100.0] * 4)))

    def test_serie_com_ruido_devolve_a_reta_e_nao_o_ultimo_ponto(self):
        # 100 -> 130 -> 170 -> 150: sobe com um recuo no fim. R² = 0,7077,
        # acima do limiar, mas a série NÃO é geométrica exata -- e é isso que
        # torna este teste capaz de distinguir a regressão de um atalho.
        #
        # Cada jeito errado de implementar devolve um número diferente daqui:
        #   nível da reta em t=3 (correto) ... 168,59
        #   último observado / values[0] .... 150,00
        #   mediana ......................... 140,00
        #   reta sem inverter a série ....... 107,99
        serie = pd.Series([150.0, 170.0, 130.0, 100.0])
        assert v._fcf_trend_base(serie) == pytest.approx(168.5923, abs=1e-3)


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

    def test_serie_com_trajetoria_usa_a_tendencia_nao_a_mediana(self):
        # 100 -> 120 -> 144 -> 172,8. A mediana (132,0) é o nível de dois anos
        # atrás: numa série que sobe todo ano ela não resiste a pico nenhum,
        # ela mede o ano errado. Esperado é o nível de hoje.
        #
        # ESTE TESTE TRAVA A DECISÃO BIDIRECIONAL: uma variante
        # min(mediana, tendência) devolveria 132,0 e quebraria aqui.
        serie = pd.Series([172.8, 144.0, 120.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(172.8)

    def test_trajetoria_de_queda_tambem_usa_a_tendencia(self):
        # Mesma série invertida. O nível fica ABAIXO da mediana (132,0) -- a
        # mediana estava inflando uma empresa em declínio.
        serie = pd.Series([100.0, 120.0, 144.0, 172.8])
        assert v.compute_fcf_base(serie) == pytest.approx(100.0)

    def test_serie_sem_trajetoria_continua_na_mediana(self):
        # R² = 0,4498: sem trajetória, a mediana segue valendo.
        serie = pd.Series([160.0, 130.0, 163.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(145.0)

    def test_tres_pontos_continuam_na_mediana(self):
        serie = pd.Series([144.0, 120.0, 100.0])
        assert v.compute_fcf_base(serie) == pytest.approx(120.0)


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

    def test_negative_growth_yields_a_positive_finite_price(self):
        # Sem piso, o estágio 1 pode começar em queda: o fluxo encolhe nos
        # primeiros anos e converge para o crescimento terminal. O resultado
        # precisa continuar sendo um número, só menor.
        got = v.discount_fcf_to_equity(10e6, -0.10, 0.20, 1e6)
        assert got > 0 and np.isfinite(got)
        assert got < v.discount_fcf_to_equity(10e6, 0.0, 0.20, 1e6)

    def test_returns_nan_at_minus_one_hundred_percent_growth(self):
        # -100% zera o fluxo do primeiro ano: não há empresa a avaliar.
        assert np.isnan(v.discount_fcf_to_equity(10e6, -1.0, 0.20, 1e6))

    def test_returns_nan_below_minus_one_hundred_percent_growth(self):
        assert np.isnan(v.discount_fcf_to_equity(10e6, -1.5, 0.20, 1e6))


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


class TestResolveForwardGrowth:
    """
    O crescimento forward vem da linha do CSV (já coletado pelo screener), não
    de uma requisição nova por ticker. O driver escolhe a coluna: receita
    (default) ou lucro.

    Receita é o default porque o DCF projeta fluxo de caixa livre — receita
    menos custos caixa e capex — e não lucro contábil, que oscila muito mais
    para a mesma variação de receita (alavancagem operacional, itens não
    recorrentes, efeitos fiscais).
    """

    @staticmethod
    def _row(receita=14.8, lucro=9.2):
        return pd.Series({
            'crescimento_receita_pct': receita,
            'crescimento_lucro_pct': lucro,
        })

    def test_reads_revenue_column_by_default(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row()) == pytest.approx(0.148)

    def test_reads_earnings_column_when_driver_is_earnings(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert v.resolve_forward_growth(self._row()) == pytest.approx(0.092)

    def test_converts_percentage_points_to_decimal(self, monkeypatch):
        # O CSV guarda pontos percentuais; o DCF trabalha em decimal.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row(receita=100.0)) == pytest.approx(1.0)

    def test_keeps_negative_growth(self, monkeypatch):
        # PETR4 com -4,61%: declínio é dado válido, não valor a ser corrigido.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert v.resolve_forward_growth(self._row(receita=-4.61)) == pytest.approx(-0.0461)

    def test_returns_nan_when_column_is_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert np.isnan(v.resolve_forward_growth(self._row(receita=np.nan)))

    def test_returns_nan_when_column_is_absent(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        assert np.isnan(v.resolve_forward_growth(pd.Series({'ticker': 'X'})))

    def test_returns_nan_at_minus_one_hundred_percent(self, monkeypatch):
        # Lucro que vira prejuízo: o denominador |realizado| cruza zero e a
        # razão deixa de significar uma taxa. Dado inválido, não valor extremo.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert np.isnan(v.resolve_forward_growth(self._row(lucro=-100.0)))

    def test_returns_nan_below_minus_one_hundred_percent(self, monkeypatch):
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        assert np.isnan(v.resolve_forward_growth(self._row(lucro=-250.0)))

    def test_invalid_driver_in_env_falls_back_to_revenue(self, monkeypatch):
        # A validação acontece na leitura da env (import do módulo), por isso o
        # reload: é o único jeito de reexecutar essa linha no teste.
        monkeypatch.setenv('FORWARD_GROWTH_DRIVER', 'ebitda')
        importlib.reload(v)
        assert v.FORWARD_GROWTH_DRIVER == 'revenue'
        monkeypatch.delenv('FORWARD_GROWTH_DRIVER')
        importlib.reload(v)  # devolve o módulo ao estado normal p/ os outros testes

    def test_earnings_driver_is_read_from_env(self, monkeypatch):
        monkeypatch.setenv('FORWARD_GROWTH_DRIVER', 'earnings')
        importlib.reload(v)
        assert v.FORWARD_GROWTH_DRIVER == 'earnings'
        monkeypatch.delenv('FORWARD_GROWTH_DRIVER')
        importlib.reload(v)


class TestForwardGrowth:
    """
    Com USE_FORWARD_ESTIMATES ligado, o estágio 1 usa o crescimento forward que
    o chamador passa (vindo de resolve_forward_growth); senão, o CAGR histórico.
    O forward não é mais buscado dentro do DCF — o valor já vem do CSV.

    O limiar de projetabilidade NÃO substitui a taxa: forward acima dele é
    descartado em favor do histórico.
    """

    FCF = pd.Series([121e6, 110e6, 100e6])  # CAGR histórico = 10%

    def test_uses_forward_growth_when_enabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.05)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(0.05)

    def test_ignores_forward_when_disabled(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', False)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.05)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_falls_back_to_historical_when_forward_nan(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                              forward_growth=float('nan'))
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_falls_back_to_historical_when_forward_is_not_projectable(self, monkeypatch):
        # ONCO3 com lucro +1410%: antes virava seed de 20% (preço justo máximo).
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=14.107)
        assert res['growth_source'] == 'historical'
        assert res['growth_rate'] == pytest.approx(0.10, abs=1e-6)

    def test_uses_negative_forward_growth_as_is(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=-0.10)
        assert res['growth_source'] == 'forward'
        assert res['growth_rate'] == pytest.approx(-0.10)

    def test_negative_growth_yields_lower_price_than_zero_growth(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self.FCF)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)
        declining = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                                    forward_growth=-0.10)
        flat = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0, forward_growth=0.0)
        assert declining['preco_justo_dcf'] < flat['preco_justo_dcf']

    def test_no_price_when_historical_is_not_projectable_and_no_forward(self, monkeypatch):
        # CAGR de +900% -> NaN. Sem forward utilizável, o DCF não sai: o
        # chamador recai no DDM e metodo_valuation registra a substituição.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: pd.Series([1000e6, 100e6]))
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', False)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert np.isnan(res['preco_justo_dcf'])
        assert np.isnan(res['growth_rate'])


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
            'fcf_base_source': 'median',
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
            'fcf_base_source': 'median',
        })
        df = pd.DataFrame({
            'ticker': ['INCORP'], 'ticker_sa': ['INCORP.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert out.loc[0, 'metodo_valuation'] == 'ddm'

    def test_passes_row_forward_growth_to_dcf(self, monkeypatch):
        # O crescimento vem da LINHA (já no CSV), não de uma busca por ticker.
        captured = {}

        def fake_dcf(ticker_sa, shares_total=None, beta=None, forward_growth=None,
                     moeda='BRL'):
            captured['forward_growth'] = forward_growth
            return {'preco_justo_dcf': 20.0, 'growth_rate': 0.148, 'fcf_base': 1.0,
                    'cost_of_equity': 0.18, 'growth_source': 'forward',
                    'fcf_base_source': 'median'}

        monkeypatch.setattr(v, 'dcf_valuation', fake_dcf)
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'revenue')
        df = pd.DataFrame({
            'ticker': ['X'], 'ticker_sa': ['X.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
            'crescimento_receita_pct': [14.8], 'crescimento_lucro_pct': [9.2],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert captured['forward_growth'] == pytest.approx(0.148)
        assert out.loc[0, 'growth_source'] == 'forward'

    def test_stock_propaga_fcf_base_source(self, monkeypatch):
        monkeypatch.setattr(v, 'dcf_valuation', lambda *a, **k: {
            'preco_justo_dcf': 20.0, 'growth_rate': 0.1, 'fcf_base': 1.0,
            'cost_of_equity': 0.18, 'growth_source': 'historical',
            'fcf_base_source': 'trend',
        })
        df = pd.DataFrame({
            'ticker': ['X'], 'ticker_sa': ['X.SA'], 'setor': ['Retail'],
            'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Retail'), model='stock')
        assert out.loc[0, 'fcf_base_source'] == 'trend'

    def test_banco_nao_tem_fcf_base_source(self, monkeypatch):
        # Banco não passa por DCF; a coluna existe e fica vazia, como
        # growth_source já faz.
        df = pd.DataFrame({
            'ticker': ['B'], 'ticker_sa': ['B.SA'], 'setor': ['Banks'],
            'roe_pct': [25.0], 'vpa': [10.0], 'lpa': [2.0], 'preco': [5.0],
            'dividend_rate': [1.0], 'shares_total': [1e6],
        })
        out = v.apply_valuation(df, self._fundamentals('Banks'), model='bank')
        assert out.loc[0, 'fcf_base_source'] == ''


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
                    'terminal_growth', 'use_forward_estimates',
                    'forward_growth_driver'):
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

    def test_snapshots_fcf_base_source(self, tmp_path):
        # A origem da base viaja até o CSV pelo mesmo caminho que a origem do
        # crescimento: sem ela, um salto no preço justo entre duas rodadas
        # fica indistinguível de uma mudança de fundamento.
        p = tmp_path / 'hist.csv'
        df = self._valued()
        df['fcf_base_source'] = ['trend']
        v.append_snapshot(df, path=p, snapshot_date='2026-08-17')
        out = pd.read_csv(p)
        assert out.loc[0, 'fcf_base_source'] == 'trend'

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

    def test_records_the_forward_growth_driver(self, monkeypatch, tmp_path):
        # Sem isso, uma rodada com 'earnings' fica indistinguível de uma com
        # 'revenue' no histórico — e o preço justo difere entre as duas.
        monkeypatch.setattr(v, 'FORWARD_GROWTH_DRIVER', 'earnings')
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-08-06')
        out = pd.read_csv(p)
        assert out.loc[0, 'forward_growth_driver'] == 'earnings'

    def test_snapshots_lpa_estimado(self, tmp_path):
        p = tmp_path / 'hist.csv'
        df = self._valued()
        df['crescimento_lucro_pct'] = [15.27]
        df['lpa_estimado'] = [2.49874]
        v.append_snapshot(df, path=p, snapshot_date='2026-08-10')
        out = pd.read_csv(p)
        assert out.loc[0, 'lpa_estimado'] == pytest.approx(2.49874)

    def test_lpa_estimado_aligns_with_history_written_before_it(self, tmp_path):
        """As 277 linhas já gravadas recebem NaN, não desalinham."""
        p = tmp_path / 'hist.csv'
        v.append_snapshot(self._valued(), path=p, snapshot_date='2026-08-06')

        df = self._valued()
        df['lpa_estimado'] = [2.49874]
        v.append_snapshot(df, path=p, snapshot_date='2026-08-10')

        out = pd.read_csv(p)
        assert len(out) == 2
        assert pd.isna(out.loc[0, 'lpa_estimado'])
        assert out.loc[1, 'lpa_estimado'] == pytest.approx(2.49874)


class TestRsul4Regression:
    """
    Regressão do caso que originou a correção: com os dados reais da RSUL4 o
    modelo dizia undervalued (R$ 309,53 vs preço R$ 47,36) enquanto a Simply
    Wall St dizia overvalued.

    A série passa pelo prejuízo (-9,8M), então não há trajetória de crescimento
    composto para projetar: o DCF não se aplica e o chamador recai no DDM. É
    uma afirmação mais forte que a anterior (o preço justo ficar abaixo do
    mercado), porque nenhum preço de DCF chega a ser emitido.
    """

    FCF_SERIES = pd.Series([41.82e6, -9.786e6, 35.426e6, 21.185e6])
    PRICE = 47.36
    TOTAL_SHARES = 6_072_128
    BETA = 1.09

    def test_growth_is_not_projectable(self):
        assert np.isnan(v._compute_fcf_growth(self.FCF_SERIES))

    def test_dcf_emits_no_price(self):
        base = v.compute_fcf_base(self.FCF_SERIES)
        growth = v._compute_fcf_growth(self.FCF_SERIES)
        coe = v.cost_of_equity(beta=self.BETA)
        fv = v.discount_fcf_to_equity(base, growth, coe, self.TOTAL_SHARES)
        assert np.isnan(fv), f"esperado NaN (DCF inaplicável), obtido R$ {fv:.2f}"


class TestMacroPorMoeda:
    """
    Descontar fluxo em dólar a 12,4% (juro brasileiro) embute inflação de reais
    num fluxo que não a tem. Cada moeda carrega o próprio juro livre de risco e
    prêmio de risco, e o crescimento na perpetuidade acompanha o juro dela.
    """

    def test_brl_usa_as_constantes_sem_sufixo(self):
        m = v.macro_for('BRL')
        assert m['risk_free_rate'] == v.RISK_FREE_RATE
        assert m['equity_risk_premium'] == v.EQUITY_RISK_PREMIUM

    def test_usd_tem_premissas_embutidas(self):
        m = v.macro_for('USD')
        assert m['risk_free_rate'] == pytest.approx(0.042)
        assert m['equity_risk_premium'] == pytest.approx(0.045)

    def test_terminal_growth_acompanha_o_juro_da_moeda(self):
        for moeda in ('BRL', 'USD'):
            m = v.macro_for(moeda)
            assert m['terminal_growth'] == m['risk_free_rate']

    def test_moeda_sem_premissas_devolve_none(self):
        assert v.macro_for('TWD') is None
        assert v.macro_for('') is None
        assert v.macro_for(None) is None

    def test_env_habilita_uma_moeda_nova(self, monkeypatch):
        monkeypatch.setenv('RISK_FREE_RATE_EUR', '0.028')
        monkeypatch.setenv('EQUITY_RISK_PREMIUM_EUR', '0.055')

        m = v.macro_for('EUR')

        assert m['risk_free_rate'] == pytest.approx(0.028)
        assert m['equity_risk_premium'] == pytest.approx(0.055)
        assert m['terminal_growth'] == pytest.approx(0.028)

    def test_env_com_so_uma_das_duas_nao_habilita(self, monkeypatch):
        monkeypatch.setenv('RISK_FREE_RATE_EUR', '0.028')
        assert v.macro_for('EUR') is None

    def test_moeda_e_normalizada(self, monkeypatch):
        assert v.macro_for('usd') == v.macro_for('USD')
        assert v.macro_for(' USD ') == v.macro_for('USD')


class TestCostOfEquityPorMoeda:
    def test_default_continua_sendo_reais(self):
        assert v.cost_of_equity(1.0) == pytest.approx(
            v.RISK_FREE_RATE + v.EQUITY_RISK_PREMIUM)

    def test_dolar_usa_as_premissas_do_dolar(self):
        assert v.cost_of_equity(1.0, moeda='USD') == pytest.approx(0.042 + 0.045)

    def test_moeda_sem_premissas_devolve_nan(self):
        assert pd.isna(v.cost_of_equity(1.0, moeda='TWD'))

    def test_clamp_de_beta_continua_valendo(self):
        assert v.cost_of_equity(9.0, moeda='USD') == pytest.approx(
            0.042 + v.MAX_BETA * 0.045)


class TestAppendSnapshotPorRegiao:
    """
    O docstring de append_snapshot diz que as premissas existem para atribuir
    uma divergência futura a mudança de dado OU de premissa. Gravá-las a partir
    de constantes do módulo faria a linha em dólar registrar o juro brasileiro:
    uma premissa que não foi usada, indistinguível de uma verdadeira.
    """

    def _df(self, **cols):
        base = {'ticker': ['X'], 'preco': [10.0], 'preco_justo_dcf': [12.0]}
        base.update(cols)
        return pd.DataFrame(base)

    def test_grava_na_pasta_da_regiao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        out = v.append_snapshot(self._df(moeda=['USD']), region='us')
        assert out == tmp_path / 'us' / 'valuation_history.csv'
        assert out.exists()

    def test_premissas_saem_da_moeda_da_linha(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        df = self._df(ticker=['PETR4', 'AAPL'], preco=[10.0, 10.0],
                      preco_justo_dcf=[12.0, 12.0], moeda=['BRL', 'USD'])

        v.append_snapshot(df, region='us')

        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        por_moeda = dict(zip(hist['moeda'], hist['risk_free_rate']))
        assert por_moeda['BRL'] == pytest.approx(v.RISK_FREE_RATE)
        assert por_moeda['USD'] == pytest.approx(0.042)

    def test_terminal_growth_tambem_e_por_linha(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        df = self._df(ticker=['PETR4', 'AAPL'], preco=[10.0, 10.0],
                      preco_justo_dcf=[12.0, 12.0], moeda=['BRL', 'USD'])

        v.append_snapshot(df, region='us')

        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert dict(zip(hist['moeda'], hist['terminal_growth']))['USD'] == pytest.approx(0.042)

    def test_sem_coluna_moeda_assume_brl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(), region='br')
        hist = pd.read_csv(tmp_path / 'br' / 'valuation_history.csv')
        assert list(hist['moeda']) == ['BRL']
        assert hist['risk_free_rate'].iloc[0] == pytest.approx(v.RISK_FREE_RATE)

    def test_regiao_vira_coluna(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(moeda=['USD']), region='us')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert list(hist['regiao']) == ['us']

    def test_tipo_de_banco_americano_continua_banco(self, tmp_path, monkeypatch):
        # `tipo` separa banco de não-banco; `regiao` diz de onde veio. Se `tipo`
        # virasse 'bdr', o JPMorgan ficaria indistinguível de uma varejista.
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(tipo=['banco'], moeda=['USD']), region='us')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert list(hist['tipo']) == ['banco']
        assert list(hist['regiao']) == ['us']

    def test_append_preserva_as_linhas_anteriores(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(moeda=['USD']), region='us', snapshot_date='2026-01-01')
        v.append_snapshot(self._df(moeda=['USD']), region='us', snapshot_date='2026-02-01')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert sorted(hist['data_snapshot'].unique()) == ['2026-01-01', '2026-02-01']


class TestApplyValuationUsaAMoedaDaLinha:
    """
    A taxa usada e a premissa gravada têm que ser a mesma coisa.

    `append_snapshot` grava o juro e o prêmio de risco da moeda de cada linha.
    Se o valuation descontar tudo a juro brasileiro, a linha em dólar sai com
    custo de capital de 19,9% ao lado de um juro de 4,2% — dois números que não
    podem ser verdadeiros ao mesmo tempo, e um histórico que mente exatamente
    sobre o que foi feito para não mentir.
    """

    @staticmethod
    def _fundamentals(setor='Retail'):
        # >=2 linhas por setor para as medianas de Graham; beta_raw = 1,0 para
        # o beta setorial ser previsível e a conta do CAPM ficar à vista.
        return pd.DataFrame({
            'ticker': ['A', 'B'], 'setor': [setor, setor],
            'pl': [8.0, 9.0], 'pvp': [1.0, 1.1], 'beta_raw': [1.0, 1.0],
        })

    @staticmethod
    def _linha(**cols):
        base = {'ticker': ['X'], 'ticker_sa': ['X'], 'setor': ['Retail'],
                'lpa': [2.0], 'vpa': [10.0], 'preco': [5.0],
                'dividend_rate': [1.0], 'shares_total': [1e6]}
        base.update(cols)
        return pd.DataFrame(base)

    @pytest.fixture(autouse=True)
    def _sem_rede(self, monkeypatch):
        # Sem série de FCF o DCF sai cedo, sem tocar na rede. O que este bloco
        # testa é a taxa de desconto, não o modelo.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: pd.Series(dtype=float))

    def test_linha_em_dolar_usa_o_custo_de_capital_do_dolar(self):
        out = v.apply_valuation(self._linha(moeda=['USD']), self._fundamentals())
        assert out.loc[0, 'cost_of_equity_pct'] == pytest.approx(
            (0.042 + 1.0 * 0.045) * 100)

    def test_linha_em_dolar_nao_usa_as_premissas_brasileiras(self):
        out = v.apply_valuation(self._linha(moeda=['USD']), self._fundamentals())
        brasileiro = (v.RISK_FREE_RATE + 1.0 * v.EQUITY_RISK_PREMIUM) * 100
        assert out.loc[0, 'cost_of_equity_pct'] != pytest.approx(brasileiro)

    def test_linha_em_reais_continua_como_antes(self):
        esperado = (v.RISK_FREE_RATE + 1.0 * v.EQUITY_RISK_PREMIUM) * 100
        out = v.apply_valuation(self._linha(moeda=['BRL']), self._fundamentals())
        assert out.loc[0, 'cost_of_equity_pct'] == pytest.approx(esperado)

    def test_sem_coluna_moeda_assume_reais(self):
        # Regra de compatibilidade do resto do pipeline: as linhas gravadas
        # antes da coluna existir são todas brasileiras.
        assert 'moeda' not in self._linha().columns
        esperado = (v.RISK_FREE_RATE + 1.0 * v.EQUITY_RISK_PREMIUM) * 100
        out = v.apply_valuation(self._linha(), self._fundamentals())
        assert out.loc[0, 'cost_of_equity_pct'] == pytest.approx(esperado)

    def test_custo_de_capital_e_premissas_do_snapshot_batem(self, tmp_path,
                                                            monkeypatch):
        # A trava contra o defeito voltar: em CADA linha do histórico,
        # custo de capital = juro livre de risco + beta × prêmio de risco,
        # com os três números saindo do mesmo arquivo.
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        df = self._linha(ticker=['PETR4', 'AAPL'], ticker_sa=['PETR4.SA', 'AAPL'],
                         setor=['Retail', 'Retail'], lpa=[2.0, 2.0],
                         vpa=[10.0, 10.0], preco=[5.0, 5.0],
                         dividend_rate=[1.0, 1.0], shares_total=[1e6, 1e6],
                         moeda=['BRL', 'USD'])

        valued = v.apply_valuation(df, self._fundamentals())
        v.append_snapshot(valued, region='us')

        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        beta_setorial = 1.0
        assert set(hist['moeda']) == {'BRL', 'USD'}
        for _, linha in hist.iterrows():
            esperado = (linha['risk_free_rate']
                        + beta_setorial * linha['equity_risk_premium']) * 100
            assert linha['cost_of_equity_pct'] == pytest.approx(esperado)

    def test_banco_em_dolar_desconta_com_crescimento_terminal_do_dolar(self):
        banco = self._linha(ticker=['JPM'], ticker_sa=['JPM'], setor=['Banks'],
                            roe_pct=[25.0], moeda=['USD'])
        out = v.apply_valuation(banco, self._fundamentals('Banks'), model='bank')

        coe_usd = 0.042 + 1.0 * 0.045
        esperado = v.excess_returns_valuation(0.25, 10.0, coe=coe_usd,
                                              terminal_growth=0.042)
        assert out.loc[0, 'preco_justo_dcf'] == pytest.approx(esperado)
        # Com o crescimento na perpetuidade em reais (12,4%, acima do custo de
        # capital em dólar de 8,7%) o modelo nem sairia com preço: os dois
        # números são de mundos diferentes.
        assert pd.isna(v.excess_returns_valuation(
            0.25, 10.0, coe=coe_usd, terminal_growth=v.TERMINAL_GROWTH))

    def test_moeda_sem_premissas_nao_produz_preco(self):
        # Sem juro livre de risco não existe taxa de desconto, e um preço justo
        # tirado de taxa faltante chega ao usuário como recomendação.
        out = v.apply_valuation(self._linha(moeda=['TWD']), self._fundamentals())
        assert pd.isna(out.loc[0, 'cost_of_equity_pct'])
        assert pd.isna(out.loc[0, 'preco_justo_dcf'])
        assert not out.loc[0, 'undervalued']

    def test_dcf_recebe_a_moeda_da_linha(self, monkeypatch):
        capturado = {}

        def fake_dcf(ticker_sa, shares_total=None, beta=None, forward_growth=None,
                     moeda='BRL'):
            capturado['moeda'] = moeda
            return {'preco_justo_dcf': 20.0, 'growth_rate': 0.1, 'fcf_base': 1.0,
                    'cost_of_equity': 0.087, 'growth_source': 'historical',
                    'fcf_base_source': 'median'}

        monkeypatch.setattr(v, 'dcf_valuation', fake_dcf)
        v.apply_valuation(self._linha(moeda=['USD']), self._fundamentals())
        assert capturado['moeda'] == 'USD'


class TestFcfBaseSource:
    """
    A origem da base viaja junto com o preço justo, pelo mesmo motivo que
    growth_source: sem ela, o preço justo de uma ação salta entre duas rodadas
    do histórico sem nada no arquivo explicando por quê, e o salto fica
    indistinguível de uma mudança de fundamento.
    """

    def test_serie_com_trajetoria_e_rotulada_trend(self, monkeypatch):
        serie = pd.Series([172.8e6, 144e6, 120e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == 'trend'

    def test_serie_curta_e_rotulada_median(self, monkeypatch):
        # 3 pontos: _fcf_trend_base recusa por n < 4 e a base cai na mediana,
        # mas _compute_fcf_growth aceita (basta 2 pontos), então o DCF chega
        # ao fim e o rótulo é gravado. É a via simples até 'median'.
        serie = pd.Series([121e6, 110e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == 'median'
        assert pd.notna(res['preco_justo_dcf'])

    def test_serie_erratica_so_chega_a_median_com_forward(self, monkeypatch):
        # 100 -> 163 -> 130 -> 160, R² = 0,4498. O MESMO MIN_TREND_R2 governa
        # duas decisões: recusa a base (certo, cai na mediana) E zera a taxa
        # histórica. Sem uma taxa vinda de fora, o DCF sai antes de rotular e
        # o rótulo fica '' -- não 'median'. Só o crescimento forward resgata o
        # lado da taxa e deixa a rodada chegar ao fim.
        serie = pd.Series([160e6, 130e6, 163e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        monkeypatch.setattr(v, 'USE_FORWARD_ESTIMATES', True)

        sem_forward = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert sem_forward['fcf_base_source'] == ''

        com_forward = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0,
                                      forward_growth=0.05)
        assert com_forward['fcf_base_source'] == 'median'

    def test_sem_base_escolhida_fica_vazio(self, monkeypatch):
        # Série vazia: o DCF sai antes de escolher qualquer base. String vazia
        # é diferente de 'median' -- aqui não houve escolha nenhuma.
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: pd.Series(dtype=float))
        res = v.dcf_valuation('X.SA', shares_total=1e6, beta=1.0)
        assert res['fcf_base_source'] == ''
        assert pd.isna(res['preco_justo_dcf'])

    def test_sem_contagem_de_acoes_fica_vazio(self, monkeypatch):
        # Série boa, com trajetória: a base chega a ser calculada. Mas o papel
        # não tem contagem de ações e o DCF sai sem preço. O rótulo precisa
        # continuar vazio -- gravar 'trend' aqui faria o histórico afirmar que
        # um DCF rodou onde ele não rodou.
        serie = pd.Series([172.8e6, 144e6, 120e6, 100e6])
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: serie)
        monkeypatch.setattr(v, 'resolve_share_count', lambda info: None)
        monkeypatch.setattr(v, 'yf', type('_YF', (), {
            'Ticker': staticmethod(lambda t: type('_T', (), {'info': {}})())})())
        res = v.dcf_valuation('X.SA', shares_total=None, beta=1.0)
        assert pd.isna(res['preco_justo_dcf'])
        assert res['fcf_base_source'] == ''


class TestDcfValuationPorMoeda:
    """O DCF desconta e projeta a perpetuidade na moeda do balanço."""

    @staticmethod
    def _serie_crescendo():
        # yfinance entrega do mais recente ao mais antigo: +10% ao ano.
        return pd.Series([133.1e6, 121e6, 110e6, 100e6])

    def _esperado(self, discount_rate, terminal_growth):
        # A base sai de compute_fcf_base, não de uma cópia da regra aqui: o
        # que estes testes verificam é qual MOEDA decide o juro de desconto e
        # a perpetuidade. Duplicar a fórmula da base fazia eles quebrarem a
        # cada mudança nela, por um motivo que não é o deles.
        return v.discount_fcf_to_equity(
            fcf_base=v.compute_fcf_base(self._serie_crescendo()),
            growth=0.10, discount_rate=discount_rate, shares=1e6,
            terminal_growth=terminal_growth)

    def test_dolar_usa_juro_e_perpetuidade_do_dolar(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self._serie_crescendo())

        res = v.dcf_valuation('AAPL', shares_total=1e6, beta=1.0, moeda='USD')

        assert res['cost_of_equity'] == pytest.approx(0.042 + 0.045)
        assert res['preco_justo_dcf'] == pytest.approx(
            self._esperado(0.042 + 0.045, 0.042))

    def test_default_continua_em_reais(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self._serie_crescendo())

        res = v.dcf_valuation('PETR4.SA', shares_total=1e6, beta=1.0)

        coe_br = v.RISK_FREE_RATE + v.EQUITY_RISK_PREMIUM
        assert res['cost_of_equity'] == pytest.approx(coe_br)
        assert res['preco_justo_dcf'] == pytest.approx(
            self._esperado(coe_br, v.TERMINAL_GROWTH))

    def test_moeda_sem_premissas_sai_sem_preco(self, monkeypatch):
        monkeypatch.setattr(v, 'get_fcf_series', lambda t: self._serie_crescendo())

        res = v.dcf_valuation('TSM', shares_total=1e6, beta=1.0, moeda='TWD')

        assert pd.isna(res['preco_justo_dcf'])
        assert pd.isna(res['cost_of_equity'])
