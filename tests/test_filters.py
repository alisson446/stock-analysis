import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import filters


def _cfg(exigir_estimativa=False, exigir_num_analistas=False,
         exigir_lpa_estimado=False,
         receita_min=0, lucro_min=0, analistas_min=2, lpa_est_min=0):
    return {
        'crescimento_receita_pct_min': receita_min,
        'crescimento_lucro_pct_min': lucro_min,
        'num_analistas_min': analistas_min,
        'lpa_estimado_min': lpa_est_min,
        'exigir_num_analistas': exigir_num_analistas,
        'exigir_estimativa': exigir_estimativa,
        'exigir_lpa_estimado': exigir_lpa_estimado,
    }


def _df(rows, lpa_estimado=1.0):
    """rows: lista de (crescimento_receita_pct, crescimento_lucro_pct, num_analistas)

    `lpa_estimado` entra como coluna à parte, escalar ou lista. O default
    positivo mantém os testes que não são sobre esse critério indiferentes a
    ele; os testes do critério novo passam os valores explicitamente.
    """
    df = pd.DataFrame(
        rows,
        columns=['crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas'],
    )
    df['lpa_estimado'] = lpa_estimado
    return df


class TestEstimatesMaskBothFlagsOn:
    """Com as duas flags ligadas, ambos os critérios valem em conjunto."""

    def _mask(self, rows):
        return filters._estimates_mask(
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


class TestEstimatesMaskFlagsIndependent:
    """Cada flag liga somente o seu critério; nenhuma altera a outra."""

    def test_both_off_passes_everything(self):
        rows = [(-50.0, -80.0, 1), (np.nan, np.nan, np.nan)]
        mask = filters._estimates_mask(_df(rows), _cfg())
        assert mask.tolist() == [True, True]

    def test_estimativa_off_ignores_negative_growth(self):
        mask = filters._estimates_mask(
            _df([(14.8, -2.4, 7)]), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True]

    def test_analistas_off_ignores_analyst_count(self):
        mask = filters._estimates_mask(
            _df([(6.0, 12.8, np.nan)]), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [True]

    def test_only_analysts_decide_when_estimativa_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._estimates_mask(_df(rows), _cfg(exigir_num_analistas=True))
        assert mask.tolist() == [True, False]

    def test_only_growth_decides_when_analistas_off(self):
        rows = [(-50.0, -80.0, 5), (99.0, 99.0, 1)]
        mask = filters._estimates_mask(_df(rows), _cfg(exigir_estimativa=True))
        assert mask.tolist() == [False, True]


class TestEstimatesMaskThresholds:

    def test_custom_thresholds_are_respected(self):
        mask = filters._estimates_mask(
            _df([(9.0, 9.0, 5), (11.0, 11.0, 5)]),
            _cfg(exigir_estimativa=True, receita_min=10, lucro_min=10),
        )
        assert mask.tolist() == [False, True]

    def test_mask_preserves_dataframe_index(self):
        df = _df([(15.0, 15.0, 5), (15.0, 15.0, 5)])
        df.index = [7, 42]
        mask = filters._estimates_mask(df, _cfg(exigir_estimativa=True))
        assert mask.index.tolist() == [7, 42]


class TestEstimatesMaskLpaEstimado:
    """Guarda de sinal: o nível projetado, não a variação."""

    def _mask(self, rows, lpa_estimado):
        return filters._estimates_mask(
            _df(rows, lpa_estimado=lpa_estimado),
            _cfg(exigir_lpa_estimado=True),
        )

    def test_positive_lpa_estimado_passes(self):
        assert self._mask([(6.0, 12.0, 4)], 2.49874).tolist() == [True]

    def test_negative_lpa_estimado_fails(self):
        assert self._mask([(6.0, 12.0, 4)], -0.14196).tolist() == [False]

    def test_zero_lpa_estimado_fails_strict_comparison(self):
        assert self._mask([(6.0, 12.0, 4)], 0.0).tolist() == [False]

    def test_nan_lpa_estimado_fails(self):
        assert self._mask([(6.0, 12.0, 4)], np.nan).tolist() == [False]

    def test_positive_growth_with_projected_loss_fails(self):
        """O caso que motiva a spec: AURE3 e HBRE3.

        'Crescimento de lucro' de +88,6% que é um prejuízo encolhendo de
        -1,25 para -0,14 por ação. Sem esta guarda a linha passa.
        """
        mask = filters._estimates_mask(
            _df([(8.0, 88.64, 3)], lpa_estimado=-0.14196),
            _cfg(exigir_estimativa=True, exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [False]

    def test_same_row_passes_without_the_guard(self):
        """Prova que a guarda é o que reprova, não os cortes de crescimento."""
        mask = filters._estimates_mask(
            _df([(8.0, 88.64, 3)], lpa_estimado=-0.14196),
            _cfg(exigir_estimativa=True),
        )
        assert mask.tolist() == [True]

    def test_custom_threshold_is_respected(self):
        mask = filters._estimates_mask(
            _df([(6.0, 12.0, 4), (6.0, 12.0, 4)], lpa_estimado=[0.40, 0.60]),
            _cfg(exigir_lpa_estimado=True, lpa_est_min=0.5),
        )
        assert mask.tolist() == [False, True]


class TestEstimatesMaskLpaEstimadoIndependent:
    """A terceira flag não altera o significado das outras duas."""

    def test_off_ignores_negative_lpa_estimado(self):
        mask = filters._estimates_mask(
            _df([(6.0, 12.0, 4)], lpa_estimado=-1.0), _cfg())
        assert mask.tolist() == [True]

    def test_only_lpa_decides_when_other_flags_off(self):
        mask = filters._estimates_mask(
            _df([(-50.0, -80.0, 1), (99.0, 99.0, 9)], lpa_estimado=[2.0, -2.0]),
            _cfg(exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [True, False]

    def test_all_three_flags_on_combine(self):
        mask = filters._estimates_mask(
            _df([(6.0, 12.0, 4), (6.0, 12.0, 1), (6.0, -2.0, 4)],
                lpa_estimado=[2.0, 2.0, 2.0]),
            _cfg(exigir_estimativa=True, exigir_num_analistas=True,
                 exigir_lpa_estimado=True),
        )
        assert mask.tolist() == [True, False, False]

    def test_all_three_flags_off_passes_everything(self):
        mask = filters._estimates_mask(
            _df([(-50.0, -80.0, 1), (np.nan, np.nan, np.nan)],
                lpa_estimado=[-5.0, np.nan]),
            _cfg(),
        )
        assert mask.tolist() == [True, True]


def test_shipped_config_has_the_lpa_guard_on():
    """A guarda do LPA projetado protege o DCF (ver resolve_forward_growth em
    src/valuation.py) contra tratar prejuízo encolhendo como crescimento de
    lucro. Os testes acima cobrem `_estimates_mask` com configs fabricadas em
    memória; nenhum lia o `config/filters.json` publicado. Como o código
    acessa a flag via `cfg.get('exigir_lpa_estimado')`, se a chave sumir do
    JSON o `.get` retorna None e a guarda para de aplicar em silêncio — sem
    este teste, ninguém notaria antes de uma empresa com prejuízo projetado
    voltar a passar no screener.
    """
    cfg = filters._load_config('br')
    for block in ('stock_filters', 'bank_filters'):
        assert cfg[block]['exigir_lpa_estimado'] is True
        assert cfg[block]['lpa_estimado_min'] == 0


class TestLoadConfigPorRegiao:
    """A região escolhe a pasta do filters.json; região sem arquivo é erro."""

    def test_br_carrega_o_arquivo_movido(self):
        cfg = filters._load_config('br')
        assert 'stock_filters' in cfg and 'bank_filters' in cfg

    def test_regiao_sem_arquivo_levanta_erro_nomeando_o_caminho(self):
        with pytest.raises(FileNotFoundError) as exc:
            filters._load_config('zz')
        # A mensagem precisa dizer QUAL arquivo criar — senão o usuário só
        # descobre que "faltou algo" sem saber onde.
        assert 'zz' in str(exc.value) and 'filters.json' in str(exc.value)

    def test_regiao_malformada_levanta_value_error(self):
        with pytest.raises(ValueError):
            filters._load_config('../etc')


class TestLiquidityMask:
    """
    O corte de liquidez muda de coluna conforme a região: no Brasil filtra a
    liquidez da própria ação; na região alcançada via BDR filtra a do BDR, em
    reais, porque é o que se consegue negociar. A chave do config carrega a
    unidade no nome para não ser lida como dólar.
    """

    def test_usa_liq_media_diaria_quando_a_chave_e_a_local(self):
        df = pd.DataFrame({'liq_media_diaria': [50_000, 150_000],
                           'liq_media_diaria_bdr': [999_999, 1]})
        mask = filters._liquidity_mask(df, {'liq_media_diaria_min': 100_000})
        assert list(mask) == [False, True]

    def test_usa_liq_media_diaria_bdr_quando_a_chave_e_a_do_bdr(self):
        df = pd.DataFrame({'liq_media_diaria': [999_999, 1],
                           'liq_media_diaria_bdr': [50_000, 150_000]})
        mask = filters._liquidity_mask(df, {'liq_media_diaria_bdr_min': 100_000})
        assert list(mask) == [False, True]

    def test_config_sem_chave_de_liquidez_levanta_erro(self):
        df = pd.DataFrame({'liq_media_diaria': [1]})
        with pytest.raises(KeyError):
            filters._liquidity_mask(df, {'pl_max': 10})


class TestConfigDaRegiaoUS:
    """
    Os limiares são idênticos aos brasileiros por premissa: o critério de
    'barato' é do investidor, não do mercado. Se a bolsa americana quase não
    produz empresa a 10x lucro, a lista vem curta — e lista curta é informação.
    """

    def test_existe_e_tem_as_mesmas_chaves_de_bloco(self):
        us = filters._load_config('us')
        br = filters._load_config('br')
        assert set(us) == set(br) == {'stock_filters', 'bank_filters'}

    def test_limiares_de_barato_sao_iguais_aos_do_br(self):
        us = filters._load_config('us')['stock_filters']
        br = filters._load_config('br')['stock_filters']
        for chave in ('pl_max', 'pvp_max', 'roe_pct_min', 'margem_liquida_pct_min'):
            assert us[chave] == br[chave]

    def test_liquidez_e_a_do_bdr_em_reais(self):
        for bloco in ('stock_filters', 'bank_filters'):
            cfg = filters._load_config('us')[bloco]
            assert 'liq_media_diaria_bdr_min' in cfg
            assert 'liq_media_diaria_min' not in cfg

    def test_bank_filters_do_us_nao_tem_dy_pct_min(self):
        # O dividendo estrangeiro sofre retenção antes de chegar ao detentor do
        # BDR, então o dividendYield do yfinance é o do acionista de lá.
        assert 'dy_pct_min' not in filters._load_config('us')['bank_filters']
        assert 'dy_pct_min' in filters._load_config('br')['bank_filters']
