import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import scraper


class TestGetTickersPorRegiao:
    """
    O cache de tickers passa a viver em data/<regiao>/tickers.csv. O teste usa
    monkeypatch na raiz de dados para não tocar no cache real do repo.
    """

    def test_le_o_cache_da_regiao_pedida(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scraper.paths, 'DATA_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        pd.DataFrame({'ticker': ['AAPL'], 'ticker_bdr': ['AAPL34.SA']}).to_csv(
            tmp_path / 'us' / 'tickers.csv', index=False)

        df = scraper.get_tickers(region='us')

        assert list(df['ticker']) == ['AAPL']

    def test_regioes_diferentes_leem_arquivos_diferentes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scraper.paths, 'DATA_ROOT', tmp_path)
        for regiao, ticker in [('br', 'PETR4'), ('us', 'AAPL')]:
            (tmp_path / regiao).mkdir()
            pd.DataFrame({'ticker': [ticker]}).to_csv(
                tmp_path / regiao / 'tickers.csv', index=False)

        assert list(scraper.get_tickers(region='br')['ticker']) == ['PETR4']
        assert list(scraper.get_tickers(region='us')['ticker']) == ['AAPL']

    def test_regiao_invalida_levanta_erro(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scraper.paths, 'DATA_ROOT', tmp_path)
        with pytest.raises(ValueError):
            scraper.get_tickers(region='../etc')
