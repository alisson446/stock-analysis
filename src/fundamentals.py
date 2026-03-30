import pandas as pd
import numpy as np
import yfinance as yf
import time
from tqdm import tqdm


def _safe_get(info: dict, key: str, default=np.nan):
    """Retorna valor do dict ou default se None/ausente."""
    val = info.get(key)
    return val if val is not None else default


def _extract_financial_value(df_fin, labels):
    """Extrai o primeiro valor encontrado de um DataFrame financeiro dado uma lista de labels possíveis."""
    if df_fin is None or df_fin.empty:
        return np.nan
    for label in labels:
        if label in df_fin.index:
            val = df_fin.loc[label].dropna()
            if not val.empty:
                return val.iloc[0]
    return np.nan


def _extract_financial_series(df_fin, labels):
    """Extrai série temporal completa (últimos anos) de um DataFrame financeiro."""
    if df_fin is None or df_fin.empty:
        return pd.Series(dtype=float)
    for label in labels:
        if label in df_fin.index:
            series = df_fin.loc[label].dropna()
            if not series.empty:
                return series
    return pd.Series(dtype=float)


def fetch_fundamentals(tickers_sa: list[str], delay: float = 0.5) -> pd.DataFrame:
    """
    Coleta dados fundamentalistas de cada ticker via yfinance.

    Args:
        tickers_sa: Lista de tickers com sufixo .SA (ex: ['PETR4.SA', 'VALE3.SA'])
        delay: Tempo de espera entre requisições (segundos)

    Returns:
        DataFrame com todas as métricas calculadas
    """
    records = []

    for ticker_sa in tqdm(tickers_sa, desc="Coletando fundamentals"):
        try:
            stock = yf.Ticker(ticker_sa)
            info = stock.info or {}

            # Dados básicos do .info
            current_price = _safe_get(info, 'currentPrice')
            pe_ratio = _safe_get(info, 'trailingPE')
            pb_ratio = _safe_get(info, 'priceToBook')
            profit_margin = _safe_get(info, 'profitMargins')
            roe = _safe_get(info, 'returnOnEquity')
            current_ratio = _safe_get(info, 'currentRatio')
            eps = _safe_get(info, 'trailingEps')
            avg_volume = _safe_get(info, 'averageDailyVolume10Day')
            shares_outstanding = _safe_get(info, 'sharesOutstanding')
            sector = _safe_get(info, 'sector', '')
            industry = _safe_get(info, 'industry', '')
            company_name = _safe_get(info, 'shortName', ticker_sa)
            dividend_yield = _safe_get(info, 'dividendYield')
            dividend_rate = _safe_get(info, 'dividendRate')  # DPS anual em R$
            total_debt = _safe_get(info, 'totalDebt')
            total_cash = _safe_get(info, 'totalCash')

            # Converter percentuais
            margin_liquida_pct = profit_margin * 100 if pd.notna(profit_margin) else np.nan
            roe_pct = roe * 100 if pd.notna(roe) else np.nan
            # dividendYield para .SA tickers já vem como % (ex: 10.38 = 10.38%)
            dy_pct = dividend_yield if pd.notna(dividend_yield) and dividend_yield > 1 else (
                dividend_yield * 100 if pd.notna(dividend_yield) else np.nan
            )

            # Liquidez média diária em R$
            liq_media_diaria = (avg_volume * current_price
                                if pd.notna(avg_volume) and pd.notna(current_price)
                                else np.nan)

            # --- Dados dos demonstrativos financeiros ---
            financials = stock.financials  # DRE anual
            balance = stock.balance_sheet  # Balanço Patrimonial
            cashflow = stock.cashflow      # Fluxo de Caixa

            # Margem EBIT = EBIT / Receita Total
            ebit = _extract_financial_value(financials, ['EBIT', 'Ebit'])
            total_revenue = _extract_financial_value(financials, [
                'Total Revenue', 'TotalRevenue', 'Operating Revenue'
            ])
            margem_ebit_pct = (
                (ebit / total_revenue) * 100
                if pd.notna(ebit) and pd.notna(total_revenue) and total_revenue != 0
                else np.nan
            )

            # Dívida Líquida = Total Debt - Cash
            # Tentar do balance_sheet se não veio do .info
            if pd.notna(total_debt) and pd.notna(total_cash):
                divida_liquida = total_debt - total_cash
            else:
                bs_debt = _extract_financial_value(balance, [
                    'Total Debt', 'TotalDebt', 'Long Term Debt', 'LongTermDebt'
                ])
                bs_cash = _extract_financial_value(balance, [
                    'Cash And Cash Equivalents', 'CashAndCashEquivalents',
                    'Cash Cash Equivalents And Short Term Investments'
                ])
                divida_liquida = (
                    bs_debt - bs_cash
                    if pd.notna(bs_debt) and pd.notna(bs_cash)
                    else np.nan
                )

            # Dívida Líquida / EBIT
            dl_ebit = (
                divida_liquida / ebit
                if pd.notna(divida_liquida) and pd.notna(ebit) and ebit != 0
                else np.nan
            )

            # Patrimônio Líquido (Stockholders Equity)
            stockholders_equity = _extract_financial_value(balance, [
                'Stockholders Equity', 'StockholdersEquity',
                'Total Stockholders Equity', 'Ordinary Shares Number'
            ])

            # Dívida Líquida / Patrimônio Líquido
            dl_pl = (
                divida_liquida / stockholders_equity
                if pd.notna(divida_liquida) and pd.notna(stockholders_equity)
                and stockholders_equity != 0
                else np.nan
            )

            # Passivos / Ativos
            total_liabilities = _extract_financial_value(balance, [
                'Total Liabilities Net Minority Interest',
                'TotalLiabilitiesNetMinorityInterest',
                'Total Liabilities', 'TotalLiabilities'
            ])
            total_assets = _extract_financial_value(balance, [
                'Total Assets', 'TotalAssets'
            ])
            passivos_ativos = (
                total_liabilities / total_assets
                if pd.notna(total_liabilities) and pd.notna(total_assets)
                and total_assets != 0
                else np.nan
            )

            # VPA (Valor Patrimonial por Ação) = Preço / P/PV
            vpa = (
                current_price / pb_ratio
                if pd.notna(current_price) and pd.notna(pb_ratio) and pb_ratio != 0
                else np.nan
            )

            # Free Cash Flow (série para DCF)
            fcf_series = _extract_financial_series(cashflow, [
                'Free Cash Flow', 'FreeCashFlow'
            ])
            fcf_latest = fcf_series.iloc[0] if not fcf_series.empty else np.nan

            records.append({
                'ticker_sa': ticker_sa,
                'ticker': ticker_sa.replace('.SA', ''),
                'nome': company_name,
                'setor': sector,
                'industria': industry,
                'preco': current_price,
                'pl': pe_ratio,
                'pvp': pb_ratio,
                'margem_ebit_pct': margem_ebit_pct,
                'margem_liquida_pct': margin_liquida_pct,
                'dl_ebit': dl_ebit,
                'dl_pl': dl_pl,
                'roe_pct': roe_pct,
                'liquidez_corrente': current_ratio,
                'passivos_ativos': passivos_ativos,
                'liq_media_diaria': liq_media_diaria,
                'lpa': eps,
                'vpa': vpa,
                'dy_pct': dy_pct,
                'divida_liquida': divida_liquida,
                'ebit': ebit,
                'fcf_latest': fcf_latest,
                'shares_outstanding': shares_outstanding,
                'dividend_rate': dividend_rate,
            })

        except Exception as e:
            print(f"[fundamentals] Erro ao processar {ticker_sa}: {e}")
            records.append({
                'ticker_sa': ticker_sa,
                'ticker': ticker_sa.replace('.SA', ''),
                'nome': ticker_sa,
                'setor': '',
                'industria': '',
                **{k: np.nan for k in [
                    'preco', 'pl', 'pvp', 'margem_ebit_pct', 'margem_liquida_pct',
                    'dl_ebit', 'dl_pl', 'roe_pct', 'liquidez_corrente', 'passivos_ativos',
                    'liq_media_diaria', 'lpa', 'vpa', 'dy_pct', 'divida_liquida',
                    'ebit', 'fcf_latest', 'shares_outstanding',
                    'dividend_rate',
                ]}
            })

        time.sleep(delay)

    df = pd.DataFrame(records)
    print(f"\n[fundamentals] {len(df)} tickers processados, "
          f"{df['preco'].notna().sum()} com dados de preço")
    return df


def get_fcf_series(ticker_sa: str) -> pd.Series:
    """Retorna série histórica de Free Cash Flow para cálculo de DCF."""
    try:
        stock = yf.Ticker(ticker_sa)
        cf = stock.cashflow
        return _extract_financial_series(cf, ['Free Cash Flow', 'FreeCashFlow'])
    except Exception:
        return pd.Series(dtype=float)
