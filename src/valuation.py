import pandas as pd
import numpy as np
import yfinance as yf
from src.fundamentals import get_fcf_series

# Parâmetros de valuation
SELIC = 0.1425           # Taxa de desconto (WACC proxy)
TERMINAL_GROWTH = 0.035  # Crescimento terminal (inflação LP Brasil)
PROJECTION_YEARS = 5
MAX_GROWTH_RATE = 0.20   # Cap de crescimento anual
MIN_GROWTH_RATE = 0.0    # Floor de crescimento


def _compute_fcf_cagr(fcf_series: pd.Series) -> float:
    """
    Calcula CAGR do Free Cash Flow a partir da série histórica.
    A série vem do yfinance com o mais recente primeiro.
    """
    if len(fcf_series) < 2:
        return 0.0

    # Ordenar do mais antigo ao mais recente
    values = fcf_series.values[::-1]

    # Precisamos de valores positivos para calcular CAGR
    first_positive = None
    last_positive = None
    first_idx = None
    last_idx = None

    for i, v in enumerate(values):
        if v > 0:
            if first_positive is None:
                first_positive = v
                first_idx = i
            last_positive = v
            last_idx = i

    if first_positive is None or last_positive is None or first_idx == last_idx:
        return 0.0

    n_years = last_idx - first_idx
    if n_years <= 0:
        return 0.0

    cagr = (last_positive / first_positive) ** (1 / n_years) - 1

    # Aplicar limites
    return max(MIN_GROWTH_RATE, min(MAX_GROWTH_RATE, cagr))


def dcf_valuation(ticker_sa: str, shares_outstanding: float = None) -> dict:
    """
    Calcula preço justo via Fluxo de Caixa Descontado (DCF).

    Args:
        ticker_sa: Ticker com sufixo .SA
        shares_outstanding: Número de ações. Se None, busca do yfinance.

    Returns:
        dict com 'preco_justo_dcf', 'growth_rate', 'fcf_base', status info
    """
    result = {
        'preco_justo_dcf': np.nan,
        'growth_rate': np.nan,
        'fcf_base': np.nan,
    }

    try:
        fcf_series = get_fcf_series(ticker_sa)

        if fcf_series.empty:
            return result

        fcf_base = fcf_series.iloc[0]  # FCF mais recente
        if pd.isna(fcf_base) or fcf_base <= 0:
            return result

        # Obter shares_outstanding se não fornecido
        if shares_outstanding is None or pd.isna(shares_outstanding):
            info = yf.Ticker(ticker_sa).info
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding is None or shares_outstanding <= 0:
                return result

        # Taxa de crescimento histórica
        growth_rate = _compute_fcf_cagr(fcf_series)

        # Projetar FCF para os próximos anos
        projected_fcfs = []
        fcf = fcf_base
        for year in range(1, PROJECTION_YEARS + 1):
            fcf = fcf * (1 + growth_rate)
            projected_fcfs.append(fcf)

        # Valor presente dos FCFs projetados
        pv_fcfs = sum(
            fcf_t / (1 + SELIC) ** t
            for t, fcf_t in enumerate(projected_fcfs, start=1)
        )

        # Valor terminal (Gordon Growth Model)
        terminal_value = (
            projected_fcfs[-1] * (1 + TERMINAL_GROWTH) /
            (SELIC - TERMINAL_GROWTH)
        )
        pv_terminal = terminal_value / (1 + SELIC) ** PROJECTION_YEARS

        # Valor total da empresa
        enterprise_value = pv_fcfs + pv_terminal

        # Preço justo por ação
        fair_price = enterprise_value / shares_outstanding

        result['preco_justo_dcf'] = fair_price if fair_price > 0 else np.nan
        result['growth_rate'] = growth_rate
        result['fcf_base'] = fcf_base

    except Exception as e:
        print(f"[valuation] DCF erro para {ticker_sa}: {e}")

    return result


def compute_sector_averages(df: pd.DataFrame) -> dict:
    """
    Calcula médias de P/L e P/PV por setor.

    Args:
        df: DataFrame com colunas 'setor', 'pl', 'pvp'

    Returns:
        dict {setor: {'avg_pe': float, 'avg_pb': float}}
    """
    # Filtrar apenas valores positivos válidos
    valid = df[(df['pl'] > 0) & (df['pvp'] > 0)].copy()

    sector_stats = {}
    for sector, group in valid.groupby('setor'):
        if not sector or len(group) < 2:
            continue
        sector_stats[sector] = {
            'avg_pe': group['pl'].median(),   # Mediana é mais robusta que média
            'avg_pb': group['pvp'].median(),
        }

    return sector_stats


def graham_valuation(lpa: float, vpa: float,
                     sector_avg_pe: float, sector_avg_pb: float) -> float:
    """
    Calcula preço justo pela fórmula de Graham modificada (com médias setoriais).

    Fórmula: V = sqrt(sector_avg_PE × sector_avg_PB × LPA × VPA)

    A fórmula original de Graham usa 22.5 (= 15 × 1.5), mas aqui substituímos
    pelos valores médios do setor da empresa para maior precisão.

    Args:
        lpa: Lucro por Ação
        vpa: Valor Patrimonial por Ação
        sector_avg_pe: P/L médio do setor
        sector_avg_pb: P/PV médio do setor

    Returns:
        Preço justo de Graham ou NaN se inválido
    """
    if any(pd.isna(v) or v <= 0 for v in [lpa, vpa, sector_avg_pe, sector_avg_pb]):
        return np.nan

    radicand = sector_avg_pe * sector_avg_pb * lpa * vpa
    if radicand <= 0:
        return np.nan

    return np.sqrt(radicand)


def apply_valuation(df: pd.DataFrame, all_fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula valuation DCF e Graham para cada ação do DataFrame e adiciona colunas.

    Args:
        df: DataFrame de ações filtradas
        all_fundamentals: DataFrame completo para cálculo de médias setoriais

    Returns:
        DataFrame com colunas adicionais: preco_justo_dcf, preco_justo_graham,
        margem_seguranca_dcf, margem_seguranca_graham, undervalued
    """
    # Calcular médias setoriais a partir de TODOS os dados (não só os filtrados)
    sector_avgs = compute_sector_averages(all_fundamentals)

    dcf_prices = []
    graham_prices = []

    for _, row in df.iterrows():
        # DCF
        dcf_result = dcf_valuation(row['ticker_sa'], row.get('shares_outstanding'))
        dcf_prices.append(dcf_result['preco_justo_dcf'])

        # Graham
        sector = row.get('setor', '')
        avgs = sector_avgs.get(sector, {})
        avg_pe = avgs.get('avg_pe', np.nan)
        avg_pb = avgs.get('avg_pb', np.nan)

        graham_price = graham_valuation(
            lpa=row.get('lpa', np.nan),
            vpa=row.get('vpa', np.nan),
            sector_avg_pe=avg_pe,
            sector_avg_pb=avg_pb,
        )
        graham_prices.append(graham_price)

    df = df.copy()
    df['preco_justo_dcf'] = dcf_prices
    df['preco_justo_graham'] = graham_prices

    # Margem de segurança: (preço_justo - preço_mercado) / preço_justo × 100
    df['margem_seg_dcf_pct'] = (
        (df['preco_justo_dcf'] - df['preco']) / df['preco_justo_dcf'] * 100
    )
    df['margem_seg_graham_pct'] = (
        (df['preco_justo_graham'] - df['preco']) / df['preco_justo_graham'] * 100
    )

    # Undervalued: preço de mercado abaixo de AMBOS os preços justos
    df['undervalued'] = (
        (df['preco'] < df['preco_justo_dcf']) &
        (df['preco'] < df['preco_justo_graham'])
    )

    n_under = df['undervalued'].sum()
    print(f"[valuation] {n_under}/{len(df)} ações estão abaixo do preço justo (DCF + Graham)")

    return df
