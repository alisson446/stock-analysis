import pandas as pd


def apply_stock_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os 11 critérios fundamentalistas para ações não-bancárias.

    Critérios:
        1. P/L entre 0 e 10
        2. P/PV entre 0 e 1,5
        3. Margem EBIT positiva
        4. Margem líquida > 10%
        5. Dívida Líquida/EBIT < 3
        6. Dívida Líquida/PL < 2
        7. ROE > 10%
        8. Liquidez corrente > 1
        9. Passivos/Ativos < 1
       10. Liquidez média diária > R$ 100.000
       11. LPA > 0
    """
    mask = (
        (df['pl'] > 0) & (df['pl'] <= 10) &
        (df['pvp'] > 0) & (df['pvp'] <= 1.5) &
        (df['margem_ebit_pct'] > 0) &
        (df['margem_liquida_pct'] > 10) &
        (df['dl_ebit'] < 3) &
        (df['dl_pl'] < 2) &
        (df['roe_pct'] > 10) &
        (df['liquidez_corrente'] > 1) &
        (df['passivos_ativos'] < 1) &
        (df['liq_media_diaria'] > 100_000) &
        (df['lpa'] > 0)
    )

    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[filters] Ações: {len(filtered)}/{len(df)} passaram nos 11 critérios")
    return filtered


def apply_bank_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios de screening adaptados para bancos.

    Critérios:
        1. P/L entre 0 e 10
        2. P/PV entre 0 e 2,0
        3. ROE > 15%
        4. Margem líquida > 10%
        5. LPA > 0
        6. Liquidez média diária > R$ 100.000
        7. Dividend Yield > 3%
    """
    mask = (
        (df['pl'] > 0) & (df['pl'] <= 10) &
        (df['pvp'] > 0) & (df['pvp'] <= 2.0) &
        (df['roe_pct'] > 15) &
        (df['margem_liquida_pct'] > 10) &
        (df['lpa'] > 0) &
        (df['liq_media_diaria'] > 100_000) &
        (df['dy_pct'] > 3)
    )

    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[filters] Bancos: {len(filtered)}/{len(df)} passaram nos critérios")
    return filtered
