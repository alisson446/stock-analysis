import json
from pathlib import Path
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'filters.json'


def _load_config() -> dict:
    """Carrega configuração de filtros do JSON."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def apply_stock_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios fundamentalistas para ações não-bancárias.
    Os limites são lidos de config/filters.json (chave 'stock_filters').
    """
    cfg = _load_config()['stock_filters']

    mask = (
        (df['pl'] > cfg['pl_min']) & (df['pl'] <= cfg['pl_max']) &
        (df['pvp'] > cfg['pvp_min']) & (df['pvp'] <= cfg['pvp_max']) &
        (df['margem_ebit_pct'] > cfg['margem_ebit_pct_min']) &
        (df['margem_liquida_pct'] > cfg['margem_liquida_pct_min']) &
        (df['dl_ebit'] < cfg['dl_ebit_max']) &
        (df['dl_pl'] < cfg['dl_pl_max']) &
        (df['roe_pct'] > cfg['roe_pct_min']) &
        (df['liquidez_corrente'] > cfg['liquidez_corrente_min']) &
        (df['passivos_ativos'] < cfg['passivos_ativos_max']) &
        (df['liq_media_diaria'] > cfg['liq_media_diaria_min']) &
        (df['lpa'] > cfg['lpa_min'])
    )

    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[filters] Ações: {len(filtered)}/{len(df)} passaram nos critérios")
    return filtered


def apply_bank_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios de screening adaptados para bancos.
    Os limites são lidos de config/filters.json (chave 'bank_filters').
    """
    cfg = _load_config()['bank_filters']

    mask = (
        (df['pl'] > cfg['pl_min']) & (df['pl'] <= cfg['pl_max']) &
        (df['pvp'] > cfg['pvp_min']) & (df['pvp'] <= cfg['pvp_max']) &
        (df['roe_pct'] > cfg['roe_pct_min']) &
        (df['margem_liquida_pct'] > cfg['margem_liquida_pct_min']) &
        (df['lpa'] > cfg['lpa_min']) &
        (df['liq_media_diaria'] > cfg['liq_media_diaria_min']) &
        (df['dy_pct'] > cfg['dy_pct_min'])
    )

    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[filters] Bancos: {len(filtered)}/{len(df)} passaram nos critérios")
    return filtered
