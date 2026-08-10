import json
from pathlib import Path
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'filters.json'


def _load_config() -> dict:
    """Carrega configuração de filtros do JSON."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def _growth_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Máscara de estimativas de analistas.

    Três flags independentes, cada uma ligando o seu próprio critério:
    `exigir_estimativa` aplica os cortes de crescimento de receita e lucro;
    `exigir_num_analistas` aplica o mínimo de analistas; `exigir_lpa_estimado`
    aplica o corte sobre o nível de lucro por ação projetado. Nenhuma altera o
    significado da outra, e flag desligada significa critério não aplicado.

    A guarda do `lpa_estimado` existe porque `crescimento_lucro_pct` é
    estimativa sobre estimativa (o `yearAgoEps` da linha '+1y' do yfinance é a
    estimativa do exercício corrente, não o lucro realizado). Com as duas
    negativas a razão fica positiva, então um prejuízo encolhendo passa no
    corte de crescimento: a AURE3 projeta -1,25 -> -0,14 por ação e aparece
    como +88,6%. Comparar o NÍVEL contra zero é o que separa lucro crescendo de
    prejuízo encolhendo — a variação sozinha não separa.

    Com a flag ligada, valor NaN reprova: sem dado não há como atestar o
    critério, e é justamente isso que a flag exige. Comparações do pandas com
    NaN já retornam False, então esse comportamento sai de graça.

    Args:
        df: DataFrame com as colunas `crescimento_receita_pct`,
            `crescimento_lucro_pct`, `lpa_estimado` e `num_analistas`.
        cfg: bloco de configuração (`stock_filters` ou `bank_filters`).

    Returns:
        Série booleana indexada como `df`.
    """
    mask = pd.Series(True, index=df.index)

    if cfg.get('exigir_estimativa'):
        mask &= (
            (df['crescimento_receita_pct'] > cfg['crescimento_receita_pct_min']) &
            (df['crescimento_lucro_pct'] > cfg['crescimento_lucro_pct_min'])
        )

    # '>=' porque é contagem: num_analistas_min = 2 significa "pelo menos dois".
    if cfg.get('exigir_num_analistas'):
        mask &= df['num_analistas'] >= cfg['num_analistas_min']

    # '>' estrito, como os demais cortes `_min` do projeto: LPA projetado
    # exatamente zero não é lucro.
    if cfg.get('exigir_lpa_estimado'):
        mask &= df['lpa_estimado'] > cfg['lpa_estimado_min']

    return mask


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

    growth = _growth_mask(df, cfg)
    filtered = df[mask & growth].copy().reset_index(drop=True)

    # Quantas passaram nos fundamentos e caíram só pelo crescimento projetado
    por_crescimento = int((mask & ~growth).sum())
    print(f"[filters] Ações: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_crescimento} reprovadas por crescimento projetado)")
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

    growth = _growth_mask(df, cfg)
    filtered = df[mask & growth].copy().reset_index(drop=True)

    por_crescimento = int((mask & ~growth).sum())
    print(f"[filters] Bancos: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_crescimento} reprovados por crescimento projetado)")
    return filtered
