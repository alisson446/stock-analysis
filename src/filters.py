import json
import pandas as pd

from src import paths


def _load_config(region: str) -> dict:
    """
    Carrega os filtros da região a partir de `config/<regiao>/filters.json`.

    Região sem arquivo é erro, não default: filtrar por critérios que ninguém
    escolheu produz uma lista que parece válida e não é. A mensagem nomeia o
    caminho para o usuário saber exatamente qual arquivo criar.
    """
    path = paths.filters_file(region)
    if not path.exists():
        raise FileNotFoundError(
            f"filtros da região {region!r} não encontrados em {path}. "
            f"Crie o arquivo antes de rodar o screener nessa região."
        )
    with open(path, 'r') as f:
        return json.load(f)


# A chave do config decide QUAL coluna de liquidez filtrar, e o sufixo carrega
# a moeda: `liq_media_diaria_bdr_min` está em reais mesmo num arquivo de região
# estrangeira, porque o que se negocia é o BDR em B3.
_LIQUIDEZ_POR_CHAVE = {
    'liq_media_diaria_min': 'liq_media_diaria',
    'liq_media_diaria_bdr_min': 'liq_media_diaria_bdr',
}


def _liquidity_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Máscara do corte de liquidez, na coluna que a chave do config indica."""
    for chave, coluna in _LIQUIDEZ_POR_CHAVE.items():
        if chave in cfg:
            return df[coluna] > cfg[chave]
    raise KeyError(
        f"config sem chave de liquidez. Use uma de: "
        f"{sorted(_LIQUIDEZ_POR_CHAVE)}"
    )


def _estimates_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Máscara de estimativas de analistas.

    Três flags independentes, cada uma ligando o seu próprio critério:
    `exigir_estimativa` aplica os cortes de crescimento de receita e lucro;
    `exigir_num_analistas` aplica o mínimo de analistas; `exigir_lpa_estimado`
    aplica o corte sobre o nível de lucro por ação projetado. Nenhuma altera o
    significado da outra, e flag desligada significa critério não aplicado.

    A guarda do `lpa_estimado` existe porque `crescimento_lucro_pct` é
    estimativa sobre estimativa: compara a estimativa do próximo exercício com
    a do exercício corrente, não com o lucro realizado. Com as duas negativas
    a razão fica positiva, então um prejuízo encolhendo passa no corte de
    crescimento: a AURE3 projeta -1,25 -> -0,14 por ação e aparece como
    +88,6%. Comparar o NÍVEL contra zero é o que separa lucro crescendo de
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


def apply_stock_filters(df: pd.DataFrame, region: str = 'br') -> pd.DataFrame:
    """
    Aplica critérios fundamentalistas para ações não-bancárias.
    Os limites são lidos de config/<region>/filters.json (chave 'stock_filters').
    """
    cfg = _load_config(region)['stock_filters']

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
        _liquidity_mask(df, cfg) &
        (df['lpa'] > cfg['lpa_min'])
    )

    estimates = _estimates_mask(df, cfg)
    filtered = df[mask & estimates].copy().reset_index(drop=True)

    # Quantas passaram nos fundamentos e caíram só pelas estimativas de
    # analistas (crescimento de receita/lucro, nº de analistas ou LPA
    # projetado — não é só crescimento, ver _estimates_mask)
    por_estimativa = int((mask & ~estimates).sum())
    print(f"[filters] Ações: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_estimativa} reprovadas por estimativas de analistas)")
    return filtered


def apply_bank_filters(df: pd.DataFrame, region: str = 'br') -> pd.DataFrame:
    """
    Aplica critérios de screening adaptados para bancos.
    Os limites são lidos de config/<region>/filters.json (chave 'bank_filters').
    """
    cfg = _load_config(region)['bank_filters']

    mask = (
        (df['pl'] > cfg['pl_min']) & (df['pl'] <= cfg['pl_max']) &
        (df['pvp'] > cfg['pvp_min']) & (df['pvp'] <= cfg['pvp_max']) &
        (df['roe_pct'] > cfg['roe_pct_min']) &
        (df['margem_liquida_pct'] > cfg['margem_liquida_pct_min']) &
        (df['lpa'] > cfg['lpa_min']) &
        _liquidity_mask(df, cfg)
    )

    # dy_pct é opcional por região: fora do Brasil o dividendo sofre retenção
    # na fonte antes de chegar ao detentor do BDR, então o dividendYield do
    # yfinance é o rendimento do acionista local, não o seu. Filtrar por um
    # número inflado é pior que não filtrar.
    if 'dy_pct_min' in cfg:
        mask &= df['dy_pct'] > cfg['dy_pct_min']

    estimates = _estimates_mask(df, cfg)
    filtered = df[mask & estimates].copy().reset_index(drop=True)

    por_estimativa = int((mask & ~estimates).sum())
    print(f"[filters] Bancos: {len(filtered)}/{len(df)} passaram nos critérios"
          f" ({por_estimativa} reprovados por estimativas de analistas)")
    return filtered
