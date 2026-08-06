import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from src.fundamentals import get_fcf_series, resolve_share_count, get_forward_growth

# Carrega variáveis de um .env se python-dotenv estiver instalado. O .env é
# gitignored e guarda os dados que mudam com o tempo e que o yfinance NÃO
# fornece (macro), para você editar manualmente sem tocar no código.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_float(name: str, default: float) -> float:
    """Lê um float da env; usa o default se ausente/vazio/inválido."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[valuation] {name}={raw!r} inválido, usando default {default}")
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Lê um booleano da env ('1', 'true', 'yes', 'on' -> True)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


# Parâmetros de valuation
# Todas as taxas são NOMINAIS (em R$ corrente), coerentes entre si.
# RF e ERP são macro: mudam com o cenário e são editáveis via env (o yfinance
# não fornece esses dados).
RISK_FREE_RATE = _env_float('RISK_FREE_RATE', 0.124)     # Título longo do governo BR (média 5a, fonte SWS)
EQUITY_RISK_PREMIUM = _env_float('EQUITY_RISK_PREMIUM', 0.075)  # Prêmio de risco BR (S&P Global, via SWS)
TERMINAL_GROWTH = RISK_FREE_RATE  # Perpetuidade não pode exceder a economia
# Liga o uso de estimativas forward de crescimento (analistas via yfinance) no
# estágio 1 do DCF. Desligado por padrão -> comportamento = CAGR histórico.
USE_FORWARD_ESTIMATES = _env_bool('USE_FORWARD_ESTIMATES', False)
PROJECTION_YEARS = 10      # Horizonte de projeção (2 estágios: decay linear)
MAX_GROWTH_RATE = 0.20     # Cap de crescimento anual
MIN_GROWTH_RATE = 0.0      # Floor de crescimento
MIN_BETA = 0.5             # Clamp de beta (regressões de small caps produzem outliers)
MAX_BETA = 2.5
MIN_BETA_OBSERVATIONS = 30  # Mínimo de semanas para uma regressão de beta confiável
MIN_MARKET_VARIANCE = 1e-12  # Abaixo disso a série é degenerada (retorno constante)
MIN_SAFETY_MARGIN_PCT = 20.0  # Margem de segurança mínima para "forte desconto" (inspirado SWS)

MARKET_INDEX = '^BVSP'     # Benchmark para o beta (Ibovespa)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta via regressão dos retornos da ação contra o mercado: cov / var.

    Não usamos o campo `beta` do yfinance: para ações brasileiras ele vem
    inutilizável (PETR4 = -0,139, WEGE3 = -0,077), aparentemente medido contra
    um benchmark errado.
    """
    joined = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
    if len(joined) < MIN_BETA_OBSERVATIONS:
        return np.nan

    stock_col, market_col = joined.iloc[:, 0], joined.iloc[:, 1]
    market_var = market_col.var()
    # Comparar com epsilon, não com zero: a variância de uma série constante
    # sai como ~3e-36 (ruído de ponto flutuante), e cov/var devolveria um beta
    # fabricado a partir desse ruído em vez de NaN.
    if pd.isna(market_var) or market_var < MIN_MARKET_VARIANCE:
        return np.nan

    return stock_col.cov(market_col) / market_var


def compute_sector_betas(df: pd.DataFrame) -> dict:
    """
    Mediana do beta por setor, a partir da coluna `beta_raw`.

    Usamos o beta do SETOR, não o da empresa: papéis ilíquidos medem beta
    artificialmente baixo porque simplesmente não negociam (RSUL4 regride 0,155
    negociando R$ 273k/dia — artefato de não-negociação, não baixo risco). A
    Simply Wall St usa beta setorial pelo mesmo motivo.
    """
    if 'beta_raw' not in df.columns:
        return {}

    valid = df[df['beta_raw'].notna()]

    sector_betas = {}
    for sector, group in valid.groupby('setor'):
        if not sector:
            continue
        sector_betas[sector] = float(group['beta_raw'].median())

    return sector_betas


def cost_of_equity(beta: float = None) -> float:
    """
    Custo de capital próprio via CAPM: RF + beta × ERP.

    A versão anterior usava a Selic pura, que é a taxa LIVRE DE RISCO — descontar
    fluxos de equity a ela omite o prêmio de risco inteiro e superestima
    sistematicamente o preço justo.
    """
    if beta is None or pd.isna(beta):
        beta = 1.0
    beta = max(MIN_BETA, min(MAX_BETA, float(beta)))
    return RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM


def compute_fcf_base(fcf_series: pd.Series) -> float:
    """
    Base de FCF para o DCF: mediana da série histórica.

    Usar o ano mais recente ancora a projeção no pico do ciclo (RSUL4: 41,8M no
    último ano vs. mediana de 28,3M). A mediana resiste tanto ao pico quanto ao
    ano negativo isolado.
    """
    if fcf_series.empty:
        return np.nan
    base = float(np.median(fcf_series.values))
    return base if base > 0 else np.nan


def _compute_fcf_cagr(fcf_series: pd.Series) -> float:
    """
    Calcula CAGR do Free Cash Flow a partir da série histórica.
    A série vem do yfinance com o mais recente primeiro.

    Qualquer ano negativo zera o crescimento: uma série que passa pelo prejuízo
    não sustenta extrapolação de crescimento composto. A versão anterior pulava
    os negativos e media apenas entre os pontos positivos, o que transformava a
    série cíclica da RSUL4 (21,2M → -9,8M → 41,8M) em "25% a.a.".
    """
    if len(fcf_series) < 2:
        return 0.0

    # Ordenar do mais antigo ao mais recente
    values = fcf_series.values[::-1]

    if any(v <= 0 for v in values):
        return 0.0

    first, last = values[0], values[-1]
    n_years = len(values) - 1

    cagr = (last / first) ** (1 / n_years) - 1

    # Aplicar limites
    return max(MIN_GROWTH_RATE, min(MAX_GROWTH_RATE, cagr))


def discount_fcf_to_equity(fcf_base: float, growth: float, discount_rate: float,
                           shares: float, terminal_growth: float = TERMINAL_GROWTH,
                           years: int = PROJECTION_YEARS) -> float:
    """
    DCF 2-estágios sobre Free Cash Flow ALAVANCADO, retornando preço por ação.

    O 'Free Cash Flow' do yfinance é OCF − CapEx, e o OCF já é líquido de juros
    pagos: é FCFE, não FCFF. Portanto o valor presente já é o equity value e NÃO
    se subtrai dívida líquida dele — o código anterior subtraía, e como a RSUL4
    tem caixa líquido (dívida líquida negativa) isso somava R$ 4,84/ação.

    Estágio 1: taxa decai linearmente de `growth` até `terminal_growth`.
    Estágio 2: perpetuidade de Gordon a `terminal_growth`.
    """
    if any(pd.isna(x) for x in [fcf_base, growth, discount_rate, shares, terminal_growth]):
        return np.nan
    if fcf_base <= 0 or shares <= 0:
        return np.nan
    if discount_rate <= terminal_growth:
        return np.nan

    projected = []
    fcf = fcf_base
    for year in range(1, years + 1):
        if years > 1:
            rate = growth - (growth - terminal_growth) * (year - 1) / (years - 1)
        else:
            rate = growth
        fcf = fcf * (1 + rate)
        projected.append(fcf)

    pv_fcfs = sum(f / (1 + discount_rate) ** t for t, f in enumerate(projected, start=1))

    terminal_value = projected[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** years

    equity_value = pv_fcfs + pv_terminal
    fair_price = equity_value / shares

    return fair_price if fair_price > 0 else np.nan


def dcf_valuation(ticker_sa: str, shares_total: float = None,
                  beta: float = None, forward_growth: float = None) -> dict:
    """
    Calcula preço justo via DCF de 2 estágios sobre FCF alavancado (estilo SWS).

    Estágio 1: taxa decai linearmente do crescimento inicial até TERMINAL_GROWTH
    ao longo de PROJECTION_YEARS anos.
    Estágio 2: perpetuidade de Gordon a TERMINAL_GROWTH.

    Crescimento inicial: por padrão é o CAGR histórico do FCF. Com
    USE_FORWARD_ESTIMATES ligado, usa a estimativa forward de analistas
    (yfinance) quando disponível, caindo de volta no CAGR histórico se não for.

    O 'Free Cash Flow' do yfinance é OCF − CapEx, com o OCF já líquido de juros
    pagos: é FCFE. O valor presente já é o equity value, sem ajuste de dívida.

    Args:
        ticker_sa: Ticker com sufixo .SA
        shares_total: Total de ações (ON + PN). Se None, resolve via yfinance.
        beta: Beta da ação. Se None, busca do yfinance (fallback 1,0).
        forward_growth: Crescimento forward já resolvido (decimal). Se None e
            USE_FORWARD_ESTIMATES estiver ligado, busca via get_forward_growth.

    Returns:
        dict com 'preco_justo_dcf', 'growth_rate', 'fcf_base', 'cost_of_equity',
        'growth_source' ('forward' | 'historical').
    """
    result = {
        'preco_justo_dcf': np.nan,
        'growth_rate': np.nan,
        'fcf_base': np.nan,
        'cost_of_equity': np.nan,
        'growth_source': 'historical',
    }

    try:
        fcf_series = get_fcf_series(ticker_sa)
        if fcf_series.empty:
            return result

        fcf_base = compute_fcf_base(fcf_series)
        if pd.isna(fcf_base):
            return result

        if (shares_total is None or pd.isna(shares_total)
                or beta is None or pd.isna(beta)):
            info = yf.Ticker(ticker_sa).info or {}
            if shares_total is None or pd.isna(shares_total):
                shares_total = resolve_share_count(info)
            if beta is None or pd.isna(beta):
                beta = info.get('beta')

        if shares_total is None or pd.isna(shares_total) or shares_total <= 0:
            return result

        coe = cost_of_equity(beta)

        # Crescimento inicial: forward (analistas) se ligado e disponível,
        # senão CAGR histórico.
        initial_growth = _compute_fcf_cagr(fcf_series)
        growth_source = 'historical'
        if USE_FORWARD_ESTIMATES:
            if forward_growth is None:
                forward_growth = get_forward_growth(ticker_sa)
            if forward_growth is not None and pd.notna(forward_growth):
                initial_growth = max(
                    MIN_GROWTH_RATE, min(MAX_GROWTH_RATE, float(forward_growth))
                )
                growth_source = 'forward'

        result['preco_justo_dcf'] = discount_fcf_to_equity(
            fcf_base=fcf_base,
            growth=initial_growth,
            discount_rate=coe,
            shares=shares_total,
        )
        result['growth_rate'] = initial_growth
        result['fcf_base'] = fcf_base
        result['cost_of_equity'] = coe
        result['growth_source'] = growth_source

    except Exception as e:
        print(f"[valuation] DCF erro para {ticker_sa}: {e}")

    return result


def excess_returns_valuation(roe_decimal: float, vpa: float,
                             coe: float = None,
                             terminal_growth: float = TERMINAL_GROWTH) -> float:
    """
    Calcula preço justo pelo modelo de Excess Returns (usado para bancos).
    Inspirado no modelo da Simply Wall St para instituições financeiras.

    Fórmula:
        excess_return = (ROE - CoE) × VPA
        terminal_value = excess_return / (CoE - g)
        fair_value = VPA + terminal_value

    Args:
        roe_decimal: ROE como decimal (ex: 0.15 para 15%)
        vpa: Valor Patrimonial por Ação
        coe: Custo de capital próprio. Se None, usa CAPM com beta 1,0.
        terminal_growth: Taxa de crescimento perpétuo (default: TERMINAL_GROWTH)

    Returns:
        Preço justo por ação ou NaN se inválido
    """
    if coe is None:
        coe = cost_of_equity()
    if any(pd.isna(v) for v in [roe_decimal, vpa, coe, terminal_growth]):
        return np.nan
    if vpa <= 0 or roe_decimal <= coe:
        return np.nan
    if coe <= terminal_growth:
        return np.nan

    excess_return = (roe_decimal - coe) * vpa
    terminal_value = excess_return / (coe - terminal_growth)
    fair_value = vpa + terminal_value

    return fair_value if fair_value > 0 else np.nan


def ddm_valuation(dps: float, discount_rate: float = None,
                  growth_rate: float = TERMINAL_GROWTH) -> float:
    """
    Calcula preço justo pelo Dividend Discount Model (Gordon Growth).
    Usado como fallback quando DCF não tem dados de FCF.

    Fórmula: V = DPS / (discount_rate - growth_rate)

    Args:
        dps: Dividendo por ação anual (R$)
        discount_rate: Taxa de desconto. Se None, usa CAPM com beta 1,0.
        growth_rate: Taxa de crescimento dos dividendos (default: TERMINAL_GROWTH)

    Returns:
        Preço justo por ação ou NaN se inválido
    """
    if discount_rate is None:
        discount_rate = cost_of_equity()
    if pd.isna(dps) or dps <= 0:
        return np.nan
    if pd.isna(discount_rate) or pd.isna(growth_rate):
        return np.nan
    if discount_rate <= growth_rate:
        return np.nan

    return dps / (discount_rate - growth_rate)


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


def apply_valuation(df: pd.DataFrame, all_fundamentals: pd.DataFrame,
                    model: str = 'stock') -> pd.DataFrame:
    """
    Calcula valuation para cada ação/banco do DataFrame e adiciona colunas.

    Modelos disponíveis:
    - 'stock': DCF 2-estágios (primário), DDM fallback, Graham (secundário)
    - 'bank': Excess Returns (primário), Graham (secundário)

    Args:
        df: DataFrame de ações/bancos filtrados
        all_fundamentals: DataFrame completo para cálculo de médias setoriais
        model: 'stock' ou 'bank'

    Returns:
        DataFrame com colunas: preco_justo_primario, preco_justo_graham,
        margem_seg_primario_pct, margem_seg_graham_pct, undervalued, forte_desconto
    """
    sector_avgs = compute_sector_averages(all_fundamentals)
    sector_betas = compute_sector_betas(all_fundamentals)

    primary_prices = []
    graham_prices = []
    coes = []
    methods = []
    growth_sources = []

    for _, row in df.iterrows():
        # Beta do SETOR, não da empresa: small caps ilíquidas medem beta
        # artificialmente baixo por não negociarem (ver compute_sector_betas).
        beta = sector_betas.get(row.get('setor', ''), np.nan)
        coe = cost_of_equity(beta)
        coes.append(coe)

        growth_source = ''

        # --- Modelo primário ---
        if model == 'bank':
            # Excess Returns para bancos
            roe_decimal = row.get('roe_pct', np.nan)
            if pd.notna(roe_decimal):
                roe_decimal = roe_decimal / 100.0
            vpa = row.get('vpa', np.nan)
            primary_price = excess_returns_valuation(roe_decimal, vpa, coe=coe)
            method = 'excess_returns' if pd.notna(primary_price) else 'none'
        else:
            # DCF 2-estágios para ações
            dcf_result = dcf_valuation(
                row['ticker_sa'],
                row.get('shares_total'),
                beta,
            )
            primary_price = dcf_result['preco_justo_dcf']

            if pd.notna(primary_price):
                method = 'dcf'
                growth_source = dcf_result['growth_source']
            else:
                # Fallback DDM quando o DCF não é aplicável (ex.: FCF histórico
                # negativo em incorporadoras). Rotulado explicitamente para não
                # se confundir com um DCF de verdade.
                dps = row.get('dividend_rate', np.nan)
                primary_price = ddm_valuation(dps, discount_rate=coe)
                method = 'ddm' if pd.notna(primary_price) else 'none'

        primary_prices.append(primary_price)
        methods.append(method)
        growth_sources.append(growth_source)

        # --- Graham (secundário, igual para ambos) ---
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
    df['preco_justo_dcf'] = primary_prices
    df['metodo_valuation'] = methods
    df['growth_source'] = growth_sources
    df['preco_justo_graham'] = graham_prices
    df['cost_of_equity_pct'] = [c * 100 for c in coes]

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

    # Margem de segurança média
    df['margem_seg_media_pct'] = (
        df[['margem_seg_dcf_pct', 'margem_seg_graham_pct']].mean(axis=1)
    )

    # Forte desconto: margem média >= 20% (inspirado SWS)
    df['forte_desconto'] = df['margem_seg_media_pct'] >= MIN_SAFETY_MARGIN_PCT

    label = 'bancos' if model == 'bank' else 'ações'
    n_under = df['undervalued'].sum()
    n_forte = df['forte_desconto'].sum()
    print(f"[valuation] {n_under}/{len(df)} {label} abaixo do preço justo | "
          f"{n_forte} com forte desconto (≥{MIN_SAFETY_MARGIN_PCT:.0f}%)")

    return df


# Histórico de valuation: uma linha por (data, ticker), append-only, para
# medir o drift do preço justo conforme fundamentos/estimativas/macro mudam.
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
VALUATION_HISTORY = DATA_DIR / 'valuation_history.csv'

# Colunas do resultado que são snapshotadas (as que existirem no df).
_SNAPSHOT_RESULT_COLS = [
    'tipo', 'ticker', 'nome', 'setor', 'preco',
    'preco_justo_dcf', 'metodo_valuation', 'growth_source',
    'preco_justo_graham', 'margem_seg_dcf_pct', 'margem_seg_graham_pct',
    'margem_seg_media_pct', 'undervalued', 'forte_desconto',
    'cost_of_equity_pct',
    'crescimento_receita_pct', 'crescimento_lucro_pct', 'num_analistas',
]


def append_snapshot(df: pd.DataFrame, path: Path = None,
                    snapshot_date: str = None) -> Path:
    """
    Anexa o resultado de valuation ao histórico append-only.

    Cada linha carrega, além do preço justo e da margem, as PREMISSAS usadas na
    rodada (RF, ERP, crescimento terminal, flag forward) — assim uma divergência
    futura pode ser atribuída a mudança de dado ou de premissa.

    Args:
        df: DataFrame vindo de apply_valuation (opcionalmente com 'tipo').
        path: Destino. Default: data/valuation_history.csv.
        snapshot_date: Data ISO (YYYY-MM-DD). Default: hoje.

    Returns:
        O Path do arquivo escrito.
    """
    path = Path(path) if path is not None else VALUATION_HISTORY
    if snapshot_date is None:
        snapshot_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    if df is None or len(df) == 0:
        print("[valuation] snapshot vazio, nada a gravar")
        return path

    cols = [c for c in _SNAPSHOT_RESULT_COLS if c in df.columns]
    snap = df[cols].copy()
    snap.insert(0, 'data_snapshot', snapshot_date)

    # Premissas da rodada (mesmas para todas as linhas).
    snap['risk_free_rate'] = RISK_FREE_RATE
    snap['equity_risk_premium'] = EQUITY_RISK_PREMIUM
    snap['terminal_growth'] = TERMINAL_GROWTH
    snap['use_forward_estimates'] = USE_FORWARD_ESTIMATES

    path.parent.mkdir(parents=True, exist_ok=True)
    n_linhas = len(snap)
    # Reescreve em vez de 'mode=a': quando o conjunto de colunas muda entre
    # rodadas, um append cru grava os valores fora das colunas do header antigo.
    # O concat alinha por nome e preenche o que falta com NaN.
    # ponytail: reescreve o arquivo inteiro; só vira problema se o histórico
    # crescer a ponto de não caber em memória (dezenas de linhas por rodada).
    if path.exists():
        snap = pd.concat([pd.read_csv(path), snap], ignore_index=True)
    snap.to_csv(path, index=False)

    print(f"[valuation] snapshot de {n_linhas} linhas anexado a {path} "
          f"(data={snapshot_date})")
    return path
