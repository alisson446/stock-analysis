import pandas as pd
import numpy as np
import yfinance as yf
import time
from pathlib import Path
from tqdm import tqdm

from src import paths


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


def resolve_share_count(info: dict, balance_shares: float = None) -> float:
    """
    Retorna o número TOTAL de ações da empresa (ON + PN).

    `sharesOutstanding` do yfinance traz apenas a classe do ticker cotado — para
    RSUL4 (PN) retorna 2.495.225, enquanto a Riosulense tem 6.072.128 no total.
    Como o DCF produz o equity value da empresa inteira, dividir pela classe
    isolada inflava o preço justo em 2,4x.
    """
    for candidate in (info.get('impliedSharesOutstanding'),
                      balance_shares,
                      info.get('sharesOutstanding')):
        if candidate is not None and pd.notna(candidate) and candidate > 0:
            return float(candidate)
    return np.nan


# Lucro líquido atribuível aos controladores, do mais específico ao mais amplo.
# Nunca usamos 'Net Income Including Noncontrolling Interests' — para holdings
# (EVEN3: R$273,8M com minoritários vs R$216,3M dos controladores) isso inflaria
# o LPA. Os sites BR (Status Invest/Fundamentus) usam o lucro dos controladores.
_NET_INCOME_COMMON_LABELS = ('Net Income Common Stockholders', 'Net Income')


def compute_ttm_net_income(info: dict, quarterly_income) -> float:
    """
    Lucro líquido dos últimos 12 meses (TTM) atribuível aos controladores.

    O `trailingEps`/`trailingPE` do `.info` usam uma contagem de ações errada
    para várias ações BR — EVEN3: o trailingEps implica ~109M ações vs 198M
    reais; SEER3: ~34M vs 128M — subestimando o P/L em 2 a 4x (EVEN3 aparece
    como 2,49 quando o real é 4,55; SEER3 como 1,58 vs 5,95). Por isso
    recalculamos o P/L a partir deste lucro dividido pelo total de ações
    (`resolve_share_count`), em vez de confiar no número pronto do Yahoo.

    Prioridade:
        1. `netIncomeToCommon` do `.info` (já é TTM e dos controladores);
        2. soma dos últimos 4 trimestres de `quarterly_income_stmt`.

    Retorna NaN quando nenhuma fonte confiável existe — nunca caímos de volta no
    `trailingEps` do Yahoo, que é justamente o valor defeituoso.
    """
    v = info.get('netIncomeToCommon')
    if v is not None and pd.notna(v) and v != 0:
        return float(v)

    for label in _NET_INCOME_COMMON_LABELS:
        series = _extract_financial_series(quarterly_income, [label])
        if len(series) >= 4:
            return float(series.iloc[:4].sum())

    return np.nan


# Período das estimativas de analistas. A Yahoo só entrega '0q', '+1q', '0y'
# e '+1y' — o '+1y' (próximo exercício) é o horizonte mais longo disponível.
# A linha 'LTG' existe no schema mas vem NaN para a ação em todos os tickers
# testados, BR e US.
_ESTIMATE_PERIOD = '+1y'


def _estimate_cell(frame, column: str) -> float:
    """
    Lê uma célula da linha '+1y' de um frame de estimativas do yfinance.

    Defensiva por design: layout e cobertura variam entre tickers e versões
    do yfinance, e uma exceção aqui interromperia a coleta dos 372 tickers.
    """
    try:
        if frame is None or frame.empty or _ESTIMATE_PERIOD not in frame.index:
            return np.nan
        value = frame.loc[_ESTIMATE_PERIOD, column]
        return float(value) if pd.notna(value) else np.nan
    except Exception:
        return np.nan


def _extract_analyst_estimates(stock) -> tuple[float, float, float, float]:
    """
    Estimativas de analistas para o próximo exercício.

    Usa `revenue_estimate` e `earnings_estimate`, que saem do mesmo módulo
    `earningsTrend` do quoteSummary e ficam em cache no objeto Ticker: acessar
    as duas custa 1 requisição HTTP. Não usar `growth_estimates`, que dispara
    uma segunda requisição para buscar dados de índice que não consumimos.

    O `lpa_estimado` é o NÍVEL de lucro por ação projetado (coluna `avg`, em R$
    por ação), não uma variação — sai do mesmo frame já em memória, então não
    custa requisição nenhuma.

    Ele existe porque o `crescimento_lucro_pct` ao lado é estimativa sobre
    estimativa: o `yearAgoEps` da linha '+1y' é o `avg` da linha '0y', não o
    lucro realizado. Com as duas estimativas negativas a razão fica POSITIVA, e
    um prejuízo encolhendo vira "crescimento de lucro" — a AURE3 projeta -1,25
    -> -0,14 por ação e isso aparece como +88,6%. Sem o nível ao lado, isso é
    indistinguível de lucro crescendo.

    O número de analistas vem do `earnings_estimate` — o `revenue_estimate` não
    expõe contagem — e governa as duas métricas de crescimento.

    Args:
        stock: objeto `yf.Ticker` já construído.

    Returns:
        (crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado,
        num_analistas). Os dois primeiros em pontos percentuais, o terceiro em
        R$ por ação (sem escalar), o último em contagem. Qualquer campo
        indisponível vem como NaN.
    """
    try:
        revenue = stock.revenue_estimate
        earnings = stock.earnings_estimate
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    return (
        _estimate_cell(revenue, 'growth') * 100,
        _estimate_cell(earnings, 'growth') * 100,
        _estimate_cell(earnings, 'avg'),
        _estimate_cell(earnings, 'numberOfAnalysts'),
    )


def fetch_betas(tickers_sa: list[str], index_symbol: str = '^BVSP',
                period: str = '5y') -> pd.Series:
    """
    Calcula o beta de cada ticker por regressão dos retornos semanais contra o
    Ibovespa, em um único download em lote.

    O campo `beta` do yfinance é inutilizável para ações brasileiras
    (PETR4 = -0,139, WEGE3 = -0,077), por isso regredimos nós mesmos.

    Returns:
        Series indexada por ticker_sa com o beta (NaN quando indeterminável)
    """
    from src.valuation import compute_beta

    symbols = list(dict.fromkeys(list(tickers_sa) + [index_symbol]))
    px = yf.download(symbols, period=period, interval='1wk',
                     progress=False, auto_adjust=True)['Close']

    returns = px.pct_change()
    if index_symbol not in returns.columns:
        print(f"[fundamentals] {index_symbol} indisponível — betas não calculados")
        return pd.Series(dtype=float)

    market = returns[index_symbol].dropna()

    betas = {}
    for ticker_sa in tickers_sa:
        if ticker_sa not in returns.columns:
            betas[ticker_sa] = np.nan
            continue
        betas[ticker_sa] = compute_beta(returns[ticker_sa].dropna(), market)

    result = pd.Series(betas, name='beta_raw')
    print(f"[fundamentals] beta calculado vs {index_symbol} para "
          f"{result.notna().sum()}/{len(result)} tickers")
    return result


def fetch_fundamentals(tickers_sa: list[str], delay: float = 0.5,
                       force_refresh: bool = False, region: str = 'br',
                       index_symbol: str = '^BVSP',
                       incluir_liq_local: bool = True) -> pd.DataFrame:
    """
    Coleta dados fundamentalistas de cada ticker via yfinance.
    Usa cache local (data/<region>/fundamentals.csv) se existir.

    Args:
        tickers_sa: Lista de tickers como o yfinance os conhece. Para 'br' é
            com sufixo .SA ('PETR4.SA'); para 'us' é o ticker puro ('AAPL').
        delay: Tempo de espera entre requisições (segundos)
        force_refresh: Se True, ignora cache e busca dados novos
        region: Pasta de dados de destino
        index_symbol: Índice contra o qual o beta é regredido. '^BVSP' para
            papel brasileiro, '^GSPC' para americano — o beta precisa medir o
            papel contra o mercado dele, não contra outro.
        incluir_liq_local: Se False, a coluna `liq_media_diaria` não é gravada.
            É o caso da região alcançada via BDR: a liquidez do subjacente em
            Nova York não é negociável daqui, e deixá-la no arquivo criaria
            duas colunas quase homônimas em moedas diferentes — sendo a SEM
            sufixo a estrangeira, justamente a que se assume ser a principal.

    Returns:
        DataFrame com todas as métricas calculadas
    """
    cache = paths.data_file(region, 'fundamentals.csv')
    if not force_refresh and cache.exists():
        df = pd.read_csv(cache)
        # Linhas gravadas antes da coluna existir são todas brasileiras.
        if 'moeda' not in df.columns:
            df['moeda'] = 'BRL'
        print(f"[fundamentals] {len(df)} tickers carregados do cache ({cache})")
        return df

    return _fetch_fundamentals_from_api(tickers_sa, delay, cache, index_symbol,
                                        incluir_liq_local)


def _fetch_fundamentals_from_api(tickers_sa: list[str], delay: float, cache: Path,
                                 index_symbol: str,
                                 incluir_liq_local: bool) -> pd.DataFrame:
    """Busca dados fundamentalistas via yfinance e salva em cache."""
    records = []

    for ticker_sa in tqdm(tickers_sa, desc="Coletando fundamentals"):
        try:
            stock = yf.Ticker(ticker_sa)
            info = stock.info or {}

            # Dados básicos do .info. P/L e LPA NÃO vêm do .info: o trailingPE/
            # trailingEps do Yahoo usam contagem de ações errada em várias ações
            # BR e são recalculados abaixo (ver compute_ttm_net_income).
            current_price = _safe_get(info, 'currentPrice')
            pb_ratio = _safe_get(info, 'priceToBook')
            profit_margin = _safe_get(info, 'profitMargins')
            roe = _safe_get(info, 'returnOnEquity')
            current_ratio = _safe_get(info, 'currentRatio')
            avg_volume = _safe_get(info, 'averageDailyVolume10Day')
            shares_outstanding = _safe_get(info, 'sharesOutstanding')
            sector = _safe_get(info, 'sector', '')
            industry = _safe_get(info, 'industry', '')
            company_name = _safe_get(info, 'shortName', ticker_sa)
            # Moeda do BALANÇO, não a do pregão. É ela que rege lucro, EBIT,
            # FCF e o preço justo — e é a que precisa bater com a do preço
            # para o P/L recalculado significar alguma coisa.
            moeda = _safe_get(info, 'financialCurrency', '')
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
            financials = stock.financials              # DRE anual
            quarterly_income = stock.quarterly_income_stmt  # DRE trimestral (TTM)
            balance = stock.balance_sheet              # Balanço Patrimonial
            cashflow = stock.cashflow                  # Fluxo de Caixa

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

            # Total de ações (ON + PN). sharesOutstanding traz só a classe do
            # ticker cotado, o que subestima o denominador do DCF em empresas
            # com duas classes (RSUL4: 2,5M PN vs 6,07M no total).
            balance_shares = _extract_financial_value(balance, [
                'Ordinary Shares Number', 'Share Issued'
            ])
            shares_total = resolve_share_count(info, balance_shares)

            # P/L e LPA recalculados: lucro TTM dos controladores / total de ações.
            # Substitui o trailingPE/trailingEps do Yahoo, que erra a contagem de
            # ações em várias ações BR (ver compute_ttm_net_income).
            lucro_ttm = compute_ttm_net_income(info, quarterly_income)
            eps = (lucro_ttm / shares_total
                   if pd.notna(lucro_ttm) and pd.notna(shares_total) and shares_total != 0
                   else np.nan)
            pe_ratio = (current_price / eps
                        if pd.notna(current_price) and pd.notna(eps) and eps > 0
                        else np.nan)

            # Estimativas de analistas para o próximo exercício. Reaproveita o
            # objeto `stock` já criado: 1 requisição HTTP a mais por ticker.
            crescimento_receita_pct, crescimento_lucro_pct, lpa_estimado, num_analistas = (
                _extract_analyst_estimates(stock)
            )

            records.append({
                'ticker_sa': ticker_sa,
                'ticker': ticker_sa.replace('.SA', ''),
                'nome': company_name,
                'setor': sector,
                'industria': industry,
                'moeda': moeda,
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
                'shares_total': shares_total,
                'dividend_rate': dividend_rate,
                'crescimento_receita_pct': crescimento_receita_pct,
                'crescimento_lucro_pct': crescimento_lucro_pct,
                'lpa_estimado': lpa_estimado,
                'num_analistas': num_analistas,
            })

        except Exception as e:
            print(f"[fundamentals] Erro ao processar {ticker_sa}: {e}")
            records.append({
                'ticker_sa': ticker_sa,
                'ticker': ticker_sa.replace('.SA', ''),
                'nome': ticker_sa,
                'setor': '',
                'industria': '',
                'moeda': '',
                **{k: np.nan for k in [
                    'preco', 'pl', 'pvp', 'margem_ebit_pct', 'margem_liquida_pct',
                    'dl_ebit', 'dl_pl', 'roe_pct', 'liquidez_corrente', 'passivos_ativos',
                    'liq_media_diaria', 'lpa', 'vpa', 'dy_pct', 'divida_liquida',
                    'ebit', 'fcf_latest', 'shares_outstanding', 'shares_total',
                    'dividend_rate', 'crescimento_receita_pct',
                    'crescimento_lucro_pct', 'lpa_estimado', 'num_analistas',
                ]}
            })

        time.sleep(delay)

    df = pd.DataFrame(records)

    # Beta por regressão vs o índice da região (download em lote, fora do laço).
    df['beta_raw'] = df['ticker_sa'].map(fetch_betas(tickers_sa, index_symbol))

    if not incluir_liq_local:
        df = df.drop(columns=['liq_media_diaria'], errors='ignore')

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)

    print(f"\n[fundamentals] {len(df)} tickers processados, "
          f"{df['preco'].notna().sum()} com dados de preço (salvo em {cache})")
    return df


def get_fcf_series(ticker_sa: str) -> pd.Series:
    """Retorna série histórica de Free Cash Flow para cálculo de DCF."""
    try:
        stock = yf.Ticker(ticker_sa)
        cf = stock.cashflow
        return _extract_financial_series(cf, ['Free Cash Flow', 'FreeCashFlow'])
    except Exception:
        return pd.Series(dtype=float)
