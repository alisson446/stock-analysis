import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time

# Tickers bancários conhecidos (fallback para classificação)
KNOWN_BANK_TICKERS = {
    'BBAS3', 'ITUB3', 'ITUB4', 'BBDC3', 'BBDC4', 'SANB11', 'SANB3', 'SANB4',
    'BRSR3', 'BRSR5', 'BRSR6', 'ABCB4', 'BPAC11', 'BPAC3', 'BPAC5',
    'BMGB4', 'BIDI3', 'BIDI4', 'BIDI11', 'BPAN4', 'PINE4', 'BGIP3', 'BGIP4',
    'BMEB3', 'BMEB4', 'BNBR3', 'BSLI3', 'BSLI4', 'IDVL3', 'IDVL4',
    'MODL3', 'MODL4', 'MODL11', 'BBSE3', 'CXSE3', 'ITSA3', 'ITSA4',
    'WIZC3', 'BPRE3', 'CSAB3', 'CSAB4', 'CRIV3', 'CRIV4', 'MERC3', 'MERC4',
    'BCSA34', 'NUBR33', 'ROXO34', 'BMOB3', 'APTS3',
}

BANK_INDUSTRIES = {
    'banks', 'banking', 'banks—regional', 'banks—diversified',
    'banks - regional', 'banks - diversified',
    'financial conglomerates', 'mortgage finance',
}


def get_tickers() -> pd.DataFrame:
    """Scrape stock tickers from dadosdemercado.com.br/acoes."""
    url = 'https://www.dadosdemercado.com.br/acoes'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'lxml')

    # Extrair tickers dos links na página
    tickers = []
    for link in soup.select('a[href*="/acoes/"]'):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        # Links no formato /acoes/PETR4
        parts = href.rstrip('/').split('/')
        if len(parts) >= 3 and parts[-2] == 'acoes':
            ticker = parts[-1].upper()
            # Filtrar tickers válidos (4-6 chars alfanuméricos)
            if 3 <= len(ticker) <= 6 and ticker[:-1].isalpha() or ticker.isalnum():
                tickers.append(ticker)

    # Remover duplicatas mantendo ordem
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    df = pd.DataFrame({
        'ticker': unique_tickers,
        'ticker_sa': [f'{t}.SA' for t in unique_tickers],
    })

    print(f"[scraper] {len(df)} tickers obtidos de dadosdemercado.com.br")
    return df


def classify_banks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classifica ações em bancos e não-bancos.
    Usa campo sector/industry do yfinance + fallback para lista hard-coded.

    Returns:
        (stocks_df, banks_df) — DataFrames separados
    """
    is_bank = []

    for _, row in df.iterrows():
        ticker = row['ticker']
        ticker_sa = row['ticker_sa']

        # Check hard-coded list first (fast)
        if ticker in KNOWN_BANK_TICKERS:
            is_bank.append(True)
            continue

        # Check via yfinance sector/industry
        try:
            info = yf.Ticker(ticker_sa).info
            sector = (info.get('sector', '') or '').lower()
            industry = (info.get('industry', '') or '').lower()

            if sector == 'financial services' and industry in BANK_INDUSTRIES:
                is_bank.append(True)
            else:
                is_bank.append(False)
        except Exception:
            is_bank.append(False)

        time.sleep(0.3)

    df = df.copy()
    df['is_bank'] = is_bank

    stocks = df[~df['is_bank']].drop(columns=['is_bank']).reset_index(drop=True)
    banks = df[df['is_bank']].drop(columns=['is_bank']).reset_index(drop=True)

    print(f"[scraper] Classificação: {len(stocks)} ações | {len(banks)} bancos")
    return stocks, banks
