# BDRs no screener — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer BDRs aparecerem no screener com fundamento coletado do ativo subjacente estrangeiro, avaliados e exibidos na moeda do próprio balanço, com `data/` e `config/` reorganizados em pastas por região.

**Architecture:** O universo sai do lado brasileiro (`yf.screen(region='br')`, marcador `DR[N123]\b`); cada BDR é ligado ao seu subjacente por `yf.Search`, e o par só é aceito se duas medidas independentes da razão do BDR concordarem. Fundamentos, filtros e valuation rodam sobre o subjacente, na moeda do balanço dele, sem nenhuma conversão. Cada região vira um pipeline autocontido em `data/<r>/` e `config/<r>/`.

**Tech Stack:** Python 3.14, pandas, numpy, yfinance 1.2.0, pytest. Nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-08-12-bdrs-no-screener-design.md`

## Global Constraints

- **Nenhuma conversão de moeda em nenhum ponto do pipeline.** Papel cujo `currency != financialCurrency` é excluído, nunca convertido.
- **Guideline 3 (`docs/GUIDELINES.md`):** constantes saem de premissa explícita, nunca de ajuste à amostra. Testes usam dados construídos à mão, **nunca** os CSVs de `data/`.
- **Guideline 4:** quando houver dúvida, excluir o papel — nunca incluí-lo com dado parcial.
- **Guideline 5:** docstrings e comentários explicam o **porquê**, em linguagem sem jargão financeiro.
- Regex do marcador de BDR: **`DR[N123]\b`** — sem `\b` inicial. Um `\b` na frente descarta 305 dos 625 BDRs em silêncio.
- Tolerâncias do portão: razão inteira dentro de **±0,02**; cotação implícita a menos de **3%** da mediana da sua moeda de pregão; mínimo de **3 pares** por moeda para a mediana valer.
- Premissas macro embutidas: `BRL = (0.124, 0.075)`, `USD = (0.042, 0.045)`. Qualquer outra moeda exige `RISK_FREE_RATE_<MOEDA>` **e** `EQUITY_RISK_PREMIUM_<MOEDA>` no `.env`, senão o papel é excluído.
- `terminal_growth` de uma moeda é sempre igual ao `risk_free_rate` dela.
- Rodar pytest via `rtk proxy` (o hook do RTK quebra a invocação direta): `rtk proxy python3 -m pytest ...`

## Estrutura de arquivos

| arquivo | responsabilidade |
|---|---|
| `src/paths.py` (novo) | única fonte dos caminhos `data/<r>/<arquivo>` e `config/<r>/filters.json`; valida o nome da região |
| `src/bdrs.py` (novo) | universo de BDRs, marcador, resolução do par, portão de qualidade, elegibilidade por moeda |
| `src/filters.py` | passa a receber `region`; nenhuma função nova |
| `src/scraper.py` | `get_tickers` passa a receber `region` |
| `src/fundamentals.py` | `fetch_fundamentals` recebe `region` e `index_symbol`; grava `moeda` |
| `src/valuation.py` | `macro_for(moeda)`; `cost_of_equity` recebe moeda; `append_snapshot` recebe região e grava premissas por linha |
| `config/br/filters.json` | movido de `config/filters.json` |
| `config/us/filters.json` | novo |
| `data/br/*.csv` | movidos de `data/*.csv` |
| `tests/test_paths.py`, `tests/test_bdrs.py` | novos |

---

### Task 1: `src/paths.py` — caminhos por região

**Files:**
- Create: `src/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nada
- Produces: `validate_region(region: str) -> str`, `data_file(region: str, name: str) -> Path`, `filters_file(region: str) -> Path`, constantes `DATA_ROOT: Path` e `CONFIG_ROOT: Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import paths


class TestValidateRegion:
    """
    O nome da região vira parte de um caminho de arquivo. Se qualquer string
    passar, um nome com '..' ou '/' escreve fora da pasta de dados — por isso
    o formato é restrito ao código de duas a oito letras minúsculas que o
    yf.screen usa ('br', 'us').
    """

    @pytest.mark.parametrize('region', ['br', 'us', 'gb', 'jp'])
    def test_accepts_lowercase_codes(self, region):
        assert paths.validate_region(region) == region

    @pytest.mark.parametrize('region', ['', '   ', 'BR', 'Us', 'b', 'a' * 9])
    def test_rejects_malformed(self, region):
        with pytest.raises(ValueError):
            paths.validate_region(region)

    @pytest.mark.parametrize('region', ['..', '../etc', 'a/b', 'a\\b', 'br/../us'])
    def test_rejects_path_escape(self, region):
        with pytest.raises(ValueError):
            paths.validate_region(region)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            paths.validate_region(None)


class TestPathBuilders:
    def test_data_file_is_under_region_folder(self):
        p = paths.data_file('us', 'fundamentals.csv')
        assert p.parent.name == 'us'
        assert p.parent.parent == paths.DATA_ROOT
        assert p.name == 'fundamentals.csv'

    def test_filters_file_is_under_region_folder(self):
        p = paths.filters_file('br')
        assert p.parent.name == 'br'
        assert p.parent.parent == paths.CONFIG_ROOT
        assert p.name == 'filters.json'

    def test_builders_validate_the_region(self):
        with pytest.raises(ValueError):
            paths.data_file('../etc', 'fundamentals.csv')
        with pytest.raises(ValueError):
            paths.filters_file('')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_paths.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/paths.py
"""
Único lugar do projeto que sabe montar os caminhos de `data/` e `config/`.

Cada região é um pipeline autocontido: `data/br/` guarda as ações brasileiras,
`data/us/` guarda os subjacentes americanos alcançados via BDR. Concentrar a
montagem aqui é o que permite acrescentar uma região criando pastas, sem
procurar caminho hard-coded espalhado pelos módulos.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / 'data'
CONFIG_ROOT = ROOT / 'config'

# Duas a oito letras minúsculas: o formato do código de região do yf.screen.
# A restrição não é cosmética — o nome vira segmento de caminho, e aceitar
# '..' ou '/' deixaria a região escrever fora da pasta de dados.
_REGION_RE = re.compile(r'^[a-z]{2,8}$')


def validate_region(region: str) -> str:
    """Devolve a região se ela for um código válido; levanta ValueError se não."""
    if not isinstance(region, str) or not _REGION_RE.match(region):
        raise ValueError(
            f"região inválida: {region!r}. Use o código do yf.screen em "
            f"minúsculas, de 2 a 8 letras (ex.: 'br', 'us')."
        )
    return region


def data_file(region: str, name: str) -> Path:
    """Caminho de um arquivo de dados da região (ex.: 'fundamentals.csv')."""
    return DATA_ROOT / validate_region(region) / name


def filters_file(region: str) -> Path:
    """Caminho do filters.json da região."""
    return CONFIG_ROOT / validate_region(region) / 'filters.json'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_paths.py -v`
Expected: PASS (14 testes)

- [ ] **Step 5: Commit**

```bash
git add src/paths.py tests/test_paths.py
git commit -m "feat: caminhos de data/ e config/ resolvidos por regiao"
```

---

### Task 2: `config/br/filters.json` e `filters.py` por região

**Files:**
- Move: `config/filters.json` → `config/br/filters.json`
- Modify: `src/filters.py:5-11` (`CONFIG_PATH`, `_load_config`), `src/filters.py:64-120` (assinaturas)
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `paths.filters_file(region)` da Task 1
- Produces: `filters._load_config(region: str) -> dict`, `apply_stock_filters(df, region: str = 'br')`, `apply_bank_filters(df, region: str = 'br')`, `filters._liquidity_mask(df, cfg) -> pd.Series`

- [ ] **Step 1: Write the failing test**

Acrescente ao final de `tests/test_filters.py`:

```python
class TestLoadConfigPorRegiao:
    """A região escolhe a pasta do filters.json; região sem arquivo é erro."""

    def test_br_carrega_o_arquivo_movido(self):
        cfg = filters._load_config('br')
        assert 'stock_filters' in cfg and 'bank_filters' in cfg

    def test_regiao_sem_arquivo_levanta_erro_nomeando_o_caminho(self):
        with pytest.raises(FileNotFoundError) as exc:
            filters._load_config('zz')
        # A mensagem precisa dizer QUAL arquivo criar — senão o usuário só
        # descobre que "faltou algo" sem saber onde.
        assert 'zz' in str(exc.value) and 'filters.json' in str(exc.value)

    def test_regiao_malformada_levanta_value_error(self):
        with pytest.raises(ValueError):
            filters._load_config('../etc')


class TestLiquidityMask:
    """
    O corte de liquidez muda de coluna conforme a região: no Brasil filtra a
    liquidez da própria ação; na região alcançada via BDR filtra a do BDR, em
    reais, porque é o que se consegue negociar. A chave do config carrega a
    unidade no nome para não ser lida como dólar.
    """

    def test_usa_liq_media_diaria_quando_a_chave_e_a_local(self):
        df = pd.DataFrame({'liq_media_diaria': [50_000, 150_000],
                           'liq_media_diaria_bdr': [999_999, 1]})
        mask = filters._liquidity_mask(df, {'liq_media_diaria_min': 100_000})
        assert list(mask) == [False, True]

    def test_usa_liq_media_diaria_bdr_quando_a_chave_e_a_do_bdr(self):
        df = pd.DataFrame({'liq_media_diaria': [999_999, 1],
                           'liq_media_diaria_bdr': [50_000, 150_000]})
        mask = filters._liquidity_mask(df, {'liq_media_diaria_bdr_min': 100_000})
        assert list(mask) == [False, True]

    def test_config_sem_chave_de_liquidez_levanta_erro(self):
        df = pd.DataFrame({'liq_media_diaria': [1]})
        with pytest.raises(KeyError):
            filters._liquidity_mask(df, {'pl_max': 10})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_filters.py -v`
Expected: FAIL — `_load_config() takes 0 positional arguments` e `module 'src.filters' has no attribute '_liquidity_mask'`

- [ ] **Step 3: Mover o arquivo e implementar**

```bash
mkdir -p config/br
git mv config/filters.json config/br/filters.json
```

Substitua o topo de `src/filters.py` (linhas 1-11):

```python
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
```

Em `apply_stock_filters`, troque a assinatura e a linha do corte de liquidez:

```python
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
```

Em `apply_bank_filters`, troque a assinatura, a leitura do config, o corte de liquidez, e torne `dy_pct_min` opcional:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_filters.py -v`
Expected: PASS — os testes existentes continuam passando e os 6 novos passam

- [ ] **Step 5: Commit**

```bash
git add config/br/filters.json src/filters.py tests/test_filters.py
git commit -m "feat: filtros lidos por regiao, com corte de liquidez por chave"
```

---

### Task 3: `data/br/tickers.csv` e `scraper.py` por região

**Files:**
- Move: `data/tickers.csv` → `data/br/tickers.csv`
- Modify: `src/scraper.py:8-9`, `src/scraper.py:29-39`, `src/scraper.py:42-86`
- Test: `tests/test_scraper.py` (novo)

**Interfaces:**
- Consumes: `paths.data_file(region, name)` da Task 1
- Produces: `scraper.get_tickers(force_refresh: bool = False, region: str = 'br') -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_scraper.py -v`
Expected: FAIL — `get_tickers() got an unexpected keyword argument 'region'`

- [ ] **Step 3: Mover o arquivo e implementar**

```bash
mkdir -p data/br
# mv simples, nao git mv: data/ esta no .gitignore e os CSVs nao sao
# rastreados, entao nao ha historico a preservar e git mv falharia.
mv data/tickers.csv data/br/tickers.csv
```

Em `src/scraper.py`, troque as linhas 8-9 e as funções de cache:

```python
from src import paths

# (remova DATA_DIR e TICKERS_CACHE — os caminhos agora saem de paths)


def get_tickers(force_refresh: bool = False, region: str = 'br') -> pd.DataFrame:
    """
    Retorna DataFrame de tickers da região. Usa cache local
    (data/<region>/tickers.csv) se existir. Se não existir ou
    force_refresh=True, faz scraping e salva o resultado.

    O scraping só existe para a região 'br' — as demais são populadas por
    outros caminhos (a região 'us' vem de src/bdrs.py). Pedir refresh de uma
    região sem scraper é erro explícito em vez de arquivo vazio.
    """
    cache = paths.data_file(region, 'tickers.csv')
    if not force_refresh and cache.exists():
        df = pd.read_csv(cache)
        print(f"[scraper] {len(df)} tickers da região {region} carregados do cache ({cache})")
        return df

    if region != 'br':
        raise ValueError(
            f"não há scraper para a região {region!r}. Popule "
            f"{cache} pelo pipeline da região (ex.: src/bdrs.py para 'us')."
        )
    return _scrape_tickers(cache)
```

Em `_scrape_tickers`, troque a assinatura e a gravação:

```python
def _scrape_tickers(cache: Path) -> pd.DataFrame:
    """Scrape stock tickers from dadosdemercado.com.br/acoes e salva em cache."""
```

e, no final da função, troque as três últimas linhas por:

```python
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)

    print(f"[scraper] {len(df)} tickers obtidos de dadosdemercado.com.br (salvo em {cache})")
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_scraper.py -v`
Expected: PASS (3 testes)

- [ ] **Step 5: Commit**

```bash
git add src/scraper.py tests/test_scraper.py   # data/ e gitignored
git commit -m "feat: cache de tickers por regiao em data/<r>/tickers.csv"
```

---

### Task 4: `data/br/fundamentals.csv`, coluna `moeda` e índice do beta

**Files:**
- Move: `data/fundamentals.csv` → `data/br/fundamentals.csv`
- Modify: `src/fundamentals.py:8-9`, `src/fundamentals.py:203-231`, `src/fundamentals.py:245` (bloco do `.info`), `src/fundamentals.py:376-406` (record), `src/fundamentals.py:428-439` (beta e gravação)
- Test: `tests/test_fundamentals.py`

**Interfaces:**
- Consumes: `paths.data_file(region, name)` da Task 1
- Produces: `fetch_fundamentals(tickers_sa, delay=0.5, force_refresh=False, region='br', index_symbol='^BVSP', incluir_liq_local=True) -> pd.DataFrame`; coluna `moeda` no DataFrame

- [ ] **Step 1: Write the failing test**

Acrescente ao final de `tests/test_fundamentals.py`:

```python
class TestLeituraDoCachePorRegiao:
    """
    O cache passa a viver em data/<regiao>/fundamentals.csv, e a coluna `moeda`
    é nova. As 372 linhas gravadas antes desta mudança não a têm, então
    ausência significa BRL — senão uma rodada que só queria reaproveitar dado
    quebraria com KeyError.
    """

    def _grava(self, tmp_path, regiao, df):
        (tmp_path / regiao).mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp_path / regiao / 'fundamentals.csv', index=False)

    def test_le_o_cache_da_regiao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'us', pd.DataFrame({'ticker': ['AAPL'], 'moeda': ['USD']}))

        out = f.fetch_fundamentals([], region='us')

        assert list(out['ticker']) == ['AAPL']
        assert list(out['moeda']) == ['USD']

    def test_cache_antigo_sem_coluna_moeda_e_lido_como_brl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'br', pd.DataFrame({'ticker': ['PETR4'], 'pl': [4.2]}))

        out = f.fetch_fundamentals([], region='br')

        assert list(out['moeda']) == ['BRL']

    def test_moeda_ja_gravada_nao_e_sobrescrita(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f.paths, 'DATA_ROOT', tmp_path)
        self._grava(tmp_path, 'us', pd.DataFrame({'ticker': ['SAP'], 'moeda': ['EUR']}))

        out = f.fetch_fundamentals([], region='us')

        assert list(out['moeda']) == ['EUR']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_fundamentals.py -v -k Regiao`
Expected: FAIL — `fetch_fundamentals() got an unexpected keyword argument 'region'`

- [ ] **Step 3: Mover o arquivo e implementar**

```bash
mv data/fundamentals.csv data/br/fundamentals.csv   # nao rastreado: mv simples
```

Em `src/fundamentals.py`, troque as linhas 8-9 por:

```python
from src import paths
```

Substitua `fetch_fundamentals` (linhas 203-223) por:

```python
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
```

Troque a assinatura de `_fetch_fundamentals_from_api` (linha 226):

```python
def _fetch_fundamentals_from_api(tickers_sa: list[str], delay: float, cache: Path,
                                 index_symbol: str,
                                 incluir_liq_local: bool) -> pd.DataFrame:
    """Busca dados fundamentalistas via yfinance e salva em cache."""
```

Logo após a linha `company_name = _safe_get(info, 'shortName', ticker_sa)`, acrescente:

```python
            # Moeda do BALANÇO, não a do pregão. É ela que rege lucro, EBIT,
            # FCF e o preço justo — e é a que precisa bater com a do preço
            # para o P/L recalculado significar alguma coisa.
            moeda = _safe_get(info, 'financialCurrency', '')
```

No dicionário `records.append({...})`, acrescente após `'industria': industry,`:

```python
                'moeda': moeda,
```

e no bloco `except` do mesmo laço, acrescente após `'industria': '',`:

```python
                'moeda': '',
```

Troque as linhas 430-435 (beta e gravação) por:

```python
    # Beta por regressão vs o índice da região (download em lote, fora do laço).
    df['beta_raw'] = df['ticker_sa'].map(fetch_betas(tickers_sa, index_symbol))

    if not incluir_liq_local:
        df = df.drop(columns=['liq_media_diaria'])

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
```

e ajuste o print final para usar `cache` em vez de `FUNDAMENTALS_CACHE`.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_fundamentals.py -v`
Expected: PASS — os testes existentes continuam passando e os 3 novos passam

- [ ] **Step 5: Commit**

```bash
git add src/fundamentals.py tests/test_fundamentals.py   # data/ e gitignored
git commit -m "feat: fundamentals por regiao, com coluna moeda e indice do beta"
```

---

### Task 5: `macro_for(moeda)` e custo de capital por moeda

**Files:**
- Modify: `src/valuation.py:42-44` (constantes), `src/valuation.py:134-145` (`cost_of_equity`)
- Test: `tests/test_valuation.py`

**Interfaces:**
- Consumes: nada
- Produces: `valuation.macro_for(moeda: str) -> dict | None` com chaves `risk_free_rate`, `equity_risk_premium`, `terminal_growth`; `cost_of_equity(beta: float = None, moeda: str = 'BRL') -> float`

- [ ] **Step 1: Write the failing test**

Acrescente ao final de `tests/test_valuation.py`:

```python
class TestMacroPorMoeda:
    """
    Descontar fluxo em dólar a 12,4% (juro brasileiro) embute inflação de reais
    num fluxo que não a tem. Cada moeda carrega o próprio juro livre de risco e
    prêmio de risco, e o crescimento na perpetuidade acompanha o juro dela.
    """

    def test_brl_usa_as_constantes_sem_sufixo(self):
        m = v.macro_for('BRL')
        assert m['risk_free_rate'] == v.RISK_FREE_RATE
        assert m['equity_risk_premium'] == v.EQUITY_RISK_PREMIUM

    def test_usd_tem_premissas_embutidas(self):
        m = v.macro_for('USD')
        assert m['risk_free_rate'] == pytest.approx(0.042)
        assert m['equity_risk_premium'] == pytest.approx(0.045)

    def test_terminal_growth_acompanha_o_juro_da_moeda(self):
        for moeda in ('BRL', 'USD'):
            m = v.macro_for(moeda)
            assert m['terminal_growth'] == m['risk_free_rate']

    def test_moeda_sem_premissas_devolve_none(self):
        assert v.macro_for('TWD') is None
        assert v.macro_for('') is None
        assert v.macro_for(None) is None

    def test_env_habilita_uma_moeda_nova(self, monkeypatch):
        monkeypatch.setenv('RISK_FREE_RATE_EUR', '0.028')
        monkeypatch.setenv('EQUITY_RISK_PREMIUM_EUR', '0.055')

        m = v.macro_for('EUR')

        assert m['risk_free_rate'] == pytest.approx(0.028)
        assert m['equity_risk_premium'] == pytest.approx(0.055)
        assert m['terminal_growth'] == pytest.approx(0.028)

    def test_env_com_so_uma_das_duas_nao_habilita(self, monkeypatch):
        monkeypatch.setenv('RISK_FREE_RATE_EUR', '0.028')
        assert v.macro_for('EUR') is None

    def test_moeda_e_normalizada(self, monkeypatch):
        assert v.macro_for('usd') == v.macro_for('USD')
        assert v.macro_for(' USD ') == v.macro_for('USD')


class TestCostOfEquityPorMoeda:
    def test_default_continua_sendo_reais(self):
        assert v.cost_of_equity(1.0) == pytest.approx(
            v.RISK_FREE_RATE + v.EQUITY_RISK_PREMIUM)

    def test_dolar_usa_as_premissas_do_dolar(self):
        assert v.cost_of_equity(1.0, moeda='USD') == pytest.approx(0.042 + 0.045)

    def test_moeda_sem_premissas_devolve_nan(self):
        assert pd.isna(v.cost_of_equity(1.0, moeda='TWD'))

    def test_clamp_de_beta_continua_valendo(self):
        assert v.cost_of_equity(9.0, moeda='USD') == pytest.approx(
            0.042 + v.MAX_BETA * 0.045)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py -v -k Moeda`
Expected: FAIL — `module 'src.valuation' has no attribute 'macro_for'`

- [ ] **Step 3: Write minimal implementation**

Em `src/valuation.py`, logo após a linha `TERMINAL_GROWTH = RISK_FREE_RATE`, acrescente:

```python
# Premissas macro embutidas por moeda. BRL e USD sobem configurados; qualquer
# outra moeda exige as duas variáveis no .env, e o papel que reporta nela é
# excluído até que existam. É de propósito: um juro chutado produz preço justo
# com aparência de calculado.
_MACRO_EMBUTIDO = {
    'BRL': (RISK_FREE_RATE, EQUITY_RISK_PREMIUM),
    'USD': (0.042, 0.045),   # Treasury longo e prêmio de risco EUA
}


def macro_for(moeda: str) -> dict | None:
    """
    Premissas de custo de capital da moeda do BALANÇO do ativo.

    Descontar fluxo em dólar ao juro brasileiro embutiria inflação de reais num
    fluxo que não a tem, e subavaliaria a empresa de forma sistemática. Por isso
    cada moeda carrega o próprio par (juro livre de risco, prêmio de risco), e
    o crescimento na perpetuidade acompanha o juro dela — a regra de sempre:
    a perpetuidade não pode crescer mais que a economia.

    Returns:
        dict com `risk_free_rate`, `equity_risk_premium` e `terminal_growth`,
        ou None quando a moeda não tem premissas definidas.
    """
    moeda = (moeda or '').strip().upper()
    if not moeda:
        return None

    embutido = _MACRO_EMBUTIDO.get(moeda)
    rf_env = os.getenv(f'RISK_FREE_RATE_{moeda}')
    erp_env = os.getenv(f'EQUITY_RISK_PREMIUM_{moeda}')

    if embutido is None and (not rf_env or not erp_env):
        return None

    base_rf, base_erp = embutido if embutido else (0.0, 0.0)
    rf = _env_float(f'RISK_FREE_RATE_{moeda}', base_rf)
    erp = _env_float(f'EQUITY_RISK_PREMIUM_{moeda}', base_erp)
    return {'risk_free_rate': rf, 'equity_risk_premium': erp,
            'terminal_growth': rf}
```

Substitua `cost_of_equity` (linhas 134-145) por:

```python
def cost_of_equity(beta: float = None, moeda: str = 'BRL') -> float:
    """
    Custo de capital próprio via CAPM: RF + beta × ERP, nas premissas da moeda.

    A versão anterior usava a Selic pura, que é a taxa LIVRE DE RISCO — descontar
    fluxos de equity a ela omite o prêmio de risco inteiro e superestima
    sistematicamente o preço justo.

    Moeda sem premissas definidas devolve NaN: sem juro livre de risco não há
    taxa de desconto, e chutar uma produziria preço justo com aparência de
    calculado.
    """
    macro = macro_for(moeda)
    if macro is None:
        return np.nan
    if beta is None or pd.isna(beta):
        beta = 1.0
    beta = max(MIN_BETA, min(MAX_BETA, float(beta)))
    return macro['risk_free_rate'] + beta * macro['equity_risk_premium']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py -v`
Expected: PASS — os testes existentes continuam passando e os 11 novos passam

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py tests/test_valuation.py
git commit -m "feat: premissas macro e custo de capital resolvidos por moeda"
```

---

### Task 6: `append_snapshot` por região, com premissas por linha

**Files:**
- Move: `data/valuation_history.csv` → `data/br/valuation_history.csv`
- Modify: `src/valuation.py:660-661` (paths), `src/valuation.py:664-672` (`_SNAPSHOT_RESULT_COLS`), `src/valuation.py:675-720` (`append_snapshot`)
- Test: `tests/test_valuation.py`

**Interfaces:**
- Consumes: `macro_for` da Task 5, `paths.data_file` da Task 1
- Produces: `append_snapshot(df, path=None, snapshot_date=None, region='br') -> Path`; colunas `regiao` e `moeda` no snapshot

- [ ] **Step 1: Write the failing test**

Acrescente ao final de `tests/test_valuation.py`:

```python
class TestAppendSnapshotPorRegiao:
    """
    O docstring de append_snapshot diz que as premissas existem para atribuir
    uma divergência futura a mudança de dado OU de premissa. Gravá-las a partir
    de constantes do módulo faria a linha em dólar registrar o juro brasileiro:
    uma premissa que não foi usada, indistinguível de uma verdadeira.
    """

    def _df(self, **cols):
        base = {'ticker': ['X'], 'preco': [10.0], 'preco_justo_dcf': [12.0]}
        base.update(cols)
        return pd.DataFrame(base)

    def test_grava_na_pasta_da_regiao(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        out = v.append_snapshot(self._df(moeda=['USD']), region='us')
        assert out == tmp_path / 'us' / 'valuation_history.csv'
        assert out.exists()

    def test_premissas_saem_da_moeda_da_linha(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        df = self._df(ticker=['PETR4', 'AAPL'], preco=[10.0, 10.0],
                      preco_justo_dcf=[12.0, 12.0], moeda=['BRL', 'USD'])

        v.append_snapshot(df, region='us')

        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        por_moeda = dict(zip(hist['moeda'], hist['risk_free_rate']))
        assert por_moeda['BRL'] == pytest.approx(v.RISK_FREE_RATE)
        assert por_moeda['USD'] == pytest.approx(0.042)

    def test_terminal_growth_tambem_e_por_linha(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        df = self._df(ticker=['PETR4', 'AAPL'], preco=[10.0, 10.0],
                      preco_justo_dcf=[12.0, 12.0], moeda=['BRL', 'USD'])

        v.append_snapshot(df, region='us')

        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert dict(zip(hist['moeda'], hist['terminal_growth']))['USD'] == pytest.approx(0.042)

    def test_sem_coluna_moeda_assume_brl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(), region='br')
        hist = pd.read_csv(tmp_path / 'br' / 'valuation_history.csv')
        assert list(hist['moeda']) == ['BRL']
        assert hist['risk_free_rate'].iloc[0] == pytest.approx(v.RISK_FREE_RATE)

    def test_regiao_vira_coluna(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(moeda=['USD']), region='us')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert list(hist['regiao']) == ['us']

    def test_tipo_de_banco_americano_continua_banco(self, tmp_path, monkeypatch):
        # `tipo` separa banco de não-banco; `regiao` diz de onde veio. Se `tipo`
        # virasse 'bdr', o JPMorgan ficaria indistinguível de uma varejista.
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(tipo=['banco'], moeda=['USD']), region='us')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert list(hist['tipo']) == ['banco']
        assert list(hist['regiao']) == ['us']

    def test_append_preserva_as_linhas_anteriores(self, tmp_path, monkeypatch):
        monkeypatch.setattr(v.paths, 'DATA_ROOT', tmp_path)
        v.append_snapshot(self._df(moeda=['USD']), region='us', snapshot_date='2026-01-01')
        v.append_snapshot(self._df(moeda=['USD']), region='us', snapshot_date='2026-02-01')
        hist = pd.read_csv(tmp_path / 'us' / 'valuation_history.csv')
        assert sorted(hist['data_snapshot'].unique()) == ['2026-01-01', '2026-02-01']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py -v -k AppendSnapshotPorRegiao`
Expected: FAIL — `append_snapshot() got an unexpected keyword argument 'region'`

- [ ] **Step 3: Mover o arquivo e implementar**

```bash
mv data/valuation_history.csv data/br/valuation_history.csv   # nao rastreado: mv simples
```

Em `src/valuation.py`, troque as linhas 660-661 por:

```python
from src import paths
```

(mantenha o `import` junto dos outros no topo do arquivo e remova `DATA_DIR`/`VALUATION_HISTORY`.)

Acrescente `'regiao'` e `'moeda'` a `_SNAPSHOT_RESULT_COLS`, logo após `'tipo'`:

```python
_SNAPSHOT_RESULT_COLS = [
    'tipo', 'regiao', 'moeda', 'ticker', 'nome', 'setor', 'preco',
    'preco_justo_dcf', 'metodo_valuation', 'growth_source',
    'preco_justo_graham', 'margem_seg_dcf_pct', 'margem_seg_graham_pct',
    'margem_seg_media_pct', 'undervalued', 'forte_desconto',
    'cost_of_equity_pct',
    'crescimento_receita_pct', 'crescimento_lucro_pct', 'lpa_estimado',
    'num_analistas',
]
```

Substitua a assinatura e o bloco de premissas de `append_snapshot`:

```python
def append_snapshot(df: pd.DataFrame, path: Path = None,
                    snapshot_date: str = None, region: str = 'br') -> Path:
    """
    Anexa o resultado de valuation ao histórico append-only da região.

    Cada linha carrega, além do preço justo e da margem, as PREMISSAS usadas na
    rodada (RF, ERP, crescimento terminal, flag forward) — assim uma divergência
    futura pode ser atribuída a mudança de dado ou de premissa.

    As premissas saem da MOEDA DE CADA LINHA, não de constantes do módulo: uma
    linha avaliada em dólar que registrasse o juro brasileiro guardaria uma
    premissa que não foi usada, indistinguível de uma verdadeira — destruindo
    exatamente a garantia que estas colunas existem para dar.

    Args:
        df: DataFrame vindo de apply_valuation (opcionalmente com 'tipo').
        path: Destino. Default: data/<region>/valuation_history.csv.
        snapshot_date: Data ISO (YYYY-MM-DD). Default: hoje.
        region: Região de destino, gravada na coluna `regiao`.

    Returns:
        O Path do arquivo escrito.
    """
    path = Path(path) if path is not None else paths.data_file(
        region, 'valuation_history.csv')
    if snapshot_date is None:
        snapshot_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    if df is None or len(df) == 0:
        print("[valuation] snapshot vazio, nada a gravar")
        return path

    df = df.copy()
    # Linhas sem moeda são brasileiras: é o que valia antes desta coluna existir.
    if 'moeda' not in df.columns:
        df['moeda'] = 'BRL'
    df['regiao'] = region

    cols = [c for c in _SNAPSHOT_RESULT_COLS if c in df.columns]
    snap = df[cols].copy()
    snap.insert(0, 'data_snapshot', snapshot_date)

    # Premissas da rodada, resolvidas pela moeda de cada linha.
    macros = [macro_for(m) or {} for m in snap['moeda']]
    snap['risk_free_rate'] = [m.get('risk_free_rate', np.nan) for m in macros]
    snap['equity_risk_premium'] = [m.get('equity_risk_premium', np.nan) for m in macros]
    snap['terminal_growth'] = [m.get('terminal_growth', np.nan) for m in macros]
    snap['use_forward_estimates'] = USE_FORWARD_ESTIMATES
    snap['forward_growth_driver'] = FORWARD_GROWTH_DRIVER
```

O restante da função (gravação com `concat`) fica como está.

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_valuation.py -v`
Expected: PASS — os existentes continuam passando e os 7 novos passam

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py tests/test_valuation.py   # data/ e gitignored
git commit -m "feat: snapshot por regiao, com premissas resolvidas por linha"
```

---

### Task 7: `src/bdrs.py` — universo e marcador de BDR

**Files:**
- Create: `src/bdrs.py`
- Test: `tests/test_bdrs.py`

**Interfaces:**
- Consumes: nada
- Produces: `bdrs.MARCADOR_BDR` (regex compilado), `bdrs.e_bdr(short_name: str) -> bool`, `bdrs.selecionar_bdrs(quotes: list[dict]) -> list[dict]`, `bdrs.buscar_universo(region: str = 'br', mcap_min: int = 500_000_000) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bdrs.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import bdrs


class TestMarcadorBDR:
    """
    O shortName do screener é campo de largura fixa. Quando o nome da empresa
    ocupa a largura inteira, o marcador fica COLADO nele — e um \\b inicial no
    regex descartaria 305 dos 625 BDRs em silêncio. O sufixo ED (ex-dividendo)
    aparece depois do marcador em 36 papéis, então ancorar no fim também falha.
    """

    @pytest.mark.parametrize('short_name', [
        'ZOETIS INC  DRN',
        'JBS N.V.    DR2',
        'XP INC      DR1',
        'AURA 360    DR3',
    ])
    def test_aceita_marcador_separado_por_espaco(self, short_name):
        assert bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', [
        'ZILLOW GROUPDRN',
        'WHIRLPOOL CODRN',
        'ZEBRA TECHNODRN',
    ])
    def test_aceita_marcador_colado_ao_nome(self, short_name):
        # Regressão do \b inicial.
        assert bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', [
        'WELLS FARGO DRN ED',
        'UNILEVER    DRN ED',
    ])
    def test_aceita_marcador_seguido_de_ed(self, short_name):
        assert bdrs.e_bdr(short_name)

    def test_aceita_os_dois_problemas_juntos(self):
        assert bdrs.e_bdr('CONSTELLATIODRN ED')

    @pytest.mark.parametrize('short_name', [
        'YDUQS PART  ON      NM',
        'PETROBRAS   PN      N2',
        'FII ZION    CI',
        'ITAUUNIBANCOPN      N1',
    ])
    def test_rejeita_acao_brasileira_e_fii(self, short_name):
        assert not bdrs.e_bdr(short_name)

    @pytest.mark.parametrize('short_name', ['', None])
    def test_rejeita_vazio(self, short_name):
        assert not bdrs.e_bdr(short_name)


class TestSelecionarBDRs:
    def test_exige_longname_preenchido(self):
        # O longName é o que alimenta a busca do subjacente. Sem ele o papel
        # não tem como ser resolvido, então não entra no universo.
        quotes = [
            {'symbol': 'AAPL34.SA', 'shortName': 'APPLE       DRN', 'longName': 'Apple Inc.'},
            {'symbol': 'XXXX34.SA', 'shortName': 'SEM NOME    DRN', 'longName': ''},
            {'symbol': 'YYYY34.SA', 'shortName': 'SEM NOME2   DRN'},
        ]
        assert [q['symbol'] for q in bdrs.selecionar_bdrs(quotes)] == ['AAPL34.SA']

    def test_descarta_nao_bdr(self):
        quotes = [
            {'symbol': 'PETR4.SA', 'shortName': 'PETROBRAS   PN      N2',
             'longName': 'Petróleo Brasileiro S.A.'},
            {'symbol': 'MSFT34.SA', 'shortName': 'MICROSOFT   DRN',
             'longName': 'Microsoft Corporation'},
        ]
        assert [q['symbol'] for q in bdrs.selecionar_bdrs(quotes)] == ['MSFT34.SA']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'src.bdrs'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/bdrs.py
"""
Descoberta e validação dos pares BDR ↔ ativo subjacente.

O ticker do BDR não sustenta o pipeline: `earnings_estimate` volta vazio em
100% dos BDRs testados, e o `.info` mistura preço em reais com lucro na moeda
do balanço da empresa estrangeira — o que faria o P/L recalculado da Apple sair
178 em vez de 33. Então o BDR entra como preço e o subjacente entra como
empresa.
"""
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as EQ

# Marcador do tipo de recibo no shortName: DRN é não-patrocinado, DR1/DR2/DR3
# são os patrocinados.
#
# SEM \b na frente, de propósito. O shortName é campo de largura fixa e o
# marcador fica colado no nome quando ele ocupa a largura inteira
# ('ZILLOW GROUPDRN'); exigir a borda inicial descartaria 305 dos 625 BDRs sem
# erro nenhum. A borda final é necessária porque 'ED' (ex-dividendo) aparece
# depois do marcador em 36 papéis, o que também derruba uma âncora de fim.
MARCADOR_BDR = re.compile(r'DR[N123]\b')

_TAMANHO_PAGINA = 250


def e_bdr(short_name: str) -> bool:
    """True se o shortName do screener carrega o marcador de recibo."""
    return bool(short_name) and bool(MARCADOR_BDR.search(short_name))


def selecionar_bdrs(quotes: list[dict]) -> list[dict]:
    """
    Filtra o universo do screener para os BDRs resolvíveis.

    Exige `longName` preenchido: é ele que alimenta a busca do subjacente, e
    sem ele o papel não tem como ser ligado a empresa nenhuma.
    """
    return [q for q in quotes
            if e_bdr(q.get('shortName')) and (q.get('longName') or '').strip()]


def buscar_universo(region: str = 'br', mcap_min: int = 500_000_000,
                    delay: float = 0.3) -> list[dict]:
    """
    Pagina o `yf.screen` da região e devolve os quotes crus.

    O universo sai do lado brasileiro porque `yf.screen(region='us')` não
    enumera o mercado americano: com `total=7511` ele omite TGT, HD, LOW e MDT
    mesmo numa faixa estreita de valor de mercado.

    Falha de rede aqui é propagada: sem universo não há o que fazer, e devolver
    lista vazia seria indistinguível de "nenhum papel passou".
    """
    query = EQ('and', [EQ('eq', ['region', region]),
                       EQ('gt', ['intradaymarketcap', mcap_min])])
    quotes, offset = [], 0
    while True:
        resposta = yf.screen(query, size=_TAMANHO_PAGINA, offset=offset)
        pagina = resposta.get('quotes', [])
        if not pagina:
            break
        quotes += pagina
        offset += len(pagina)
        if offset >= (resposta.get('total') or 0):
            break
        time.sleep(delay)

    print(f"[bdrs] {len(quotes)} papéis na região {region}")
    return quotes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v`
Expected: PASS (18 testes)

- [ ] **Step 5: Commit**

```bash
git add src/bdrs.py tests/test_bdrs.py
git commit -m "feat: universo de BDRs e marcador DR[N123] sem borda inicial"
```

---

### Task 8: Resolução do ticker subjacente

**Files:**
- Modify: `src/bdrs.py`
- Test: `tests/test_bdrs.py`

**Interfaces:**
- Consumes: nada da Task 7 além do módulo
- Produces: `bdrs.resolver_subjacente(long_name: str, buscar=None) -> str | None`

- [ ] **Step 1: Write the failing test**

Acrescente a `tests/test_bdrs.py`:

```python
class _BuscaFake:
    """Substitui yf.Search nos testes: devolve quotes fixos por nome."""

    def __init__(self, por_nome):
        self.por_nome = por_nome

    def __call__(self, nome, max_results=12):
        if nome not in self.por_nome:
            raise RuntimeError('sem resultado')
        return type('R', (), {'quotes': self.por_nome[nome]})()


class TestResolverSubjacente:
    """
    A busca por nome não é confiável sozinha — ela casou Fomento Económico
    Mexicano com Vista Energy. O portão da Task 9 é que torna isso utilizável;
    aqui só garantimos que o candidato escolhido é uma ação estrangeira.
    """

    def test_escolhe_a_acao_em_bolsa_estrangeira(self):
        busca = _BuscaFake({'Apple Inc.': [
            {'symbol': 'AAPL34.SA', 'exchange': 'SAO', 'quoteType': 'EQUITY'},
            {'symbol': 'AAPL', 'exchange': 'NMS', 'quoteType': 'EQUITY'},
        ]})
        assert bdrs.resolver_subjacente('Apple Inc.', buscar=busca) == 'AAPL'

    def test_ignora_o_proprio_bdr(self):
        busca = _BuscaFake({'X': [{'symbol': 'X34.SA', 'exchange': 'SAO',
                                   'quoteType': 'EQUITY'}]})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None

    def test_ignora_preferenciais_com_hifen(self):
        # WFC-PY e WFC-PC são preferenciais; a ordinária é WFC.
        busca = _BuscaFake({'Wells Fargo & Company': [
            {'symbol': 'WFC-PY', 'exchange': 'NYQ', 'quoteType': 'EQUITY'},
            {'symbol': 'WFC', 'exchange': 'NYQ', 'quoteType': 'EQUITY'},
        ]})
        assert bdrs.resolver_subjacente('Wells Fargo & Company', buscar=busca) == 'WFC'

    def test_ignora_o_que_nao_e_acao(self):
        busca = _BuscaFake({'X': [
            {'symbol': 'XOPT', 'exchange': 'NYQ', 'quoteType': 'OPTION'},
            {'symbol': 'XETF', 'exchange': 'NYQ', 'quoteType': 'ETF'},
        ]})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None

    def test_busca_que_falha_devolve_none(self):
        busca = _BuscaFake({})
        assert bdrs.resolver_subjacente('Inexistente', buscar=busca) is None

    def test_sem_candidato_devolve_none(self):
        busca = _BuscaFake({'X': []})
        assert bdrs.resolver_subjacente('X', buscar=busca) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v -k Resolver`
Expected: FAIL — `module 'src.bdrs' has no attribute 'resolver_subjacente'`

- [ ] **Step 3: Write minimal implementation**

Acrescente a `src/bdrs.py`:

```python
# A bolsa do próprio BDR. Qualquer outra praça é candidata a subjacente.
_BOLSA_DO_BDR = 'SAO'


def resolver_subjacente(long_name: str, buscar=None) -> str | None:
    """
    Encontra o ticker do ativo subjacente a partir do nome legal da empresa.

    A busca do Yahoo não é confiável sozinha: ela casou 'Fomento Económico
    Mexicano' com 'VIST' (Vista Energy). Quem torna isso utilizável é o portão
    de qualidade, que rejeita o par por duas medidas independentes da razão do
    BDR. Aqui só descartamos o que nem candidato é.

    Símbolos com '-' são preferenciais e classes especiais (WFC-PY, WFC-PC) —
    queremos a ordinária, que é a que o BDR referencia.

    Args:
        long_name: nome legal vindo do `longName` do screener.
        buscar: injetável nos testes; por padrão `yf.Search`.

    Returns:
        O ticker do subjacente, ou None quando nada serve.
    """
    buscar = buscar if buscar is not None else yf.Search
    try:
        quotes = buscar(long_name, max_results=12).quotes
    except Exception:
        return None

    for q in quotes:
        simbolo = q.get('symbol') or ''
        if (q.get('quoteType') == 'EQUITY'
                and q.get('exchange') != _BOLSA_DO_BDR
                and '-' not in simbolo):
            return simbolo
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v`
Expected: PASS (24 testes)

- [ ] **Step 5: Commit**

```bash
git add src/bdrs.py tests/test_bdrs.py
git commit -m "feat: resolucao do ticker subjacente a partir do longName"
```

---

### Task 9: Portão de qualidade — razão inteira e cotação implícita

**Files:**
- Modify: `src/bdrs.py`
- Test: `tests/test_bdrs.py`

**Interfaces:**
- Consumes: nada
- Produces: `bdrs.TOLERANCIA_RAZAO = 0.02`, `bdrs.DESVIO_MAXIMO = 0.03`, `bdrs.MIN_PARES_POR_MOEDA = 3`, `bdrs.razao_e_inteira(razao: float) -> bool`, `bdrs.cotacao_implicita(razao, preco_bdr, preco_subjacente) -> float`, `bdrs.aprovar_pelo_portao(candidatos: list[dict]) -> tuple[list[dict], dict[str, int]]`

Cada candidato é um dict com `ticker_bdr`, `ticker`, `razao`, `preco_bdr`, `preco_subjacente`, `moeda_pregao`.

- [ ] **Step 1: Write the failing test**

Acrescente a `tests/test_bdrs.py`:

```python
def _cand(ticker, razao, preco_bdr, preco_sub, moeda_pregao='USD'):
    return {'ticker_bdr': f'{ticker}34.SA', 'ticker': ticker, 'razao': razao,
            'preco_bdr': preco_bdr, 'preco_subjacente': preco_sub,
            'moeda_pregao': moeda_pregao}


class TestRazaoEInteira:
    """A razão do BDR é quantos BDRs equivalem a uma ação: sempre inteira."""

    @pytest.mark.parametrize('razao', [20.0, 1.0, 120.0, 20.02, 19.98])
    def test_aceita_inteiro_dentro_da_tolerancia(self, razao):
        assert bdrs.razao_e_inteira(razao)

    @pytest.mark.parametrize('razao', [0.615, 20.05, 19.9, 0.0, -20.0])
    def test_rejeita_fora_da_tolerancia_ou_nao_positiva(self, razao):
        assert not bdrs.razao_e_inteira(razao)

    def test_rejeita_nan(self):
        assert not bdrs.razao_e_inteira(np.nan)


class TestCotacaoImplicita:
    def test_e_a_cotacao_que_o_par_implica(self):
        # 20 BDRs a R$ 78,64 equivalem a uma ação de US$ 304,76 -> R$ 5,16/US$
        assert bdrs.cotacao_implicita(20.0, 78.64, 304.76) == pytest.approx(5.16, abs=0.01)

    def test_preco_zero_ou_ausente_devolve_nan(self):
        assert pd.isna(bdrs.cotacao_implicita(20.0, 78.64, 0))
        assert pd.isna(bdrs.cotacao_implicita(20.0, np.nan, 304.76))


class TestAprovarPeloPortao:
    """
    O portão não lê cotação nenhuma: ele compara a cotação que cada par implica
    com a mediana do próprio universo. Um par errado implica um câmbio sem
    sentido — FMXB34 casado com VIST deu 15% de desvio.
    """

    def _cinco_bons(self):
        # Todos implicam ~5,16
        return [_cand('AAPL', 20.0, 78.64, 304.76),
                _cand('MSFT', 24.0, 108.22, 503.0),
                _cand('GOGL', 12.0, 148.06, 344.3),
                _cand('JPMC', 10.0, 187.07, 362.5),
                _cand('NFLX', 50.0, 7.73, 74.9)]

    def test_aprova_pares_concordantes(self):
        aprovados, descartes = bdrs.aprovar_pelo_portao(self._cinco_bons())
        assert len(aprovados) == 5
        assert descartes == {}

    def test_rejeita_razao_nao_inteira(self):
        cands = self._cinco_bons() + [_cand('FMXB', 0.615, 100.0, 30.0)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(cands)
        assert 'FMXB' not in [a['ticker'] for a in aprovados]
        assert descartes['razao_nao_inteira'] == 1

    def test_rejeita_cotacao_implicita_fora_da_tolerancia(self):
        # razão inteira, mas implica ~5,95 contra mediana ~5,16 (15% de desvio)
        cands = self._cinco_bons() + [_cand('FMXB', 1.0, 5.95, 1.0)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(cands)
        assert 'FMXB' not in [a['ticker'] for a in aprovados]
        assert descartes['cotacao_divergente'] == 1

    def test_fronteira_da_tolerancia(self):
        base = self._cinco_bons()
        mediana = 78.64 * 20 / 304.76
        for fator, esperado in [(1.0299, True), (1.0301, False)]:
            cand = _cand('BORD', 1.0, mediana * fator, 1.0)
            aprovados, _ = bdrs.aprovar_pelo_portao(base + [cand])
            assert ('BORD' in [a['ticker'] for a in aprovados]) is esperado

    def test_mediana_e_por_moeda_de_pregao(self):
        # Um grupo em EUR com cotação bem diferente não pode deslocar o de USD.
        usd = self._cinco_bons()
        eur = [_cand(f'E{i}', 1.0, 6.5, 1.0, moeda_pregao='EUR') for i in range(3)]
        aprovados, descartes = bdrs.aprovar_pelo_portao(usd + eur)
        assert len(aprovados) == 8
        assert descartes == {}

    def test_mediana_ignora_os_reprovados_pela_razao(self):
        # Seis pares com razão quebrada e cotação absurda não podem virar a
        # referência do grupo.
        bons = self._cinco_bons()
        ruins = [_cand(f'R{i}', 0.5, 99.0, 1.0) for i in range(6)]
        aprovados, _ = bdrs.aprovar_pelo_portao(bons + ruins)
        assert sorted(a['ticker'] for a in aprovados) == \
            sorted(c['ticker'] for c in bons)

    def test_grupo_pequeno_demais_nao_aprova_ninguem(self):
        dois = self._cinco_bons()[:2]
        aprovados, descartes = bdrs.aprovar_pelo_portao(dois)
        assert aprovados == []
        assert descartes['moeda_com_poucos_pares'] == 2

    def test_nao_le_cotacao_externa(self, monkeypatch):
        # Trava a separação: mesmo com qualquer variável de câmbio absurda no
        # ambiente, o conjunto aprovado é o mesmo.
        monkeypatch.setenv('USD_BRL_RATE', '999')
        aprovados, _ = bdrs.aprovar_pelo_portao(self._cinco_bons())
        assert len(aprovados) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v -k "Razao or Cotacao or Portao"`
Expected: FAIL — `module 'src.bdrs' has no attribute 'razao_e_inteira'`

- [ ] **Step 3: Write minimal implementation**

Acrescente a `src/bdrs.py`:

```python
# Tolerâncias do portão. Vêm de premissa, não de ajuste à amostra: o desvio da
# cotação implícita é dominado por defasagem de cotação (o BDR tem 15 minutos
# de atraso e o subjacente é outro instante), e 3% é folga confortável para
# descasamento intradiário sem acomodar erro de identidade — o par errado
# medido deu 15%, cinco vezes o limite.
TOLERANCIA_RAZAO = 0.02
DESVIO_MAXIMO = 0.03
# Mediana de dois pares não é referência: qualquer um dos dois a define.
MIN_PARES_POR_MOEDA = 3


def razao_e_inteira(razao: float) -> bool:
    """
    True se a razão é um inteiro positivo dentro da tolerância.

    A razão do BDR é quantos recibos equivalem a uma ação — 20 para AAPL34, 120
    para MELI34. Um valor quebrado significa que o par está errado.
    """
    if razao is None or pd.isna(razao) or razao <= 0:
        return False
    return abs(razao - round(razao)) <= TOLERANCIA_RAZAO


def cotacao_implicita(razao: float, preco_bdr: float,
                      preco_subjacente: float) -> float:
    """
    A cotação de câmbio que o par implica: razão × preço do BDR ÷ preço da ação.

    Num par correto esse número é o câmbio de mercado. Num par errado é um valor
    sem sentido — e é isso que o portão detecta, sem precisar consultar o câmbio
    de verdade em lugar nenhum.
    """
    for valor in (razao, preco_bdr, preco_subjacente):
        if valor is None or pd.isna(valor):
            return np.nan
    if preco_subjacente == 0:
        return np.nan
    return razao * preco_bdr / preco_subjacente


def aprovar_pelo_portao(candidatos: list[dict]) -> tuple[list[dict], dict]:
    """
    Aprova os pares cujas duas medidas da razão do BDR concordam.

    Duas condições sobre a mediana das cotações implícitas:

    1. É por MOEDA DE PREGÃO do subjacente. Pares que implicam BRL/USD e pares
       que implicam BRL/EUR são populações diferentes, e misturá-las produziria
       uma mediana que não é cotação de nada.
    2. Entram nela apenas os pares que já passaram no teste da razão inteira,
       para que um punhado de pares errados não desloque a referência.

    Returns:
        (aprovados, descartes por motivo)
    """
    descartes = {}

    def descartar(motivo, n=1):
        descartes[motivo] = descartes.get(motivo, 0) + n

    com_razao, aprovados = [], []
    for c in candidatos:
        if razao_e_inteira(c['razao']):
            com_razao.append(c)
        else:
            descartar('razao_nao_inteira')

    por_moeda = {}
    for c in com_razao:
        por_moeda.setdefault(c['moeda_pregao'], []).append(c)

    for moeda, grupo in por_moeda.items():
        implicitas = [cotacao_implicita(c['razao'], c['preco_bdr'],
                                        c['preco_subjacente']) for c in grupo]
        validas = [x for x in implicitas if pd.notna(x)]
        if len(validas) < MIN_PARES_POR_MOEDA:
            descartar('moeda_com_poucos_pares', len(grupo))
            print(f"[bdrs] moeda de pregão {moeda}: só {len(validas)} pares "
                  f"válidos (mínimo {MIN_PARES_POR_MOEDA}) — grupo inteiro fora")
            continue

        mediana = float(np.median(validas))
        for c, implicita in zip(grupo, implicitas):
            if pd.isna(implicita) or abs(implicita / mediana - 1) > DESVIO_MAXIMO:
                descartar('cotacao_divergente')
                continue
            aprovados.append({**c, 'cotacao_implicita': implicita})

    return aprovados, descartes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v`
Expected: PASS (39 testes)

- [ ] **Step 5: Commit**

```bash
git add src/bdrs.py tests/test_bdrs.py
git commit -m "feat: portao de qualidade auto-calibrado pela cotacao implicita"
```

---

### Task 10: Elegibilidade por moeda e os dois frames de saída

**Files:**
- Modify: `src/bdrs.py`
- Test: `tests/test_bdrs.py`

**Interfaces:**
- Consumes: `valuation.macro_for` da Task 5, `paths.filters_file` da Task 1, `aprovar_pelo_portao` da Task 9
- Produces: `bdrs.motivo_inelegibilidade(moeda_pregao, moeda_balanco, regiao, regioes_pedidas) -> str | None`, `bdrs.montar_frames(aprovados: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

Acrescente a `tests/test_bdrs.py`:

```python
class TestMotivoInelegibilidade:
    """
    Sem conversão de moeda, `preco ÷ LPA` só é um P/L quando pregão e balanço
    estão na mesma moeda. E sem premissas macro não há taxa de desconto.
    """

    def test_elegivel_devolve_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        (tmp_path / 'us' / 'filters.json').write_text('{}')
        assert bdrs.motivo_inelegibilidade('USD', 'USD', 'us', {'us'}) is None

    def test_moeda_divergente_exclui(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'us').mkdir()
        (tmp_path / 'us' / 'filters.json').write_text('{}')
        # UL negocia em USD e reporta em EUR; TSM em USD e reporta em TWD.
        assert bdrs.motivo_inelegibilidade('USD', 'EUR', 'us', {'us'}) == 'moeda_divergente'
        # AZN.L cota em pence com balanço em dólar — sai pela mesma condição,
        # então a armadilha do GBp nunca precisa de tratamento.
        assert bdrs.motivo_inelegibilidade('GBp', 'USD', 'gb', {'gb'}) == 'moeda_divergente'

    def test_moeda_sem_premissas_exclui(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        (tmp_path / 'tw').mkdir()
        (tmp_path / 'tw' / 'filters.json').write_text('{}')
        assert bdrs.motivo_inelegibilidade('TWD', 'TWD', 'tw', {'tw'}) == 'moeda_sem_premissas'

    def test_regiao_descoberta_sem_config_exclui_so_o_papel(self, tmp_path, monkeypatch):
        # Um BDR resolvido para Londres não pode derrubar a rodada inteira.
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        assert bdrs.motivo_inelegibilidade('USD', 'USD', 'gb', {'us'}) == 'regiao_sem_config'

    def test_regiao_pedida_sem_config_levanta_erro(self, tmp_path, monkeypatch):
        # Você pediu por ela: filtrar por defaults que ninguém escolheu é pior.
        monkeypatch.setattr(bdrs.paths, 'CONFIG_ROOT', tmp_path)
        with pytest.raises(FileNotFoundError):
            bdrs.motivo_inelegibilidade('USD', 'USD', 'us', {'us'})


class TestMontarFrames:
    """
    Os dois frames têm tempos de vida diferentes: o par é estável e vai para o
    tickers.csv; preço e liquidez são voláteis e não podem ser gravados lá,
    senão o arquivo "estável" guarda uma cotação velha que ninguém sabe que é
    velha.
    """

    def _aprovado(self):
        return [{'ticker': 'AAPL', 'ticker_bdr': 'AAPL34.SA', 'razao': 20.0,
                 'moeda': 'USD', 'preco_bdr': 78.64, 'volume_bdr': 468071,
                 'preco_subjacente': 304.76, 'moeda_pregao': 'USD'}]

    def test_pares_tem_so_as_colunas_estaveis(self):
        pares, _ = bdrs.montar_frames(self._aprovado())
        assert list(pares.columns) == ['ticker', 'ticker_bdr', 'razao', 'moeda']

    def test_cotacoes_tem_so_as_volateis(self):
        _, cotacoes = bdrs.montar_frames(self._aprovado())
        assert list(cotacoes.columns) == ['ticker', 'preco_bdr', 'liq_media_diaria_bdr']

    def test_liquidez_e_volume_vezes_preco_do_bdr(self):
        _, cotacoes = bdrs.montar_frames(self._aprovado())
        assert cotacoes['liq_media_diaria_bdr'].iloc[0] == pytest.approx(78.64 * 468071)

    def test_os_dois_frames_sao_disjuntos_fora_do_ticker(self):
        pares, cotacoes = bdrs.montar_frames(self._aprovado())
        comuns = set(pares.columns) & set(cotacoes.columns)
        assert comuns == {'ticker'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_bdrs.py -v -k "Inelegibilidade or MontarFrames"`
Expected: FAIL — `module 'src.bdrs' has no attribute 'motivo_inelegibilidade'`

- [ ] **Step 3: Write minimal implementation**

Acrescente ao topo de `src/bdrs.py`, junto dos imports:

```python
from src import paths
from src.valuation import macro_for
```

e ao final do módulo:

```python
def motivo_inelegibilidade(moeda_pregao: str, moeda_balanco: str, regiao: str,
                           regioes_pedidas: set) -> str | None:
    """
    Diz por que um par aprovado no portão ainda assim não entra, ou None.

    Três condições, todas eliminatórias:

    - `moeda_divergente`: pregão e balanço em moedas diferentes. Sem conversão,
      `preco ÷ LPA` só é um P/L quando as duas são a mesma. Uma comparação
      substitui uma família inteira de conversões, e a armadilha do `GBp`
      (pence, centésimo de libra) desaparece junto — `AZN.L` sai por aqui.
    - `moeda_sem_premissas`: sem juro livre de risco e prêmio de risco daquela
      moeda não existe taxa de desconto.
    - `regiao_sem_config`: o subjacente caiu numa região que não tem
      `config/<r>/filters.json`.

    A última distingue região PEDIDA de região DESCOBERTA. Se você pediu a
    região, a falta do arquivo é erro — filtrar por defaults que ninguém
    escolheu é pior que parar. Se ela apenas apareceu porque um subjacente
    resolveu para lá, derrubar a rodada inteira por causa de um papel seria
    desproporcional, e ele é apenas excluído.
    """
    if (moeda_pregao or '') != (moeda_balanco or ''):
        return 'moeda_divergente'

    if macro_for(moeda_balanco) is None:
        return 'moeda_sem_premissas'

    if not paths.filters_file(regiao).exists():
        if regiao in regioes_pedidas:
            raise FileNotFoundError(
                f"filtros da região {regiao!r} não encontrados em "
                f"{paths.filters_file(regiao)}. Crie o arquivo antes de rodar "
                f"o screener nessa região."
            )
        return 'regiao_sem_config'

    return None


def montar_frames(aprovados: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o resultado em identidade estável e cotações voláteis.

    O portão precisa do preço do BDR para validar o par, mas esse preço não pode
    ser gravado no `tickers.csv`: o arquivo fica em cache por meses e passaria a
    carregar uma cotação velha que ninguém sabe que está velha. As cotações são
    recoletadas a cada rodada e mescladas no `fundamentals.csv` por `ticker`.

    Returns:
        (pares, cotacoes) — o primeiro vai para data/<r>/tickers.csv.
    """
    if not aprovados:
        vazio_pares = pd.DataFrame(columns=['ticker', 'ticker_bdr', 'razao', 'moeda'])
        vazio_cot = pd.DataFrame(columns=['ticker', 'preco_bdr', 'liq_media_diaria_bdr'])
        return vazio_pares, vazio_cot

    df = pd.DataFrame(aprovados)
    pares = df[['ticker', 'ticker_bdr', 'razao', 'moeda']].copy()

    cotacoes = df[['ticker', 'preco_bdr']].copy()
    # Liquidez do BDR em reais: preço e volume vêm os dois do pregão de B3, e
    # é ela que diz se dá para comprar e vender o papel aqui. A liquidez do
    # subjacente em Nova York não é negociável daqui e não entra em critério.
    cotacoes['liq_media_diaria_bdr'] = df['preco_bdr'] * df['volume_bdr']

    return pares, cotacoes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/ -v`
Expected: PASS — suíte inteira verde

- [ ] **Step 5: Commit**

```bash
git add src/bdrs.py tests/test_bdrs.py
git commit -m "feat: elegibilidade por moeda e separacao dos frames estavel/volatil"
```

---

### Task 11: `config/us/filters.json` e `.env.example`

**Files:**
- Create: `config/us/filters.json`
- Modify: `.env.example`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `filters._load_config` da Task 2
- Produces: nada em código — apenas configuração

- [ ] **Step 1: Write the failing test**

Acrescente a `tests/test_filters.py`:

```python
class TestConfigDaRegiaoUS:
    """
    Os limiares são idênticos aos brasileiros por premissa: o critério de
    'barato' é do investidor, não do mercado. Se a bolsa americana quase não
    produz empresa a 10x lucro, a lista vem curta — e lista curta é informação.
    """

    def test_existe_e_tem_as_mesmas_chaves_de_bloco(self):
        us = filters._load_config('us')
        br = filters._load_config('br')
        assert set(us) == set(br) == {'stock_filters', 'bank_filters'}

    def test_limiares_de_barato_sao_iguais_aos_do_br(self):
        us = filters._load_config('us')['stock_filters']
        br = filters._load_config('br')['stock_filters']
        for chave in ('pl_max', 'pvp_max', 'roe_pct_min', 'margem_liquida_pct_min'):
            assert us[chave] == br[chave]

    def test_liquidez_e_a_do_bdr_em_reais(self):
        for bloco in ('stock_filters', 'bank_filters'):
            cfg = filters._load_config('us')[bloco]
            assert 'liq_media_diaria_bdr_min' in cfg
            assert 'liq_media_diaria_min' not in cfg

    def test_bank_filters_do_us_nao_tem_dy_pct_min(self):
        # O dividendo estrangeiro sofre retenção antes de chegar ao detentor do
        # BDR, então o dividendYield do yfinance é o do acionista de lá.
        assert 'dy_pct_min' not in filters._load_config('us')['bank_filters']
        assert 'dy_pct_min' in filters._load_config('br')['bank_filters']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python3 -m pytest tests/test_filters.py -v -k RegiaoUS`
Expected: FAIL com `FileNotFoundError: filtros da região 'us' não encontrados`

- [ ] **Step 3: Criar os arquivos**

```bash
mkdir -p config/us
```

`config/us/filters.json`:

```json
{
  "stock_filters": {
    "pl_min": 0,
    "pl_max": 10,
    "pvp_min": 0,
    "pvp_max": 1.5,
    "margem_ebit_pct_min": 0,
    "margem_liquida_pct_min": 10,
    "dl_ebit_max": 3,
    "dl_pl_max": 2,
    "roe_pct_min": 10,
    "liquidez_corrente_min": 1,
    "passivos_ativos_max": 1,
    "liq_media_diaria_bdr_min": 100000,
    "lpa_min": 0,
    "crescimento_receita_pct_min": 0,
    "crescimento_lucro_pct_min": 0,
    "num_analistas_min": 2,
    "lpa_estimado_min": 0,
    "exigir_num_analistas": false,
    "exigir_estimativa": true,
    "exigir_lpa_estimado": true
  },
  "bank_filters": {
    "pl_min": 0,
    "pl_max": 10,
    "pvp_min": 0,
    "pvp_max": 2.0,
    "roe_pct_min": 15,
    "margem_liquida_pct_min": 10,
    "lpa_min": 0,
    "liq_media_diaria_bdr_min": 100000,
    "crescimento_receita_pct_min": 0,
    "crescimento_lucro_pct_min": 0,
    "num_analistas_min": 2,
    "lpa_estimado_min": 0,
    "exigir_num_analistas": false,
    "exigir_estimativa": true,
    "exigir_lpa_estimado": true
  }
}
```

Acrescente ao final de `.env.example`:

```bash
# --- Macro por moeda (ativos estrangeiros) ---
# BRL e USD já sobem configurados no código; as linhas abaixo só existem para
# você ajustar os valores. Qualquer OUTRA moeda precisa das duas variáveis, e
# até lá o papel que reporta nela é excluído com log dizendo o que falta.
# Ex.: para habilitar a Europa, acrescente RISK_FREE_RATE_EUR e
# EQUITY_RISK_PREMIUM_EUR.
# Taxa livre de risco em dólar: Treasury longo. Default: 0.042
RISK_FREE_RATE_USD=0.042
# Prêmio de risco de mercado EUA. Default: 0.045
EQUITY_RISK_PREMIUM_USD=0.045

# Região varrida pelo yf.screen para montar o universo de BDRs. Default: br
# Não é a região de destino: os subjacentes vão para a pasta da bolsa em que
# negociam (AAPL -> data/us/).
BDR_REGION=br
```

- [ ] **Step 4: Run test to verify it passes**

Run: `rtk proxy python3 -m pytest tests/ -v`
Expected: PASS — suíte inteira verde

- [ ] **Step 5: Commit**

```bash
git add config/us/filters.json .env.example tests/test_filters.py
git commit -m "feat: filtros da regiao us e premissas macro em dolar no .env"
```

---

### Task 12: Pipeline da região `us` no notebook

**Files:**
- Modify: `analysis.ipynb` (células 3, 5, 7, 11, 13, 15 e novas)
- Test: execução do notebook via nbconvert

**Interfaces:**
- Consumes: tudo das tasks 1-11
- Produces: `data/us/tickers.csv`, `data/us/fundamentals.csv`, `data/us/valuation_history.csv`

- [ ] **Step 1: Ajustar as células existentes para passar a região**

Na célula 1 (setup), troque as duas linhas de import e reload por:

```python
from src import scraper, fundamentals, filters, valuation, bdrs, paths

# Recarregar módulos durante desenvolvimento
for mod in [scraper, fundamentals, filters, valuation, bdrs, paths]:
```

Na célula 3, troque por:

```python
# Obter tickers de ações brasileiras
tickers_df = scraper.get_tickers(region='br')
print(f"Total de tickers: {len(tickers_df)}")
tickers_df.head(10)
```

Na célula 5:

```python
# Coletar fundamentals para todas as ações brasileiras
stock_fundamentals = fundamentals.fetch_fundamentals(
    tickers_df['ticker_sa'].tolist(), delay=0.4, region='br')
stock_fundamentals.head()
```

Na célula 7, troque a chamada do filtro por `filters.apply_stock_filters(stock_fundamentals_clean, region='br')`.
Na célula 11, troque por `filters.apply_bank_filters(bank_fundamentals, region='br')`.
Na célula 15, troque por `valuation.append_snapshot(all_valued_df, region='br')`.

- [ ] **Step 2: Acrescentar a célula do pipeline da região `us`**

Nova célula, após a célula 13:

```python
# --- Região us: BDRs de B3 -> ativo subjacente ---
# O BDR é o ticker negociável; o subjacente é a empresa, na moeda dela.
import os

from src import bdrs

BDR_REGION = os.getenv('BDR_REGION', 'br')

universo = bdrs.buscar_universo(region=BDR_REGION)
candidatos_bdr = bdrs.selecionar_bdrs(universo)
print(f"BDRs no universo: {len(candidatos_bdr)}")
```

Nova célula seguinte:

```python
# Resolver o subjacente de cada BDR e validar o par pelo portão.
# Só roda quando data/us/tickers.csv não existe — a resolução é cara
# (duas requisições por BDR) e o par é estável, como o cache de tickers do BR.
import time

import yfinance as yf

from src import paths

cache_pares = paths.data_file('us', 'tickers.csv')

if cache_pares.exists():
    pares_us = pd.read_csv(cache_pares)
    print(f"[us] {len(pares_us)} pares carregados do cache ({cache_pares})")
    cotacoes_us = None
else:
    candidatos = []
    descartes = {'sem_candidato': 0}
    for q in candidatos_bdr:
        ticker_sub = bdrs.resolver_subjacente(q['longName'])
        if ticker_sub is None:
            descartes['sem_candidato'] += 1
            continue
        info_sub = yf.Ticker(ticker_sub).info or {}
        candidatos.append({
            'ticker': ticker_sub,
            'ticker_bdr': q['symbol'],
            'razao': bdrs.razao_acoes_do_par(q.get('sharesOutstanding'),
                                             info_sub.get('sharesOutstanding')),
            'preco_bdr': q.get('regularMarketPrice'),
            'volume_bdr': q.get('averageDailyVolume10Day'),
            'preco_subjacente': info_sub.get('currentPrice'),
            'moeda_pregao': info_sub.get('currency'),
            'moeda': info_sub.get('financialCurrency'),
            'regiao': bdrs.regiao_do_ticker(ticker_sub),
        })
        time.sleep(0.3)

    aprovados, descartes_portao = bdrs.aprovar_pelo_portao(candidatos)
    descartes.update(descartes_portao)

    elegiveis = []
    for a in aprovados:
        motivo = bdrs.motivo_inelegibilidade(
            a['moeda_pregao'], a['moeda'], a['regiao'], regioes_pedidas={'us'})
        if motivo:
            descartes[motivo] = descartes.get(motivo, 0) + 1
            continue
        elegiveis.append(a)

    pares_us, cotacoes_us = bdrs.montar_frames(
        [e for e in elegiveis if e['regiao'] == 'us'])
    cache_pares.parent.mkdir(parents=True, exist_ok=True)
    pares_us.to_csv(cache_pares, index=False)

    print(f"[us] {len(pares_us)} pares aprovados de {len(candidatos_bdr)} BDRs")
    for motivo, n in sorted(descartes.items()):
        print(f"[us]   descartados por {motivo}: {n}")

    # A taxa medida até aqui (21/22) valeu só para bolsas americanas. Para as
    # demais praças ela é desconhecida — este resumo é o que a revela.
    display(bdrs.resumo_por_regiao(candidatos, elegiveis))
```

Nova célula seguinte:

```python
# Fundamentos do subjacente, na moeda do balanço dele, com o beta contra o S&P.
us_fundamentals = fundamentals.fetch_fundamentals(
    pares_us['ticker'].tolist(), delay=0.4, region='us',
    index_symbol='^GSPC', incluir_liq_local=False)

# As cotações do BDR são voláteis: entram por merge, não pelo tickers.csv.
if cotacoes_us is not None:
    us_fundamentals = us_fundamentals.merge(cotacoes_us, on='ticker', how='left')
us_fundamentals = us_fundamentals.merge(
    pares_us[['ticker', 'ticker_bdr', 'razao']], on='ticker', how='left')

us_clean = us_fundamentals.copy()
for col in numeric_cols + ['preco_bdr', 'liq_media_diaria_bdr', 'razao']:
    if col in us_clean.columns:
        us_clean[col] = pd.to_numeric(us_clean[col], errors='coerce')

us_clean.head()
```

Nova célula seguinte:

```python
# Filtros e valuation da região us. As medianas setoriais saem do frame da
# PRÓPRIA região: usar as brasileiras colocaria o P/L mediano de "Technology"
# no Brasil dentro da fórmula de Graham de uma empresa americana.
from src.scraper import BANK_INDUSTRIES

us_bank_mask = us_clean['industria'].str.lower().isin(BANK_INDUSTRIES)

filtered_us_stocks = filters.apply_stock_filters(us_clean[~us_bank_mask], region='us')
filtered_us_banks = filters.apply_bank_filters(us_clean[us_bank_mask], region='us')

valued_us = []
if len(filtered_us_stocks) > 0:
    vs = valuation.apply_valuation(filtered_us_stocks, us_clean, model='stock')
    vs['tipo'] = 'ação'
    valued_us.append(vs)
if len(filtered_us_banks) > 0:
    vb = valuation.apply_valuation(filtered_us_banks, us_clean, model='bank')
    vb['tipo'] = 'banco'
    valued_us.append(vb)

us_valued_df = pd.concat(valued_us, ignore_index=True) if valued_us else pd.DataFrame()
if len(us_valued_df) > 0:
    valuation.append_snapshot(us_valued_df, region='us')
```

Nova célula seguinte, para o ranking unificado:

```python
# Ranking unificado. É a única visão em que `preco` carrega moedas diferentes
# por linha — por isso a coluna `moeda` é exibida junto. A margem de segurança
# é adimensional e comparável entre regiões sem nenhuma cotação.
ranking = pd.concat(
    [df for df in (all_valued_df.assign(regiao='br') if len(all_valued_df) else None,
                   us_valued_df.assign(regiao='us') if len(us_valued_df) else None)
     if df is not None],
    ignore_index=True,
)

if len(ranking) > 0 and ranking['undervalued'].any():
    top20 = (ranking[ranking['undervalued']]
             .sort_values('margem_seg_media_pct', ascending=False)
             .head(20))
    display(top20[['regiao', 'tipo', 'moeda', 'ticker', 'nome', 'preco',
                   'preco_justo_dcf', 'margem_seg_media_pct', 'metodo_valuation']])
else:
    print("Nenhuma ação subvalorizada nesta rodada.")
```

- [ ] **Step 3: Acrescentar os dois helpers que o notebook usa**

Em `src/bdrs.py`:

```python
# Sufixo do ticker -> região da bolsa. Ticker sem ponto é americano, que é
# onde a maioria dos subjacentes negocia, inclusive europeus e asiáticos via
# ADR (UL, TSM, UBS, YPF).
_REGIAO_POR_SUFIXO = {
    'SA': 'br', 'L': 'gb', 'DE': 'de', 'PA': 'fr', 'SW': 'ch',
    'T': 'jp', 'AX': 'au', 'TO': 'ca', 'HK': 'hk', 'MX': 'mx',
}


def regiao_do_ticker(ticker: str) -> str:
    """Região da bolsa em que o ticker negocia, pelo sufixo do símbolo."""
    if '.' not in ticker:
        return 'us'
    return _REGIAO_POR_SUFIXO.get(ticker.rsplit('.', 1)[1].upper(), 'desconhecida')


def razao_acoes_do_par(shares_bdr: float, shares_subjacente: float) -> float:
    """
    Quantos BDRs equivalem a uma ação, pelo número de ações de cada lado.

    O yfinance já entrega o `sharesOutstanding` do BDR em unidades de BDR
    (AAPL34: 291,88 bi contra 14,59 bi da AAPL), então a divisão dá a razão
    direto — 20, no caso.
    """
    for valor in (shares_bdr, shares_subjacente):
        if valor is None or pd.isna(valor) or valor == 0:
            return np.nan
    return shares_bdr / shares_subjacente


def resumo_por_regiao(candidatos: list[dict], aprovados: list[dict]) -> pd.DataFrame:
    """
    Taxa de aprovação por região de bolsa do subjacente.

    Existe porque a única medição feita até aqui restringiu os candidatos a
    bolsas americanas, e aprovou 21 de 22. Para as demais praças a taxa é
    DESCONHECIDA — imprimi-la a cada rodada é o que faz isso deixar de ser
    suposição na primeira execução real.
    """
    total = {}
    for c in candidatos:
        total[c['regiao']] = total.get(c['regiao'], 0) + 1
    ok = {}
    for a in aprovados:
        ok[a['regiao']] = ok.get(a['regiao'], 0) + 1

    linhas = [{'regiao': r, 'candidatos': n, 'aprovados': ok.get(r, 0),
               'taxa_pct': round(100 * ok.get(r, 0) / n, 1)}
              for r, n in sorted(total.items())]
    return pd.DataFrame(linhas)
```

E acrescente os testes correspondentes em `tests/test_bdrs.py`:

```python
class TestRegiaoDoTicker:
    @pytest.mark.parametrize('ticker,esperado', [
        ('AAPL', 'us'), ('JPM', 'us'), ('PETR4.SA', 'br'),
        ('AZN.L', 'gb'), ('SAP.DE', 'de'), ('7203.T', 'jp'),
        ('XXXX.ZZ', 'desconhecida'),
    ])
    def test_sufixo_decide_a_regiao(self, ticker, esperado):
        assert bdrs.regiao_do_ticker(ticker) == esperado


class TestRazaoAcoesDoPar:
    def test_divide_as_contagens(self):
        assert bdrs.razao_acoes_do_par(291_883_600_000, 14_594_180_000) == pytest.approx(20.0)

    @pytest.mark.parametrize('a,b', [(None, 10), (10, None), (10, 0), (np.nan, 10)])
    def test_faltando_um_lado_devolve_nan(self, a, b):
        assert pd.isna(bdrs.razao_acoes_do_par(a, b))


class TestResumoPorRegiao:
    """A taxa medida valeu só para bolsas americanas; o resumo é o que revela
    a das demais praças na primeira rodada real."""

    def test_conta_candidatos_e_aprovados_por_regiao(self):
        candidatos = [{'regiao': 'us'}, {'regiao': 'us'}, {'regiao': 'gb'}]
        aprovados = [{'regiao': 'us'}]

        resumo = bdrs.resumo_por_regiao(candidatos, aprovados)

        por_regiao = resumo.set_index('regiao').to_dict('index')
        assert por_regiao['us'] == {'candidatos': 2, 'aprovados': 1, 'taxa_pct': 50.0}
        assert por_regiao['gb'] == {'candidatos': 1, 'aprovados': 0, 'taxa_pct': 0.0}
```

- [ ] **Step 4: Rodar a suíte e executar o notebook**

Run: `rtk proxy python3 -m pytest tests/ -v`
Expected: PASS — suíte inteira verde

Run: `rtk proxy python3 -m jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`
Expected: execução sem exceção; a saída deve imprimir a contagem de BDRs aprovados e os descartes por motivo, e `data/us/tickers.csv`, `data/us/fundamentals.csv` e `data/us/valuation_history.csv` devem existir ao final.

- [ ] **Step 5: Commit**

```bash
git add analysis.ipynb src/bdrs.py tests/test_bdrs.py   # data/ e gitignored
git commit -m "feat: pipeline da regiao us no notebook, com ranking unificado"
```

---

## Notas de execução

**Ordem obrigatória.** As tasks 1-6 são o alicerce (caminhos por região e premissas por moeda) e precisam vir antes das 7-12. Dentro de 7-10, cada uma acrescenta uma função a `src/bdrs.py` e depende da anterior apenas para o módulo existir.

**Só `config/filters.json` é rastreado pelo git** — ele vai para `config/br/` com `git mv` (Task 2). Os três CSVs de `data/` estão no `.gitignore`: movem-se com `mv` simples, e não entram em nenhum `git add`.

**Ponto de atenção na Task 12.** A célula de resolução é a única que faz ~1200 requisições (duas por BDR). Ela é pulada quando `data/us/tickers.csv` existe. Na primeira execução, contar com dezenas de minutos.

**Correção pendente na spec.** A spec afirma que `fetch_betas` "ganha o índice como parâmetro", mas ele já tem `index_symbol='^BVSP'` desde antes — o que falta é `_fetch_fundamentals_from_api` repassá-lo, que é o que a Task 4 faz. Ajustar a linha correspondente da spec.
