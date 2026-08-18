import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from src.fundamentals import get_fcf_series, resolve_share_count
from src import paths

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
    # Moeda ausente não é moeda: uma coluna lida de CSV traz NaN (float) onde o
    # dado faltou, e NaN é "verdadeiro" para o `or`. Sem premissa não há taxa.
    if not isinstance(moeda, str):
        return None
    moeda = moeda.strip().upper()
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


# Liga o uso de estimativas forward de crescimento (analistas via yfinance) no
# estágio 1 do DCF. Desligado por padrão -> comportamento = crescimento histórico.
USE_FORWARD_ESTIMATES = _env_bool('USE_FORWARD_ESTIMATES', False)
# Qual crescimento alimenta o estágio 1 do DCF: 'revenue' (receita, default) ou
# 'earnings' (lucro). Receita é o default por razão estrutural, não amostral: o
# DCF projeta fluxo de caixa livre (receita menos custos caixa e capex), e o
# lucro contábil amplifica a mesma variação de receita via alavancagem
# operacional, itens não recorrentes e efeitos fiscais.
# Validado aqui em vez de num helper _env_str: é o único uso de string na env.
FORWARD_GROWTH_DRIVER = os.getenv('FORWARD_GROWTH_DRIVER', 'revenue').strip().lower()
if FORWARD_GROWTH_DRIVER not in ('revenue', 'earnings'):
    print(f"[valuation] FORWARD_GROWTH_DRIVER={FORWARD_GROWTH_DRIVER!r} inválido, "
          f"usando 'revenue'")
    FORWARD_GROWTH_DRIVER = 'revenue'
PROJECTION_YEARS = 10      # Horizonte de projeção (2 estágios: decay linear)
# Limiar de PROJETABILIDADE, não teto de crescimento: responde "consigo projetar
# essa taxa por 10 anos?". Quando a resposta é não, o modelo recua para outra
# fonte ou se declara inaplicável (NaN) — nunca troca a taxa por 0,20. Substituir
# silenciosamente produz um preço justo que aparenta ter modelado a empresa
# quando modelou outra, mais saudável, e isso chega ao usuário como recomendação.
# Não existe piso: crescimento negativo é projetado como negativo. Um piso
# seleciona por MAGNITUDE, e magnitude não indica falta de confiabilidade — uma
# empresa que cai todo ano por quatro anos produz um número grande e confiável.
MAX_PROJECTABLE_GROWTH = 0.20
# Segundo limiar de PROJETABILIDADE, e a outra metade da pergunta. O
# MAX_PROJECTABLE_GROWTH acima responde "essa taxa se sustenta por 10 anos?".
# Este responde "essa série descreve uma trajetória, ou são quatro números
# soltos?" — via R², a fatia da variação da série que a reta de tendência
# explica. Duas perguntas independentes, por isso duas constantes.
# 0,5 é premissa, não calibração: a tendência precisa explicar MAIS DA METADE
# da variação; abaixo disso o que a série tem é mais ruído do que trajetória.
# Qualquer 0,45 ou 0,6 só se justificaria olhando quais ações passam, que é
# exatamente o que a Guideline 3 proíbe.
MIN_TREND_R2 = 0.5
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


def _fcf_trend_base(fcf_series: pd.Series) -> float:
    """
    Nível da tendência no último ano, ou NaN se a série não tem trajetória.

    Responde "onde a empresa está hoje?". É pergunta diferente da que
    `_compute_fcf_growth` faz ("ela continua nesse ritmo?"), e por isso as duas
    podem discordar sobre a mesma reta -- ver a nota em `compute_fcf_base`.

    Ajusta a mesma reta sobre o logaritmo dos FCFs e a avalia no ano mais
    recente. Não é o último valor observado: a reta amortece um ano isolado de
    pico, que é a preocupação que fez a base ser a mediana em primeiro lugar.

    Devolve NaN em quatro situações, todas com o mesmo significado -- "não há
    trajetória aqui, use a mediana":

    - Menos de 4 pontos. Com 3, metade das séries sem tendência nenhuma passam
      no teste de R² (é a distribuição do R² sob ruído, não uma observação
      sobre estas ações); com 4, 29%. O yfinance entrega 4 anos, às vezes 3.
    - Qualquer ano com FCF <= 0. Não existe logaritmo de número negativo, e uma
      série que atravessou o prejuízo é justamente aquela em que extrapolar o
      nível é menos confiável -- a limitação técnica coincide com a prudência.
    - Série constante. A variação nula no log tornaria o R² uma divisão por
      zero; a guarda explícita evita decidir por acidente de ponto flutuante.
      (O resultado seria o mesmo: numa série constante o nível da reta é a
      mediana.)
    - R² abaixo de MIN_TREND_R2. A reta explica menos da metade da variação da
      série, ou seja, o que existe ali é mais ruído do que trajetória.
    """
    if len(fcf_series) < 4:
        return np.nan

    # Ordenar do mais antigo ao mais recente: o yfinance entrega ao contrário.
    values = fcf_series.values[::-1].astype(float)

    if any(v <= 0 for v in values):
        return np.nan

    log_values = np.log(values)
    years = np.arange(len(values))

    if np.ptp(log_values) == 0:
        return np.nan

    slope, intercept = np.polyfit(years, log_values, 1)

    r2 = float(np.corrcoef(years, log_values)[0, 1] ** 2)
    if r2 < MIN_TREND_R2:
        return np.nan

    return float(np.exp(intercept + slope * (len(values) - 1)))


def compute_fcf_base(fcf_series: pd.Series) -> float:
    """
    Base de FCF para o DCF: nível da tendência quando existe, mediana quando não.

    A mediana foi a regra única por um tempo, para não ancorar a projeção num
    ano de pico (RSUL4: 41,8M no último ano vs. mediana de 28,3M). O argumento
    vale para série ERRÁTICA e falha para série que sobe (ou desce) todo ano:
    ali a mediana não resiste a pico nenhum -- ela É, por construção, um valor
    do meio da série, ou seja, o nível de dois anos atrás.

    O efeito era grande. A SEER3 (38,1 -> 67,7 -> 116,3 -> 289,0 em R$ mi) tinha
    base 92,0 e o DCF projetava o ano 1 em 96,9 -- abaixo do que a empresa já
    havia entregue. Como o preço justo é linear na base, a subestimação passava
    inteira: R$ 10,98 contra R$ 31,08 partindo do nível atual.

    E errava nos dois sentidos: na CMIG4, caindo todo ano, a mediana ficava
    ACIMA do nível atual e inflava uma empresa em declínio.

    Quem decide qual via vale é `_fcf_trend_base`, que devolve NaN quando a
    série não tem trajetória. Na dúvida, sempre a mediana (Guideline 4).

    Nota sobre uma aparente contradição: na SEER3 esta função aceita o nível da
    reta (260,4M) enquanto `_compute_fcf_growth` rejeita a INCLINAÇÃO da mesma
    reta (+93,8% ao ano, acima do projetável). Não é incoerência -- são duas
    afirmações separáveis. "Onde a empresa está hoje" é sobre o passado
    observado, e a reta responde bem. "Se ela mantém esse ritmo" é sobre dez
    anos de futuro, e a reta responde mal. O contrário é que seria estranho:
    projetar a taxa e desprezar o nível.
    """
    if fcf_series.empty:
        return np.nan

    base = _fcf_trend_base(fcf_series)
    if pd.isna(base):
        base = float(np.median(fcf_series.values))

    return base if base > 0 else np.nan


def _compute_fcf_growth(fcf_series: pd.Series) -> float:
    """
    Crescimento anual do FCF a partir da TENDÊNCIA da série inteira.

    Ajusta uma reta sobre o logaritmo dos FCFs e usa a inclinação dela como
    taxa. O logaritmo entra porque crescimento é multiplicativo: crescer 10%
    todo ano vira uma reta perfeita no log, e a inclinação se lê direto como
    taxa (exp(inclinação) - 1).

    A versão anterior comparava só o primeiro e o último ponto e ignorava o
    caminho entre eles. A RIAA3 (519 -> 951 -> 1.087 -> 351) saía como -12,2%
    a.a., um número que não descreve nenhum dos anos: o cálculo não enxergava
    que a empresa passou por 1.087 no meio.

    Mas trocar o estimador não basta -- a regressão lê a RIAA3 como -9,9%, que
    continua sem sentido. O defeito nunca foi o VALOR da tendência: é que não
    existe tendência ali. Por isso o R² decide. Ele é a fatia da variação da
    série que a reta explica, ou seja, responde "essa série conta uma história
    ou são números soltos?". Abaixo de MIN_TREND_R2 a resposta é não e a função
    devolve NaN: o DCF se declara inaplicável, o chamador recai no DDM e
    metodo_valuation registra a troca. A taxa nunca é substituída por outra.

    R² não enxerga tamanho, só formato: uma série que sobe suavemente 0,5% ao
    ano tem R² = 1,00 e passa, por menor que seja o crescimento. Quem é
    rejeitada é a série ERRÁTICA -- que sobe e desce sem padrão --, mesmo que
    quase estável, como 100 -> 101 -> 100 -> 101. É um erro conhecido, e ele
    erra na direção barata: tira a ação da lista em vez de fazê-la parecer
    barata.

    Devolve NaN (não modelável) ou uma taxa que pode ser negativa. Nunca 0,0:
    zerar seria substituir o número por outro e, pior, não é conservador --
    o estágio 1 do DCF faz a taxa SUBIR de 0 até TERMINAL_GROWTH (12,4%), de
    modo que a série com prejuízo acabava projetada acelerando.
    """
    if len(fcf_series) < 2:
        return np.nan

    # Ordenar do mais antigo ao mais recente: o yfinance entrega ao contrário.
    values = fcf_series.values[::-1].astype(float)

    # Não existe logaritmo de número negativo -- e uma série que passa pelo
    # prejuízo não descreve trajetória de crescimento composto. Caso RSUL4:
    # 21,2M -> 35,4M -> -9,8M -> 41,8M.
    if any(v <= 0 for v in values):
        return np.nan

    log_values = np.log(values)
    years = np.arange(len(values))

    # Série constante: o R² seria uma divisão por zero (numpy devolveria NaN
    # com RuntimeWarning) -- não há variação nenhuma para a reta explicar.
    # Não modelável por DCF.
    if np.ptp(log_values) == 0:
        return np.nan

    slope = np.polyfit(years, log_values, 1)[0]
    growth = float(np.exp(slope) - 1)

    r2 = float(np.corrcoef(years, log_values)[0, 1] ** 2)
    if r2 < MIN_TREND_R2:
        return np.nan

    # Acima do limiar não é "20%": é uma taxa que não se projeta por 10 anos.
    # Abaixo de zero passa direto: declínio é dado válido.
    if growth > MAX_PROJECTABLE_GROWTH:
        return np.nan

    return growth


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


def resolve_forward_growth(row) -> float:
    """
    Crescimento forward da linha do DataFrame, em decimal.

    O dado já está em `data/fundamentals.csv` (colunas `crescimento_receita_pct`
    e `crescimento_lucro_pct`, coletadas pelo screener), então não há requisição
    nova: lê-se a coluna do driver configurado e converte de pontos percentuais
    para decimal (14,8 -> 0,148).

    `crescimento_lucro_pct` não compara a estimativa com o lucro realizado: é
    (estimativa do exercício seguinte − estimativa do exercício corrente) /
    |estimativa do exercício corrente|, as duas linhas ('+1y' e '0y') do
    mesmo consenso de analistas. Com as duas estimativas negativas a razão
    fica positiva, e um prejuízo projetado encolhendo aparece como
    "crescimento de lucro" (ex.: AURE3, −1,25 → −0,14 por ação = +88,6%). Quem
    protege esse caminho na prática é o filtro upstream `lpa_estimado > 0`
    (`exigir_lpa_estimado` em `config/filters.json`): uma linha só chega até
    aqui depois de já ter passado pelo corte de nível, então o caso do
    parágrafo acima é descartado antes de virar taxa de crescimento do DCF.

    Crescimento <= -100% é tratado como dado AUSENTE, não como valor extremo:
    quando a estimativa do exercício seguinte é negativa o bastante para
    superar em módulo a estimativa do exercício corrente, o número deixa de
    funcionar como taxa de crescimento anual (uma queda de mais de 100% ao ano
    não é uma taxa projetável, é um sinal de que o lucro projetado colapsa).
    Recai no crescimento histórico como qualquer estimativa faltante.

    Não decide projetabilidade: essa avaliação fica concentrada em
    `dcf_valuation`, com o limiar MAX_PROJECTABLE_GROWTH.

    Args:
        row: linha do DataFrame de fundamentos (pd.Series).

    Returns:
        Crescimento anual como decimal, ou NaN se ausente/inválido.
    """
    col = ('crescimento_receita_pct' if FORWARD_GROWTH_DRIVER == 'revenue'
           else 'crescimento_lucro_pct')
    value = row.get(col, np.nan)
    if pd.isna(value):
        return np.nan
    growth = float(value) / 100.0
    return growth if growth > -1.0 else np.nan


def _forward_contradicts_history(fcf_series: pd.Series, historical_growth: float,
                                 forward_growth: float) -> bool:
    """
    As duas fontes de crescimento discordam sobre a DIREÇÃO?

    O forward é crescimento de RECEITA (ver FORWARD_GROWTH_DRIVER) e o
    histórico é crescimento de CAIXA LIVRE. São grandezas diferentes, então
    comparar as magnitudes não significa nada: uma empresa pode ter receita
    subindo 4% e caixa caindo 21% sem contradição nenhuma -- basta a margem
    comprimir ou o capex subir. É o retrato normal de uma elétrica em ciclo de
    investimento.

    A DIREÇÃO, essa sim, significa a mesma coisa nas duas medidas. "O dinheiro
    que sobra para o acionista está encolhendo, com a reta explicando 93% da
    variação" e "vai crescer" são afirmações que não podem ser verdadeiras ao
    mesmo tempo sobre a mesma empresa. Quando isso acontece, o DCF projeta a
    queda: a taxa vem do histórico e o forward é descartado.

    Não é clamp e não é teto (Guideline 2): a taxa continua saindo de uma
    fonte real, apenas da outra. O divisor é o zero -- crescer contra
    encolher --, que é premissa, não um limiar ajustado à amostra
    (Guideline 3).

    Exige 4 pontos. Com 3, metade das séries SEM tendência nenhuma passam no
    teste de R² (é a distribuição do R² sob ruído, medida na spec de
    2026-08-17); com exatamente 2, o R² é sempre 1 porque a reta passa pelos
    dois pontos, e o critério de qualidade não filtra nada. Sem esse mínimo,
    dois anos de queda por acaso derrubariam uma estimativa de analista.

    O mínimo vale SÓ aqui. `_compute_fcf_growth` continua aceitando 2 pontos e
    continua sendo a taxa usada quando não há forward: este mínimo não decide
    quem recebe DCF, decide quem tem autoridade para derrubar o forward.

    A ordem da série é irrelevante -- só o tamanho dela é lido.

    Defeito declarado e aceito: uma queda real medida em 3 anos NÃO derruba o
    forward, e nesse caso o preço justo fica mais alto. Contraria a Guideline
    4 nesse caso específico. A Guideline 4 arbitra ignorância simétrica, e um
    R² de 3 pontos não é ignorância -- é uma medida que já se sabe quebrada.
    """
    return (len(fcf_series) >= 4
            and pd.notna(historical_growth) and historical_growth < 0
            and forward_growth >= 0)


def dcf_valuation(ticker_sa: str, shares_total: float = None,
                  beta: float = None, forward_growth: float = None,
                  moeda: str = 'BRL') -> dict:
    """
    Calcula preço justo via DCF de 2 estágios sobre FCF alavancado (estilo SWS).

    Estágio 1: taxa decai linearmente do crescimento inicial até o crescimento
    terminal DA MOEDA do balanço, ao longo de PROJECTION_YEARS anos.
    Estágio 2: perpetuidade de Gordon a esse mesmo crescimento terminal.

    Crescimento inicial: por padrão é o crescimento histórico do FCF (tendência
    log-linear). Com USE_FORWARD_ESTIMATES ligado, usa a estimativa forward
    passada pelo chamador quando ela existe e é projetável (<=
    MAX_PROJECTABLE_GROWTH), caindo de volta no crescimento histórico caso
    contrário. Se nem uma nem outra for projetável, sai sem preço e o chamador
    recai no DDM.

    Exceção: quando as duas fontes discordam sobre a DIREÇÃO -- o histórico
    bem ajustado diz que o caixa encolhe e o forward diz que cresce -- o
    forward é descartado e o histórico prevalece (ver
    _forward_contradicts_history). O forward é crescimento de receita usado
    como proxy do de caixa, e nenhuma das duas medidas sustenta a reversão que
    a substituição afirmaria.

    O 'Free Cash Flow' do yfinance é OCF − CapEx, com o OCF já líquido de juros
    pagos: é FCFE. O valor presente já é o equity value, sem ajuste de dívida.

    Args:
        ticker_sa: Ticker com sufixo .SA
        shares_total: Total de ações (ON + PN). Se None, resolve via yfinance.
        beta: Beta da ação. Se None, busca do yfinance (fallback 1,0).
        forward_growth: Crescimento forward já resolvido em decimal, tipicamente
            de resolve_forward_growth(row). Nunca é buscado aqui dentro: o dado
            já está no CSV de fundamentos.
        moeda: Moeda do BALANÇO da empresa, que é a moeda do fluxo projetado e
            portanto a que decide o juro de desconto e o crescimento na
            perpetuidade. Default 'BRL' — o que valia antes desta coluna
            existir, e verdade para todas as linhas gravadas até então.

    Returns:
        dict com 'preco_justo_dcf', 'growth_rate', 'fcf_base', 'cost_of_equity',
        'growth_source' ('forward' = a estimativa de analista foi usada |
        'historical' = não havia forward utilizável | 'historical_override' =
        havia forward projetável, mas ele contradizia a direção do histórico e
        foi descartado | '' quando o DCF não chegou ao fim -- mesmos quatro
        caminhos e mesmo motivo do comentário em 'fcf_base_source' logo
        abaixo),
        'fcf_base_source' ('trend' | 'median' | '' quando o DCF não chegou ao
        fim -- ver o comentário no dict abaixo).
    """
    result = {
        'preco_justo_dcf': np.nan,
        'growth_rate': np.nan,
        'fcf_base': np.nan,
        'cost_of_equity': np.nan,
        # String vazia pelo mesmo motivo de 'fcf_base_source' logo abaixo:
        # nenhuma taxa foi escolhida nos quatro caminhos de saída antecipada.
        'growth_source': '',
        # String vazia = o DCF não chegou ao fim, então nenhuma base foi
        # USADA. Quatro caminhos levam aqui: série de FCF vazia, mediana
        # não-positiva, contagem de ações ausente, e nenhuma taxa de
        # crescimento projetável. Nos dois primeiros a base nem chegou a ser
        # calculada; nos dois últimos ela foi calculada e descartada junto com
        # a rodada. É diferente de 'median', que significa "a base saiu da
        # mediana e o preço justo foi calculado com ela".
        'fcf_base_source': '',
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

        coe = cost_of_equity(beta, moeda=moeda)
        # Desconto e crescimento na perpetuidade saem da MESMA moeda: misturar
        # um fluxo em dólar com juro de real embutiria inflação de reais num
        # fluxo que não a tem. Moeda sem premissas deixa os dois NaN, e o
        # modelo sai sem preço em vez de sair com um número inventado.
        macro = macro_for(moeda)
        terminal_growth = macro['terminal_growth'] if macro else np.nan

        # Crescimento inicial: crescimento histórico do FCF (pode ser NaN),
        # substituído pela estimativa forward quando ela está ligada, existe e
        # é projetável. Forward acima do limiar NÃO é capado: recua para o
        # histórico.
        initial_growth = _compute_fcf_growth(fcf_series)
        growth_source = 'historical'
        if (USE_FORWARD_ESTIMATES
                and forward_growth is not None and pd.notna(forward_growth)
                and float(forward_growth) <= MAX_PROJECTABLE_GROWTH):
            if _forward_contradicts_history(fcf_series, initial_growth,
                                            float(forward_growth)):
                # As duas fontes discordam sobre a direção: o histórico fica
                # como está e o forward é descartado. Rótulo próprio porque
                # 'historical' já significa outra coisa -- "não havia forward
                # utilizável" --, e confundir os dois faria o histórico de
                # valuation atribuir a dado o que foi mudança de premissa.
                growth_source = 'historical_override'
            else:
                initial_growth = float(forward_growth)
                growth_source = 'forward'

        # Nenhuma taxa projetável (histórico acima do limiar e sem forward
        # utilizável): sai sem preço. O chamador recai no DDM e a coluna
        # metodo_valuation registra a substituição, em vez de o número aparecer
        # como se tivesse saído de um DCF.
        if pd.isna(initial_growth):
            return result

        result['preco_justo_dcf'] = discount_fcf_to_equity(
            fcf_base=fcf_base,
            growth=initial_growth,
            discount_rate=coe,
            shares=shares_total,
            terminal_growth=terminal_growth,
        )
        result['growth_rate'] = initial_growth
        result['fcf_base'] = fcf_base
        result['cost_of_equity'] = coe
        result['growth_source'] = growth_source
        # Mesmo helper que compute_fcf_base consultou. O ajuste é refeito de
        # propósito: são 4 pontos de numpy, e o custo é nulo perto de manter
        # compute_fcf_base com assinatura estável (devolver uma tupla
        # (base, origem) quebraria os testes e o call site sem ganho).
        result['fcf_base_source'] = (
            'trend' if pd.notna(_fcf_trend_base(fcf_series)) else 'median')

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

    O juro de desconto e o crescimento na perpetuidade saem da coluna `moeda`
    DE CADA LINHA (real quando a coluna não existe), e não de constantes do
    módulo. É a mesma moeda que `append_snapshot` usa para gravar as premissas:
    uma linha em dólar descontada a juro brasileiro sairia no histórico com
    12,4% de custo de capital ao lado de um juro de 4,2%, dois números que não
    podem ser verdadeiros ao mesmo tempo.

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
    fcf_base_sources = []

    for _, row in df.iterrows():
        # Beta do SETOR, não da empresa: small caps ilíquidas medem beta
        # artificialmente baixo por não negociarem (ver compute_sector_betas).
        beta = sector_betas.get(row.get('setor', ''), np.nan)

        # Moeda do BALANÇO da linha. É ela que decide o juro de desconto e o
        # crescimento na perpetuidade — e é a mesma que `append_snapshot` grava
        # na coluna de premissas. Se as duas divergirem, o histórico registra
        # uma premissa que não foi usada, indistinguível de uma verdadeira.
        # Coluna ausente significa real: mesma regra de compatibilidade do
        # resto do pipeline, verdadeira para todas as linhas já gravadas.
        moeda = row.get('moeda', 'BRL')
        macro = macro_for(moeda) or {}
        # NaN quando a moeda não tem premissas. Todo modelo abaixo devolve NaN
        # com taxa NaN, de propósito: preço justo tirado de juro faltante sai
        # com cara de calculado e chega ao usuário como recomendação.
        g_terminal = macro.get('terminal_growth', np.nan)
        coe = cost_of_equity(beta, moeda=moeda)
        coes.append(coe)

        growth_source = ''
        fcf_base_source = ''

        # --- Modelo primário ---
        if model == 'bank':
            # Excess Returns para bancos
            roe_decimal = row.get('roe_pct', np.nan)
            if pd.notna(roe_decimal):
                roe_decimal = roe_decimal / 100.0
            vpa = row.get('vpa', np.nan)
            primary_price = excess_returns_valuation(
                roe_decimal, vpa, coe=coe, terminal_growth=g_terminal)
            method = 'excess_returns' if pd.notna(primary_price) else 'none'
        else:
            # DCF 2-estágios para ações. O crescimento forward sai da própria
            # linha (colunas já coletadas no CSV): nenhuma requisição por ticker.
            dcf_result = dcf_valuation(
                row['ticker_sa'],
                row.get('shares_total'),
                beta,
                forward_growth=resolve_forward_growth(row),
                moeda=moeda,
            )
            primary_price = dcf_result['preco_justo_dcf']

            if pd.notna(primary_price):
                method = 'dcf'
                growth_source = dcf_result['growth_source']
                fcf_base_source = dcf_result['fcf_base_source']
            else:
                # Fallback DDM quando o DCF não é aplicável (ex.: FCF histórico
                # negativo em incorporadoras). Rotulado explicitamente para não
                # se confundir com um DCF de verdade.
                dps = row.get('dividend_rate', np.nan)
                primary_price = ddm_valuation(dps, discount_rate=coe,
                                              growth_rate=g_terminal)
                method = 'ddm' if pd.notna(primary_price) else 'none'

        primary_prices.append(primary_price)
        methods.append(method)
        growth_sources.append(growth_source)
        fcf_base_sources.append(fcf_base_source)

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
    df['fcf_base_source'] = fcf_base_sources
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

# Colunas do resultado que são snapshotadas (as que existirem no df).
_SNAPSHOT_RESULT_COLS = [
    'tipo', 'regiao', 'moeda', 'ticker', 'nome', 'setor', 'preco',
    'preco_justo_dcf', 'metodo_valuation', 'growth_source', 'fcf_base_source',
    'preco_justo_graham', 'margem_seg_dcf_pct', 'margem_seg_graham_pct',
    'margem_seg_media_pct', 'undervalued', 'forte_desconto',
    'cost_of_equity_pct',
    'crescimento_receita_pct', 'crescimento_lucro_pct', 'lpa_estimado',
    'num_analistas',
]


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
