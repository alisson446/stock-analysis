"""Configuração compartilhada da suíte."""
import os

import pytest

# Prefixos das premissas macro por moeda: RISK_FREE_RATE_USD,
# EQUITY_RISK_PREMIUM_EUR e assim por diante.
_PREFIXOS_MACRO_POR_MOEDA = ('RISK_FREE_RATE_', 'EQUITY_RISK_PREMIUM_')


@pytest.fixture(autouse=True)
def env_de_moeda_isolada(monkeypatch):
    """
    Remove do ambiente as premissas macro POR MOEDA antes de cada teste.

    `valuation.macro_for` lê a env a cada chamada — de propósito, para você
    habilitar uma moeda nova editando o `.env` sem tocar em código. O efeito
    colateral é que qualquer teste sobre premissas passa a medir a máquina de
    quem roda, não o comportamento do código.

    E o `.env.example` **ensina** a definir `RISK_FREE_RATE_USD`. Sem esta
    limpeza, seguir a documentação do próprio projeto deixa a suíte vermelha —
    com falhas que apontam para a aritmética do valuation em vez de para o
    ambiente, que é o tipo de pista que leva a depurar o lugar errado.

    As constantes SEM sufixo (`RISK_FREE_RATE`, `EQUITY_RISK_PREMIUM`, as do
    Brasil) não são tocadas: são lidas uma vez na importação do módulo, então
    apagá-las aqui não teria efeito nenhum.

    Um teste que precise de uma moeda habilitada faz o seu próprio
    `monkeypatch.setenv` — o corpo do teste roda depois desta fixture.
    """
    for nome in list(os.environ):
        if nome.startswith(_PREFIXOS_MACRO_POR_MOEDA):
            monkeypatch.delenv(nome, raising=False)
