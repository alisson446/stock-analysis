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
