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
