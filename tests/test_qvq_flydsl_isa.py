"""CPU-only regression checks for FlyDSL binary artifact decoding."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "qvq_flydsl_isa", Path(__file__).resolve().parents[1] / "scripts/analyze_qvq_flydsl_isa.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_decode_binary_preserves_mlir_hex_quote_and_backslash():
    assert _MODULE.decode_binary(r'bin = "\7FELF\00\FF\\\""') == b'\x7fELF\x00\xff\\"'


@pytest.mark.parametrize("source", [
    "", 'bin = "no ELF"', r'bin = "\7FELF\ZZ"', r'bin = "\7FELF\A"',
    r'bin = "\7FELF" bin = "\7FELF"',
])
def test_decode_binary_rejects_invalid_or_ambiguous_artifacts(source):
    with pytest.raises(ValueError):
        _MODULE.decode_binary(source)
