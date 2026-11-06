"""EXL3 fractional `bits` cannot be realised uniformly.

The trellis stores an integer number of bits per step, so ``EXL3Config(bits=2.5)``
quantizes every module at 2 bpw. That used to happen silently: the saved config
recorded 2.5 while the checkpoint was 2.0 bpw wide, so a "2.5 bpw" model was
really a 2.0 bpw model. A fractional average has to be expressed as per-module
``dynamic`` widths, which is what the RCO allocator produces.
"""

import pytest

from gptqmodel.quantization import config as config_module
from gptqmodel.quantization.config import EXL3Config


class _Capture:
    def __init__(self):
        self.messages = []

    def __call__(self, message, *args, **kwargs):
        self.messages.append(str(message))


@pytest.fixture
def capture(monkeypatch):
    sink = _Capture()
    monkeypatch.setattr(config_module.log, "warn", sink)
    return sink


def test_fractional_bits_warns_and_floors(capture):
    qc = EXL3Config(bits=2.5)

    # The requested width is preserved in the config payload...
    assert qc.bits == 2.5
    # ...but the quantizer can only build an integer trellis.
    assert qc.runtime_bits == 2
    assert any("not an integer trellis width" in message for message in capture.messages)


def test_integer_bits_is_silent(capture):
    qc = EXL3Config(bits=3)

    assert qc.runtime_bits == 3
    assert capture.messages == []


def test_dynamic_map_suppresses_the_warning(capture):
    qc = EXL3Config(
        bits=2.5,
        dynamic={"model.layers.0.mlp.down_proj": {"bits": 3.0}},
    )

    # The fallback for unlisted modules still floors, so the map must cover them.
    assert qc.runtime_bits == 2
    assert qc.dynamic_get("model.layers.0.mlp.down_proj", "bits") == 3.0
    assert capture.messages == []


def test_fractional_bits_survives_serialization(capture):
    payload = EXL3Config(bits=2.5).to_dict()

    assert payload["bits"] == 2.5
