"""EXL3 cannot encode a fractional bits-per-weight: `bits` must be an integer.

The trellis stores one integer depth (``K``) per module -- ``pack_trellis`` sizes
the packed tensor as ``256 * K // 16`` and the kernel takes ``K`` as an ``int``.
``quant_bits_width`` used to floor a fractional request, so ``EXL3Config(bits=2.5)``
quantized every module at 2 bpw while the saved ``config.json`` still recorded
``2.5``: a "2.5 bpw" checkpoint that was really 2.0 bpw. ``EXL3Config`` now
rejects the request instead of shipping a mislabelled artifact.

A fractional *average* is expressed with per-module ``dynamic`` widths, each of
which is an integer; that is what the RCO allocator produces.
"""

import pytest

from gptqmodel.quantization.config import EXL3Config


@pytest.mark.parametrize("bits", [2.5, 2.25, 3.5, 7.75, "2.5"])
def test_fractional_bits_is_rejected(bits):
    with pytest.raises(ValueError, match="not an integer trellis width"):
        EXL3Config(bits=bits)


def test_the_error_names_the_width_that_would_have_been_used():
    with pytest.raises(ValueError) as excinfo:
        EXL3Config(bits=2.5)

    message = str(excinfo.value)
    # 2.5 used to floor to 2, so the message has to say what the request would
    # have silently become rather than only that it is invalid.
    assert "2.5" in message
    assert "2 bpw" in message
    assert "dynamic" in message


@pytest.mark.parametrize("bits", [1, 2, 2.0, 3.0, 4, 8])
def test_integer_bits_is_accepted(bits):
    qc = EXL3Config(bits=bits)

    assert qc.bits == float(bits)
    assert qc.runtime_bits == int(bits)


def test_fractional_head_bits_is_rejected():
    with pytest.raises(ValueError, match="not an integer trellis width"):
        EXL3Config(bits=3, head_bits=4.5)


def test_fractional_dynamic_width_is_rejected():
    with pytest.raises(ValueError, match="not an integer trellis width"):
        EXL3Config(
            bits=3,
            dynamic={"model.layers.0.mlp.down_proj": {"bits": 2.5}},
        )


def test_integer_dynamic_map_expresses_a_fractional_average():
    qc = EXL3Config(
        bits=4,
        dynamic={
            "model.layers.0.mlp.down_proj": {"bits": 2},
            "model.layers.1.mlp.down_proj": {"bits": 3},
        },
    )

    # The global field is only the fallback for modules the map does not name.
    assert qc.runtime_bits == 4
    assert qc.dynamic_get("model.layers.0.mlp.down_proj", "bits") == 2.0
    assert qc.dynamic_get("model.layers.1.mlp.down_proj", "bits") == 3.0
    assert qc.to_dict()["bits"] == 4.0
