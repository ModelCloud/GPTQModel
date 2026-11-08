"""Every quantization method validates its integer bit width.

``quant_bits_width`` floors a fractional width, so accepting one would write a
config that advertises a precision the packed tensors cannot realize. Each
method also supports a narrow set of integer widths; reject anything outside
that set at config construction instead of failing later at kernel selection.
"""

import pytest

from gptqmodel.quantization.config import (
    METHOD,
    AWQConfig,
    BitsAndBytesConfig,
    EXL3Config,
    FP8Config,
    GGUFBits,
    GGUFConfig,
    GPTQConfig,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    VPTQConfig,
)


# The integer widths each method's packer and kernels can actually realize.
SUPPORTED_BITS = {
    GPTQConfig: (2, 3, 4, 5, 6, 7, 8),
    RTNConfig: (2, 3, 4, 5, 6, 7, 8),
    AWQConfig: (2, 3, 4, 8),
    QQQConfig: (4,),
    ParoConfig: (4,),
    FP8Config: (8,),
    BitsAndBytesConfig: (4, 8),
    EXL3Config: (1, 2, 3, 4, 5, 6, 7, 8),
    GGUFConfig: (1, 2, 3, 4, 5, 6, 8),
    VPTQConfig: (1, 2, 3, 4, 5, 6, 7, 8),
}


def _build(config_cls, bits, **kwargs):
    return config_cls(bits=bits, offload_to_disk=False, **kwargs)


@pytest.mark.parametrize("config_cls", list(SUPPORTED_BITS))
@pytest.mark.parametrize("bits", [2.5, 2.25, 4.5, "2.5"])
def test_fractional_bits_is_rejected(config_cls, bits):
    match = (
        "not an integer trellis width"
        if config_cls is EXL3Config
        else "not an integer bit width"
    )
    with pytest.raises(ValueError, match=match):
        _build(config_cls, bits)


@pytest.mark.parametrize("config_cls, supported", SUPPORTED_BITS.items())
@pytest.mark.parametrize("bits", range(0, 10))
def test_bits_outside_method_support_is_rejected(config_cls, supported, bits):
    if bits in supported:
        _build(config_cls, bits)
    else:
        match = (
            "between 1.0 and 8.0"
            if config_cls is EXL3Config
            else "supported bit widths"
        )
        with pytest.raises(ValueError, match=match):
            _build(config_cls, bits)


@pytest.mark.parametrize(
    "config_cls, bits",
    [
        (GPTQConfig, 4.0),
        (RTNConfig, 4.0),
        (AWQConfig, 4.0),
        (QQQConfig, 4.0),
        (ParoConfig, 4.0),
        (FP8Config, 8.0),
        (BitsAndBytesConfig, 8.0),
        (EXL3Config, 2.0),
        (GGUFConfig, 4.0),
        (VPTQConfig, 2.0),
    ],
)
def test_integer_valued_float_is_accepted(config_cls, bits):
    _build(config_cls, bits)


def test_awq_qqq_paro_reject_unsupported_bit_widths():
    """The config used to accept the full 2-8 GPTQ range for every method."""

    for bits in (5, 6, 7):
        with pytest.raises(ValueError, match=r"supported bit widths are `\[2, 3, 4, 8\]`"):
            _build(AWQConfig, bits)

    for config_cls in (QQQConfig, ParoConfig):
        for bits in (2, 3, 5, 6, 7, 8):
            with pytest.raises(ValueError, match=r"supported bit widths are `\[4\]`"):
                _build(config_cls, bits)


def test_awq_accepts_eight_bit_marlin_weights():
    _build(AWQConfig, 8)


@pytest.mark.parametrize("config_cls", [QQQConfig, ParoConfig])
def test_dynamic_width_must_match_method_support(config_cls):
    with pytest.raises(ValueError, match="supported bit widths"):
        _build(config_cls, 4, dynamic={"model.layers.0.mlp.down_proj": {"bits": 3}})

    _build(config_cls, 4, dynamic={"model.layers.0.mlp.down_proj": {"bits": 4}})


@pytest.mark.parametrize("bits", [2, 3, 8])
def test_awq_dynamic_supported_width_is_accepted(bits):
    _build(
        AWQConfig,
        4,
        dynamic={"model.layers.0.mlp.down_proj": {"bits": bits}},
    )


@pytest.mark.parametrize("bits", [2.5, "2.5"])
def test_gptq_dynamic_fractional_bits_is_rejected(bits):
    with pytest.raises(ValueError, match="not an integer bit width"):
        _build(GPTQConfig, 4, dynamic={"model.layers.0.mlp.down_proj": {"bits": bits}})


def test_fp8_dynamic_bits_is_validated_without_int_truncation():
    # ``int(8.5)`` is 8, so this used to pass while advertising 8.5.
    with pytest.raises(ValueError, match="not an integer bit width"):
        _build(FP8Config, 8, dynamic={"model.layers.0.mlp.down_proj": {"bits": 8.5}})

    with pytest.raises(ValueError, match="supported bit widths"):
        _build(FP8Config, 8, dynamic={"model.layers.0.mlp.down_proj": {"bits": 4}})

    _build(FP8Config, 8, dynamic={"model.layers.0.mlp.down_proj": {"bits": 8}})


def test_bitsandbytes_dynamic_bits_is_validated_without_int_truncation():
    # ``int(4.5)`` is 4, so this used to pass while advertising 4.5.
    with pytest.raises(ValueError, match="not an integer bit width"):
        _build(
            BitsAndBytesConfig,
            4,
            dynamic={"model.layers.0.self_attn.q_proj": {"bits": 4.5}},
        )

    with pytest.raises(ValueError, match="supported bit widths"):
        _build(
            BitsAndBytesConfig,
            4,
            dynamic={"model.layers.0.self_attn.q_proj": {"bits": 5}},
        )

    _build(
        BitsAndBytesConfig,
        4,
        dynamic={"model.layers.0.self_attn.q_proj": {"bits": 8}},
    )


def test_dynamic_error_names_the_layer_and_field():
    with pytest.raises(
        ValueError,
        match=r"dynamic\['model\.layers\.0\.mlp\.down_proj'\]\.bits",
    ):
        _build(
            AWQConfig,
            4,
            dynamic={"model.layers.0.mlp.down_proj": {"bits": 5}},
        )


def test_quantize_config_dispatcher_uses_method_support():
    with pytest.raises(
        ValueError,
        match=r"`bits=5` is not supported for method `awq`",
    ):
        QuantizeConfig(quant_method=METHOD.AWQ, bits=5, offload_to_disk=False)

    with pytest.raises(
        ValueError,
        match=r"`bits=3` is not supported for method `qqq`",
    ):
        QuantizeConfig(quant_method=METHOD.QQQ, bits=3, offload_to_disk=False)

    with pytest.raises(
        ValueError,
        match=r"`bits=3` is not supported for method `paroquant`",
    ):
        QuantizeConfig(quant_method=METHOD.PARO, bits=3, offload_to_disk=False)

    cfg = QuantizeConfig(quant_method=METHOD.AWQ, bits=4, offload_to_disk=False)
    assert isinstance(cfg, AWQConfig)

    for bits in (2, 3):
        assert isinstance(QuantizeConfig(quant_method=METHOD.AWQ, bits=bits, offload_to_disk=False), AWQConfig)

    cfg = QuantizeConfig(quant_method=METHOD.AWQ, bits=8, offload_to_disk=False)
    assert isinstance(cfg, AWQConfig)


def test_exl3_fractional_error_names_the_width_that_would_have_been_used():
    with pytest.raises(
        ValueError,
        match="not an integer trellis width",
    ) as excinfo:
        EXL3Config(bits=2.5, offload_to_disk=False)

    message = str(excinfo.value)
    assert "2.5" in message
    assert "2 bpw" in message
    assert "dynamic" in message


def test_exl3_fractional_head_bits_and_dynamic_widths_are_rejected():
    with pytest.raises(
        ValueError,
        match=r"`head_bits=4\.5` is not an integer trellis width",
    ):
        EXL3Config(bits=3, head_bits=4.5, offload_to_disk=False)

    with pytest.raises(
        ValueError,
        match=r"dynamic\['model\.layers\.0\.mlp\.down_proj'\]\.bits=2\.5",
    ):
        EXL3Config(
            bits=3,
            dynamic={"model.layers.0.mlp.down_proj": {"bits": 2.5}},
            offload_to_disk=False,
        )


def test_exl3_integer_dynamic_map_expresses_a_fractional_average():
    cfg = EXL3Config(
        bits=4,
        dynamic={
            "model.layers.0.mlp.up_proj": {"bits": 2},
            "model.layers.1.mlp.down_proj": {"bits": 3},
        },
        offload_to_disk=False,
    )

    assert cfg.runtime_bits == 4
    assert cfg.dynamic_get("model.layers.0.mlp.up_proj", "bits") == 2.0
    assert cfg.dynamic_get("model.layers.1.mlp.down_proj", "bits") == 3.0
    assert cfg.to_dict()["bits"] == 4.0


def test_exl3_integer_bits_round_trip():
    cfg = EXL3Config(bits=3, offload_to_disk=False)

    payload = cfg.to_dict()
    assert payload["bits"] == 3.0
    assert QuantizeConfig.from_quant_config(payload).bits == 3.0


def test_gguf_bits_outside_aliases_is_rejected():
    with pytest.raises(ValueError, match="supported bit widths"):
        GGUFConfig(bits=7, offload_to_disk=False)

    _build(GGUFConfig, "q4_0")
    _build(GGUFConfig, "q3_k")
    _build(GGUFConfig, 3)


def test_gguf_bits_spec_rejects_fractional_width():
    with pytest.raises(ValueError, match="positive integer"):
        GGUFBits(bits=2.5, version="q", variant="0")


def test_dispatcher_honors_explicit_gguf_method():
    """`method="gguf"` used to be silently rewritten to a GPTQ config, which let
    an unsupported GGUF width bypass the GGUF width check entirely."""

    with pytest.raises(ValueError, match="supported bit widths"):
        QuantizeConfig(quant_method=METHOD.GGUF, bits=7, offload_to_disk=False)

    cfg = QuantizeConfig(quant_method=METHOD.GGUF, bits=3, offload_to_disk=False)
    assert isinstance(cfg, GGUFConfig)
    assert cfg.bits == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"quant_method": "gptq", "bits": 2.5},
        {"quant_method": "awq", "bits": 2.5},
        {"quant_method": "qqq", "bits": 2.5},
        {"quant_method": "paroquant", "bits": 2.5},
        {"quant_method": "bitsandbytes", "bits": 4.5},
        {"quant_method": "fp8", "bits": 8.5},
        {"quant_method": "exl3", "bits": 2.5},
        {"quant_method": "gguf", "bits": 2.5, "format": "q_0"},
        {"quant_method": "vptq", "bits": 2.5},
    ],
)
def test_from_quant_config_rejects_fractional_bits(payload):
    """Checkpoint payloads loaded from disk take the same validation path."""

    with pytest.raises(ValueError, match=r"not an integer (bit|trellis) width"):
        QuantizeConfig.from_quant_config(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"quant_method": "awq", "bits": 5},
        {"quant_method": "qqq", "bits": 8},
        {"quant_method": "paroquant", "bits": 2},
        {"quant_method": "fp8", "bits": 4},
        {"quant_method": "gguf", "bits": 7},
        {"quant_method": "vptq", "bits": 9},
    ],
)
def test_from_quant_config_rejects_unsupported_bits(payload):
    with pytest.raises(ValueError, match="supported bit widths"):
        QuantizeConfig.from_quant_config(payload)


@pytest.mark.parametrize(
    "payload, config_cls",
    [
        ({"quant_method": "gguf", "bits": 4}, GGUFConfig),
        ({"quant_method": "fp8", "bits": 8}, FP8Config),
        ({"quant_method": "bitsandbytes", "bits": 4}, BitsAndBytesConfig),
    ],
)
def test_from_quant_config_defaults_weight_only_formats(payload, config_cls):
    """A weight-only payload without an explicit `format` still dispatches to
    the method-specific config and picks that method's default subtype."""

    assert isinstance(QuantizeConfig.from_quant_config(payload), config_cls)
