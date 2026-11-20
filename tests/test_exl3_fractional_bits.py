"""Native mul1 half-bit rates and integer EXL3 checkpoint compatibility."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.exllamav3_processor import EXL3Processor, clone_exllamav3_config_for_module
from gptqmodel.quantization.config import EXL3Config, QuantizeConfig


@pytest.mark.parametrize("bits", [1.5, 2.5, 3.5, "2.5"])
def test_mul1_native_half_rate_survives_checkpoint_config(bits):
    qc = EXL3Config(bits=bits, codebook="mul1", offload_to_disk=False)
    assert qc.runtime_bits == float(bits)
    payload = qc.to_dict()
    assert payload["bits"] == float(bits)
    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.runtime_bits == float(bits)
    assert restored.codebook == "mul1"


def test_half_rate_survives_head_and_dynamic_resolution():
    projection = "model.layers.0.mlp.down_proj"
    cfg = EXL3Config(
        bits=2.5,
        head_bits="3.5",
        codebook="mul1",
        dynamic={projection: {"bits": "1.5"}},
        offload_to_disk=False,
    )
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.head_bits == 3.5
    processor = object.__new__(EXL3Processor)
    processor.lm_head_name = "lm_head"
    for name, rate in [(projection, 1.5), ("model.layers.0.mlp.up_proj", 2.5), ("lm_head", 3.5)]:
        resolved = clone_exllamav3_config_for_module(restored, name)
        module = SimpleNamespace(name=name, full_name=name)
        args = processor._build_quant_args(module, resolved, torch.device("cuda:0"))
        assert args["K"] == rate
        assert args["mul1"] is True


def test_dynamic_half_rate_can_select_its_own_codebook():
    projection = "model.layers.0.mlp.down_proj"
    cfg = EXL3Config(
        bits=2,
        codebook="mcg",
        dynamic={projection: {"bits": 2.5, "codebook": "mul1"}},
        offload_to_disk=False,
    )
    resolved = clone_exllamav3_config_for_module(cfg, projection)
    assert (resolved.runtime_bits, resolved.codebook) == (2.5, "mul1")
    assert (cfg.runtime_bits, cfg.codebook) == (2, "mcg")


@pytest.mark.parametrize("bits", [2.5, 3])
def test_q4nx_recipe_reaches_projection_quantizer_after_config_round_trip(bits):
    defaults = QuantizeConfig.from_quant_config(EXL3Config(bits=bits, codebook="q4nx").to_dict())
    assert defaults.runtime_bits == bits and defaults.codebook == "q4nx"

    projection = "model.layers.0.mlp.experts.0.down_proj"
    cfg = EXL3Config(
        bits=4,
        head_bits=6,
        dynamic={
            projection: {"bits": bits, "codebook": " Q4NX "},
            "model.layers.0.self_attn.q_proj": {"bits": 4},
        },
        offload_to_disk=False,
    )
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    resolved = clone_exllamav3_config_for_module(restored, projection)
    processor = object.__new__(EXL3Processor)
    processor.lm_head_name = "lm_head"
    args = processor._build_quant_args(
        SimpleNamespace(name=projection, full_name=projection), resolved, torch.device("cuda:0"),
    )
    assert args["K"] == bits
    assert args["q4nx"] is True
    assert not {"mcg", "mul1"} & args.keys()
    assert restored.codebook == "mcg"


@pytest.mark.parametrize("bits", [1.5, 2, 3.5, 4])
def test_q4nx_rejects_other_rates_in_global_and_dynamic_config(bits):
    with pytest.raises(ValueError, match="q4nx.*2.5.*3.0"):
        EXL3Config(bits=bits, codebook="q4nx")
    with pytest.raises(ValueError, match="q4nx.*2.5.*3.0"):
        EXL3Config(bits=4, dynamic={"down_proj": {"codebook": "q4nx", "bits": bits}})


@pytest.mark.parametrize("bits,codebook", [(2.5, "3inst"), (2.25, "mul1"), (4.5, "mul1")])
def test_unsupported_rate_and_codebook_pairs_are_rejected(bits, codebook):
    with pytest.raises(ValueError, match="Native half-bit rates"):
        EXL3Config(bits=bits, codebook=codebook, offload_to_disk=False)


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
