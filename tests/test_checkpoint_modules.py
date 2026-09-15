# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.checkpoint_modules import (
    packed_module_spec,
    restore_packed_module,
)
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.quantization.config import (
    _GGUF_BITS_ALIAS_INFO,
    FP8Config,
    GGUFConfig,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
)


def assert_schema_roundtrip(module, config):
    spec = ContinuationCodec.loads(
        ContinuationCodec.dumps(packed_module_spec(module, config))
    )
    restored = restore_packed_module(
        spec, name="proj", config=config, kernel=type(module), lm_head_name="lm_head"
    )
    assert type(restored) is type(module)
    assert {
        key: (value.shape, value.dtype) for key, value in module.state_dict().items()
    } == {
        key: (value.shape, value.dtype) for key, value in restored.state_dict().items()
    }
    assert all(value.device.type == "meta" for value in restored.state_dict().values())
    return restored


@pytest.mark.parametrize("format", ["gptq", "gptq_v2", "gptq_p"])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_packed_schema(format, bits):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    config = QuantizeConfig(bits=bits, group_size=32, format=format, desc_act=False)
    module = TorchLinear(
        bits=bits,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=True,
        format=config.format,
        register_buffers=True,
    )
    assert_schema_roundtrip(module, config)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bitblas_compute_dtype_schema_without_optional_compiler(dtype):
    config = QuantizeConfig(bits=4, group_size=32, desc_act=False)
    source = SimpleNamespace(
        in_features=64,
        out_features=64,
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        bias=None,
        QUANT_TYPE="awq_bitblas",
        quant_config=SimpleNamespace(torch_dtype=dtype),
    )
    spec = packed_module_spec(source, config)
    restored = restore_packed_module(
        spec,
        name="proj",
        config=config,
        kernel=lambda **kwargs: SimpleNamespace(**kwargs),
        lm_head_name="lm_head",
    )
    assert restored.dtype == dtype


@pytest.mark.parametrize("scale", ["row", "tensor", "block"])
def test_fp8_scale_schema(scale):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    config = FP8Config(
        weight_scale_method=scale,
        weight_block_size=(32, 32) if scale == "block" else None,
    )
    module = TorchFP8Linear(
        bits=8,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        **config.quant_linear_init_kwargs(),
    )
    restored = assert_schema_roundtrip(module, config)
    assert restored.weight_scale_method == scale
    assert restored.weight_block_size == module.weight_block_size


@pytest.mark.parametrize("bits", sorted(_GGUF_BITS_ALIAS_INFO))
def test_gguf_bits_schema(bits):
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear

    config = GGUFConfig(bits=bits)
    module = GGUFTorchLinear(
        bits=config.runtime_bits,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=256,
    )
    restored = assert_schema_roundtrip(module, config)
    assert restored.bits == module.bits


def test_qqq_optional_group_scales_schema():
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear

    config = QQQConfig(bits=4, group_size=128)
    module = QQQLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
    )
    del module.s_group
    assert_schema_roundtrip(module, config)


def test_paro_runtime_options_schema():
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    config = ParoConfig(bits=4, group_size=32, krot=4)
    module = ParoLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        krot=4,
        fp32_accum=False,
        cache_runtime_dtype=True,
        register_buffers=True,
    )
    restored = assert_schema_roundtrip(module, config)
    assert restored.krot == 4
    assert restored.fp32_accum is False
    assert restored.cache_runtime_dtype is True


def test_exl3_explicit_buffer_schema():
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    module = ExllamaV3Linear(
        in_features=128,
        out_features=128,
        name="proj",
        out_dtype=torch.bfloat16,
        tensors={
            "trellis": torch.zeros(16, 32, dtype=torch.int16),
            "mcg": torch.tensor([7], dtype=torch.int32),
        },
    )
    restored = assert_schema_roundtrip(module, None)
    assert restored.out_dtype == torch.bfloat16


@pytest.mark.parametrize("compute_dtype", [torch.float32, None])
def test_bitsandbytes_schema_under_meta_and_inference_context(compute_dtype):
    from gptqmodel.nn_modules.qlinear.bitsandbytes import (
        BITSANDBYTES_AVAILABLE,
        BitsAndBytesLinear,
    )
    from gptqmodel.quantization.config import BitsAndBytesConfig

    if not BITSANDBYTES_AVAILABLE:
        pytest.skip("optional bitsandbytes dependency unavailable")
    config = BitsAndBytesConfig(bits=4)
    module = BitsAndBytesLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        dtype=compute_dtype,
    )
    module.pack_original(torch.nn.Linear(128, 128, bias=False), None, None)
    with torch.inference_mode():
        restored = assert_schema_roundtrip(module, config)
    assert restored.compute_dtype == compute_dtype
