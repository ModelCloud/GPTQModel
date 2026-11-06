# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.quantization.config import FORMAT, METHOD, EXL3Config, QuantizeConfig
from gptqmodel.utils.exllamav3 import build_exllamav3_tensor_storage, replace_exllamav3_placeholders
from gptqmodel.utils.model_dequant import detect_format


def test_exllamav3_quantize_config_round_trip():
    cfg = QuantizeConfig(
        quant_method=METHOD.EXL3,
        format=FORMAT.EXL3,
        bits=2,
        head_bits=4.0,
        out_scales="always",
        codebook="mul1",
    )

    assert isinstance(cfg, EXL3Config)
    assert cfg.quant_method == METHOD.EXL3
    assert cfg.format == FORMAT.EXL3
    assert cfg.runtime_bits == 2
    assert cfg.uses_weight_only_lifecycle() is False
    assert cfg.requires_calibration_dataset() is True

    payload = cfg.to_dict()
    assert payload["bits"] == 2.0
    assert payload["head_bits"] == 4.0
    assert payload["out_scales"] == "always"
    assert payload["codebook"] == "mul1"

    reloaded = QuantizeConfig.from_quant_config(payload)
    assert isinstance(reloaded, EXL3Config)
    assert reloaded.bits == 2.0
    assert reloaded.head_bits == 4.0
    assert reloaded.out_scales == "always"
    assert reloaded.codebook == "mul1"
    assert reloaded.runtime_bits == 2


def test_exllamav3_rejects_fractional_bits():
    """EXL3 packs one integer trellis depth per module; a fractional width cannot
    be realised and would silently quantize narrower than requested."""

    for bits in (2.5, 2.25, 3.5):
        with pytest.raises(ValueError, match="not an integer trellis width"):
            EXL3Config(bits=bits)

    with pytest.raises(ValueError, match="not an integer trellis width"):
        EXL3Config(bits=3, head_bits=4.5)

    with pytest.raises(ValueError, match="not an integer trellis width"):
        EXL3Config(bits=3, dynamic={"model.layers.0.mlp.down_proj": {"bits": 2.5}})


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16, bias=True)


def test_replace_exllamav3_placeholders_uses_tensor_storage_metadata():
    model = _TinyModel()
    tensor_storage = {
        "proj": {
            "stored_tensors": {
                "proj.trellis": {"shape": [1, 1, 32], "torch_dtype": "int16"},
                "proj.suh": {"shape": [16], "torch_dtype": "float16"},
                "proj.svh": {"shape": [16], "torch_dtype": "float16"},
                "proj.bias": {"shape": [16], "torch_dtype": "float16"},
                "proj.mul1": {"shape": [], "torch_dtype": "int32"},
            },
            "quant_format": "exl3",
            "bits_per_weight": 2,
        }
    }

    replace_exllamav3_placeholders(
        model=model,
        module_names=["proj"],
        tensor_storage=tensor_storage,
    )

    assert isinstance(model.proj, ExllamaV3Linear)
    assert model.proj.trellis.device.type == "meta"
    assert tuple(model.proj.trellis.shape) == (1, 1, 32)
    assert model.proj.suh.dtype == torch.float16
    assert model.proj.svh.dtype == torch.float16
    assert model.proj.bias.dtype == torch.float16
    assert model.proj.mul1.dtype == torch.int32


def test_replace_exllamav3_placeholders_supports_torch_reference_kernel():
    model = _TinyModel()
    tensor_storage = {
        "proj": {
            "stored_tensors": {
                "proj.trellis": {"shape": [1, 1, 32], "torch_dtype": "int16"},
                "proj.suh": {"shape": [16], "torch_dtype": "float16"},
                "proj.svh": {"shape": [16], "torch_dtype": "float16"},
            },
            "quant_format": "exl3",
            "bits_per_weight": 2,
        }
    }

    replace_exllamav3_placeholders(
        model=model,
        module_names=["proj"],
        tensor_storage=tensor_storage,
        module_cls=ExllamaV3TorchLinear,
    )

    assert isinstance(model.proj, ExllamaV3TorchLinear)
    assert build_exllamav3_tensor_storage(model)["proj"]["quant_format"] == "exl3"


def test_detect_format_identifies_exllamav3(tmp_path):
    shard_path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.trellis": torch.zeros((1, 1, 32), dtype=torch.int16),
            "model.layers.0.self_attn.q_proj.suh": torch.zeros((16,), dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.svh": torch.zeros((16,), dtype=torch.float16),
        },
        str(shard_path),
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "exl3",
                    "format": "exl3",
                }
            }
        ),
        encoding="utf-8",
    )

    detected = detect_format(tmp_path, json.loads(config_path.read_text(encoding="utf-8")))
    assert detected == "exl3"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_exllamav3_torch_rocm_forward_runs_and_matches_oracle(dtype):
    """Hardware test: the AUTO ROCm backend (EXL3_TORCH) must actually forward.

    On gfx1201 this failed before the Hadamard dtype fix: ``_forward_eager``
    casts activations to FP16 while the cached Hadamard matrix was FP32, so
    ``torch.matmul`` raised ``c10::Half != float`` at the first forward.
    """

    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm GPU required")

    from test_exllamav3_cpu import _independent_weight_oracle, _pack_symbols, _relative_l2

    in_features, out_features = 384, 256
    generator = torch.Generator().manual_seed(17 if dtype is torch.float16 else 29)
    symbols = torch.randint(
        0,
        1 << 4,
        (in_features // 16, out_features // 16, 256),
        dtype=torch.int64,
        generator=generator,
    )
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="rocm_forward",
        tensors={
            "trellis": _pack_symbols(symbols, 4),
            "suh": (torch.rand(in_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
            "svh": (torch.rand(out_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
            "bias": torch.randn(out_features, generator=generator, dtype=torch.float16) * 0.01,
        },
    ).eval()
    module = module.to(device="cuda", dtype=torch.float16)

    x = torch.randn(7, in_features, dtype=dtype, generator=generator).cuda()
    with torch.inference_mode():
        actual = module(x)
        repeated = module(x)

    expected = x.double().cpu() @ _independent_weight_oracle(
        symbols, 4, module.suh.cpu(), module.svh.cpu()
    ) + module.bias.double().cpu()

    assert actual.dtype == dtype
    torch.testing.assert_close(repeated, actual)
    assert _relative_l2(actual.cpu(), expected) <= 2e-3
    assert _relative_l2(actual.cpu(), expected.to(dtype)) <= 2e-3
