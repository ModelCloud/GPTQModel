# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""QQQ keeps a bias at the source model width across pack and reload (#5603).

The portable packer used to narrow every bias into the FP16 placeholder. A
finite BF16 bias outside the FP16 range (65536 here) became +inf during
``pack()``, so even the FP32-accumulating portable forward returned inf. The
bias is model-dtype storage, not quantization state: it must keep the width
the source layer had, on save and on load.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear


IN_FEATURES = 128
OUT_FEATURES = 128
BIAS = 65536.0


def _source(dtype: torch.dtype = torch.bfloat16, bias: float = BIAS) -> torch.nn.Linear:
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True, dtype=dtype)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(bias)
    return linear


def _module(cls, **kwargs):
    return cls(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        **kwargs,
    )


def _packed(cls, source: torch.nn.Linear):
    module = _module(cls)
    module.pack(source, torch.ones(IN_FEATURES, 1))
    return module


def _forward_bfloat16(module):
    x = torch.zeros(1, IN_FEATURES, dtype=torch.bfloat16)
    return module(x)


@pytest.mark.parametrize("cls", [QQQTorchLinear, QQQLinear])
def test_pack_keeps_the_source_bias_width(cls):
    source = _source(torch.bfloat16)
    module = _packed(cls, source)

    assert module.bias.dtype is torch.bfloat16
    assert module.bias_dtype is torch.bfloat16
    assert module.bias[0].item() == BIAS


def test_portable_forward_keeps_a_finite_bfloat16_bias():
    module = _packed(QQQTorchLinear, _source(torch.bfloat16))
    out = _forward_bfloat16(module)

    assert out.dtype is torch.bfloat16
    assert out[0, 0].item() == BIAS
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [QQQTorchLinear, QQQLinear])
def test_state_dict_reload_keeps_the_bias_width(cls):
    source = _source(torch.bfloat16)
    state = _packed(cls, source).state_dict()

    target = _module(cls)
    assert target.bias.dtype is torch.float16
    target.load_state_dict(state)

    assert target.bias.dtype is torch.bfloat16
    assert target.bias[0].item() == BIAS


def test_kernel_rebuild_copy_keeps_the_bias_width():
    # ``create_quant_linear`` rebuilds a loaded kernel by copying the source
    # module's state dict into a freshly constructed FP16 placeholder.
    state = _packed(QQQTorchLinear, _source(torch.bfloat16)).state_dict()
    rebuilt = _module(QQQTorchLinear)
    rebuilt.load_state_dict(state, strict=False)

    assert rebuilt.bias.dtype is torch.bfloat16
    assert _forward_bfloat16(rebuilt)[0, 0].item() == BIAS


def test_export_after_a_cpu_forward_keeps_the_bias_width():
    module = _packed(QQQTorchLinear, _source(torch.bfloat16))
    # The first CPU forward may replace B/scales with the packed cache; the
    # bias is not derived state and must survive the export unchanged.
    assert _forward_bfloat16(module)[0, 0].item() == BIAS

    exported = module.state_dict()
    assert exported["bias"].dtype is torch.bfloat16
    assert exported["bias"][0].item() == BIAS


def test_constructor_bias_dtype_is_honoured():
    module = _module(QQQTorchLinear, bias_dtype=torch.bfloat16)
    assert module.bias.dtype is torch.bfloat16


def test_float16_bias_stays_float16():
    module = _packed(QQQTorchLinear, _source(torch.float16, bias=1024.0))
    assert module.bias.dtype is torch.float16
    assert _forward_bfloat16(module).dtype is torch.bfloat16


class _Model(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.proj = module

    def tie_weights(self):
        pass


def test_accelerate_load_keeps_the_bias_width(tmp_path):
    from safetensors.torch import save_file

    from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights

    module = _packed(QQQTorchLinear, _source(torch.bfloat16))
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in _Model(module).state_dict().items()},
        str(checkpoint),
        metadata={"format": "pt"},
    )

    model = _Model(_module(QQQTorchLinear))
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    assert model.proj.bias.dtype is torch.bfloat16
    assert model.proj.bias[0].item() == BIAS
    assert torch.isfinite(_forward_bfloat16(model.proj)).all()
