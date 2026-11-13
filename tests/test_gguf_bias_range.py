# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.utils.model import (
    _restore_fp32_checkpoint_buffers,
    load_checkpoint_in_model_then_tie_weights,
)


def test_gguf_packing_preserves_fp32_bias_range():
    linear = torch.nn.Linear(32, 32, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(100000.0)
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=True,
        register_buffers=False,
    )
    module.pack_original(linear, scales=None, zeros=None)

    output = module(torch.zeros((2, 32), dtype=torch.float32))

    assert module.bias.dtype == torch.float32
    assert torch.equal(output, torch.full((2, 32), 100000.0))


def _packed_q8_bias_case(dtype: torch.dtype):
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=1,
        bias=True,
    ).eval()
    if dtype == torch.float16:
        scale, code, activation, weight = 0.125, 25, 1000.0, 3.125
    else:
        scale, code, activation, weight = 1.0, 1, 3120.0, 1.0
    module.qweight[:, :2] = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
    module.qweight[:, 2:] = code
    module.bias.fill_(-100000.0)
    module.gguf_fused_cpu_max_rows = 0
    x = torch.full((1, 32), activation, dtype=dtype)
    # The reference uses the specified Q8_0 scale/code directly; it does not
    # call the GGUF dequantizer or either forward implementation.
    expected = (
        x.double() @ torch.full((32, 1), weight, dtype=torch.float64)
        + torch.tensor([-100000.0], dtype=torch.float64)
    ).to(dtype)
    return module, x, expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_cpu_forward_preserves_fp32_bias_until_output_rounding(dtype):
    module, x, expected = _packed_q8_bias_case(dtype)
    original_qweight = module.qweight.clone()
    original_bias = module.bias.clone()

    for _ in range(2):
        actual = module(x)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert module.bias.dtype == torch.float32
    torch.testing.assert_close(module.qweight, original_qweight, rtol=0, atol=0)
    torch.testing.assert_close(module.bias, original_bias, rtol=0, atol=0)
    assert set(dict(module.named_buffers())) == {"qweight", "bias"}
    assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())


class _CheckpointModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.proj, self.x, self.expected = _packed_q8_bias_case(dtype)

    def tie_weights(self, **_kwargs):
        pass


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint_kind", ["single", "sharded", "pytorch"])
def test_gguf_bias_survives_checkpoint_reload(tmp_path, dtype, checkpoint_kind):
    source = _CheckpointModel(dtype)
    state = source.state_dict()
    if checkpoint_kind == "sharded":
        save_file({"proj.qweight": state["proj.qweight"]}, tmp_path / "weights.safetensors", metadata={"format": "pt"})
        save_file({"proj.bias": state["proj.bias"]}, tmp_path / "bias.safetensors", metadata={"format": "pt"})
        index = tmp_path / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {
            "proj.qweight": "weights.safetensors", "proj.bias": "bias.safetensors",
        }}))
        checkpoint = index
    elif checkpoint_kind == "single":
        checkpoint = tmp_path / "model.safetensors"
        save_file(state, checkpoint, metadata={"format": "pt"})
    else:
        checkpoint = tmp_path / "pytorch_model.bin"
        torch.save(state, checkpoint)

    restored = _CheckpointModel(dtype)
    restored.proj.bias.zero_()
    restored.proj.qweight.zero_()
    load_checkpoint_in_model_then_tie_weights(
        restored,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=dtype,
        keep_in_fp32_modules=["proj.bias"],
        fp32_bf16_buffer_names=["proj.bias"],
    )

    assert restored.proj.bias.dtype == torch.float32
    assert restored.proj.bias.item() == -100000.0
    torch.testing.assert_close(restored.proj.qweight, source.proj.qweight, rtol=0, atol=0)
    torch.testing.assert_close(restored.proj(restored.x), restored.expected, rtol=0, atol=0)


def test_gguf_bias_survives_checkpoint_key_remap(tmp_path):
    source = _CheckpointModel(torch.bfloat16)
    checkpoint = tmp_path / "renamed.safetensors"
    save_file(
        {f"saved.{name}": value for name, value in source.state_dict().items()},
        checkpoint,
        metadata={"format": "pt"},
    )
    restored = _CheckpointModel(torch.bfloat16)
    restored.proj.bias.zero_()
    restored.proj.qweight.zero_()

    load_checkpoint_in_model_then_tie_weights(
        restored,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
        keep_in_fp32_modules=["proj.bias"],
        fp32_bf16_buffer_names=["proj.bias"],
        key_converter=lambda name: name.removeprefix("saved."),
    )

    assert restored.proj.bias.dtype == torch.float32
    torch.testing.assert_close(restored.proj(restored.x), restored.expected, rtol=0, atol=0)


def test_gguf_bias_restore_rejects_missing_index_entry(tmp_path):
    source = _CheckpointModel(torch.bfloat16)
    save_file(
        {"proj.qweight": source.proj.qweight},
        tmp_path / "weights.safetensors",
        metadata={"format": "pt"},
    )
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"proj.qweight": "weights.safetensors"}}))

    with pytest.raises(RuntimeError, match="proj.bias"):
        _restore_fp32_checkpoint_buffers(source, index, {"proj.bias"})


def test_gguf_bias_restore_uses_same_directory_checkpoint_as_accelerate(tmp_path):
    source = _CheckpointModel(torch.bfloat16)
    torch.save(source.state_dict(), tmp_path / "pytorch_model.bin")
    alternate = {name: value.clone() for name, value in source.state_dict().items()}
    alternate["proj.bias"].fill_(1.0)
    save_file(alternate, tmp_path / "model.safetensors", metadata={"format": "pt"})
    source.proj.bias.zero_()

    _restore_fp32_checkpoint_buffers(source, tmp_path, {"proj.bias"})

    assert source.proj.bias.item() == -100000.0
