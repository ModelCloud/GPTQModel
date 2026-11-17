# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Re-save must preserve dense Parameter storage dtype and payload (#4466).

The export shell is allocated in the model dtype. Accelerate's ``dtype=None``
load copies each checkpoint tensor into that shell slot, so a dense Parameter
stored at another dtype (a norm at FP16 next to a BF16 config) was narrowed on
re-save. The shell now restores every checkpoint-owned tensor, Parameters
included, at its stored dtype.

The same module also covers the CPU int4pack repack gap: after one valid
forward the live packed operands are a private runtime ABI, while the canonical
checkpoint tensors survive in the non-persistent snapshot. A repack must use
that snapshot, not copy the transformed runtime operands.
"""

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_resave_preserves_checkpoint_dtypes_4379 import (
    CALIBRATION,
    _shard_tensors,
    _tiny_source,
)

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.packing import hf_pack_layer, hf_repack_layer


KEY = "model.norm.weight"
# Finite and exactly representable at FP16, but not at BF16: a BF16 round trip
# collapses it to 1.0.
EXPECTED_VALUE = 1.001


def _replace_tensor(directory, key, tensor):
    replaced = 0
    for filename in sorted(directory.iterdir()):
        if not filename.name.endswith(".safetensors"):
            continue
        tensors = load_file(str(filename))
        if key not in tensors:
            continue
        tensors[key] = tensor
        save_file(tensors, str(filename), metadata={"format": "pt"})
        replaced += 1
    assert replaced == 1, f"expected exactly one shard with {key}, found {replaced}"


def _quantized_tiny_model(tmp_path):
    source = _tiny_source(tmp_path / "source")
    quantized = tmp_path / "quantized"
    model = GPTQModel.load(
        str(source),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device="cpu"
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CALIBRATION,
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
        calibration_data_min_length=1,
    )
    model.save(str(quantized))
    del model
    return quantized


def test_resave_preserves_dense_parameter_dtype_and_payload(tmp_path):
    quantized = _quantized_tiny_model(tmp_path)
    resaved = tmp_path / "resaved"

    expected = torch.full((32,), EXPECTED_VALUE, dtype=torch.float16)
    _replace_tensor(quantized, KEY, expected)
    before = _shard_tensors(quantized)
    assert before[KEY][0] is torch.float16

    loaded = GPTQModel.load(str(quantized), backend=BACKEND.TORCH, device="cpu")
    # A normal load still casts dense Parameters to the requested model dtype.
    # The export shell must not copy that runtime slot: it rereads the
    # checkpoint tensor instead.
    runtime_parameter = dict(loaded.model.named_parameters())[KEY]
    assert runtime_parameter.dtype is torch.bfloat16
    assert runtime_parameter[0].item() == 1.0

    loaded.save(str(resaved))
    del loaded

    after = _shard_tensors(resaved)
    assert set(after) == set(before)
    assert after[KEY][0] is torch.float16, after[KEY][0]
    assert torch.equal(after[KEY][1], before[KEY][1]), KEY

    # The re-saved artifact must remain loadable.
    GPTQModel.load(str(resaved), backend=BACKEND.TORCH, device="cpu")


def _aten_available() -> bool:
    ok, _ = TorchAtenLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=64,
        pack_dtype=torch.int32,
        device=DEVICE.CPU,
    )
    return bool(ok)


@pytest.mark.skipif(not _aten_available(), reason="ATen CPU int4pack ops are unavailable")
def test_repack_after_cpu_forward_uses_checkpoint_snapshot():
    """An unchanged source must repack from its snapshot, not runtime operands."""

    torch.set_num_threads(1)
    module = TorchAtenLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=64,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    dense = torch.nn.Linear(128, 64, bias=False)
    with torch.no_grad():
        dense.weight.fill_(1)
    hf_pack_layer(
        module,
        dense,
        torch.ones(64, 4),
        checkpoint_format=FORMAT.GPTQ_V2,
        post_init=True,
    )
    checkpoint_qweight = module.qweight.detach().clone()
    module.eval()

    with torch.inference_mode():
        output = module(torch.ones(1, 128, dtype=torch.bfloat16))
    assert torch.equal(output, torch.full_like(output, 128))
    # The forward replaced the live operand with the int4pack runtime layout
    # and retained the checkpoint tensor in the snapshot.
    assert module._fused_train_state is not None
    assert module.qweight.shape != checkpoint_qweight.shape
    snapshot_qweight = module._buffers[module._fused_train_state["qweight"]]
    assert torch.equal(snapshot_qweight, checkpoint_qweight)

    target = hf_repack_layer(module, BACKEND.GPTQ_TORCH, device="cpu").eval()
    with torch.inference_mode():
        repacked_output = target(torch.ones(1, 128, dtype=torch.bfloat16))

    # The target received the canonical checkpoint tensor, not the transformed
    # uint8 int4pack operand, so it remains a valid checkpoint-layout layer.
    assert target.qweight.shape == checkpoint_qweight.shape
    assert torch.equal(target.qweight, checkpoint_qweight)
    assert torch.equal(repacked_output, torch.full_like(repacked_output, 128))
    # Repacking must not mutate the source's live runtime layout.
    assert module.qweight.shape != checkpoint_qweight.shape
