# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``post_init`` must regenerate the derived wf shift tables.

Transformers materializes a meta shell's non-persistent buffer slots with
``empty_like`` before handing the model to GPT-QModel. The slots exist but
hold uninitialized storage, and ``_init_wf_unsqueeze_buffers`` used to return
early whenever both names were present, so the CPU repack and portable decode
read garbage shifts. Loading the same checkpoint twice then produced
different logits (issue #5843).
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND


GROUP = 32


def _packed(bits: int = 4, seed: int = 0) -> TorchLinear:
    torch.manual_seed(seed)
    linear = nn.Linear(64, 128, bias=False).half()
    generator = torch.Generator().manual_seed(seed)
    groups = linear.in_features // GROUP
    scales = torch.rand(linear.out_features, groups, generator=generator) * 0.05 + 1e-3
    zeros = torch.randint(
        0, 1 << bits, (linear.out_features, groups), generator=generator, dtype=torch.int32
    )
    module = TorchLinear(
        bits=bits,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=FORMAT.GPTQ,
    )
    module.pack_original(linear, scales, zeros)
    module.post_init()
    return module, linear


def _expected_tables(bits: int, pack_dtype_bits: int):
    if bits in (2, 4, 8):
        wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32)
        return wf.unsqueeze(0).unsqueeze(0), wf.unsqueeze(0).unsqueeze(-1)
    wf = torch.tensor(
        [
            [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
            [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
            [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
        ],
        dtype=torch.int32,
    ).reshape(1, 3, 12)
    return wf.unsqueeze(0), wf.unsqueeze(-1)


def _poison_tables(module: TorchLinear) -> None:
    """Simulate the assignment path: slots exist but were never initialized."""

    module._buffers["wf_unsqueeze_zero"] = torch.full_like(module.wf_unsqueeze_zero, -1)
    module._buffers["wf_unsqueeze_neg_one"] = torch.full_like(module.wf_unsqueeze_neg_one, -1)


@pytest.mark.parametrize("bits", [4, 3])
def test_post_init_rebuilds_materialized_tables(bits):
    module, _ = _packed(bits)
    _poison_tables(module)

    module.post_init()

    zero, neg_one = _expected_tables(bits, module.pack_dtype_bits)
    assert torch.equal(module.wf_unsqueeze_zero, zero)
    assert torch.equal(module.wf_unsqueeze_neg_one, neg_one)


@pytest.mark.parametrize("bits", [4, 3])
def test_post_init_restores_packed_forward_parity(bits):
    reference, _ = _packed(bits)
    poisoned, _ = _packed(bits)
    _poison_tables(poisoned)
    poisoned.post_init()

    probe = torch.randn(2, 64, dtype=torch.float16)
    with torch.inference_mode():
        assert torch.equal(poisoned(probe), reference(probe))


def test_forced_refresh_does_not_materialize_tables_on_meta():
    """A disk-offloaded module must keep its live tables (#5788)."""

    module, _ = _packed(4)
    module.to("meta")
    assert module.wf_unsqueeze_zero.device.type == "cpu"
    assert module.wf_unsqueeze_neg_one.device.type == "cpu"

    module._init_wf_unsqueeze_buffers(force=True)

    assert module.wf_unsqueeze_zero.device.type == "cpu"
    assert module.wf_unsqueeze_neg_one.device.type == "cpu"
