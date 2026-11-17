# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The AVX512 CPU kernel must not keep the checkpoint ``qweight`` twice (#4783).

``TorchCpuAvx512Linear`` is what CPU ``AUTO`` selects for a bf16 GPTQ model on
an AVX512-BF16 host. Its ``transform_cpu`` replaces ``qweight`` with the packed
runtime words, but the class declared no verified inverse, so
``capture_fused_train_state`` kept a full clone of the checkpoint words next to
it: +63% resident bytes on an 8B model.

``unpack_hf_int4`` is the documented inverse of ``pack_hf_int4``, so the
checkpoint words can be rebuilt on save. These tests assert that the clone is
dropped only when the rebuild reproduces the captured bytes exactly, for both
the identity channel order and a ``desc_act`` permutation.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear._fused_train_state import (
    RebuiltCheckpointTensor,
    capture_fused_train_state,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.quantization import FORMAT


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32
SNAPSHOT_BUFFER = "_fused_train_state_qweight"


def _available() -> bool:
    try:
        ok, _ = TorchCpuAvx512Linear.cached_validate_once()
    except Exception:
        return False
    return bool(ok)


def _g_idx(desc_act: bool) -> torch.Tensor:
    canonical = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    if not desc_act:
        return canonical
    # Rotate the groups: every group keeps ``GROUP_SIZE`` columns, but the
    # channel order is no longer the canonical one, so ``ret_idx`` is a real
    # permutation and the rebuild has to undo it.
    groups = IN_FEATURES // GROUP_SIZE
    return (canonical + 1) % groups


def _module(desc_act: bool) -> TorchCpuAvx512Linear:
    torch.manual_seed(4783)
    module = TorchCpuAvx512Linear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=desc_act,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        dtype=torch.bfloat16,
    )
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        dense.weight.normal_(mean=0.0, std=0.02)
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        dense,
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.bfloat16),
        torch.full((OUT_FEATURES, groups), 1 << 3, dtype=torch.int32),
        _g_idx(desc_act),
    )
    module.post_init()
    return module.eval()


@pytest.mark.skipif(not _available(), reason="AVX512-BF16 CPU kernel unavailable")
@pytest.mark.parametrize("desc_act", [False, True], ids=["identity", "desc_act"])
def test_avx512_transform_drops_the_checkpoint_qweight_clone(desc_act):
    module = _module(desc_act)
    checkpoint_qweight = module.qweight.detach().clone()
    assert module.qweight.dtype is torch.int32

    capture_fused_train_state(module)
    assert SNAPSHOT_BUFFER in module._buffers, "capture did not clone the checkpoint words"

    module.transform_cpu()

    # The transform really produced a different layout, so the assertion below
    # is about the rebuild and not about an untouched tensor.
    assert module.qweight.shape != checkpoint_qweight.shape

    rebuilt = module._rebuild_checkpoint_qweight()
    assert rebuilt.shape == checkpoint_qweight.shape
    assert torch.equal(rebuilt, checkpoint_qweight), (
        "the rebuilt checkpoint words do not match the bytes captured before the "
        "transform; the clone must be kept in that case"
    )

    # The clone is gone and the snapshot points at the rebuilt descriptor.
    assert SNAPSHOT_BUFFER not in module._buffers
    assert isinstance(module._fused_train_state["qweight"], RebuiltCheckpointTensor)

    # A save still emits the checkpoint bytes, not the runtime layout.
    saved = module.state_dict()
    assert torch.equal(saved["qweight"], checkpoint_qweight)


@pytest.mark.skipif(not _available(), reason="AVX512-BF16 CPU kernel unavailable")
@pytest.mark.parametrize("desc_act", [False, True], ids=["identity", "desc_act"])
def test_avx512_rebuilt_words_survive_a_forward(desc_act):
    """The live runtime words are unchanged by rebuilding the checkpoint layout."""

    module = _module(desc_act)
    checkpoint_qweight = module.qweight.detach().clone()
    capture_fused_train_state(module)
    module.transform_cpu()
    live = module.qweight.detach().clone()

    rebuilt = module._rebuild_checkpoint_qweight()

    assert torch.equal(module.qweight, live)
    assert torch.equal(rebuilt, checkpoint_qweight)
