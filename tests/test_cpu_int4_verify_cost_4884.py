# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Verifying the CPU int4pack inverse must not rebuild every weight (#4884).

``#4845`` (fix of ``#4783``) drops the checkpoint ``qweight`` clone of
``TorchCpuAvx512Linear`` and verifies the pure-Torch inverse against that clone
inside ``transform_cpu``. Because the inverse was a full ``codes.t()``
transpose plus eight strided shift/OR passes, every module paid that rebuild
twice: once to verify and once on every ``state_dict()``. On an 8B model the
first forward went from 1.97 s to 23.5 s and ``state_dict()`` from 0.32 s to
22.95 s.

These tests bound both costs:

* the inverse must stay within a small multiple of the forward pack of the
  same shape (the forward pack is the pure-Torch operation it inverts);
* a transformed module's ``state_dict()`` must stay within a bounded multiple
  of copying its own checkpoint bytes.

Both bounds are ratios, so a slow or loaded host moves the numerator and the
denominator together.
"""

from __future__ import annotations

import time

import pytest
import torch

from gptqmodel.nn_modules.qlinear._cpu_int4pack_layout import (
    rebuild_avx512_qweight,
)
from gptqmodel.nn_modules.qlinear._fused_train_state import (
    RebuildCandidate,
    RebuiltCheckpointTensor,
    capture_fused_train_state,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import (
    TorchCpuAvx512Linear,
    pack_hf_int4,
    unpack_hf_int4,
)
from gptqmodel.quantization import FORMAT


# The issue measured this synthetic module: rebuild 200 ms (desc_act) / 144 ms
# (identity) against a 0.10 s C++ pack for a whole 1B model.
SYNTHETIC_OUT = 4096
SYNTHETIC_IN = 14336

# Small enough that the state_dict guard runs in a few milliseconds, large
# enough that per-op overhead does not dominate the copy baseline.
MODULE_OUT = 1024
MODULE_IN = 8192
MODULE_GROUP = 128


def _best_seconds(fn, repeats: int = 5) -> float:
    """Return the fastest of ``repeats`` runs, after one warm-up."""

    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


class _SyntheticLayout:
    """Minimal module surface ``rebuild_avx512_qweight`` reads."""

    bits = 4
    pack_factor = 8
    out_features = SYNTHETIC_OUT
    in_features = SYNTHETIC_IN

    def __init__(self, qweight: torch.Tensor):
        self.qweight = qweight

    def _ret_idx_is_identity(self) -> bool:
        return True


def test_avx512_inverse_cost_is_bounded_by_the_forward_pack():
    """The checkpoint inverse must not cost many forward packs per weight.

    The C++ ``pack_gptq_int4`` kernel is gated on AVX512-BF16 and cannot load
    on this host, so the budget uses the pure-Torch ``pack_hf_int4`` forward of
    the exact same permutation and byte count. Unmodified main rebuilds with a
    ``codes.t()`` transpose plus eight strided ``int32`` shift/OR passes and
    lands around 5-6x the forward; the contiguous inverse stays near 1x.
    """

    torch.manual_seed(4884)
    packed = torch.randint(
        0, 256, (SYNTHETIC_OUT, SYNTHETIC_IN // 2), dtype=torch.uint8
    )
    layout = _SyntheticLayout(packed)
    codes = unpack_hf_int4(packed, SYNTHETIC_OUT, SYNTHETIC_IN)

    forward = _best_seconds(lambda: pack_hf_int4(codes))
    rebuild = _best_seconds(lambda: rebuild_avx512_qweight(layout))
    unpack = _best_seconds(
        lambda: unpack_hf_int4(packed, SYNTHETIC_OUT, SYNTHETIC_IN)
    )

    assert forward > 0.0
    assert rebuild < 3 * forward, (
        "rebuild_avx512_qweight costs "
        f"{rebuild / forward:.1f}x the forward pack of the same shape "
        f"({rebuild * 1e3:.1f} ms vs {forward * 1e3:.1f} ms)"
    )
    assert unpack < 3 * forward, (
        "unpack_hf_int4 costs "
        f"{unpack / forward:.1f}x the forward pack of the same shape "
        f"({unpack * 1e3:.1f} ms vs {forward * 1e3:.1f} ms)"
    )


def _codes_from_checkpoint_words(words: torch.Tensor, out_features: int) -> torch.Tensor:
    """Unpack checkpoint ``[in // 8, out]`` int32 words into ``[out, in]`` codes."""

    in_features = words.shape[0] * 8
    codes = torch.empty((in_features, out_features), dtype=torch.uint8)
    for nibble in range(8):
        codes[nibble::8, :] = ((words.to(torch.int64) >> (4 * nibble)) & 0x0F).to(
            torch.uint8
        )
    return codes.t().contiguous()


def _fake_avx512_pack(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    ret_idx: torch.Tensor,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch stand-in for the AVX512-BF16 ``pack_gptq_int4`` kernel."""

    codes = _codes_from_checkpoint_words(qweight, out_features)
    if ret_idx.numel():
        codes = codes[:, ret_idx.to(torch.long)]
    packed = pack_hf_int4(codes.contiguous())
    zero_codes = torch.empty(
        (qzeros.shape[0], out_features), dtype=torch.uint8
    )
    words = qzeros.to(torch.int64)
    for nibble in range(8):
        zero_codes[:, nibble::8] = ((words >> (4 * nibble)) & 0x0F).to(torch.uint8)
    return packed.contiguous(), zero_codes.contiguous()


def _corrupting_avx512_pack(*args, **kwargs):
    """Pack correctly, then flip one runtime nibble to model a corrupted layout."""

    packed, zero_codes = _fake_avx512_pack(*args, **kwargs)
    packed = packed.clone()
    packed[0, 0] ^= 0x01
    return packed, zero_codes


def _g_idx(desc_act: bool) -> torch.Tensor:
    canonical = torch.arange(MODULE_IN, dtype=torch.int32) // MODULE_GROUP
    if not desc_act:
        return canonical
    groups = MODULE_IN // MODULE_GROUP
    return (canonical + 1) % groups


def _make_module(desc_act: bool = False) -> TorchCpuAvx512Linear:
    torch.manual_seed(4884)
    module = TorchCpuAvx512Linear(
        bits=4,
        group_size=MODULE_GROUP,
        sym=True,
        desc_act=desc_act,
        in_features=MODULE_IN,
        out_features=MODULE_OUT,
        bias=False,
        format=FORMAT.GPTQ_V2,
        dtype=torch.bfloat16,
    )
    dense = torch.nn.Linear(
        MODULE_IN, MODULE_OUT, bias=False, dtype=torch.bfloat16
    )
    with torch.no_grad():
        dense.weight.normal_(mean=0.0, std=0.02)
    groups = MODULE_IN // MODULE_GROUP
    module.pack(
        dense,
        torch.full((MODULE_OUT, groups), 1 / 128, dtype=torch.bfloat16),
        torch.full((MODULE_OUT, groups), 1 << 3, dtype=torch.int32),
        _g_idx(desc_act),
    )
    module.post_init()
    return module.eval()


@pytest.fixture
def avx512_stand_in(monkeypatch):
    """Run the module-level AVX512 path with pure-Torch kernels on any host."""

    monkeypatch.setattr(
        TorchCpuAvx512Linear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    yield


def _transform(module, pack_kernel) -> None:
    capture_fused_train_state(module)
    module.pack_gptq_int4_kernel = pack_kernel
    module.transform_cpu()


def _checkpoint_tensors(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


@pytest.mark.parametrize("desc_act", [False, True], ids=["identity", "desc_act"])
def test_transformed_state_dict_is_bounded_by_its_own_bytes(avx512_stand_in, desc_act):
    """``state_dict()`` must stay close to copying the checkpoint bytes.

    A transformed module cannot emit the checkpoint ``qweight`` without
    rebuilding it, so the guard bounds that rebuild instead of pretending it is
    free. The ratio uses the module's own checkpoint byte count, and a second
    ratio against the forward pack keeps the bound meaningful on a loaded host.
    """

    module = _make_module(desc_act)
    checkpoint = _checkpoint_tensors(module)
    qweight_bytes = (
        checkpoint["qweight"].numel() * checkpoint["qweight"].element_size()
    )
    _transform(module, _fake_avx512_pack)

    source = torch.empty(qweight_bytes, dtype=torch.uint8)
    destination = torch.empty(qweight_bytes, dtype=torch.uint8)
    memcpy = _best_seconds(lambda: destination.copy_(source))
    state_dict = _best_seconds(lambda: module.state_dict())
    forward = _best_seconds(
        lambda: pack_hf_int4(
            unpack_hf_int4(module.qweight, MODULE_OUT, MODULE_IN)
        )
    )

    assert memcpy > 0.0
    # A pure-Torch nibble permutation runs at a few GB/s while a raw copy
    # reaches tens of GB/s, so the honest floor is tens of copies. 150x is
    # still far below the ~350x of the old eight-pass rebuild on this host.
    assert state_dict < 150 * memcpy, (
        f"state_dict() took {state_dict * 1e3:.2f} ms for {qweight_bytes / 1e6:.1f} MB "
        f"of checkpoint bytes ({state_dict / memcpy:.0f}x a raw copy)"
    )
    assert state_dict < 3 * forward, (
        f"state_dict() took {state_dict * 1e3:.2f} ms, "
        f"{state_dict / forward:.1f}x the forward pack of the same weight"
    )
    # The emitted checkpoint is still bitwise identical.
    emitted = module.state_dict()
    for name, expected in checkpoint.items():
        assert torch.equal(emitted[name], expected), name


def test_avx512_verification_runs_once_per_layout(avx512_stand_in, monkeypatch):
    """Only the first module of a layout pays the full rebuild-compare."""

    real_rebuild = TorchCpuAvx512Linear._rebuild_checkpoint_qweight
    calls: list[int] = []

    def counting_rebuild(self):
        calls.append(1)
        return real_rebuild(self)

    monkeypatch.setattr(
        TorchCpuAvx512Linear, "_rebuild_checkpoint_qweight", counting_rebuild
    )

    first = _make_module()
    capture_fused_train_state(first)
    first.pack_gptq_int4_kernel = _fake_avx512_pack
    calls.clear()
    first.transform_cpu()
    assert len(calls) == 1, "the first module of a layout must verify once"
    assert isinstance(first._fused_train_state["qweight"], RebuiltCheckpointTensor)

    second = _make_module()
    capture_fused_train_state(second)
    second.pack_gptq_int4_kernel = _fake_avx512_pack
    calls.clear()
    second.transform_cpu()
    assert calls == [], "a later module of the same layout must not rebuild"
    assert isinstance(second._fused_train_state["qweight"], RebuiltCheckpointTensor)
    assert "_fused_train_state_qweight" not in second._buffers


@pytest.mark.parametrize("desc_act", [False, True], ids=["identity", "desc_act"])
def test_avx512_verification_rejects_corrupted_runtime_words(
    avx512_stand_in, desc_act
):
    """A corrupted packed tensor must keep the clone authoritative."""

    module = _make_module(desc_act)
    checkpoint = _checkpoint_tensors(module)
    _transform(module, _corrupting_avx512_pack)

    snapshot = module._fused_train_state
    assert isinstance(snapshot["qweight"], RebuildCandidate)
    assert "_fused_train_state_qweight" in module._buffers
    emitted = module.state_dict()
    for name, expected in checkpoint.items():
        assert torch.equal(emitted[name], expected), name


def test_avx512_verification_rejects_a_corrupted_inverse(avx512_stand_in, monkeypatch):
    """A wrong inverse must not inherit an earlier layout's proof."""

    module = _make_module()
    checkpoint = _checkpoint_tensors(module)
    real_rebuild = TorchCpuAvx512Linear._rebuild_checkpoint_qweight

    def corrupted_rebuild(self):
        tensor = real_rebuild(self).clone()
        tensor[0, 0] ^= 1
        return tensor

    monkeypatch.setattr(
        TorchCpuAvx512Linear,
        "_rebuild_checkpoint_qweight",
        corrupted_rebuild,
    )
    _transform(module, _fake_avx512_pack)
    assert isinstance(module._fused_train_state["qweight"], RebuildCandidate)
    assert "_fused_train_state_qweight" in module._buffers
    emitted = module.state_dict()
    for name, expected in checkpoint.items():
        assert torch.equal(emitted[name], expected), name
