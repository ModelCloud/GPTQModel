# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #5013: drop the CPU int4 ``qweight`` snapshot where an inverse exists.

#4918 (fix of #4873) removed the fused/Paro operands that pinned the
safetensors mapping, but two CPU int4 layouts still kept a **full
checkpoint-layout copy of ``qweight``** (``_fused_train_state_qweight``) for the
life of the model, even though a bitwise inverse exists:

* ``TorchCpuAvx512AwqLinear`` declared ``_FUSED_TRAIN_STATE_REBUILD = None``
  ("no verified inverse") for the extension's ``pack_awq_int4``.
* GPTQ ``desc_act`` on ``TorchAtenLinear``/``TorchFusedLinear`` bailed out of
  ``_can_rebuild_checkpoint_qweight`` whenever ``ret_idx`` was not the identity,
  although the module keeps ``ret_idx`` and scattering the unpacked codes back
  through it is the exact inverse of the transform's ``index_select``.

``AUTO`` selects ``TorchAtenLinear`` for a FP16 CPU load, so the second case is
not opt-in: a ``desc_act`` checkpoint kept ~3.3 GiB of duplicate ``qweight`` on
an 8B model (measured in the issue).

Both inverses are only ever trusted after ``finalize_fused_train_state``
compares the rebuilt words bitwise with the clone captured before the
transform; a mismatch keeps the clone (fail closed). These tests pin the
inverse, the drop, and the fail-closed behaviour.

**Host caveat:** this machine has no AVX512-BF16, so the AWQ AVX512 extension
kernel cannot load and its forward cannot run. The AWQ packer is verified
against a line-by-line Python mirror of
``gptqmodel_ext/int4_qzero_group_scaled_cpu/int4_qzero_group_scaled_pack.cpp``
(and against the real kernel when it is available), and the module-level test
installs the runtime words the kernel would produce.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear._cpu_int4pack_layout import (
    int4pack_block_size,
    rebuild_gptq_qweight,
)
from gptqmodel.nn_modules.qlinear._fused_train_state import (
    RebuildCandidate,
    RebuiltCheckpointTensor,
    capture_fused_train_state,
    finalize_fused_train_state,
)
from gptqmodel.nn_modules.qlinear._runtime_cache import build_group_ret_idx
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization.awq.utils.packing_utils import AWQ_REVERSE_ORDER
from gptqmodel.utils.backend import BACKEND


SNAPSHOT_BUFFER = "_fused_train_state_qweight"


def _tensor_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def _require_aten_int4pack() -> None:
    if int4pack_block_size() is None:
        pytest.skip("aten CPU int4pack op is unavailable")


def _permuted_g_idx(in_features: int, group_size: int, seed: int) -> torch.Tensor:
    """A checkpoint-valid ``desc_act`` g_idx that is not the identity order."""

    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    generator = torch.Generator().manual_seed(seed)
    return g_idx[torch.randperm(in_features, generator=generator)]


def _checkpoint_words(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Independent oracle: pack ``[out, in]`` codes into checkpoint words.

    ``qweight[word, out]`` nibble ``i`` holds the code of input row
    ``word * (32 // bits) + i`` (the layout ``transform_cpu`` unpacks).
    """

    out_features, in_features = codes.shape
    pack_factor = 32 // bits
    words = in_features // pack_factor
    expected = torch.zeros((words, out_features), dtype=torch.int32)
    for index in range(pack_factor):
        # ``codes[:, index::pack_factor]`` is ``[out, words]``; the checkpoint
        # word is ``[word, out]``.
        column = codes[:, index::pack_factor].t().to(torch.int32)
        expected |= column << (index * bits)
    return expected


def _desc_act_gptq_module(
    layer_cls, backend, *, in_features, group_size, out_features=64, seed=11
):
    """A packed ``desc_act`` module whose ``ret_idx`` reorders the input rows."""

    from tests.test_checkpoint_state_invariant import _materialize

    module = layer_cls(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=backend,
        bias=False,
        register_buffers=True,
    )
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.05 + 1e-3
    zeros = torch.randint(0, 16, (out_features, groups), dtype=torch.int32)
    module.pack_original(
        linear, scales, zeros, _permuted_g_idx(in_features, group_size, seed=seed)
    )
    module._init_wf_unsqueeze_buffers()
    return _materialize(module).eval()


class _GptqRuntimeLayout:
    """The module surface ``rebuild_gptq_qweight`` reads."""

    bits = 4
    pack_factor = 8

    def __init__(self, qweight, out_features, in_features, ret_idx=None):
        self.qweight = qweight
        self.out_features = out_features
        self.in_features = in_features
        self._ret_idx = ret_idx

    def _ret_idx_is_identity(self) -> bool:
        if self._ret_idx is None:
            return True
        return torch.equal(
            self._ret_idx,
            torch.arange(self._ret_idx.numel(), dtype=self._ret_idx.dtype),
        )

    def _build_ret_idx(self) -> torch.Tensor:
        if self._ret_idx is None:
            return torch.arange(self.in_features, dtype=torch.int32)
        return self._ret_idx


@pytest.mark.parametrize(
    ("in_features", "group_size", "out_features"),
    [(128, 32, 64), (256, 32, 64), (256, 128, 64), (512, 128, 64)],
)
def test_gptq_rebuild_undoes_the_desc_act_row_reorder(
    in_features, group_size, out_features
):
    """The runtime codes are group-ordered; the checkpoint keeps source order."""

    _require_aten_int4pack()
    torch.manual_seed(5013 + in_features + group_size)
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.int32)
    ret_idx = build_group_ret_idx(_permuted_g_idx(in_features, group_size, seed=7))
    assert not torch.equal(
        ret_idx, torch.arange(in_features, dtype=torch.int32)
    ), "the fixture must exercise a non-identity ret_idx"

    # ``transform_cpu``: index_select the input rows, transpose, then pack.
    runtime_codes = codes.index_select(1, ret_idx.to(torch.long))
    runtime_words = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
        runtime_codes.contiguous(), 1
    ).contiguous()

    rebuilt = rebuild_gptq_qweight(
        _GptqRuntimeLayout(runtime_words, out_features, in_features, ret_idx)
    )

    assert torch.equal(rebuilt, _checkpoint_words(codes, bits=4))


@pytest.mark.parametrize("layer_cls", (TorchAtenLinear, TorchFusedLinear))
@pytest.mark.parametrize(("in_features", "group_size"), ((128, 32), (256, 128)))
def test_desc_act_layer_drops_the_qweight_snapshot(layer_cls, in_features, group_size):
    """A non-identity ``ret_idx`` must not keep a resident checkpoint copy."""

    _require_aten_int4pack()
    backend = (
        BACKEND.GPTQ_TORCH_ATEN
        if layer_cls is TorchAtenLinear
        else BACKEND.GPTQ_TORCH_FUSED
    )
    dtype = torch.float16 if layer_cls is TorchAtenLinear else torch.bfloat16
    module = _desc_act_gptq_module(
        layer_cls, backend, in_features=in_features, group_size=group_size
    )
    assert not module._ret_idx_is_identity(), "the fixture must reorder input rows"

    checkpoint = _tensor_state(module)
    with torch.inference_mode():
        module(torch.randn(2, in_features, dtype=dtype))

    snapshot = module._fused_train_state
    assert snapshot is not None, "the int4pack transform did not run"
    assert isinstance(snapshot["qweight"], RebuiltCheckpointTensor), (
        "a desc_act module with a verified inverse still kept the qweight clone"
    )
    assert SNAPSHOT_BUFFER not in module._buffers

    emitted = _tensor_state(module)
    assert set(emitted) == set(checkpoint)
    for name, expected in checkpoint.items():
        assert torch.equal(emitted[name], expected), f"{name} is not bitwise identical"


def test_desc_act_layer_keeps_the_clone_when_the_inverse_disagrees(monkeypatch):
    """Fail closed: a wrong permutation must never reach a saved checkpoint."""

    _require_aten_int4pack()
    in_features, group_size = 128, 32
    module = _desc_act_gptq_module(
        TorchAtenLinear,
        BACKEND.GPTQ_TORCH_ATEN,
        in_features=in_features,
        group_size=group_size,
        seed=13,
    )
    checkpoint = _tensor_state(module)

    real_rebuild = TorchAtenLinear._rebuild_checkpoint_qweight

    def wrong_order(self):
        # Drop the scatter: the group-ordered codes are emitted as if they were
        # the checkpoint order.
        tensor = real_rebuild(self).clone()
        tensor[0, 0] ^= 1
        return tensor

    monkeypatch.setattr(TorchAtenLinear, "_rebuild_checkpoint_qweight", wrong_order)
    with torch.inference_mode():
        module(torch.randn(2, in_features, dtype=torch.float16))

    snapshot = module._fused_train_state
    assert isinstance(snapshot["qweight"], RebuildCandidate)
    assert SNAPSHOT_BUFFER in module._buffers

    emitted = _tensor_state(module)
    for name, expected in checkpoint.items():
        assert torch.equal(emitted[name], expected), f"{name} changed"


def _pack_awq_int4_mirror(qweight, qzeros, out_features, in_features):
    """Line-by-line mirror of the extension's ``pack_awq_int4`` (#5013).

    ``gptqmodel_ext/int4_qzero_group_scaled_cpu/int4_qzero_group_scaled_pack.cpp``
    writes, for each input feature ``k``, output group ``g`` and row ``r``::

        byte = code(g * 32 + r) | code(g * 32 + r + 16) << 4

    at ``g * (in_features // 2) * 32 + (k // 2) * 32 + r * 2 + (k % 2)``, reading
    the checkpoint words through ``kAwqReverseOrder`` (``AWQ_REVERSE_ORDER``).
    The zero points are unpacked to one byte per output column.
    """

    groups = out_features // 32
    k_half = in_features // 2
    packed = torch.zeros((out_features, k_half), dtype=torch.uint8)
    flat = packed.view(-1)
    source = qweight.to(torch.int64)
    for k in range(in_features):
        for group in range(groups):
            base = group * k_half * 32 + (k >> 1) * 32 + (k & 1)
            for row in range(16):
                low_col = group * 32 + row
                high_col = low_col + 16
                low_word = int(source[k, low_col >> 3])
                high_word = int(source[k, high_col >> 3])
                low = (low_word >> (4 * AWQ_REVERSE_ORDER[low_col & 7])) & 0xF
                high = (high_word >> (4 * AWQ_REVERSE_ORDER[high_col & 7])) & 0xF
                flat[base + row * 2] = torch.tensor(low | (high << 4), dtype=torch.uint8)

    zero_codes = torch.zeros((qzeros.shape[0], out_features), dtype=torch.uint8)
    zero_source = qzeros.to(torch.int64)
    for group in range(qzeros.shape[0]):
        for word_idx in range(out_features // 8):
            word = int(zero_source[group, word_idx])
            for col in range(8):
                zero_codes[group, word_idx * 8 + col] = torch.tensor(
                    (word >> (4 * AWQ_REVERSE_ORDER[col])) & 0xF, dtype=torch.uint8
                )
    return packed, zero_codes


class _AwqAvx512RuntimeLayout:
    bits = 4

    def __init__(self, qweight, out_features, in_features):
        self.qweight = qweight
        self.out_features = out_features
        self.in_features = in_features


@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [(32, 32), (64, 32), (128, 64), (256, 64), (16, 96)],
)
def test_avx512_awq_inverse_round_trips_every_checkpoint_bit(in_features, out_features):
    """The AWQ AVX512 packer is a pure nibble permutation with an exact inverse."""

    # Imported here so the GPTQ nodes still collect on a tree that predates the
    # AWQ inverse (their own failures are then reported individually).
    from gptqmodel.nn_modules.qlinear._cpu_int4pack_layout import (
        rebuild_avx512_awq_qweight,
    )

    torch.manual_seed(5013 + in_features + out_features)
    # Every bit of a checkpoint word is a code nibble, so random full-width
    # words prove the inverse recovers all 32 bits.
    qweight = torch.randint(
        0, 2**31 - 1, (in_features, out_features // 8), dtype=torch.int32
    )
    qzeros = torch.zeros((in_features // 32, out_features // 8), dtype=torch.int32)
    packed, _zeros = _pack_awq_int4_mirror(
        qweight, qzeros, out_features, in_features
    )

    rebuilt = rebuild_avx512_awq_qweight(
        _AwqAvx512RuntimeLayout(packed, out_features, in_features)
    )

    assert torch.equal(rebuilt, qweight)


def test_avx512_awq_inverse_matches_the_extension_when_it_is_available():
    """On an AVX512-BF16 host, cross-check the mirror and the real kernel."""

    from gptqmodel.extension import op as extension_op
    from gptqmodel.nn_modules.qlinear._cpu_int4pack_layout import (
        rebuild_avx512_awq_qweight,
    )
    from gptqmodel.utils.cpu_features import has_avx512_bf16

    if not has_avx512_bf16():
        pytest.skip("the AVX512-BF16 pack kernel cannot load on this host")

    pack = extension_op("int4_qzero_group_scaled_cpu", "pack_awq_int4")
    in_features, out_features = 128, 64
    torch.manual_seed(5013)
    qweight = torch.randint(
        0, 2**31 - 1, (in_features, out_features // 8), dtype=torch.int32
    )
    qzeros = torch.randint(
        0, 2**31 - 1, (in_features // 32, out_features // 8), dtype=torch.int32
    )

    real_packed, real_zeros = pack(qweight, qzeros, out_features, in_features)
    mirror, mirror_zeros = _pack_awq_int4_mirror(
        qweight, qzeros, out_features, in_features
    )
    assert torch.equal(real_packed, mirror)
    assert torch.equal(real_zeros, mirror_zeros)

    rebuilt = rebuild_avx512_awq_qweight(
        _AwqAvx512RuntimeLayout(real_packed, out_features, in_features)
    )
    assert torch.equal(rebuilt, qweight)


class _UnvalidatedAvx512Awq(TorchCpuAvx512AwqLinear):
    """Construct the module without the AVX512-BF16 runtime gate.

    The gate protects the extension kernel, which cannot load on this host (no
    AVX512-BF16), so the module would be impossible to build. The tests below
    never run a forward: they install the runtime words the kernel would
    produce and exercise capture/finalize/serialization, which are pure Torch.
    """

    @classmethod
    def validate_once(cls):
        return True, None


# ``BaseQuantLinear.__init__`` requires every subclass to restate the kernel
# declarations in its own ``__dict__``; this test double is the real AVX512 AWQ
# class with only the runtime gate relaxed.
for _name, _value in list(TorchCpuAvx512AwqLinear.__dict__.items()):
    if _name.startswith("SUPPORTS") and not callable(_value):
        setattr(_UnvalidatedAvx512Awq, _name, _value)


def _awq_avx512_module(in_features=128, group_size=32, out_features=64):
    """An AVX512 AWQ module holding the same checkpoint words an AWQ pack produces.

    ``AwqTorchLinear.pack`` writes the identical checkpoint layout (``qweight``
    ``[in, out // 8]`` int32, ``qzeros`` ``[in // group_size, out // 8]`` int32,
    ``scales`` ``[in // group_size, out]``), so its buffers can seed this
    module without the extension kernel.
    """

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    source = AwqTorchLinear(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.AWQ_TORCH,
        bias=False,
        register_buffers=True,
    )
    torch.manual_seed(5013)
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.05 + 1e-3
    zeros = torch.randint(0, 16, (out_features, groups), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    source.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)

    module = _UnvalidatedAvx512Awq(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        register_buffers=True,
    )
    # Clone outside ``inference_mode`` so the buffers are ordinary tensors, like
    # a module loaded from a checkpoint.
    module.qweight = source.qweight.detach().clone()
    module.qzeros = source.qzeros.detach().clone()
    module.scales = source.scales.detach().clone()
    # Stand in for the AVX512-BF16 extension: ``transform_cpu`` calls this
    # instance attribute, so the real transform runs end to end (capture,
    # pack, finalize) without the kernel.
    module.pack_awq_int4_kernel = lambda qweight, qzeros, out, inp: list(
        _pack_awq_int4_mirror(qweight, qzeros, out, inp)
    )
    return module


def test_avx512_awq_module_drops_the_qweight_snapshot():
    """Capture -> runtime words -> finalize must drop the clone and save exactly."""

    module = _awq_avx512_module()
    checkpoint = _tensor_state(module)
    capture_fused_train_state(module)

    snapshot = module._fused_train_state
    assert isinstance(snapshot["qweight"], RebuildCandidate), (
        "the AVX512 AWQ layout declared no inverse, so the clone was kept"
    )

    # The real transform: pack with the stand-in kernel, then finalize.
    module.transform_cpu()

    assert isinstance(module._fused_train_state["qweight"], RebuiltCheckpointTensor)
    assert SNAPSHOT_BUFFER not in module._buffers

    emitted = _tensor_state(module)
    assert set(emitted) == set(checkpoint)
    assert torch.equal(emitted["qweight"], checkpoint["qweight"])


def test_avx512_awq_module_keeps_the_clone_for_a_foreign_layout(monkeypatch):
    """A runtime layout the inverse does not describe must keep the snapshot.

    The proof is per *layout*, not per module: once one module of a layout has
    been verified, later modules of the same layout only check shape/dtype
    (#4884). This test therefore clears that process-level cache, which is the
    state a fresh process (or the first module of a new layout) sees.
    """

    from gptqmodel.nn_modules.qlinear import _fused_train_state

    _require_aten_int4pack()
    monkeypatch.setattr(_fused_train_state, "_VERIFIED_REBUILD_LAYOUTS", set())
    module = _awq_avx512_module()
    checkpoint = _tensor_state(module)
    capture_fused_train_state(module)

    # A GPTQ-style int4pack word layout is not the AWQ packer's output.
    codes = torch.randint(
        0, 16, (module.out_features, module.in_features), dtype=torch.int32
    )
    module.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
        codes.contiguous(), 1
    ).contiguous()
    module._cpu_transformed = True

    assert finalize_fused_train_state(module) is False
    assert isinstance(module._fused_train_state["qweight"], RebuildCandidate)
    assert SNAPSHOT_BUFFER in module._buffers

    emitted = _tensor_state(module)
    assert torch.equal(emitted["qweight"], checkpoint["qweight"])
