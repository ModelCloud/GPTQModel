# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``ret_idx`` must be rebuilt when ``g_idx`` is replaced by a checkpoint load.

``TorchFusedLinear`` and ``TorchAtenLinear`` both derive an activation-reorder
index from ``g_idx``. The cache used to be keyed on ``numel()`` alone, so
``load_state_dict`` copying a different ``g_idx`` (a desc_act checkpoint) into
the same buffer silently reused the previous permutation. These tests drive the
real load path and compare against an independent re-derivation of the index.
"""

from __future__ import annotations

import weakref

import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32


def _module(cls):
    module = cls(
        bits=4, group_size=GROUP_SIZE, sym=True, desc_act=True,
        in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False,
        pack_dtype=torch.int32,
    )
    module.qweight = torch.zeros((IN_FEATURES // 8, OUT_FEATURES), dtype=torch.int32)
    module.qzeros = torch.zeros((IN_FEATURES // GROUP_SIZE, OUT_FEATURES // 8), dtype=torch.int32)
    module.scales = torch.ones((IN_FEATURES // GROUP_SIZE, OUT_FEATURES), dtype=torch.float16)
    return module


def _group_labels(order) -> torch.Tensor:
    return torch.cat([torch.full((GROUP_SIZE,), i, dtype=torch.int32) for i in order])


def _oracle_ret_idx(g_idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent re-derivation of the permutation (flattened group offsets)."""
    total = g_idx.shape[0]
    labels = g_idx.to(torch.int32)
    flat = labels * group_size
    for label in range(int(labels.max()) + 1):
        mask = labels == label
        # Each group label restarts its within-group offset at zero.
        flat[mask] += torch.arange(int(mask.sum()), dtype=torch.int32)
    out = torch.zeros(total, dtype=torch.int32)
    out[flat] = torch.arange(total, dtype=torch.int32)
    return out


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_load_state_dict_with_new_g_idx_rebuilds_ret_idx(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    first = module._build_ret_idx().clone()

    state = {name: value.clone() for name, value in module.state_dict().items()}
    new_g_idx = _group_labels((3, 0, 2, 1))
    state["g_idx"] = new_g_idx
    module.load_state_dict(state)

    assert torch.equal(module.g_idx, new_g_idx)
    rebuilt = module._build_ret_idx()
    expected = _oracle_ret_idx(new_g_idx, GROUP_SIZE)
    assert not torch.equal(rebuilt, first), "the stale permutation was reused"
    assert torch.equal(rebuilt, expected)


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_explicit_clear_weight_cache_drops_ret_idx(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    module._build_ret_idx()

    assert callable(getattr(module, "clear_weight_cache", None))
    module.clear_weight_cache()

    assert module.ret_idx is None
    assert getattr(module, "_ret_idx_identity", None) is None


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_repeated_build_reuses_the_cache(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))

    first = module._build_ret_idx()
    second = module._build_ret_idx()

    assert first is second


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_in_place_g_idx_mutation_is_observed(cls):
    module = _module(cls)
    module.g_idx = _group_labels((0, 1, 2, 3))
    module._build_ret_idx()

    with torch.no_grad():
        module.g_idx.copy_(_group_labels((3, 0, 2, 1)))

    rebuilt = module._build_ret_idx()
    assert torch.equal(rebuilt, _oracle_ret_idx(module.g_idx, GROUP_SIZE))


def test_ret_idx_is_identity_lives_on_the_packable_base():
    assert TorchFusedLinear._ret_idx_is_identity is PackableQuantLinear._ret_idx_is_identity
    assert TorchAtenLinear._ret_idx_is_identity is PackableQuantLinear._ret_idx_is_identity


def test_fused_op_skips_the_gather_for_an_identity_ret_idx():
    """``desc_act=False`` must not pay for ``x[:, arange].contiguous()``.

    The fused op is stubbed so the captured tensor shows whether
    ``_fused_op_forward`` reordered/copied the activation.
    """
    module = _module(TorchFusedLinear)
    module.g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    assert module._ret_idx_is_identity()

    captured = {}

    def _capture(x):
        captured["x"] = x
        return torch.zeros((x.shape[0], OUT_FEATURES), dtype=x.dtype)

    module.torch_fused_op = _capture

    # A contiguous activation is forwarded unchanged: no gather, no copy.
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    module._fused_op_forward(x)
    assert captured["x"] is x

    # A strided activation is still made contiguous for the fused kernel.
    strided = torch.randn(IN_FEATURES, 4, dtype=torch.bfloat16).t()
    assert not strided.is_contiguous()
    module._fused_op_forward(strided)
    assert captured["x"] is not strided
    assert captured["x"].is_contiguous()


def test_fused_op_gathers_for_a_non_identity_ret_idx():
    module = _module(TorchFusedLinear)
    module.g_idx = _group_labels((3, 0, 2, 1))
    assert not module._ret_idx_is_identity()

    captured = {}

    def _capture(x):
        captured["x"] = x
        return torch.zeros((x.shape[0], OUT_FEATURES), dtype=x.dtype)

    module.torch_fused_op = _capture
    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    module._fused_op_forward(x)

    assert captured["x"] is not x
    assert captured["x"].is_contiguous()


def test_xpu_keeps_the_main_branch_gather_for_an_identity_ret_idx():
    """XPU must not take the CPU-only identity fast path.

    The XPU kernel has no coverage for activation views with a non-zero storage
    offset, so it keeps the main-branch unconditional gather even when the
    permutation is the identity.
    """
    module = _module(TorchFusedLinear)
    module.g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    assert module._ret_idx_is_identity()
    assert module._needs_activation_gather("xpu") is True
    assert module._needs_activation_gather("cpu") is False


def test_identity_fast_path_forwards_a_storage_offset_fp16_activation():
    """A contiguous FP16 view with a non-zero storage offset is forwarded as-is.

    The old unconditional ``x[:, arange].contiguous()`` always materialized a
    zero-offset copy, so a fused kernel that quietly required offset-zero
    storage would have been masked. The fast path hands the original view to
    ``torch_fused_op``; pin both the identity of the forwarded tensor and its
    numerical equivalence with the forced-gather path.
    """
    storage = torch.randn(10, IN_FEATURES, dtype=torch.float16)
    activation = storage[2:6]
    assert activation.storage_offset() != 0
    assert activation.is_contiguous()

    # 1) The fast path forwards the same tensor object: no gather, no copy.
    module = _fused_module(TorchFusedLinear, _fused_checkpoint(seed=5, order=(0, 1, 2, 3)))
    assert module._ret_idx_is_identity()
    captured = {}

    def _capture(x):
        captured["x"] = x
        return torch.zeros((x.shape[0], OUT_FEATURES), dtype=x.dtype)

    module.torch_fused_op = _capture
    module._fused_op_forward(activation)

    assert captured["x"] is activation

    # 2) Same weights through the pre-change gather path: bit-exact outputs.
    checkpoint = _fused_checkpoint(seed=5, order=(0, 1, 2, 3))

    fast = _fused_module(TorchFusedLinear, checkpoint)
    with torch.no_grad():
        fast_out = fast(activation).clone()

    gathered = _fused_module(TorchFusedLinear, checkpoint)
    gathered._ret_idx_is_identity = lambda: False  # force the old gather
    with torch.no_grad():
        gathered_out = gathered(activation).clone()

    torch.testing.assert_close(fast_out, gathered_out, rtol=0, atol=0)
    assert fast_out.dtype is torch.float16

    # 3) And the fast path still matches an independent float64 oracle.
    oracle = _dense_oracle(checkpoint, activation)
    torch.testing.assert_close(fast_out.float(), oracle, rtol=2e-3, atol=2e-3)


def test_clear_weight_cache_drops_the_cached_identity_flag():
    module = _module(TorchFusedLinear)
    module.g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    assert module._ret_idx_is_identity()
    assert module._ret_idx_identity is True

    module.clear_weight_cache()

    assert module.ret_idx is None
    assert module._ret_idx_identity is None


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
@pytest.mark.parametrize("mutation", ["in_place", "replace"])
def test_rebuilding_ret_idx_after_an_identity_forward_drops_the_identity_flag(cls, mutation):
    """An explicit rebuild must not leave ``_ret_idx_identity`` stale.

    ``forward`` caches ``True`` for a canonical ``g_idx``. Replacing ``g_idx``
    and calling ``_build_ret_idx()`` refreshes ``_ret_idx_key``; if the cached
    identity flag survives, the next forward skips a required gather and
    silently returns wrong values.

    The control omits the explicit rebuild: there the key mismatch forces
    ``_ret_idx_is_identity()`` to recompute, which is the known-correct path.
    """
    activation = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)

    def _run(explicit_rebuild):
        module = _fused_module(cls, _fused_checkpoint(seed=7, order=(0, 1, 2, 3)))
        with torch.no_grad():
            module(activation)  # identity g_idx: caches the identity flag
            assert module._ret_idx_is_identity()

            new_g_idx = _group_labels((3, 0, 2, 1))
            if mutation == "in_place":
                module.g_idx.copy_(new_g_idx)
            else:
                module.g_idx = new_g_idx
            if explicit_rebuild:
                module._build_ret_idx()
                assert not module._ret_idx_is_identity()
            return module(activation).clone()

    expected = _run(explicit_rebuild=False)
    actual = _run(explicit_rebuild=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Derived fused metadata must follow the packed buffers a load replaces.
# ---------------------------------------------------------------------------


def _fused_checkpoint(seed, order, dtype=torch.float16):
    generator = torch.Generator().manual_seed(seed)
    return {
        "qweight": torch.randint(
            -(2 ** 31), 2 ** 31 - 1, (IN_FEATURES // 8, OUT_FEATURES),
            dtype=torch.int32, generator=generator,
        ),
        # Symmetric GPTQ zero code (8) packed eight times into every word.
        "qzeros": torch.full(
            (IN_FEATURES // GROUP_SIZE, OUT_FEATURES // 8),
            8 * 0x11111111 - 2 ** 32,
            dtype=torch.int32,
        ),
        "scales": torch.rand(
            (IN_FEATURES // GROUP_SIZE, OUT_FEATURES), dtype=dtype,
            generator=generator,
        ) * 0.002 + 0.001,
        "g_idx": _group_labels(order),
    }


def _fused_module(cls, checkpoint):
    module = cls(
        bits=4, group_size=GROUP_SIZE, sym=True, desc_act=True,
        in_features=IN_FEATURES, out_features=OUT_FEATURES, bias=False,
        pack_dtype=torch.int32, enable_wf_unsqueeze=True,
    )
    for name, value in checkpoint.items():
        setattr(module, name, value.clone())
    module.eval()
    module._init_wf_unsqueeze_buffers()
    return module


def _dense_oracle(checkpoint, activation):
    """Independent float64 dequantization of a checkpoint-layout layer."""
    word = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = (
        (checkpoint["qweight"].unsqueeze(1) >> word.view(1, -1, 1)) & 0xF
    ).double().reshape(-1, OUT_FEATURES)
    zeros = (
        (checkpoint["qzeros"].unsqueeze(2) >> word.view(1, 1, -1)) & 0xF
    ).double().reshape(checkpoint["qzeros"].shape[0], -1)
    g_idx = checkpoint["g_idx"].long()
    weight = checkpoint["scales"].double()[g_idx] * (codes - zeros[g_idx])
    return (activation.double() @ weight).float()


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_derived_fused_metadata_follows_loaded_checkpoint(cls):
    """A load must not keep serving scales/zero points the module no longer owns.

    ``load_state_dict`` copies ``scales``/``qzeros`` into the existing buffers
    after the one-time transform built ``scales_and_zeros``. Without a refresh
    the fused op keeps the pre-load derived tensor, so the output silently
    ignores the checkpoint that was just loaded.
    """
    activation = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    checkpoint = _fused_checkpoint(seed=2, order=(3, 0, 2, 1))

    source = _fused_module(cls, checkpoint)
    with torch.no_grad():
        expected = source(activation).clone()

    target = _fused_module(cls, _fused_checkpoint(seed=1, order=(0, 1, 2, 3)))
    with torch.no_grad():
        _ = target(activation)  # build the one-time fused state
    target.load_state_dict(source.state_dict())

    with torch.no_grad():
        actual = target(activation)

    # The loaded checkpoint has a different g_idx, so the activation-reorder
    # index must be rebuilt for it as well as the derived scales/zero points.
    assert not torch.equal(target.g_idx, _group_labels((0, 1, 2, 3)))
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    oracle = _dense_oracle(checkpoint, activation)
    # ``actual`` keeps the activation dtype; compare the values, not the dtype.
    torch.testing.assert_close(
        actual.float(), oracle, rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_cache_clear_keeps_packed_storage_only(cls):
    """Refreshing derived metadata must not create a dense weight cache."""
    module = _fused_module(cls, _fused_checkpoint(seed=5, order=(0, 1, 2, 3)))
    activation = torch.randn(2, IN_FEATURES, dtype=torch.bfloat16)
    with torch.no_grad():
        module(activation)

    packed = module.qweight
    module.clear_weight_cache()

    assert module.qweight is packed
    assert module.qweight.dtype in (torch.uint8, torch.int32)
    # Include unregistered attributes and child modules, and reject expanded
    # integer weights as well as every floating-point dense representation.
    for tensor in state_tensors(module):
        assert tensor.numel() < IN_FEATURES * OUT_FEATURES
    derived = getattr(module, "scales_and_zeros", None)
    if derived is not None:
        assert derived.numel() < IN_FEATURES * OUT_FEATURES


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
@pytest.mark.parametrize("name", ["qweight", "scales", "qzeros"])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_forward_observes_buffer_changes(cls, name, replace, dtype):
    checkpoint = _fused_checkpoint(seed=29, order=(3, 0, 2, 1))
    checkpoint["scales"].fill_(1 / 128)
    module = _fused_module(cls, checkpoint)
    activation = torch.ones((2, IN_FEATURES), dtype=dtype)
    with torch.no_grad():
        module(activation)
        if name == "qweight":
            replacement_checkpoint = _fused_checkpoint(seed=47, order=(3, 0, 2, 1))
            replacement_checkpoint["scales"].fill_(1 / 128)
            other = _fused_module(cls, replacement_checkpoint)
            other(activation)
            value = other.qweight.clone()
            expected = _dense_oracle(replacement_checkpoint, activation)
        elif name == "scales":
            value = module.scales * 2
            checkpoint["scales"].mul_(2)
            expected = _dense_oracle(checkpoint, activation)
        else:
            # After conversion qzeros contains affine offsets, not packed codes.
            value = module.qzeros + 1 / 128
            expected = _dense_oracle(checkpoint, activation) + activation.float().sum(-1, keepdim=True) / 128
        if replace:
            setattr(module, name, value)
        else:
            getattr(module, name).copy_(value)
        for _ in range(2):
            actual = module(activation)
            assert actual.dtype == activation.dtype
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
@pytest.mark.parametrize("operation", ["clear", "move", "load_assign"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_cache_lifecycle_with_inference_tensors(cls, operation, dtype):
    with torch.inference_mode():
        checkpoint = _fused_checkpoint(seed=19, order=(3, 0, 2, 1))
        checkpoint["scales"].fill_(1 / 128)
        module = _fused_module(cls, checkpoint)
        activation = torch.ones((2, IN_FEATURES), dtype=dtype)
        module(activation)
        old_metadata = weakref.ref(module.scales_and_zeros)
        if operation == "load_assign":
            state = {name: value.clone() for name, value in module.state_dict().items()}
            state["scales"].mul_(2)
            module.load_state_dict(state, assign=True)
        else:
            module.scales.mul_(2)
            if operation == "clear":
                module.clear_weight_cache()
            else:
                module.to(dtype=torch.float32)
        checkpoint["scales"].mul_(2)
        assert old_metadata() is None
        expected = _dense_oracle(checkpoint, activation)
        for _ in range(2):
            actual = module(activation)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
            assert all(t.numel() < IN_FEATURES * OUT_FEATURES for t in state_tensors(module))


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
def test_unchanged_fused_forward_reuses_metadata(cls):
    module = _fused_module(cls, _fused_checkpoint(seed=5, order=(0, 1, 2, 3)))
    activation = torch.ones((1, IN_FEATURES), dtype=torch.bfloat16)
    with torch.no_grad():
        module(activation)
        metadata = module.scales_and_zeros
        packed = module.qweight
        for _ in range(2):
            module(activation)
            assert module.scales_and_zeros is metadata
            assert module.qweight is packed


@pytest.mark.parametrize("cls", [TorchFusedLinear, TorchAtenLinear])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operation", ["forward", "clear", "load_assign"])
def test_fused_refresh_preserves_wider_source_metadata(cls, dtype, operation):
    checkpoint = _fused_checkpoint(seed=5, order=(0, 1, 2, 3))
    checkpoint["qweight"].fill_(-2004318072)  # All codes 8.
    checkpoint["qweight"][0].add_(1)  # Code 9 for the first input feature.
    checkpoint["scales"].fill_(1)
    module = _fused_module(cls, checkpoint)
    activation = torch.zeros((2, IN_FEATURES), dtype=dtype)
    activation[:, 0] = 1
    with torch.no_grad():
        module(activation)
        if operation == "load_assign":
            state = {name: value.clone() for name, value in module.state_dict().items()}
            state["scales"] = state["scales"].float().fill_(2049)
            state["qzeros"] = state["qzeros"].float().fill_(-2048)
            module.load_state_dict(state, assign=True)
        else:
            module.scales = module.scales.float().fill_(2049)
            module.qzeros = module.qzeros.float().fill_(-2048)
            if operation == "clear":
                module.clear_weight_cache()
        # The transformed affine representation decodes this active row as
        # (9 - 8) * 2049 - 2048 = 1, exactly in FP64 and both output dtypes.
        expected = torch.ones((2, OUT_FEATURES), dtype=torch.float64)
        actual = module(activation)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0)
