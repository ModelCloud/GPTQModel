# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A Marlin layer built or loaded without a ``g_idx`` must not invent act-order.

``MarlinLinear`` used to register ``g_idx`` with ``torch.empty``. ``post_init()``
reads that tensor through the #4053 contiguity guard to decide whether a
checkpoint stores a real activation-order index, so uninitialised memory was
decoded as a permutation (or rejected as "non-contiguous" for shapes that cannot
run the activation-order kernel).

The placeholder is now the canonical ``arange(K) // group_size`` mapping: the
identity ordering, the same index the base class stores for ``desc_act=False``,
and a valid shape for ``load_state_dict()``. A checkpoint that omits ``g_idx``
therefore stays on the contiguous fast path, while a checkpoint that really
stores a permutation still switches to the activation-order path.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import marlin as marlin_qlinear_module
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear, _g_idx_is_contiguous


_CUDA_MARLIN = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and marlin_qlinear_module.marlin_import_exception is None
    and marlin_qlinear_module.marlin_runtime_available(torch.float16)
)


def _canonical_g_idx(in_features: int, group_size: int) -> torch.Tensor:
    group = group_size if group_size > 0 else in_features
    return torch.arange(in_features, dtype=torch.int32) // group


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[K, N]`` integer codes into GPTQ int32 words ``[K / pf, N]``.

    GPTQ stores code ``k`` in word ``k // pf`` at bit offset ``bits * (k % pf)``;
    this helper is the test's own packer and does not call the code under test.
    """

    pack_factor = 32 // bits
    in_features, out_features = codes.shape
    viewed = codes.reshape(in_features // pack_factor, pack_factor, out_features).to(torch.int64)
    shifts = (torch.arange(pack_factor, dtype=torch.int64) * bits).view(1, pack_factor, 1)
    return (viewed << shifts).sum(dim=1).to(torch.int32)


def _fp64_reference(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    zero_point: int,
) -> torch.Tensor:
    """Independent FP64 oracle: ``x @ (scales[g_idx] * (codes - zero)).T``."""

    weight = scales.double()[g_idx.to(torch.long)] * (codes.double() - zero_point)
    return x.double() @ weight


def _quantized_state_dict(
    *,
    in_features: int,
    out_features: int,
    bits: int,
    group_size: int,
    seed: int,
    g_idx: torch.Tensor = None,
):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (in_features, out_features), generator=generator, dtype=torch.int64)
    scales = (
        torch.rand(in_features // group_size, out_features, generator=generator, dtype=torch.float32) * 0.02
        + 0.001
    ).half()
    pack_factor = 32 // bits
    state = {
        "qweight": _pack_codes(codes, bits),
        "qzeros": torch.full(
            (in_features // group_size, out_features // pack_factor), 0, dtype=torch.int32
        ),
        "scales": scales,
    }
    if g_idx is not None:
        state["g_idx"] = g_idx
    return state, codes, scales


@pytest.mark.parametrize(
    "in_features, group_size, desc_act",
    [
        (256, 32, False),
        (288, 32, False),
        (128, 128, False),
        (256, -1, False),
        (256, 64, True),
    ],
)
def test_placeholder_g_idx_is_the_canonical_contiguous_index(monkeypatch, in_features, group_size, desc_act):
    # The placeholder is a construction-time invariant, so no kernel is needed.
    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)

    module = MarlinLinear(
        bits=4,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,
        in_features=in_features,
        out_features=64,
        bias=False,
        dtype=torch.float16,
    )

    expected = _canonical_g_idx(in_features, group_size)
    assert module.g_idx.dtype == torch.int32
    assert torch.equal(module.g_idx, expected)
    assert _g_idx_is_contiguous(module.g_idx, in_features, group_size)


def test_post_init_without_checkpoint_g_idx_keeps_the_contiguous_path(monkeypatch):
    """K=288/N=200 is not tile aligned: garbage g_idx used to raise here."""

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_validate_runtime_device", lambda *args, **kwargs: (8, 0))
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits, dtype=None):
        pack_factor = 32 // num_bits
        return torch.zeros(
            (size_k // 16, size_n * 16 // pack_factor),
            dtype=torch.int32,
            device=b_q_weight.device,
        )

    monkeypatch.setattr(marlin_qlinear_module, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_permute_scales",
        lambda scales, size_k, size_n, group_size: scales,
    )

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=288,
        out_features=200,
        bias=False,
        dtype=torch.float16,
    )

    module.post_init()

    assert module.desc_act is False
    assert module.is_k_full is True
    assert module._marlin_tile_padding == (256, 320)
    assert module.g_idx.numel() == 0
    assert module.g_idx_sort_indices.numel() == 0


@pytest.mark.skipif(not _CUDA_MARLIN, reason="requires CUDA Marlin kernels")
def test_checkpoint_without_g_idx_decodes_contiguous_groups():
    bits, in_features, out_features, group_size = 4, 256, 256, 32
    state, codes, scales = _quantized_state_dict(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        seed=1234,
    )

    module = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    ).cuda()
    # A desc_act=False checkpoint that omits g_idx: keep the placeholder.
    module.load_state_dict(state, strict=False)
    module.post_init()

    # Assert the decoded layout before running the kernel: on the buggy build
    # post_init() flips desc_act to True and would decode garbage indices.
    assert module.desc_act is False
    assert module.is_k_full is True
    assert module.g_idx_sort_indices.numel() == 0

    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        actual = module(x).double()
    expected = _fp64_reference(
        x.cpu(),
        codes,
        scales,
        _canonical_g_idx(in_features, group_size).to(torch.long),
        zero_point=8,
    )

    relative = ((actual.cpu() - expected).norm() / expected.norm()).item()
    assert relative < 1e-3, relative
    assert (actual.cpu() - expected).abs().max().item() < 5e-3


@pytest.mark.skipif(not _CUDA_MARLIN, reason="requires CUDA Marlin kernels")
def test_desc_act_module_without_checkpoint_g_idx_starts_from_identity_order():
    """A ``desc_act=True`` shell has no permutation yet: identity, not garbage."""

    bits, in_features, out_features, group_size = 4, 256, 256, 32
    state, codes, scales = _quantized_state_dict(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        seed=11,
    )

    module = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=True,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    ).cuda()
    module.load_state_dict(state, strict=False)
    module.post_init()

    assert module.desc_act is True
    assert module.g_idx.numel() == in_features
    assert module.g_idx_sort_indices.numel() == in_features
    # The canonical index is the identity ordering, so sorting it is a no-op.
    assert torch.equal(module.g_idx.cpu(), _canonical_g_idx(in_features, group_size))

    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        actual = module(x).double()
    expected = _fp64_reference(
        x.cpu(),
        codes,
        scales,
        _canonical_g_idx(in_features, group_size).to(torch.long),
        zero_point=8,
    )

    relative = ((actual.cpu() - expected).norm() / expected.norm()).item()
    assert relative < 1e-3, relative


@pytest.mark.skipif(not _CUDA_MARLIN, reason="requires CUDA Marlin kernels")
def test_noncontiguous_checkpoint_g_idx_still_selects_the_act_order_path():
    """The #4053 behaviour must survive the deterministic placeholder."""

    bits, in_features, out_features, group_size = 4, 256, 256, 32
    groups = in_features // group_size
    canonical = _canonical_g_idx(in_features, group_size).to(torch.long)
    generator = torch.Generator().manual_seed(7)
    permuted_groups = torch.randperm(groups, generator=generator)
    checkpoint_g_idx = permuted_groups[canonical].to(torch.int32)
    assert not torch.all(checkpoint_g_idx[1:] >= checkpoint_g_idx[:-1])

    state, codes, scales = _quantized_state_dict(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        group_size=group_size,
        seed=7,
        g_idx=checkpoint_g_idx,
    )

    module = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    ).cuda()
    module.load_state_dict(state, strict=False)
    module.post_init()

    assert module.desc_act is True
    assert module.g_idx.numel() == in_features

    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        actual = module(x).double()
    expected = _fp64_reference(x.cpu(), codes, scales, checkpoint_g_idx.to(torch.long), zero_point=8)

    relative = ((actual.cpu() - expected).norm() / expected.norm()).item()
    assert relative < 1e-3, relative
    assert (actual.cpu() - expected).abs().max().item() < 5e-3
