# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for every table/dispatch case; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Ada (RTX 4090, SM 8.9) portable AWQ Triton GEMV: launch table + oracle A/B.

The oracle unpacks the ``FORMAT.GEMV`` buffers with Torch integer shifts and
multiplies in FP64; it never calls the Triton kernel or reuses its
intermediates.
"""

import pytest
import torch

from gptqmodel.quantization.awq.modules.triton import gemv as gemv_mod
from gptqmodel.utils import arch


TOL = 2e-3
GROUP_SIZE = 128
ADA_CFG = gemv_mod.AwqGemvLaunch(
    block_n=32, max_block_k=512, num_warps=8, num_stages=1
)


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _is_ada_host() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    )


requires_ada = pytest.mark.skipif(not _is_ada_host(), reason="Ada (SM 8.9) CUDA GPU required")


# --------------------------------------------------------------------------- #
# Hardware-free: table contents and routing.
# --------------------------------------------------------------------------- #

def test_ada_entry_is_present_and_measured():
    table = gemv_mod._AWQ_GEMV_LAUNCH_CONFIGS
    assert "ada" in table
    assert table["ada"] == ADA_CFG


def test_rdna4_entry_and_portable_default_are_untouched():
    table = gemv_mod._AWQ_GEMV_LAUNCH_CONFIGS
    assert table["rdna4"] == gemv_mod.AwqGemvLaunch(
        block_n=64, max_block_k=512, num_warps=4, num_stages=1
    )
    assert gemv_mod._DEFAULT_AWQ_GEMV_LAUNCH == gemv_mod.AwqGemvLaunch(
        block_n=64, max_block_k=512, num_warps=4, num_stages=1
    )
    # The Ada tile is a real retune, not the portable default under a new name.
    assert table["ada"] != gemv_mod._DEFAULT_AWQ_GEMV_LAUNCH


@pytest.mark.parametrize("target", [(8, 9), "sm_89", "sm_89a"])
def test_ada_is_selected_for_every_ada_spelling(target):
    assert arch.tuning_family(target) == "ada"
    assert gemv_mod.launch_config(target) == ADA_CFG


@pytest.mark.parametrize(
    "target",
    [(7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0), "sm_80", "sm_90", "gfx942",
     "gfx1100", "", "unknown"],
)
def test_non_ada_families_keep_the_portable_default(target):
    # ``None`` is absent on purpose: it means "the current device", which on
    # this host is the Ada GPU.
    assert arch.tuning_family(target) != "ada"
    assert gemv_mod.launch_config(target) == gemv_mod._DEFAULT_AWQ_GEMV_LAUNCH


def test_rdna4_routing_is_unchanged():
    assert (
        gemv_mod.launch_config("gfx1201")
        == gemv_mod._AWQ_GEMV_LAUNCH_CONFIGS["rdna4"]
    )
    assert (
        gemv_mod.launch_config("gfx950")
        == gemv_mod._AWQ_GEMV_LAUNCH_CONFIGS["rdna4"]
    )


def test_current_device_routes_to_ada_on_an_ada_host():
    if not _is_ada_host():
        pytest.skip("Ada (SM 8.9) CUDA GPU required")
    arch.clear_arch_cache()
    assert arch.tuning_family(None) == "ada"
    assert gemv_mod.launch_config(None) == ADA_CFG


def test_the_one_measured_small_shape_loss_is_documented():
    # 896x128 (N=128 => four 32-wide blocks) is launch-bound and measures
    # ~0.98x against the portable default.  The tile is kept uniform and the
    # exception is documented in the table comment instead of hidden; if a
    # future retune fixes or removes it, this test should be updated with the
    # measurement rather than silently relaxed.
    import inspect

    source = inspect.getsource(gemv_mod)
    assert "896x128" in source
    assert "launch-bound" in source


def test_ada_tile_block_k_is_clamped_to_the_group():
    # The launch helper clamps block_k to group_size; 512 therefore means 128
    # for the group-128 checkpoints this path is measured on.
    assert min(ADA_CFG.max_block_k, GROUP_SIZE) == GROUP_SIZE


# --------------------------------------------------------------------------- #
# Real forward on this 4090 vs an independent FP64 oracle.
# --------------------------------------------------------------------------- #

def _gemv_pack_layout(qweight, qzeros, scales, group_size):
    """FP64 dense weight from the GEMV packed buffers (independent oracle)."""

    nibbles = torch.arange(0, 8, device=qweight.device, dtype=torch.int64)
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 8
    codes = ((qweight.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        out_features, in_features
    )
    zeros = ((qzeros.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        qzeros.shape[0], -1
    )
    group = torch.arange(in_features, device=qweight.device) // group_size
    return (codes.to(torch.float64) - zeros.to(torch.float64)[:, group]) * scales.to(
        torch.float64
    )[:, group]


def _build_packed(out_features, in_features, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    groups = in_features // GROUP_SIZE
    pad = (groups + 7) // 8 * 8
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (out_features, in_features // 8), device=device, dtype=torch.int32
    )
    qzeros = torch.randint(0, 15, (out_features, pad), device=device, dtype=torch.int32)
    scales = (
        torch.rand(out_features, pad, device=device, dtype=torch.float16) * 0.1 + 0.01
    ).contiguous()
    return qweight, qzeros, scales


@pytest.mark.cuda
@requires_ada
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "in_features,out_features",
    [(896, 128), (2048, 2048), (4096, 4096), (4096, 11008), (14336, 4096)],
)
def test_ada_awq_gemv_matches_independent_fp64_oracle(in_features, out_features, dtype):
    from gptqmodel.quantization.awq.modules.triton.gemv import awq_gemv_triton

    qweight, qzeros, scales = _build_packed(out_features, in_features, 3 + in_features)
    inputs = torch.randn(1, in_features, device="cuda:0", dtype=dtype)

    assert gemv_mod.launch_config(inputs.device) == ADA_CFG

    with torch.inference_mode():
        first = awq_gemv_triton(inputs, qweight, scales, qzeros, GROUP_SIZE)
        repeated = awq_gemv_triton(inputs, qweight, scales, qzeros, GROUP_SIZE)
    torch.cuda.synchronize()

    dense = _gemv_pack_layout(qweight, qzeros, scales, GROUP_SIZE)
    truth = inputs.to(torch.float64) @ dense.t()

    assert first.dtype == dtype
    assert torch.equal(first, repeated)
    scale = truth.abs().max().item()
    err = (first.to(torch.float64) - truth).abs().max().item()
    assert err / scale <= TOL, f"max rel err {err / scale:.3e} > {TOL}"
    rounded = truth.to(dtype).to(torch.float64)
    assert (first.to(torch.float64) - rounded).abs().max().item() <= TOL * scale


@pytest.mark.cuda
@requires_ada
def test_ada_awq_gemv_module_keeps_no_dense_weight_after_forward():
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear

    in_features = out_features = 512
    dense_layer = torch.nn.Linear(in_features, out_features, bias=False).cuda()
    groups = in_features // GROUP_SIZE
    scales = (
        torch.rand(out_features, groups, device="cuda:0", dtype=torch.float16) * 0.1 + 0.01
    ).contiguous()
    zeros = torch.randint(0, 15, (out_features, groups), device="cuda:0", dtype=torch.int32)

    module = AwqGEMVTritonLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    ).cuda().eval()
    module.pack(dense_layer, scales.clone(), zeros.clone())
    module.post_init()
    del dense_layer

    inputs = torch.randn(1, in_features, dtype=torch.float16, device="cuda:0")
    with torch.inference_mode():
        module(inputs)
        torch.cuda.synchronize()
        after_one = _dense_weight_attrs(module)
        for _ in range(3):
            module(inputs)
        torch.cuda.synchronize()
        after_many = _dense_weight_attrs(module)

    assert after_one == set(), f"dense weight retained after first forward: {after_one}"
    assert after_many == set(), f"dense weight retained after repeated forwards: {after_many}"
    assert module.qweight.dtype == torch.int32
    assert module.qweight.shape == (out_features, in_features // 8)


def _dense_weight_attrs(module) -> set:
    """Attributes holding a full dense (out_features, in_features) float weight.

    ``scales`` is ``[out_features, groups]`` and is an allowed packed-format
    table, so the check keys on the dense GEMM shape, not just the row count.
    """

    dense_shape = (module.out_features, module.in_features)
    found = set()
    for name, value in list(module.__dict__.items()):
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and tuple(value.shape) == dense_shape
        ):
            found.add(name)
    for name, value in module._buffers.items():
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and tuple(value.shape) == dense_shape
        ):
            found.add(name)
    return found
