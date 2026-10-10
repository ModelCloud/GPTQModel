# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU/GPU coverage for the stable `hf_` packing/repacking APIs.

These entry points are the external contract used by transformers/optimum,
auto-round and other integrators, so the tests assert:

1. feasibility probing accepts valid contracts and rejects unsupported ones
2. best-format probing returns a feasible format for the target device
3. layer packing matches the internal `pack_original`/`pack_block` paths,
   including synthesized symmetric zero points and asymmetric zero points
4. repacking copies packed state between kernels that share a layout and
   rejects unsupported targets
5. GPU packing/repacking (`pack_impl="gpu"`, Marlin, ExllamaV2) matches the
   CPU reference path when CUDA is available
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.packing import (
    hf_check_best_packing_format,
    hf_check_packing_feasibility,
    hf_pack_layer,
    hf_post_init,
    hf_repack_layer,
)
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
    convert_gptq_v2_to_v1_format_module,
)


pytestmark = [pytest.mark.cpu]


def _make_inputs(bits: int, in_features: int = 128, out_features: int = 64, group_size: int = 32):
    torch.manual_seed(1000 + bits)
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)

    groups = in_features // group_size
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << bits, (out_features, groups), dtype=torch.int32)
    return linear, scales, zeros, g_idx


def _make_gptq_module(
    bits: int = 4,
    group_size: int = 32,
    sym: bool = False,
    in_features: int = 128,
    out_features: int = 64,
) -> TorchLinear:
    return TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
    )


def _make_awq_module(
    bits: int = 4,
    group_size: int = 32,
    sym: bool = False,
    in_features: int = 128,
    out_features: int = 64,
) -> AwqTorchLinear:
    return AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.AWQ_TORCH,
        bias=False,
        register_buffers=True,
    )


def _packed_tensors(module: BaseQuantLinear):
    return {
        key: getattr(module, key)
        for key in ("qweight", "qzeros", "scales", "g_idx")
        if getattr(module, key, None) is not None
    }


def _assert_packed_equal(left: BaseQuantLinear, right: BaseQuantLinear):
    left_state = _packed_tensors(left)
    right_state = _packed_tensors(right)
    assert left_state.keys() == right_state.keys()
    for key, value in left_state.items():
        assert torch.equal(value, right_state[key]), f"packed buffer `{key}` differs"


def _torch_aten_available() -> bool:
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


def test_feasibility_accepts_supported_cpu_kernel():
    assert hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_TORCH, device=DEVICE.CPU
    )


def test_feasibility_rejects_cuda_only_kernel_on_cpu():
    assert not hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_MARLIN, device=DEVICE.CPU
    )
    # Marlin is a repack target, but not on CPU.
    assert not hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_MARLIN, device=DEVICE.CPU, repack=True
    )


def test_feasibility_repack_targets_include_plain_gptq_kernels():
    assert hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_TORCH, device=DEVICE.CPU, repack=True
    )


def test_feasibility_rejects_unsupported_group_size():
    assert not hf_check_packing_feasibility(
        4, 7, False, True, backend=BACKEND.GPTQ_TORCH, device=DEVICE.CPU
    )


def test_feasibility_checks_layer_shape_when_provided():
    # AwqTorchLinear requires out_features % 8 == 0.
    assert not hf_check_packing_feasibility(
        4,
        32,
        False,
        True,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        backend=BACKEND.AWQ_TORCH,
        device=DEVICE.CPU,
        out_features=60,
    )
    assert hf_check_packing_feasibility(
        4,
        32,
        False,
        True,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        backend=BACKEND.AWQ_TORCH,
        device=DEVICE.CPU,
        out_features=64,
    )


def test_feasibility_raises_for_unknown_backend():
    # An unknown backend string is a caller error, not an infeasible contract.
    with pytest.raises(ValueError):
        hf_check_packing_feasibility(
            4, 32, False, True, backend="not_a_real_backend", device=DEVICE.CPU
        )


def test_best_format_is_feasible_on_cpu():
    best = hf_check_best_packing_format(4, 32, False, True, device=DEVICE.CPU)
    assert best == FORMAT.GPTQ
    assert hf_check_packing_feasibility(
        4, 32, False, True, format=best, device=DEVICE.CPU
    )


def test_best_format_respects_candidate_restriction():
    best = hf_check_best_packing_format(
        4, 32, False, True, device=DEVICE.CPU, formats=[FORMAT.GPTQ_P]
    )
    assert best == FORMAT.GPTQ_P


def test_best_format_for_awq_returns_gemm():
    best = hf_check_best_packing_format(
        4, 32, False, False, quant_method=METHOD.AWQ, device=DEVICE.CPU
    )
    assert best == FORMAT.GEMM


def test_best_format_raises_when_nothing_is_feasible():
    # No kernel supports 1-bit packing.
    with pytest.raises(ValueError):
        hf_check_best_packing_format(1, 32, False, True, device=DEVICE.CPU)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_hf_pack_layer_matches_pack_original(bits):
    linear, scales, zeros, g_idx = _make_inputs(bits)

    expected = _make_gptq_module(bits=bits)
    expected.pack_original(linear, scales, zeros, g_idx)

    actual = _make_gptq_module(bits=bits)
    returned = hf_pack_layer(actual, linear, scales, zeros, g_idx)

    assert returned is actual
    _assert_packed_equal(expected, actual)


def test_hf_pack_layer_synthesizes_symmetric_zeros():
    linear, scales, zeros, g_idx = _make_inputs(4)

    explicit = _make_gptq_module(sym=True)
    explicit.pack_original(linear, scales, torch.full_like(zeros, 1 << 3), g_idx)

    module = _make_gptq_module(sym=True)
    hf_pack_layer(module, linear, scales, None, g_idx)

    _assert_packed_equal(explicit, module)


def test_hf_pack_layer_requires_zeros_for_asymmetric():
    linear, scales, _, g_idx = _make_inputs(4)
    module = _make_gptq_module(sym=False)

    with pytest.raises(ValueError):
        hf_pack_layer(module, linear, scales, None, g_idx)


def test_hf_pack_layer_rejects_non_quant_linear():
    linear, scales, zeros, g_idx = _make_inputs(4)

    with pytest.raises(TypeError):
        hf_pack_layer(linear, linear, scales, zeros, g_idx)


def test_hf_pack_layer_block_impl_matches_original():
    linear, scales, zeros, g_idx = _make_inputs(4)

    original = _make_gptq_module()
    hf_pack_layer(original, linear, scales, zeros, g_idx, pack_impl="original")

    block = _make_gptq_module()
    hf_pack_layer(block, linear, scales, zeros, g_idx, pack_impl="block")

    _assert_packed_equal(original, block)


def test_hf_pack_layer_gpu_impl_falls_back_on_cpu():
    if torch.cuda.is_available():
        pytest.skip("CPU fallback path is only exercised without CUDA")

    linear, scales, zeros, g_idx = _make_inputs(4)

    original = _make_gptq_module()
    hf_pack_layer(original, linear, scales, zeros, g_idx)

    fallback = _make_gptq_module()
    hf_pack_layer(fallback, linear, scales, zeros, g_idx, pack_impl="gpu")

    _assert_packed_equal(original, fallback)


def test_hf_pack_layer_runs_post_init_when_requested(monkeypatch):
    linear, scales, zeros, g_idx = _make_inputs(4)
    module = _make_gptq_module()

    calls = []
    monkeypatch.setattr(TorchLinear, "post_init", lambda self: calls.append(self))

    hf_pack_layer(module, linear, scales, zeros, g_idx, post_init=True)

    assert calls == [module]


def test_hf_pack_layer_dispatches_awq_kernels():
    linear, scales, zeros, g_idx = _make_inputs(4)

    expected = _make_awq_module()
    expected.pack(linear, scales, zeros, g_idx)

    actual = _make_awq_module()
    hf_pack_layer(actual, linear, scales, zeros, g_idx)

    _assert_packed_equal(expected, actual)


def test_hf_pack_layer_awq_synthesizes_symmetric_zeros():
    linear, scales, zeros, g_idx = _make_inputs(4)

    explicit = _make_awq_module(sym=True)
    explicit.pack(linear, scales, torch.full_like(zeros, 1 << 3), g_idx)

    module = _make_awq_module(sym=True)
    hf_pack_layer(module, linear, scales, None, g_idx)

    _assert_packed_equal(explicit, module)


def test_hf_pack_method_matches_module_level_api():
    linear, scales, zeros, g_idx = _make_inputs(4)

    method_module = _make_gptq_module()
    returned = method_module.hf_pack(linear, scales, zeros, g_idx)

    function_module = _make_gptq_module()
    hf_pack_layer(function_module, linear, scales, zeros, g_idx)

    assert returned is method_module
    _assert_packed_equal(method_module, function_module)


def test_hf_post_init_initializes_single_layer(monkeypatch):
    module = _make_gptq_module()
    calls = []
    monkeypatch.setattr(TorchLinear, "post_init", lambda self: calls.append(self))

    assert hf_post_init(module) is module
    assert calls == [module]


def test_hf_post_init_initializes_container_children(monkeypatch):
    container = nn.Module()
    container.first = _make_gptq_module()
    container.second = _make_gptq_module()
    calls = []
    monkeypatch.setattr(TorchLinear, "post_init", lambda self: calls.append(self))

    assert hf_post_init(container) is container
    assert calls == [container.first, container.second]


def test_hf_post_init_rejects_non_module():
    with pytest.raises(TypeError):
        hf_post_init("not-a-module")


def test_hf_repack_layer_same_backend_is_noop():
    linear, scales, zeros, g_idx = _make_inputs(4)
    module = _make_gptq_module()
    hf_pack_layer(module, linear, scales, zeros, g_idx)

    assert hf_repack_layer(module, BACKEND.GPTQ_TORCH) is module


def test_hf_repack_layer_between_compatible_kernels():
    if not _torch_aten_available():
        pytest.skip("torch aten int4 CPU ops are unavailable in this build")

    linear, scales, zeros, g_idx = _make_inputs(4)
    source = _make_gptq_module()
    hf_pack_layer(source, linear, scales, zeros, g_idx, checkpoint_format=FORMAT.GPTQ_V2)
    # Model packing commonly exports a v1 GPTQ checkpoint after packing.
    convert_gptq_v2_to_v1_format_module(source, QuantizeConfig(bits=4))

    target = hf_repack_layer(source, BACKEND.GPTQ_TORCH_ATEN)

    assert isinstance(target, TorchAtenLinear)
    for key in ("qweight", "scales", "g_idx"):
        assert torch.equal(getattr(source, key), getattr(target, key))
    assert target.qzero_format() == 2

    # Compare inference with an independently packed v2 reference, rather
    # than with the v1 checkpoint source itself.
    reference = _make_gptq_module()
    hf_pack_layer(
        reference,
        linear,
        scales,
        zeros,
        g_idx,
        checkpoint_format=FORMAT.GPTQ_V2,
        post_init=True,
    )

    x = torch.randn(3, 128)
    with torch.no_grad():
        torch.testing.assert_close(target(x), reference(x))


def test_hf_repack_layer_rejects_unsupported_target():
    linear, scales, zeros, g_idx = _make_inputs(4)
    module = _make_gptq_module()
    hf_pack_layer(module, linear, scales, zeros, g_idx)

    # Marlin requires CUDA, so the CPU-only target must be rejected.
    with pytest.raises(ValueError):
        hf_repack_layer(module, BACKEND.GPTQ_MARLIN, device=DEVICE.CPU)


def test_hf_repack_layer_rejects_non_quant_linear():
    linear, _, _, _ = _make_inputs(4)

    with pytest.raises(TypeError):
        hf_repack_layer(linear, BACKEND.GPTQ_TORCH)


def test_hf_repack_layer_moves_target_to_requested_device():
    if not _torch_aten_available():
        pytest.skip("torch aten int4 CPU ops are unavailable in this build")

    linear, scales, zeros, g_idx = _make_inputs(4)
    source = _make_gptq_module()
    hf_pack_layer(source, linear, scales, zeros, g_idx)

    target = hf_repack_layer(source, BACKEND.GPTQ_TORCH_ATEN, device="cpu")

    assert target.qweight.device.type == "cpu"


def test_hf_repack_layer_requires_packed_source():
    if not _torch_aten_available():
        pytest.skip("torch aten int4 CPU ops are unavailable in this build")

    unpacked = TorchLinear(
        bits=4,
        group_size=32,
        sym=False,
        desc_act=False,
        in_features=128,
        out_features=64,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=False,
    )

    with pytest.raises(ValueError):
        hf_repack_layer(unpacked, BACKEND.GPTQ_TORCH_ATEN)


def test_hf_repack_layer_rejects_abi_transforming_source(monkeypatch):
    linear, scales, zeros, g_idx = _make_inputs(4)
    source = _make_gptq_module()
    hf_pack_layer(source, linear, scales, zeros, g_idx)

    # Marlin-style kernels repack their buffers into a private runtime ABI in
    # post_init(); copying those buffers out would be silently wrong.
    monkeypatch.setattr(TorchLinear, "QUANT_TYPE", "marlin")

    with pytest.raises(NotImplementedError):
        hf_repack_layer(source, BACKEND.GPTQ_TORCH_ATEN)


def test_hf_repack_layer_aligns_v1_v2_qzero_flavor(monkeypatch):
    if not _torch_aten_available():
        pytest.skip("torch aten int4 CPU ops are unavailable in this build")

    linear, scales, zeros, g_idx = _make_inputs(4)
    source = _make_gptq_module()
    hf_pack_layer(source, linear, scales, zeros, g_idx)
    convert_gptq_v1_to_v2_format_module(source, bits=4, pack_dtype=torch.int32)

    # Simulate a v1-native target (like Marlin): the copied v2 qzeros must be
    # converted back to the checkpoint v1 domain.
    monkeypatch.setattr(TorchAtenLinear, "REQUIRES_FORMAT_V2", False)
    target = hf_repack_layer(source, BACKEND.GPTQ_TORCH_ATEN)

    expected = source.qzeros - 0b00010001000100010001000100010001
    assert torch.equal(target.qzeros, expected)
    assert target.qzero_format() == 1


def test_hf_pack_layer_honors_checkpoint_format():
    linear, scales, zeros, g_idx = _make_inputs(4)
    module = _make_gptq_module()

    hf_pack_layer(
        module,
        linear,
        scales,
        zeros,
        g_idx,
        checkpoint_format=FORMAT.GPTQ_V2,
    )

    assert module.qzero_format() == 2


def test_hf_pack_layer_reports_missing_scales():
    module = _make_gptq_module(sym=True)
    with pytest.raises(ValueError, match="scales.*required"):
        hf_pack_layer(module, nn.Linear(128, 64, bias=False), None, None, None)


# ---------------------------------------------------------------------------
# CUDA coverage
# ---------------------------------------------------------------------------


def _cuda_device(index: int = 0) -> str:
    return f"cuda:{index}"


@pytest.mark.cuda
def test_gpu_feasibility_and_best_format():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU feasibility checks")

    device = _cuda_device()
    # Marlin has no raw-weight packer; it is a repack target that consumes
    # GPTQ-packed tensors.
    assert not hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_MARLIN, device=device
    )
    assert hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_MARLIN, device=device, repack=True
    )
    assert not hf_check_packing_feasibility(
        4, 32, False, False, backend=BACKEND.GPTQ_MARLIN, device=device, repack=True
    )

    best = hf_check_best_packing_format(4, 32, False, True, device=device)
    assert hf_check_packing_feasibility(
        4, 32, False, True, format=best, device=device
    )


@pytest.mark.cuda
def test_hf_pack_layer_gpu_impl_matches_original():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU packing")

    linear, scales, zeros, g_idx = _make_inputs(4)

    original = _make_gptq_module()
    hf_pack_layer(original, linear, scales, zeros, g_idx, pack_impl="original")

    gpu = _make_gptq_module()
    hf_pack_layer(gpu, linear, scales, zeros, g_idx, pack_impl="gpu", device=_cuda_device())

    # GPU packing is a compute accelerator only: like the CPU packers, the
    # packed buffers are registered on CPU for later device placement.
    assert gpu.qweight.device.type == "cpu"
    _assert_packed_equal(original, gpu)


@pytest.mark.cuda
def test_hf_repack_layer_to_marlin_matches_source():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Marlin repack target")

    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    if not marlin_runtime_available(torch.float16):
        pytest.skip(f"Marlin runtime unavailable: {marlin_runtime_error(torch.float16)}")

    device = _cuda_device()
    linear, scales, _, g_idx = _make_inputs(4, group_size=128)

    # Marlin supports symmetric 4-bit weights only.
    source = _make_gptq_module(group_size=128, sym=True)
    hf_pack_layer(source, linear, scales, None, g_idx)
    source.to(device)

    target = hf_repack_layer(source, BACKEND.GPTQ_MARLIN, device=device)

    assert isinstance(target, MarlinLinear)
    assert target.qweight.device.type == "cuda"

    x = torch.randn(4, 128, dtype=torch.float16, device=device)
    with torch.no_grad():
        torch.testing.assert_close(target(x), source(x), atol=1e-2, rtol=1e-2)


@pytest.mark.cuda
def test_hf_repack_layer_to_exllama_v2_with_scratch_space():
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required for the non-zero ordinal check")

    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from gptqmodel.utils.exllamav2 import (
        ScratchSpace,
        exllamav2_gptq_runtime_available,
        exllamav2_gptq_runtime_error,
    )

    if not exllamav2_gptq_runtime_available():
        pytest.skip(f"ExllamaV2 runtime unavailable: {exllamav2_gptq_runtime_error()}")

    device = _cuda_device(1)
    linear, scales, zeros, g_idx = _make_inputs(4, group_size=32)

    source = _make_gptq_module(group_size=32)
    hf_pack_layer(
        source,
        linear,
        scales,
        zeros,
        g_idx,
        checkpoint_format=FORMAT.GPTQ_V2,
    )
    convert_gptq_v2_to_v1_format_module(source, QuantizeConfig(bits=4))
    source.to(device)

    # ExllamaV2 needs a scratch space sized from the target module, so the
    # caller builds it and post-inits through the stable API.
    target = hf_repack_layer(
        source, BACKEND.GPTQ_EXLLAMA_V2, device=device, post_init=False
    )
    assert isinstance(target, ExllamaV2Linear)
    assert target.qweight.device.type == "cuda"

    hf_post_init(
        target,
        scratch_space=ScratchSpace(scratch_bytes=target.temp_dq_size(), dev=device),
    )

    reference = _make_gptq_module(group_size=32)
    hf_pack_layer(
        reference,
        linear,
        scales,
        zeros,
        g_idx,
        checkpoint_format=FORMAT.GPTQ_V2,
    )
    reference.to(device)

    x = torch.randn(4, 128, dtype=torch.float16, device=device)
    with torch.no_grad():
        torch.testing.assert_close(target(x), reference(x), atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
def test_hf_packing_api_accepts_device_enum_and_ordinal():
    if torch.cuda.device_count() < 3:
        pytest.skip("Three CUDA devices are required for the ordinal check")

    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    if not marlin_runtime_available(torch.float16):
        pytest.skip(f"Marlin runtime unavailable: {marlin_runtime_error(torch.float16)}")

    # DEVICE.CUDA covers every visible ordinal.
    assert hf_check_packing_feasibility(
        4, 32, False, True, backend=BACKEND.GPTQ_MARLIN, device=DEVICE.CUDA, repack=True
    )

    linear, scales, _, g_idx = _make_inputs(4, group_size=32)
    source = _make_gptq_module(group_size=32, sym=True)
    hf_pack_layer(source, linear, scales, None, g_idx)

    # Public integer device values are accelerator ordinals.
    target = hf_repack_layer(source, BACKEND.GPTQ_MARLIN, device=2)

    assert isinstance(target, MarlinLinear)
    assert target.qweight.device == torch.device("cuda:2")
