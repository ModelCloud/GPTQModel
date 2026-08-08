# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Accuracy, edge-case, and free-threaded tests for the fused Triton GPTQ block kernel."""

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import gptqmodel
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig
from gptqmodel.utils.python import has_gil_disabled


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@dataclass(frozen=True)
class RawCase:
    rows: int
    count: int
    group_size: int
    bits: int
    weight_kind: str
    hessian_kind: str
    groupwise: bool = False


def _serial_block(W1, Q1, Err1, Hinv1, scale, zero, maxq, group_size, groupwise=False):
    """Independent eager reference for one grouped-GPTQ block."""

    for i in range(W1.shape[1]):
        w = W1[:, i]
        q_scale = scale[:, i // group_size]
        q_zero = zero[:, i // group_size]
        if groupwise:
            q = q_scale * torch.clamp(torch.round(w / q_scale), -maxq, maxq)
        else:
            q = q_scale * (torch.clamp(torch.round(w / q_scale) + q_zero, 0, maxq) - q_zero)
        err = (w - q) / Hinv1[i, i]
        Q1[:, i] = q
        Err1[:, i] = err
        W1[:, i:] = torch.addr(W1[:, i:], err, Hinv1[i, i:], alpha=-1.0)


def _case_inputs(case: RawCase, seed: int = 0, *, strided_operands: bool = False):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = case.count // case.group_size

    scale = torch.rand(case.rows, groups, generator=generator, device=device, dtype=torch.float32) + 0.125
    zero = torch.randint(
        0,
        2**case.bits,
        (case.rows, groups),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).float()

    if case.weight_kind == "gaussian":
        weights = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    elif case.weight_kind == "zeros":
        weights = torch.zeros(case.rows, case.count, device=device, dtype=torch.float32)
    elif case.weight_kind == "constants":
        constants = torch.linspace(-3.5, 3.5, case.rows, device=device, dtype=torch.float32)
        weights = constants[:, None].expand(-1, case.count).clone()
    elif case.weight_kind == "rounding_ties":
        # Powers-of-two scales keep exact half-bin ratios representable. A diagonal
        # Hessian prevents previous columns from perturbing those boundaries.
        scale = torch.full((case.rows, groups), 0.25, device=device, dtype=torch.float32)
        zero = torch.full(
            (case.rows, groups),
            2 ** (case.bits - 1),
            device=device,
            dtype=torch.float32,
        )
        ratios = torch.tensor(
            [-9.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 7.5, 8.5, 9.5],
            device=device,
            dtype=torch.float32,
        )
        weights = ratios.repeat((case.rows * case.count + ratios.numel() - 1) // ratios.numel())
        weights = (weights[: case.rows * case.count] * 0.25).reshape(case.rows, case.count)
    elif case.weight_kind == "outliers":
        weights = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        weights[:, ::11] *= 1_000.0
        weights[:, 1::17] *= 1e-3
    else:
        raise AssertionError(f"Unknown weight kind: {case.weight_kind}")

    if case.hessian_kind == "identity":
        hessian_inverse = torch.eye(case.count, device=device, dtype=torch.float32)
    elif case.hessian_kind == "diagonal":
        diagonal = torch.logspace(-1, 1, case.count, device=device, dtype=torch.float32)
        hessian_inverse = torch.diag(diagonal)
    elif case.hessian_kind == "correlated":
        factor = torch.randn(
            case.count,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        covariance = factor @ factor.T + 0.5 * torch.eye(case.count, device=device)
        hessian_inverse = torch.linalg.cholesky(torch.linalg.inv(covariance), upper=True)
    else:
        raise AssertionError(f"Unknown Hessian kind: {case.hessian_kind}")

    if strided_operands:
        hessian_storage = torch.empty(case.count, case.count * 2, device=device, dtype=torch.float32)
        hessian_storage[:, ::2] = hessian_inverse
        hessian_inverse = hessian_storage[:, ::2]
        scale_storage = torch.empty(case.rows, groups * 2, device=device, dtype=torch.float32)
        scale_storage[:, ::2] = scale
        scale = scale_storage[:, ::2]
        zero_storage = torch.empty(case.rows, groups * 2, device=device, dtype=torch.float32)
        zero_storage[:, ::2] = zero
        zero = zero_storage[:, ::2]
        assert not hessian_inverse.is_contiguous()
        assert not scale.is_contiguous()
        assert not zero.is_contiguous()

    return weights, hessian_inverse, scale, zero


def _logical_codes(quantized, scale, zero, group_size, *, groupwise):
    group_index = torch.arange(quantized.shape[1], device=quantized.device) // group_size
    expanded_scale = scale[:, group_index]
    if groupwise:
        return torch.round(quantized / expanded_scale).to(torch.int32)
    expanded_zero = zero[:, group_index]
    return torch.round(quantized / expanded_scale + expanded_zero).to(torch.int32)


def _run_raw_case(
    case: RawCase,
    seed: int = 0,
    *,
    strided_operands: bool = False,
    before_synchronize=None,
):
    import gptqmodel.quantization._gptq_block_triton as block_module

    if block_module.gptq_block_triton is None:
        pytest.skip("Triton GPTQ block kernel is unavailable")

    weights, hessian_inverse, scale, zero = _case_inputs(case, seed, strided_operands=strided_operands)
    maxq = 2 ** (case.bits - 1) - 1 if case.groupwise else 2**case.bits - 1

    reference_weights = weights.clone()
    reference_quantized = torch.empty_like(reference_weights)
    reference_errors = torch.empty_like(reference_weights)
    _serial_block(
        reference_weights,
        reference_quantized,
        reference_errors,
        hessian_inverse.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        maxq,
        case.group_size,
        case.groupwise,
    )

    actual_weights = weights.clone()
    actual_quantized = torch.empty_like(actual_weights)
    actual_errors = torch.empty_like(actual_weights)
    block_module.gptq_block_triton(
        actual_weights,
        actual_quantized,
        actual_errors,
        hessian_inverse,
        scale,
        zero,
        maxq,
        case.group_size,
        groupwise=case.groupwise,
    )
    if before_synchronize is not None:
        before_synchronize()
    torch.cuda.synchronize(weights.device)

    assert torch.isfinite(actual_quantized).all()
    assert torch.isfinite(actual_errors).all()
    torch.testing.assert_close(actual_quantized, reference_quantized, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(actual_errors, reference_errors, atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(
        _logical_codes(actual_quantized, scale, zero, case.group_size, groupwise=case.groupwise),
        _logical_codes(reference_quantized, scale, zero, case.group_size, groupwise=case.groupwise),
        atol=0,
        rtol=0,
    )
    return actual_quantized, actual_errors


def _run_gptq_quantize(group_size, use_triton, *, bits=4, sym=False, dtype=torch.float16):
    """Run one production GPTQ layer and return its quantization outputs."""
    env_key = "GPTQMODEL_TRITON_BLOCK"
    previous = os.environ.get(env_key)
    os.environ[env_key] = "1" if use_triton else "0"
    try:
        import gptqmodel.quantization.gptq as gptq_module

        importlib.reload(gptq_module)
        torch.manual_seed(42)
        device = "cuda:0"
        layer = nn.Linear(512, 512, bias=False, dtype=dtype, device=device)
        dense_weight = layer.weight.detach().float().clone()
        qcfg = QuantizeConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=False,
            act_group_aware=False,
            offload_to_disk=False,
            mse=2.0,
            scale_search=ScaleSearchConfig.ACTIVATION,
        )
        quantizer = gptq_module.GPTQ(layer, qcfg=qcfg)
        quantizer.quantizer.configure(perchannel=True)
        calibration = torch.randn(8, 512, dtype=dtype, device=device)
        quantizer.add_batch(calibration, None)
        return dense_weight, quantizer.quantize(blocksize=128)
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _output_metrics(actual, reference):
    delta = (actual - reference).float()
    reference_fp32 = reference.float()
    return {
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_fp32)).item(),
        "max_abs": delta.abs().max().item(),
    }


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            RawCase(1, 32, 32, 2, "gaussian", "correlated"),
            id="w2-single-row-correlated",
        ),
        pytest.param(RawCase(7, 64, 64, 3, "zeros", "identity"), id="w3-zeros"),
        pytest.param(RawCase(33, 64, 32, 4, "constants", "diagonal"), id="w4-constants"),
        pytest.param(RawCase(65, 128, 64, 4, "rounding_ties", "identity"), id="w4-rounding-ties"),
        pytest.param(RawCase(96, 128, 128, 8, "outliers", "diagonal"), id="w8-outliers"),
        pytest.param(RawCase(17, 64, 32, 4, "constants", "diagonal", True), id="signed-groupwise"),
    ],
)
def test_gptq_triton_block_matches_eager_reference_across_semantic_cases(case):
    """Bits, shapes, ties, saturation, Hessian spectra, and both formulas match eager math."""
    _run_raw_case(case, seed=17)


@pytest.mark.parametrize(
    ("bits", "group_size", "sym", "dtype"),
    [
        pytest.param(2, 32, False, torch.float16, id="w2-g32-asym-fp16"),
        pytest.param(3, 64, False, torch.bfloat16, id="w3-g64-asym-bf16"),
        pytest.param(4, 64, True, torch.float16, id="w4-g64-sym-fp16"),
        pytest.param(4, 128, False, torch.bfloat16, id="w4-g128-asym-bf16"),
        pytest.param(8, 128, True, torch.bfloat16, id="w8-g128-sym-bf16"),
    ],
)
def test_gptq_triton_block_matches_production_quantize_and_held_out_outputs(bits, group_size, sym, dtype):
    """The fused production path preserves weights, codes, loss, and independent held-out outputs."""
    dense_weight, eager = _run_gptq_quantize(group_size, False, bits=bits, sym=sym, dtype=dtype)
    dense_weight_fused, fused = _run_gptq_quantize(group_size, True, bits=bits, sym=sym, dtype=dtype)
    eager_weight, eager_scale, eager_zero, eager_g_idx, _, eager_loss, _, _ = eager
    fused_weight, fused_scale, fused_zero, fused_g_idx, _, fused_loss, _, _ = fused

    torch.testing.assert_close(dense_weight, dense_weight_fused, atol=0, rtol=0)
    torch.testing.assert_close(eager_scale, fused_scale, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(eager_zero, fused_zero, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(eager_g_idx, fused_g_idx, atol=0, rtol=0)
    torch.testing.assert_close(eager_weight, fused_weight, atol=5e-2, rtol=1e-2)
    assert isinstance(eager_loss, float) and isinstance(fused_loss, float)
    assert abs(eager_loss - fused_loss) < 1e-6

    held_out_generator = torch.Generator(device="cuda:0").manual_seed(1_000_003)
    held_out = torch.randn(16, 512, generator=held_out_generator, device="cuda:0", dtype=torch.float32)
    dense_output = F.linear(held_out, dense_weight)
    eager_output = F.linear(held_out, eager_weight.float())
    fused_output = F.linear(held_out, fused_weight.float())
    parity = _output_metrics(fused_output, eager_output)
    eager_dense_error = _output_metrics(eager_output, dense_output)
    fused_dense_error = _output_metrics(fused_output, dense_output)

    assert all(torch.isfinite(output).all() for output in (dense_output, eager_output, fused_output))
    assert parity["mae"] < 2e-4
    assert parity["rmse"] < 5e-4
    # BF16 source weights have a wider representable spacing than FP16. Keep
    # implementation parity far below one source ULP while separately requiring
    # that dense-reference quality does not regress.
    assert parity["relative_l2"] < max(1e-4, torch.finfo(dtype).eps / 32)
    assert parity["max_abs"] < 2e-2
    for metric in ("mae", "rmse", "relative_l2", "max_abs"):
        numerical_slack = max(2e-6, eager_dense_error[metric] * 5e-4)
        assert fused_dense_error[metric] <= eager_dense_error[metric] + numerical_slack

    eager_log_probs = F.log_softmax(eager_output, dim=-1)
    fused_log_probs = F.log_softmax(fused_output, dim=-1)
    kl_divergence = F.kl_div(fused_log_probs, eager_log_probs, log_target=True, reduction="batchmean")
    assert abs(kl_divergence.item()) < 1e-6
    assert torch.equal(eager_output.argmax(dim=-1), fused_output.argmax(dim=-1))


def test_gptq_triton_block_repeats_exactly_across_seeds():
    """Ten same-seed launches for each of three seeds must have zero output spread."""
    case = RawCase(32, 128, 64, 4, "gaussian", "correlated")
    for seed in (7, 101, 65_537):
        first_quantized = None
        first_errors = None
        for _ in range(10):
            quantized, errors = _run_raw_case(case, seed)
            if first_quantized is None:
                first_quantized = quantized
                first_errors = errors
            else:
                torch.testing.assert_close(quantized, first_quantized, atol=0, rtol=0)
                torch.testing.assert_close(errors, first_errors, atol=0, rtol=0)


def test_gptq_triton_block_records_prepared_operands_on_non_default_stream():
    """Strided wrapper-local copies stay live under non-default-stream allocator pressure."""
    import gptqmodel.quantization._gptq_block_triton as block_module

    case = RawCase(96, 128, 64, 4, "gaussian", "correlated")
    recorded_operands = []
    original_record_stream = torch.Tensor.record_stream

    class TrackingLock:
        def __init__(self):
            self.enter_count = 0

        def __enter__(self):
            self.enter_count += 1

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    launch_lock = TrackingLock()

    def capture_record_stream(tensor, stream):
        recorded_operands.append((tensor.is_contiguous(), stream))
        return original_record_stream(tensor, stream)

    pressure = []

    def allocate_on_default_stream():
        default_stream = torch.cuda.default_stream("cuda:0")
        with torch.cuda.stream(default_stream):
            pressure.extend(torch.empty(128, 128, device="cuda:0", dtype=torch.float32) for _ in range(32))

    stream = torch.cuda.Stream(device="cuda:0")
    with (
        torch.cuda.stream(stream),
        patch.object(block_module, "_TRITON_LAUNCH_LOCK", launch_lock),
        patch.object(torch.Tensor, "record_stream", capture_record_stream),
    ):
        _run_raw_case(
            case,
            seed=7,
            strided_operands=True,
            before_synchronize=allocate_on_default_stream,
        )

    stream.synchronize()
    pressure.clear()

    assert launch_lock.enter_count == 1
    assert len(recorded_operands) == 6
    assert all(is_contiguous for is_contiguous, _ in recorded_operands)
    assert all(recorded_stream == stream for _, recorded_stream in recorded_operands)


@pytest.mark.skipif(not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0")
def test_gptq_triton_block_is_safe_on_process_device_thread_pool():
    """Concurrent production ThreadX workers launch safely and retain exact numerical parity."""
    case = RawCase(48, 128, 64, 4, "gaussian", "correlated")
    futures = [
        gptqmodel.DEVICE_THREAD_POOL.submit(torch.device("cuda:0"), _run_raw_case, case, seed) for seed in range(8)
    ]
    results = [future.result(timeout=120) for future in futures]
    assert len(results) == 8
    assert all(torch.isfinite(quantized).all() and torch.isfinite(errors).all() for quantized, errors in results)


def test_gptq_triton_block_rejects_invalid_inputs_before_launch():
    """Invalid shapes, dtypes, devices, and grouping fail on the host instead of reaching Triton."""
    import gptqmodel.quantization._gptq_block_triton as block_module

    if block_module.gptq_block_triton is None:
        pytest.skip("Triton GPTQ block kernel is unavailable")

    device = torch.device("cuda:0")

    def operands(rows=4, count=64, group_size=64):
        groups = count // group_size
        weights = torch.randn(rows, count, device=device, dtype=torch.float32)
        return (
            weights,
            torch.empty_like(weights),
            torch.empty_like(weights),
            torch.eye(count, device=device, dtype=torch.float32),
            torch.ones(rows, groups, device=device, dtype=torch.float32),
            torch.zeros(rows, groups, device=device, dtype=torch.float32),
        )

    valid = operands()
    with patch.object(block_module, "_gptq_block_kernel") as launch:
        with pytest.raises(ValueError, match="two-dimensional"):
            block_module.gptq_block_triton(
                *(tensor[0] if index < 3 else tensor for index, tensor in enumerate(valid)),
                15,
                64,
            )
        with pytest.raises(ValueError, match="count <= 128"):
            block_module.gptq_block_triton(*operands(count=256), 15, 64)
        with pytest.raises(ValueError, match="positive"):
            block_module.gptq_block_triton(*valid, 15, 0)
        with pytest.raises(ValueError, match="must divide"):
            block_module.gptq_block_triton(*valid, 15, 48)

        bad_dtype = list(valid)
        bad_dtype[1] = bad_dtype[1].half()
        with pytest.raises(TypeError, match="W1/Q1/Err1"):
            block_module.gptq_block_triton(*bad_dtype, 15, 64)

        bad_output_shape = list(valid)
        bad_output_shape[2] = torch.empty(4, 63, device=device)
        with pytest.raises(ValueError, match="must match W1 shape"):
            block_module.gptq_block_triton(*bad_output_shape, 15, 64)

        bad_device = list(valid)
        bad_device[5] = bad_device[5].cpu()
        with pytest.raises(ValueError, match="share one device"):
            block_module.gptq_block_triton(*bad_device, 15, 64)

        cpu_operands = tuple(tensor.cpu() for tensor in valid)
        with pytest.raises(ValueError, match="must be CUDA"):
            block_module.gptq_block_triton(*cpu_operands, 15, 64)

        bad_read_dtype = list(valid)
        bad_read_dtype[3] = bad_read_dtype[3].half()
        with pytest.raises(TypeError, match="Hinv1/scale/zero"):
            block_module.gptq_block_triton(*bad_read_dtype, 15, 64)

        bad_hessian = list(valid)
        bad_hessian[3] = torch.eye(63, device=device)
        with pytest.raises(ValueError, match="Hinv1 must have shape"):
            block_module.gptq_block_triton(*bad_hessian, 15, 64)

        bad_scale = list(valid)
        bad_scale[4] = torch.ones(4, 2, device=device)
        with pytest.raises(ValueError, match="scale/zero must have shape"):
            block_module.gptq_block_triton(*bad_scale, 15, 64)

    launch.assert_not_called()


def test_gptq_triton_availability_handles_runtime_and_import_failures():
    import builtins

    import gptqmodel.quantization._gptq_block_triton as block_module

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert not block_module._triton_available()

    original_import = builtins.__import__

    def fail_triton_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("deliberate test failure")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fail_triton_import):
        assert not block_module._triton_available()

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert importlib.reload(block_module).gptq_block_triton is None
    assert importlib.reload(block_module).gptq_block_triton is not None


def test_gptq_triton_block_crash_probe_synchronizes_at_launch_site():
    import gptqmodel.quantization._gptq_block_triton as block_module

    if block_module.gptq_block_triton is None:
        pytest.skip("Triton GPTQ block kernel is unavailable")

    case = RawCase(4, 32, 32, 4, "gaussian", "identity")
    weights, hessian_inverse, scale, zero = _case_inputs(case, seed=31)
    quantized = torch.empty_like(weights)
    errors = torch.empty_like(weights)
    original_synchronize = torch.cuda.synchronize
    with (
        patch.object(block_module, "_CUDA_CRASH_PROBE", True),
        patch.object(torch.cuda, "synchronize", wraps=original_synchronize) as synchronize,
    ):
        block_module.gptq_block_triton(
            weights,
            quantized,
            errors,
            hessian_inverse,
            scale,
            zero,
            15,
            case.group_size,
        )
    synchronize.assert_called_once_with(weights.device)


def test_gptq_triton_block_env_disable_falls_back():
    """With GPTQMODEL_TRITON_BLOCK=0 the production path does not need the Triton kernel."""
    _, outputs = _run_gptq_quantize(128, False)
    assert outputs[0] is not None
    assert outputs[0].numel() > 0


@pytest.mark.skipif(os.environ.get("GPTQMODEL_TRITON_FRESH_PROBE") != "1", reason="fresh-process probe")
def test_gptq_triton_block_fresh_process_probe():
    _run_raw_case(RawCase(8, 64, 64, 4, "rounding_ties", "identity"), seed=23)


@pytest.mark.skipif(not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0")
def test_gptq_triton_block_initializes_cleanly_in_fresh_processes():
    """Triton compile/cache initialization is stable in independent free-threaded processes."""
    test_id = f"{__file__}::test_gptq_triton_block_fresh_process_probe"
    environment = os.environ.copy()
    environment["GPTQMODEL_TRITON_FRESH_PROBE"] = "1"
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_id],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
