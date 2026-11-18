# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import gptqmodel.utils.model as model_utils
from gptqmodel import GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    _BITSANDBYTES_4BIT_STATE_BUFFER_NAMES,
    BITSANDBYTES_AVAILABLE,
    BitsAndBytesLinear,
    _cpu_int8_dense_should_use,
    _cpu_packed_kernels,
    _cpu_packed_max_rows,
    _dequantize_int8_weight,
    _int8_chunked_fp32_epilogue,
)
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import BitsAndBytesConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.kernel import KernelConfig, configure_kernel
from gptqmodel.utils.model_dequant import dequantize_model, detect_format


def _build_linear(bits: int) -> nn.Linear:
    torch.manual_seed(100 + bits)
    linear = nn.Linear(32, 24, bias=True).eval()
    if bits == 8:
        linear = linear.half()
    return linear


def _build_kernel(bits: int, linear: nn.Linear) -> BitsAndBytesLinear:
    kwargs = {
        "bits": bits,
        "group_size": -1,
        "sym": True,
        "desc_act": False,
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bias": linear.bias is not None,
        "register_buffers": False,
    }
    if bits == 4:
        kwargs.update(
            {
                "format": "nf4",
                "block_size": 128,
                "compress_statistics": False,
            }
        )

    kernel = BitsAndBytesLinear(**kwargs)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    return kernel.eval()


def _build_cpu_packed_kernel(
    *,
    in_features: int = 128,
    out_features: int = 64,
    register_buffers: bool = False,
    bnb_format: str = "nf4",
    compress_statistics: bool = False,
    block_size: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Linear, BitsAndBytesLinear]:
    torch.manual_seed(321)
    linear = nn.Linear(in_features, out_features, bias=True).to(compute_dtype).eval()
    kernel = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=register_buffers,
        format=bnb_format,
        block_size=block_size,
        compress_statistics=compress_statistics,
        dtype=compute_dtype,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    return linear, kernel.eval()


def _relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    difference = output.float() - reference.float()
    return (difference.norm() / reference.float().norm().clamp_min(1e-12)).item()


def _enable_packed_candidate(monkeypatch, *, accuracy_tol: float = 1.0) -> None:
    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    configure_kernel(KernelConfig(bitsandbytes_cpu_packed=True))
    monkeypatch.setattr(bnb_module, "_CPU_PACKED_REL_TOL", accuracy_tol)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_kernel_selection():
    assert get_kernel_for_backend(BACKEND.BITSANDBYTES, METHOD.BITSANDBYTES, FORMAT.BITSANDBYTES) is BitsAndBytesLinear

    for bits in (4, 8):
        selected = select_quant_linear(
            bits=bits,
            group_size=-1,
            desc_act=False,
            sym=True,
            backend=BACKEND.BITSANDBYTES,
            format=FORMAT.BITSANDBYTES,
            quant_method=METHOD.BITSANDBYTES,
            device=DEVICE.CPU,
            pack_dtype=torch.int32,
        )
        assert selected is BitsAndBytesLinear


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_packing_and_schema_support_inference_and_meta_contexts():
    from gptqmodel.nn_modules.qlinear.bitsandbytes import _buffer_spec_4bit

    _buffer_spec_4bit.cache_clear()
    kwargs = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "in_features": 64,
        "out_features": 128,
        "dtype": torch.float32,
    }
    with torch.inference_mode(), torch.device("meta"):
        shell = BitsAndBytesLinear(**kwargs)
    assert all(value.device.type == "meta" for value in shell.state_dict().values())
    kernel = BitsAndBytesLinear(**kwargs)
    with torch.inference_mode():
        source = torch.nn.Linear(64, 128, bias=False)
        kernel.pack_original(source, None, None)
    assert kernel.dequantize_weight().shape == source.weight.shape
    assert torch.equal(kernel._weight_to_matrix(source), source.weight)


def test_create_quant_module_uses_dynamic_bits_for_bitsandbytes_format_normalization():
    seen = {}

    class _DummyBitsAndBytesLinear(nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            seen["validate_bits"] = kwargs.get("bits")
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            seen["init_bits"] = kwargs.get("bits")
            seen["format"] = kwargs.get("format")
            self.bias = None

    module = nn.Module()
    module.proj = nn.Linear(32, 32, bias=False)

    model_utils.create_quant_module(
        name="proj",
        linear_cls=_DummyBitsAndBytesLinear,
        bits=4,
        desc_act=False,
        dynamic={
            r"proj": {
                "bits": 8,
                "bnb_quant_type": "int8",
            }
        },
        group_size=-1,
        module=module,
        submodule=module.proj,
        sym=True,
        device=None,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.BITSANDBYTES,
        backend=BACKEND.BITSANDBYTES,
    )

    assert seen["validate_bits"] == 8
    assert seen["init_bits"] == 8
    assert seen["format"] == "int8"
    assert isinstance(module.proj, _DummyBitsAndBytesLinear)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_bitsandbytes_forward_matches_dequantized_reference(bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    x = torch.randn(7, linear.in_features, dtype=torch.float32)
    dequant_weight = kernel.dequantize_weight().to(torch.float32)
    expected = torch.matmul(x, dequant_weight.t())
    if kernel.bias is not None:
        expected = expected + kernel.bias.to(torch.float32)

    with torch.inference_mode():
        out = kernel(x)

    atol = 1e-3 if bits == 4 else 1e-2
    rtol = 1e-3 if bits == 4 else 5e-2
    torch.testing.assert_close(out.to(torch.float32), expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the bitsandbytes INT8 M=0 abort is CUDA-only; the CPU path already returns empty",
)
def test_bitsandbytes_zero_row_input_does_not_terminate_process():
    """#4745: a zero-row activation must not reach the native M=0 launch.

    ``int8_vectorwise_quant``/``int8_linear_matmul`` answer an M=0 problem with
    ``Error invalid argument ... ops.cu`` and terminate the interpreter, so a
    regression cannot be observed from inside this process. Run the forward in
    a child and require it to exit cleanly for both bit widths and ranks.
    """

    script = textwrap.dedent(
        """
        import json
        import torch
        from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

        in_features = out_features = 256
        linear = torch.nn.Linear(in_features, out_features, bias=False)
        results = {}
        for bits in (4, 8):
            kernel = BitsAndBytesLinear(
                bits=bits,
                group_size=-1,
                sym=True,
                desc_act=False,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                dtype=torch.float16,
            )
            kernel.pack_original(linear=linear, scales=None, zeros=None)
            kernel = kernel.eval().cuda()
            for shape in ((0, in_features), (2, 0, in_features)):
                x = torch.randn(*shape, device="cuda", dtype=torch.float16)
                with torch.inference_mode():
                    out = kernel(x)
                results[f"{bits}:{shape}"] = [list(out.shape), str(out.dtype), out.numel()]
        print("ZERO_ROW_JSON:" + json.dumps(results))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    markers = [
        line for line in completed.stdout.splitlines() if line.startswith("ZERO_ROW_JSON:")
    ]
    assert markers, completed.stdout[-4000:]
    results = json.loads(markers[-1].split("ZERO_ROW_JSON:", 1)[1])

    for bits in (4, 8):
        for leading in ((0,), (2, 0)):
            shape = [*leading, 256]
            key = f"{bits}:{tuple(shape)}"
            assert results[key] == [shape, "torch.float16", 0], (key, results[key])


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bitsandbytes_int8_chunked_epilogue_matches_single_expression(
    monkeypatch,
    bias: bool,
    out_dtype: torch.dtype,
):
    """#5097: chunking must not change a value the FP32 epilogue produces."""

    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    torch.manual_seed(5097)
    rows, cols = 37, 19
    mm_out = torch.randint(-(2**30), 2**30, (rows, cols), dtype=torch.int32)
    x_stats = torch.rand(rows, dtype=torch.float32) * 32.0
    weight_scb = torch.rand(cols, dtype=torch.float32) * 64.0 + 1.0
    bias_tensor = torch.randn(cols, dtype=torch.float32) * 3.0 if bias else None

    expected = mm_out.to(torch.float32) * x_stats.reshape(-1, 1) * weight_scb.reshape(1, -1) / (127.0 * 127.0)
    if bias_tensor is not None:
        expected = expected + bias_tensor

    # Force a ragged 5-row chunking so the boundary rows are exercised.
    monkeypatch.setattr(bnb_module, "_INT8_EPILOGUE_CHUNK_ELEMENTS", 5 * cols)
    actual = _int8_chunked_fp32_epilogue(
        mm_out,
        x_stats,
        weight_scb,
        bias=bias_tensor,
        out_dtype=out_dtype,
    )

    assert actual.shape == (rows, cols)
    assert actual.dtype == out_dtype
    torch.testing.assert_close(actual, expected.to(out_dtype), rtol=0, atol=0)


def _int8_boundary_child_script(limit_rows: int, over_limit_rows: int) -> str:
    """Forward a no-bias FP16 INT8 layer at and just below the 2**31 limit.

    ``int8_mm_dequant`` answers an output with 2**31 elements or more with
    ``Error invalid argument ... ops.cu``, which exits the interpreter instead
    of raising, so the forward runs in a child process.
    """

    return textwrap.dedent(
        f"""
        import json
        import torch
        import torch.nn as nn
        from gptqmodel.nn_modules.qlinear.bitsandbytes import (
            BitsAndBytesLinear,
            import_bitsandbytes,
        )

        in_features, out_features = 64, 2048
        bnb = import_bitsandbytes()
        native_calls = []
        native = bnb.functional.int8_mm_dequant

        def counted_native(*args, **kwargs):
            native_calls.append(1)
            return native(*args, **kwargs)

        bnb.functional.int8_mm_dequant = counted_native

        torch.manual_seed(5097)
        linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16).eval()
        kernel = BitsAndBytesLinear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=torch.float16,
        )
        kernel.pack_original(linear=linear, scales=None, zeros=None)
        kernel.post_init()
        kernel = kernel.eval().cuda()

        results = {{}}
        for rows in ({limit_rows}, {over_limit_rows}):
            native_calls.clear()
            x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                out = kernel(x)
            rows_out = out.reshape(-1, out_features)
            # Independent FP64 A8W8 oracle over the first and last rows: it
            # quantizes the activation rows itself and never reads a kernel
            # intermediate.
            weight64 = kernel.weight.to(torch.float64) * (
                kernel.weight_scb.to(torch.float64) / 127.0
            ).unsqueeze(1)
            probe = torch.cat([x[:1], x[-1:]]).to(torch.float64)
            absmax = probe.abs().amax(dim=1)
            scale = (absmax / 127.0).clamp_min(torch.finfo(torch.float64).tiny)
            codes = (probe / scale[:, None]).round().clamp(-127, 127)
            expected = (codes * scale[:, None]) @ weight64.t()
            expected = expected.to(torch.float16)
            actual = torch.cat([rows_out[:1], rows_out[-1:]]).to(torch.float64)
            reference = expected.to(torch.float64)
            error = ((actual - reference).abs().max() / reference.abs().max()).item()
            results[str(rows)] = {{
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "finite": bool(torch.isfinite(out).all()),
                "max_rel_error": error,
                "native_epilogue_calls": len(native_calls),
            }}
            del out, rows_out
            torch.cuda.empty_cache()
        print("INT8_LIMIT_JSON:" + json.dumps(results))
        """
    )


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the bitsandbytes INT8 epilogue limit is CUDA/HIP-only",
)
def test_bitsandbytes_int8_epilogue_above_int32_elements_does_not_terminate_process():
    """#5097: rows x out_features >= 2**31 must not reach the native epilogue.

    The layer holds 64 x 2048 weights, so M=1048576 rows is exactly 2**31
    output elements. The native path aborts the interpreter there; the FP32
    chunked epilogue must serve the call. One row fewer stays on the native
    fused epilogue, which is the control that the guard is a boundary.
    """

    out_features = 2048
    limit_rows = (2**31) // out_features
    over_limit_rows = limit_rows - 1
    # mm_out (INT32) plus the FP16 output and the activation dominate the footprint.
    needed = limit_rows * out_features * (4 + 2)
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < needed:
        pytest.skip(f"needs about {needed // 2**30} GiB of free device memory for the 2**31-element output")

    completed = subprocess.run(
        [sys.executable, "-c", _int8_boundary_child_script(limit_rows, over_limit_rows)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    markers = [line for line in completed.stdout.splitlines() if line.startswith("INT8_LIMIT_JSON:")]
    assert markers, completed.stdout[-4000:]
    results = json.loads(markers[-1].split("INT8_LIMIT_JSON:", 1)[1])

    below = results[str(over_limit_rows)]
    assert below["shape"] == [over_limit_rows, out_features]
    assert below["dtype"] == "torch.float16"
    assert below["finite"]
    assert below["native_epilogue_calls"] == 1
    assert below["max_rel_error"] <= 2e-3

    at_limit = results[str(limit_rows)]
    assert at_limit["shape"] == [limit_rows, out_features]
    assert at_limit["dtype"] == "torch.float16"
    assert at_limit["finite"]
    # The trigger must have been the size limit, not an unrelated reroute.
    assert at_limit["native_epilogue_calls"] == 0
    assert at_limit["max_rel_error"] <= 2e-3


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_int8_dequant_matches_native_op():
    from gptqmodel.nn_modules.qlinear.bitsandbytes import import_bitsandbytes

    bnb = import_bitsandbytes()
    weight = torch.randint(-127, 128, (64, 128), dtype=torch.int8)
    weight_scb = torch.rand(64, dtype=torch.float32) + 0.5

    expected = bnb.functional.int8_vectorwise_dequant(weight, weight_scb)
    actual = _dequantize_int8_weight(weight, weight_scb)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_bitsandbytes_int8_dense_dispatch_gate(monkeypatch):
    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: True)

    assert _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert _cpu_int8_dense_should_use(
        rows=256,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=32,
        in_features=640,
        out_features=1024,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=32,
        in_features=512,
        out_features=1024,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=63,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=257,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.float16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=1024,
        out_features=1024,
        compute_dtype=torch.bfloat16,
    )
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=2048,
        compute_dtype=torch.bfloat16,
    )

    configure_kernel(KernelConfig(bitsandbytes_cpu_int8_dense=False))
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )

    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: False)
    assert not _cpu_int8_dense_should_use(
        rows=64,
        in_features=512,
        out_features=512,
        compute_dtype=torch.bfloat16,
    )


def test_bitsandbytes_cpu_packed_dispatch_is_shape_and_thread_aware():
    assert _cpu_packed_max_rows(in_features=512, out_features=512, threads=1) == 32
    assert _cpu_packed_max_rows(in_features=512, out_features=512, threads=2) == 64
    assert _cpu_packed_max_rows(in_features=512, out_features=512, threads=4) == 512
    assert _cpu_packed_max_rows(in_features=512, out_features=512, threads=8) == 512

    assert _cpu_packed_max_rows(in_features=512, out_features=2048, threads=1) == 32
    assert _cpu_packed_max_rows(in_features=512, out_features=2048, threads=2) == 48
    assert _cpu_packed_max_rows(in_features=512, out_features=2048, threads=8) == 48


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_int8_small_bf16_uses_tiled_recompute_without_caching(monkeypatch):
    import bitsandbytes.functional as bnb_functional

    bnb_module = sys.modules[BitsAndBytesLinear.__module__]
    monkeypatch.setattr(bnb_module, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(bnb_module, "_CPU_INT8_DENSE_TILE_FEATURES", 128)

    torch.manual_seed(501)
    linear = nn.Linear(512, 512, bias=True).to(torch.bfloat16).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=512,
        bias=True,
        dtype=torch.bfloat16,
    ).eval()
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()

    x = torch.randn(64, 512, dtype=torch.bfloat16)
    expected = (
        x.float() @ kernel.dequantize_weight().float().t()
        + kernel.bias.float()
    )

    calls = []
    original_dequant = bnb_module._dequantize_int8_weight

    def counted_dequant(weight, weight_scb):
        calls.append(weight.shape)
        return original_dequant(weight, weight_scb)

    monkeypatch.setattr(bnb_module, "_dequantize_int8_weight", counted_dequant)
    monkeypatch.setattr(
        bnb_functional,
        "int8_linear_matmul",
        lambda *_args, **_kwargs: pytest.fail("native INT8 path should not run"),
    )

    with torch.inference_mode():
        out = kernel(x)

    assert calls == [(512, 128)] * 4
    assert out.dtype == x.dtype
    assert _relative_l2(out, expected) < 2e-3
    torch.testing.assert_close(
        out,
        expected.to(torch.bfloat16),
        rtol=2e-3,
        atol=2e-3,
    )
    assert kernel.weight.dtype == torch.int8
    assert kernel.weight.numel() == kernel.in_features * kernel.out_features
    assert not any(
        name in vars(kernel)
        for name in ("_dequantized_weight", "_dense_weight", "_weight_cache")
    )

    calls.clear()
    with torch.inference_mode():
        repeated = kernel(x)
    assert calls == [(512, 128)] * 4
    assert torch.equal(repeated, out)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_bitsandbytes_int8_native_matches_independent_a8w8_oracle():
    torch.manual_seed(501)
    linear = nn.Linear(512, 512, bias=True).to(torch.bfloat16).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=512,
        bias=True,
        dtype=torch.bfloat16,
    ).eval()
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    kernel._int8_dense_disabled = True

    x = torch.randn(16, 512, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = kernel(x)

    # The INT8 path computes the row scale in FP32 from the source dtype; an
    # intermediate FP16 cast would overflow any magnitude above 65504.
    x_fp32 = x.float()
    absmax = x_fp32.abs().amax(dim=1)
    x_scale = (absmax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
    x_int8 = (x_fp32 / x_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    weight_scale = kernel.weight_scb.float() / 127.0
    expected = (
        (x_int8.float() @ kernel.weight.float().t())
        * x_scale[:, None]
        * weight_scale[None, :]
        + kernel.bias.float()
    )

    assert out.dtype == x.dtype
    assert _relative_l2(out, expected) < 2e-3
    assert _relative_l2(out, expected.to(torch.float16)) < 2e-3


# #4746: shapes whose 2-D cuBLASLt INT8 IMMA launch raises for M > 16. Both
# K % 16 != 0 (with K % 4 == 0) and N % 4 != 0 must stay covered.
_INT8_IMMA_MISALIGNED_SHAPES = [(100, 37), (100, 2050), (104, 37), (104, 2050)]


def _build_int8_kernel(in_features: int, out_features: int) -> BitsAndBytesLinear:
    torch.manual_seed(4746 + in_features * 1000 + out_features)
    linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    return kernel.eval()


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the bitsandbytes INT8 cuBLASLt IMMA path is CUDA/HIP-only",
)
@pytest.mark.parametrize(("in_features", "out_features"), _INT8_IMMA_MISALIGNED_SHAPES)
def test_bitsandbytes_int8_misaligned_imma_shapes_match_dequantized_reference(
    in_features: int,
    out_features: int,
):
    """#4746: M > 16 must not reach cuBLASLt IMMA for a shape it rejects."""

    kernel = _build_int8_kernel(in_features, out_features).to("cuda")
    dequant32 = kernel.dequantize_weight()
    assert dequant32.dtype == torch.float32
    dequant64 = dequant32.to(torch.float64)

    for rows in (1, 17):
        torch.manual_seed(4746 + rows)
        x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            out = kernel(x)

        expected32 = x.to(torch.float32) @ dequant32.t()
        expected64 = x.to(torch.float64) @ dequant64.t()

        assert out.dtype == torch.float16
        # The M=1 path still quantizes activations to INT8, so it only tracks
        # the dequantized-weight reference to A8W8 precision. The M=17
        # fallback multiplies the full-precision activation by the dequantized
        # weight, so it must match both references tightly.
        torch.testing.assert_close(
            out.to(torch.float32),
            expected64.to(torch.float32),
            rtol=5e-2 if rows == 1 else 2e-3,
            atol=2e-2 if rows == 1 else 2e-3,
        )
        if rows == 17:
            torch.testing.assert_close(
                out,
                expected32.to(torch.float16),
                rtol=1e-3,
                atol=1e-3,
            )


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the bitsandbytes INT8 cuBLASLt IMMA path is CUDA/HIP-only",
)
def test_bitsandbytes_int8_imma_guard_only_reroutes_misaligned_shapes(monkeypatch):
    """#4746: the fallback is exact; aligned shapes keep the native INT8 kernel."""

    from gptqmodel.nn_modules.qlinear.bitsandbytes import import_bitsandbytes

    bnb = import_bitsandbytes()
    native_calls = []
    fallback_calls = []
    native = bnb.functional.int8_linear_matmul

    def counted_native(*args, **kwargs):
        native_calls.append(1)
        return native(*args, **kwargs)

    def counted_fallback(self, x, bias):
        fallback_calls.append(1)
        return fallback(self, x, bias)

    monkeypatch.setattr(bnb.functional, "int8_linear_matmul", counted_native)
    fallback = getattr(BitsAndBytesLinear, "_int8_dequant_tiled_forward", None)
    monkeypatch.setattr(
        BitsAndBytesLinear,
        "_int8_dequant_tiled_forward",
        counted_fallback,
        raising=False,
    )

    aligned = _build_int8_kernel(128, 64).to("cuda")
    x = torch.randn(17, 128, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        aligned(x)
    assert native_calls and not fallback_calls

    native_calls.clear()
    fallback_calls.clear()
    misaligned = _build_int8_kernel(100, 37).to("cuda")
    x = torch.randn(17, 100, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        misaligned(x)
    assert fallback_calls == [1]
    assert not native_calls


def _install_fake_bitsandbytes(monkeypatch) -> None:
    """Install a rank-2-faithful fake of the optional bitsandbytes module.

    Used only when bitsandbytes is absent (the CUDA venv). The stubs mirror the
    native kernels: the vectorwise quantizer accepts rank 2 only, while
    ``int8_linear_matmul`` broadcasts the rank-3 product exactly like Torch
    does, which is what surfaces the missing flatten in the FP32 epilogue.
    """

    fake = types.ModuleType("bitsandbytes")
    fake.__version__ = "0.50.2"
    function = types.ModuleType("bitsandbytes.functional")

    def int8_vectorwise_quant(x, threshold=0.0):
        del threshold
        assert x.dim() == 2, f"native int8_vectorwise_quant expects rank 2, got {tuple(x.shape)}"
        # The native kernel reads data_ptr() as packed [M, K] rows and ignores
        # the tensor's strides. as_strided reproduces that read so a row-strided
        # view yields the same wrong codes the real kernel produces.
        x_packed = torch.as_strided(x, (x.numel(),), (1,)).reshape(x.shape)
        x_fp32 = x_packed.to(torch.float32)
        row_absmax = x_fp32.abs().amax(dim=1)
        scale = row_absmax / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = (x_fp32 / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        return codes, row_absmax, None

    def int8_linear_matmul(x_int8, weight):
        # Exact for INT8 inputs: the products and their K-sum fit in FP32.
        return torch.matmul(x_int8.to(torch.float32), weight.to(torch.float32).t())

    def int8_mm_dequant(mm_out, row_stats, col_stats):
        return (
            mm_out.to(torch.float32)
            * row_stats.to(torch.float32).reshape(-1, 1)
            * col_stats.to(torch.float32).reshape(1, -1)
            / (127.0 * 127.0)
        )

    function.int8_vectorwise_quant = int8_vectorwise_quant
    function.int8_linear_matmul = int8_linear_matmul
    function.int8_mm_dequant = int8_mm_dequant
    fake.functional = function

    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", function)
    monkeypatch.setattr(
        sys.modules[BitsAndBytesLinear.__module__],
        "_VALIDATED_BITSANDBYTES",
        fake,
    )
    # Skip only the install probe; the shape/dtype validation still runs.
    monkeypatch.setattr(
        BitsAndBytesLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )


@pytest.mark.parametrize(
    ("dtype", "bias"),
    [
        (torch.float32, False),
        (torch.bfloat16, True),
        (torch.float16, False),
    ],
)
@pytest.mark.parametrize("leading", [(2, 3), (4, 2, 3), ()])
def test_bitsandbytes_int8_forward_preserves_batched_activation_dims(
    monkeypatch,
    dtype: torch.dtype,
    bias: bool,
    leading: tuple[int, ...],
):
    if not BITSANDBYTES_AVAILABLE:
        _install_fake_bitsandbytes(monkeypatch)

    torch.manual_seed(4479)
    linear = nn.Linear(128, 64, bias=bias).to(dtype).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=64,
        bias=bias,
        dtype=dtype,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    kernel.eval()

    x = torch.randn(*leading, 128, dtype=dtype)
    with torch.inference_mode():
        flattened = kernel(x.reshape(-1, 128))
        batched = kernel(x)

    assert batched.shape == leading + (64,)
    assert torch.equal(batched, flattened.reshape(batched.shape))

    if dtype is torch.float32:
        # Independent Torch oracle for the quantize -> INT8 matmul -> dequant
        # arithmetic; it does not call the kernel or reuse its intermediates.
        x_flat = x.reshape(-1, 128).float()
        absmax = x_flat.abs().amax(dim=1)
        x_scale = (absmax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        x_codes = (x_flat / x_scale[:, None]).round().clamp(-127, 127)
        expected = (
            (x_codes * x_scale[:, None]) @ kernel.dequantize_weight().float().t()
        ).reshape(batched.shape)
        torch.testing.assert_close(batched, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bitsandbytes INT8 vectorwise quant requires CUDA/HIP",
)
def test_bitsandbytes_int8_forward_handles_row_strided_activations(monkeypatch):
    """A row-strided activation must be packed before the native INT8 ops."""

    if not BITSANDBYTES_AVAILABLE:
        _install_fake_bitsandbytes(monkeypatch)

    torch.manual_seed(4584)
    linear = nn.Linear(128, 64, bias=False).to(torch.float16).eval()
    kernel = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.float16,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    kernel.eval()
    kernel = kernel.to("cuda")

    # ``base[:, -1:, :]`` has row stride 4 * K, so ``reshape`` returns a view
    # rather than a packed [M, K] buffer.
    base = torch.randn(3, 4, 128, dtype=torch.float16, device="cuda")
    x = base[:, -1:, :]
    x_flat = x.reshape(-1, 128)
    assert not x_flat.is_contiguous()

    with torch.inference_mode():
        actual = kernel(x)
        contiguous = kernel(x_flat.contiguous()).reshape(actual.shape)

    assert actual.shape == (3, 1, 64)
    assert torch.equal(actual, contiguous)

    # Independent FP64 oracle: quantize the logical rows, multiply by the
    # dequantized packed weight, and dequantize without calling forward.
    x64 = x_flat.to(torch.float64)
    absmax = x64.abs().amax(dim=1)
    scale = (absmax / 127.0).clamp_min(torch.finfo(torch.float64).tiny)
    codes = (x64 / scale[:, None]).round().clamp(-127, 127)
    weight64 = kernel.weight.to(torch.float64) * (
        kernel.weight_scb.to(torch.float64) / 127.0
    ).unsqueeze(1)
    expected = ((codes * scale[:, None]) @ weight64.t()).reshape(actual.shape)
    torch.testing.assert_close(
        actual.to(torch.float64), expected, rtol=2e-3, atol=2e-3
    )


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
@pytest.mark.parametrize(
    ("bnb_format", "compress_statistics"),
    [
        ("nf4", False),
        ("nf4", True),
        ("fp4", False),
        ("fp4", True),
    ],
)
def test_bitsandbytes_cpu_packed_kernel_matches_oracle_and_preserves_checkpoint_schema(
    bnb_format: str,
    compress_statistics: bool,
    monkeypatch,
):
    _enable_packed_candidate(monkeypatch)
    linear, kernel = _build_cpu_packed_kernel(
        bnb_format=bnb_format,
        compress_statistics=compress_statistics,
    )
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    canonical_state = {name: tensor.clone() for name, tensor in kernel.state_dict().items()}
    dequant_weight = kernel.dequantize_weight().float()
    expected = torch.matmul(x.float(), dequant_weight.t())
    if kernel.bias is not None:
        expected = expected + kernel.bias.float()

    with torch.inference_mode():
        out = kernel(x)

    assert kernel._cpu_packing_active
    assert out.dtype == torch.bfloat16
    assert tuple(kernel.weight.shape) == (linear.out_features, linear.in_features // 2)
    relative_error = _relative_l2(out, expected)
    assert relative_error < 4e-3
    assert relative_error > 2e-3
    torch.testing.assert_close(out.float(), expected, rtol=4e-3, atol=4e-3)

    with torch.inference_mode():
        repeated = kernel(x)
    assert torch.equal(repeated, out)

    state_dict = kernel.state_dict()
    assert not kernel._cpu_packing_active
    for name, tensor in canonical_state.items():
        assert torch.equal(state_dict[name], tensor)
    assert tuple(state_dict["weight"].shape) == (
        linear.out_features * linear.in_features // 2,
        1,
    )
    torch.testing.assert_close(
        kernel.dequantize_weight().float(),
        dequant_weight,
        rtol=1e-6,
        atol=1e-6,
    )

    reloaded = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
        register_buffers=True,
        format=bnb_format,
        block_size=128,
        compress_statistics=compress_statistics,
        dtype=torch.bfloat16,
    ).eval()
    reloaded.load_state_dict(state_dict, strict=True)
    reloaded.post_init()

    with torch.inference_mode():
        reloaded_out = reloaded(x)
    assert reloaded._cpu_packing_active
    assert torch.equal(reloaded_out, out)


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
@pytest.mark.parametrize("block_size", [32, 64, 128, 256, 512])
def test_bitsandbytes_cpu_packed_kernel_supports_bnb_block_sizes(block_size: int, monkeypatch):
    _enable_packed_candidate(monkeypatch)
    linear, kernel = _build_cpu_packed_kernel(
        in_features=512,
        block_size=block_size,
    )
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = kernel(x)

    assert kernel._cpu_packing_active
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
def test_bitsandbytes_cpu_packed_kernel_skips_unsupported_shapes_and_env(monkeypatch):
    _enable_packed_candidate(monkeypatch)
    linear, kernel = _build_cpu_packed_kernel(out_features=24)
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        kernel(x)
    assert not kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel()
    with torch.inference_mode():
        kernel(x)
    assert kernel._cpu_packing_active

    _, wide_kernel = _build_cpu_packed_kernel(in_features=2048, out_features=512)
    wide_x = torch.randn(7, wide_kernel.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        wide_kernel(wide_x)
    assert wide_kernel._cpu_packing_active
    with torch.inference_mode():
        wide_kernel(torch.randn(49, wide_kernel.in_features, dtype=torch.bfloat16))
    assert not wide_kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel(compute_dtype=torch.float16)
    with torch.inference_mode():
        kernel(x.to(torch.float16))
    assert not kernel._cpu_packing_active

    _, kernel = _build_cpu_packed_kernel()
    configure_kernel(KernelConfig(bitsandbytes_cpu_packed=False))
    with torch.inference_mode():
        kernel(x)
    assert not kernel._cpu_packing_active


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
def test_bitsandbytes_cpu_packed_defaults_to_opt_in(monkeypatch):
    configure_kernel(KernelConfig())
    linear, kernel = _build_cpu_packed_kernel()
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = kernel(x)

    assert not kernel._cpu_packing_active
    assert not kernel._cpu_packing_disabled
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    _cpu_packed_kernels() is None,
    reason="bitsandbytes CPU packed kernel unavailable",
)
def test_bitsandbytes_cpu_packed_kernel_falls_back_when_accuracy_gate_fails(monkeypatch):
    _enable_packed_candidate(monkeypatch, accuracy_tol=2e-3)
    linear, kernel = _build_cpu_packed_kernel()
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    dequant_weight = kernel.dequantize_weight().float()
    expected = torch.matmul(x.float(), dequant_weight.t()) + kernel.bias.float()

    with torch.inference_mode():
        out = kernel(x)

    assert not kernel._cpu_packing_active
    assert kernel._cpu_packing_disabled
    assert _relative_l2(out, expected) < 2e-3
    assert tuple(kernel.weight.shape) == (
        linear.out_features * linear.in_features // 2,
        1,
    )
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_bitsandbytes_state_dict_round_trip(bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    reload_kwargs = {
        "bits": bits,
        "group_size": -1,
        "sym": True,
        "desc_act": False,
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bias": linear.bias is not None,
        "register_buffers": True,
    }
    if bits == 4:
        reload_kwargs.update(
            {
                "format": "nf4",
                "block_size": 128,
                "compress_statistics": False,
            }
        )

    reloaded = BitsAndBytesLinear(**reload_kwargs).eval()
    reloaded.load_state_dict(kernel.state_dict(), strict=True)
    reloaded.post_init()

    x = torch.randn(5, linear.in_features, dtype=torch.float32)
    with torch.inference_mode():
        torch.testing.assert_close(
            reloaded.dequantize_weight().to(torch.float32),
            kernel.dequantize_weight().to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            reloaded(x).to(torch.float32),
            kernel(x).to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
@pytest.mark.parametrize("bits", [4, 8])
def test_detect_and_dequantize_bitsandbytes_checkpoint(tmp_path, bits: int):
    linear = _build_linear(bits)
    kernel = _build_kernel(bits, linear)

    prefix = "model.layers.0.mlp.up_proj"
    state_dict = {f"{prefix}.{name}": tensor for name, tensor in kernel.state_dict().items()}

    model_dir = tmp_path / f"bnb_{bits}bit"
    model_dir.mkdir()
    save_file(state_dict, str(model_dir / "model.safetensors"))

    quant_cfg = BitsAndBytesConfig(
        bits=bits,
        format="nf4" if bits == 4 else "int8",
        block_size=128 if bits == 4 else 64,
        compress_statistics=False if bits == 4 else True,
    )
    config_payload = {"quantization_config": quant_cfg.to_dict()}
    (model_dir / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")

    detected = detect_format(model_dir, config_payload)
    assert detected == "bitsandbytes"

    output_dir = tmp_path / f"bnb_{bits}bit_dequantized"
    dequantize_model(model_dir, output_dir, target_dtype=torch.float16)

    with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as reader:
        weight = reader.get_tensor(f"{prefix}.weight")
        bias = reader.get_tensor(f"{prefix}.bias")

    torch.testing.assert_close(
        weight.to(torch.float32),
        kernel.dequantize_weight().to(torch.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        bias.to(torch.float32),
        kernel.bias.to(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def _save_tiny_tokenizer(directory) -> None:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["the quick brown fox", "a tiny llama checkpoint"],
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


@pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable")
def test_gptqmodel_load_reloads_bitsandbytes_4bit_checkpoint(tmp_path):
    """A GPTQModel-saved BnB checkpoint must reload through ``GPTQModel.load``.

    The state buffers are checkpoint data, not activations: ``weight_quant_state``
    is variable-length JSON bytes and the float state tables must stay FP32.
    Accelerate sizes/casts them during the real load path, which is why the
    module-level ``load_state_dict`` round trip never exercised this.
    """

    torch.manual_seed(0)
    vocab_size = 32
    hidden_size = 32
    intermediate_size = 64
    block_size = 64
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    source = LlamaForCausalLM(config).eval().half()

    quantized_names = []
    for name, linear in list(source.named_modules()):
        if not isinstance(linear, nn.Linear) or name == "lm_head":
            continue
        quant = BitsAndBytesLinear(
            bits=4,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            register_buffers=True,
            format="nf4",
            block_size=block_size,
            compress_statistics=True,
            dtype=torch.float16,
        )
        quant.pack_original(linear=linear, scales=None, zeros=None)
        quant.post_init()
        source.set_submodule(name, quant)
        quantized_names.append(name)

    state = source.state_dict()
    model_dir = tmp_path / "bnb4"
    model_dir.mkdir()
    save_file(
        {name: value.contiguous() for name, value in state.items()},
        str(model_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )

    # The module's template-sized slot must be smaller than the stored JSON,
    # otherwise the shape-mismatch regression is not exercised.
    probe_name = quantized_names[0]
    probe = source.get_submodule(probe_name)
    template = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=probe.in_features,
        out_features=probe.out_features,
        bias=probe.bias is not None,
        register_buffers=True,
        format="nf4",
        block_size=block_size,
        compress_statistics=True,
        dtype=torch.float16,
    )
    assert tuple(state[f"{probe_name}.weight_quant_state"].shape) != tuple(
        template.state_dict()["weight_quant_state"].shape
    )

    quant_config = BitsAndBytesConfig(
        bits=4,
        format="nf4",
        block_size=block_size,
        compress_statistics=True,
    )
    quant_config.save_pretrained(str(model_dir))
    config.architectures = ["LlamaForCausalLM"]
    config.quantization_config = quant_config.to_dict()
    config.save_pretrained(str(model_dir))
    _save_tiny_tokenizer(str(model_dir))

    loaded = GPTQModel.load(
        str(model_dir),
        backend=BACKEND.BITSANDBYTES,
        device="cpu",
        dtype=torch.float16,
        local_files_only=True,
    )

    for name in quantized_names:
        module = loaded.model.get_submodule(name)
        assert isinstance(module, BitsAndBytesLinear)
        for buffer_name in _BITSANDBYTES_4BIT_STATE_BUFFER_NAMES:
            expected = state[f"{name}.{buffer_name}"]
            actual = getattr(module, buffer_name)
            assert tuple(actual.shape) == tuple(expected.shape), (
                f"{name}.{buffer_name} shape changed across GPTQModel.load"
            )
            assert actual.dtype == expected.dtype, (
                f"{name}.{buffer_name} dtype changed across GPTQModel.load"
            )

    # First decode step (M=1) is the path that rejects a narrowed absmax.
    with torch.inference_mode():
        logits = loaded.model(torch.tensor([[1]], dtype=torch.long)).logits
    assert logits.shape == (1, 1, vocab_size)
    assert torch.isfinite(logits).all()
