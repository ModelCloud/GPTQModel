# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the table/gate cases; the kernel cases carry ``pytest.mark.cuda``.
"""bitsandbytes 4-bit Triton kernels: launch table, gate and FP64 oracle.

The oracle decodes the packed bytes with Torch integer shifts and accumulates
in FP64; it never calls the kernel under test.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear import bitsandbytes_triton as bt


bnb = pytest.importorskip("bitsandbytes")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(not _cuda_available(), reason="CUDA/ROCm device required")


# --------------------------------------------------------------------------- #
# CPU-only: launch table and dispatch gate.
# --------------------------------------------------------------------------- #

def test_launch_table_has_ada_and_rdna4():
    assert set(bt.tuned_families()) == {"ada", "rdna4"}
    for family, cfg in bt._BNB_GEMM_LAUNCH.items():
        assert cfg.min_m >= 1, family
        assert cfg.block_m > 0 and cfg.block_n > 0 and cfg.block_k > 0, family


def test_launch_resolution_by_family():
    assert bt.gemm_launch("gfx1201") == bt._BNB_GEMM_LAUNCH["rdna4"]
    assert bt.gemm_launch("gfx1200") == bt._BNB_GEMM_LAUNCH["rdna4"]
    assert bt.gemm_launch("sm_89") == bt._BNB_GEMM_LAUNCH["ada"]
    # Unregistered targets keep the portable default instead of inheriting a tile.
    assert bt.gemm_launch("gfx942") == bt._DEFAULT_GEMM
    assert bt.gemm_launch(None) == bt._DEFAULT_GEMM


def test_should_use_gate_is_rocm_and_row_banded(monkeypatch):
    monkeypatch.setattr(bt, "IS_ROCM", True)
    monkeypatch.setattr("gptqmodel.utils.env.env_flag", lambda *a, **k: False)
    cfg = bt._BNB_GEMM_LAUNCH["rdna4"]
    assert bt.should_use("gfx1201", torch.float16, cfg.min_m) is True
    assert bt.should_use("gfx1201", torch.bfloat16, cfg.min_m) is True
    assert bt.should_use("gfx1201", torch.float16, cfg.min_m - 1) is False
    assert bt.should_use("gfx1201", torch.float32, cfg.min_m) is False

    # CUDA stays on the vendor kernels unless explicitly forced.
    monkeypatch.setattr(bt, "IS_ROCM", False)
    assert bt.should_use("sm_89", torch.float16, 64) is False


# --------------------------------------------------------------------------- #
# Oracle helpers.
# --------------------------------------------------------------------------- #

def _decode_oracle(qweight: torch.Tensor, quant_state) -> torch.Tensor:
    """Decode packed 4-bit codes to FP64 without calling any GPTQModel kernel."""

    out_features, in_features = quant_state.shape
    n = out_features * in_features
    packed = qweight.reshape(-1)
    idx = torch.arange(n, device=packed.device)
    byte = packed[idx // 2].long()
    # Even element lives in the high nibble, odd element in the low nibble.
    nibble = torch.where(idx % 2 == 0, (byte >> 4) & 0xF, byte & 0xF)

    code = quant_state.code
    if code is None:
        code = bnb.functional.get_4bit_type(quant_state.quant_type, device=packed.device)
    code = code.to(torch.float64).reshape(-1)

    blocksize = int(quant_state.blocksize)
    if quant_state.absmax.dtype == torch.uint8:
        state2 = quant_state.state2
        nested_code = state2.code.to(torch.float64)
        nested_absmax = state2.absmax.to(torch.float64).reshape(-1)
        nblocks = quant_state.absmax.numel()
        scale = nested_code[quant_state.absmax.reshape(-1).long()] * nested_absmax[
            torch.arange(nblocks, device=packed.device) // int(state2.blocksize)
        ] + quant_state.offset.to(torch.float64)
        absmax = scale[idx // blocksize]
    else:
        absmax = quant_state.absmax.to(torch.float64).reshape(-1)[idx // blocksize]
    return (code[nibble] * absmax).reshape(out_features, in_features)


def _packed_module(in_features, out_features, *, dtype, quant_type, blocksize, compress):
    from torch import nn

    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

    torch.manual_seed(7)
    linear = nn.Linear(in_features, out_features, bias=True).to(dtype).eval()
    module = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=quant_type,
        block_size=blocksize,
        compress_statistics=compress,
        dtype=dtype,
        register_buffers=False,
    )
    module.pack_original(linear=linear, scales=None, zeros=None)
    module.post_init()
    return module


# --------------------------------------------------------------------------- #
# Kernel cases.
# --------------------------------------------------------------------------- #

@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("blocksize", [32, 64])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize_matches_fp64_oracle(quant_type, blocksize, compress, dtype):
    module = _packed_module(512, 128, dtype=dtype, quant_type=quant_type,
                            blocksize=blocksize, compress=compress)
    module = module.cuda().eval()
    qs = module._refresh_quant_state()
    oracle = _decode_oracle(module.weight, qs)
    # Compare the kernel's FP32 output against the FP64 decode; casting to the
    # activation dtype first would lose the precision this assertion checks.
    got = bt.dequantize_4bit(module.weight, qs, torch.float32).double()
    assert got.shape == oracle.shape
    torch.testing.assert_close(got, oracle, rtol=1e-6, atol=1e-6)
    # The activation-dtype output must still preserve dtype and stay within one ULP.
    typed = bt.dequantize_4bit(module.weight, qs, dtype)
    assert typed.dtype == dtype
    torch.testing.assert_close(typed.double(), oracle, rtol=8e-3, atol=8e-3)


@requires_gpu
@pytest.mark.cuda
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [16, 64])
def test_matmul_matches_fp64_oracle(quant_type, dtype, m):
    module = _packed_module(1024, 512, dtype=dtype, quant_type=quant_type,
                            blocksize=64, compress=False)
    module = module.cuda().eval()
    qs = module._refresh_quant_state()
    oracle = _decode_oracle(module.weight, qs)

    torch.manual_seed(11)
    x = torch.randn(m, module.in_features, dtype=dtype, device="cuda")
    out = bt.matmul_4bit(x, module.weight, qs, bias=module.bias, out_dtype=dtype)

    expected = (x.double() @ oracle.t() + module.bias.double().reshape(1, -1))
    assert out.dtype == dtype
    assert out.shape == (m, module.out_features)
    # FP32 accumulation against an FP64 reference; allow one activation ULP.
    torch.testing.assert_close(out.double(), expected, rtol=4e-3, atol=4e-3)


@requires_gpu
@pytest.mark.cuda
def test_module_dispatch_and_no_dense_cache():
    module = _packed_module(1024, 512, dtype=torch.float16, quant_type="nf4",
                            blocksize=64, compress=False)
    module = module.cuda().eval()
    family = bt._family(module.weight.device)
    if family not in bt.tuned_families():
        pytest.skip(f"no tuned bitsandbytes tile for {family}")

    before = {name: tuple(t.shape) for name, t in module.named_buffers()}
    torch.manual_seed(3)
    x = torch.randn(bt.gemm_launch(module.weight.device).min_m, module.in_features,
                    dtype=torch.float16, device="cuda")
    first = module(x)
    second = module(x)
    assert first.dtype == torch.float16
    assert torch.equal(first, second)
    after = {name: tuple(t.shape) for name, t in module.named_buffers()}
    assert after == before, "forward changed the registered buffer shapes"
    # No buffer may hold a dense (out_features, in_features) activation-dtype weight.
    dense = module.out_features * module.in_features
    for name, tensor in module.named_buffers():
        assert tensor.numel() < dense, f"{name} looks like a dense dequantized weight"

    # Below the row band the vendor kernel is the faster path.
    small = x[: bt.gemm_launch(module.weight.device).min_m - 1]
    assert bt.should_use(module.weight.device, torch.float16, small.shape[0]) is False


def _retained_tensors(obj, seen=None):
    """Include unregistered module/QuantState/container tensors, not just buffers."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _retained_tensors(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _retained_tensors(value, seen)
    elif isinstance(obj, (torch.nn.Module, bnb.functional.QuantState)):
        yield from _retained_tensors(vars(obj), seen)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quant_type,path", [
    ("nf4", "vendor"), ("fp4", "vendor"), ("int8", "vendor"),
    ("nf4", "triton"), ("fp4", "triton"),
])
def test_selected_gpu_module_matches_oracle_without_expanded_storage(dtype, quant_type, path, monkeypatch, record_property):
    from gptqmodel.nn_modules.qlinear import bitsandbytes as bnb_module
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    if path == "triton" and dtype == torch.float32:
        pytest.skip("FP32 activations use the vendor path")
    if path == "triton" and bt._family(torch.device("cuda:0")) not in bt.tuned_families():
        pytest.skip("no tuned bitsandbytes tile for this GPU")

    bits = 8 if quant_type == "int8" else 4
    cls = select_quant_linear(
        bits=bits, group_size=-1, desc_act=False, sym=True,
        backend=BACKEND.AUTO, format=FORMAT.BITSANDBYTES, quant_method=METHOD.BITSANDBYTES,
        device=torch.device("cuda:0"), pack_dtype=torch.int32, dtype=dtype,
    )
    assert cls is bnb_module.BitsAndBytesLinear
    torch.manual_seed(155)
    source = torch.nn.Linear(512, 128, bias=True).to(dtype).eval()
    module = cls(
        bits=bits, group_size=-1, sym=True, desc_act=False, in_features=512,
        out_features=128, bias=True, dtype=dtype, register_buffers=False,
        format=quant_type, block_size=64, compress_statistics=bits == 4,
    )
    module.pack_original(source, None, None)
    module = module.cuda().eval()
    module.post_init()
    triton_calls = []
    if path == "vendor":
        monkeypatch.setattr(bnb_module, "_triton_backend", lambda: None)
    else:
        monkeypatch.setenv("GPTQMODEL_BNB_TRITON", "1")
        triton_matmul = bt.matmul_4bit

        def counted_matmul(*args, **kwargs):
            triton_calls.append(True)
            return triton_matmul(*args, **kwargs)

        monkeypatch.setattr(bt, "matmul_4bit", counted_matmul)

    # Exactly representable inputs give int8 quantization a known, exact answer:
    # each row has absmax=127/1024 and every value is an integer multiple of 1/1024.
    x = torch.randint(-127, 128, (64, 512), device="cuda").to(dtype) / 1024
    x[:, 0] = 127 / 1024
    if bits == 4:
        weight = _decode_oracle(module.weight, module._refresh_quant_state())
    else:
        weight = module.weight.double() * (module.weight_scb.double()[:, None] / 127)
    expected = x.double() @ weight.t() + module.bias.double()

    def retained_bytes():
        tensors = list(_retained_tensors(module))
        dense_elements = module.in_features * module.out_features
        for tensor in tensors:
            if tensor.numel() >= dense_elements:
                # Native INT8 weights are allowed; any expanded 4-bit or dense
                # floating weight retained through any object is a violation.
                assert bits == 8 and tensor is module.weight and tensor.dtype == torch.int8
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
        return sum(storages.values())

    before = retained_bytes()
    max_abs_error = 0.0
    peak_extra_bytes = 0
    with torch.inference_mode():
        for _ in range(2):
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            actual = module(x)
            torch.cuda.synchronize()
            peak_extra_bytes = max(peak_extra_bytes, torch.cuda.max_memory_allocated() - allocated)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=2e-3)
            max_abs_error = max(max_abs_error, (actual.double() - expected).abs().max().item())
            assert retained_bytes() == before
    assert len(triton_calls) == (2 if path == "triton" else 0)
    record_property("max_abs_error", max_abs_error)
    record_property("retained_bytes", before)
    record_property("additional_retained_bytes", retained_bytes() - before)
    # Includes the output and allocator-visible temporary tensors, not native
    # vendor workspaces outside the Torch allocator or the separate FP64 oracle.
    record_property("peak_extra_torch_bytes", peak_extra_bytes)
