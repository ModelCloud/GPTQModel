import os
import time

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
import gptqmodel.nn_modules.qlinear.gemm_awq_triton as gemm_awq_triton


def _fake_quant_tensors(in_features: int = 32, out_features: int = 8, group_size: int = 32):
    qweight = torch.ones((in_features, out_features // 8), dtype=torch.int32)
    scales = torch.ones((in_features // group_size, out_features), dtype=torch.float16)
    qzeros = torch.zeros((in_features // group_size, out_features // 8), dtype=torch.int32)
    return qweight, scales, qzeros


def _patch_for_triton(monkeypatch, calls):
    monkeypatch.setattr(gemm_awq, "awq_ext", None)

    monkeypatch.setattr(gemm_awq_triton.tritonv2, "TRITON_AVAILABLE", True)

    def fake_dequant(qweight, scales, qzeros):
        calls["dequant"] += 1
        return torch.ones(qweight.shape[0], qweight.shape[1] * 8, dtype=torch.float16)

    def fake_gemm(input, qweight, scales, qzeros, split_k_iters, **_):
        calls["gemm"] += 1
        out_features = qweight.shape[1] * 8
        return torch.ones(input.shape[0], out_features, device=input.device, dtype=input.dtype)

    monkeypatch.setattr(gemm_awq_triton, "awq_dequantize_triton", fake_dequant, raising=False)
    monkeypatch.setattr(gemm_awq_triton, "awq_gemm_triton", fake_gemm, raising=False)


def test_fp16_matmul_heuristic_prefers_dequant_for_large_matrices(monkeypatch):
    calls = {"dequant": 0, "gemm": 0}
    _patch_for_triton(monkeypatch, calls)

    group_size = 32
    out_features = 8
    qweight, scales, qzeros = _fake_quant_tensors(in_features=32, out_features=out_features, group_size=group_size)

    # Large batch x sequence activates the dequantize-then-matmul path.
    x = torch.ones((33, 32, qweight.shape[0]), dtype=torch.float16)

    out = gemm_awq_triton.AwqGEMMTritonQuantLinear._Function.apply(
        x, qweight, qzeros, scales, 4, group_size, None, out_features,
    )

    assert calls == {"dequant": 1, "gemm": 0}
    assert out.shape == (33, 32, out_features)


def test_fp16_matmul_heuristic_prefers_fused_gemm_for_small_matrices(monkeypatch):
    calls = {"dequant": 0, "gemm": 0}
    _patch_for_triton(monkeypatch, calls)

    group_size = 32
    out_features = 8
    qweight, scales, qzeros = _fake_quant_tensors(in_features=32, out_features=out_features, group_size=group_size)

    # Small batch x sequence stays on the fused GEMM kernel.
    x = torch.ones((1, 1, qweight.shape[0]), dtype=torch.float16)

    out = gemm_awq_triton.AwqGEMMTritonQuantLinear._Function.apply(
        x, qweight, qzeros, scales, 4, group_size, None, out_features,
    )

    assert calls == {"dequant": 0, "gemm": 1}
    assert out.shape == (1, 1, out_features)


SEQ_LENS = [128, 256, 512, 1024, 1280, 1536, 2048, 4096, 8192]

# Each entry: (case_name, batch, in_features, out_features)
BENCH_CASES = [
    # Llama 3.2-style shapes (hidden=4096)
    ("llama3.2_qkv", 1, 4096, 4096 * 3),
    ("llama3.2_up_proj", 1, 4096, 11008),
    ("llama3.2_down_proj", 1, 11008, 4096),
    # Qwen3-style shapes (hidden≈3584)
    ("qwen3_qkv", 1, 3584, 3584 * 3),
    ("qwen3_up_proj", 1, 3584, 14336),  # 4x hidden for MLP expansion
    ("qwen3_down_proj", 1, 14336, 3584),
]


@pytest.mark.parametrize(
    ("case_name", "batch", "seq", "in_features", "out_features"),
    [(case, batch, seq, inf, outf) for (case, batch, inf, outf) in BENCH_CASES for seq in SEQ_LENS],
    ids=[f"{case}_s{seq}" for (case, _, _, _) in BENCH_CASES for seq in SEQ_LENS],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend required for AWQ benchmark")
def test_fp16_matmul_heuristic_benchmark(case_name, batch, seq, in_features, out_features):
    if os.getenv("RUN_AWQ_FP16_HEURISTIC_BENCH") != "1":
        pytest.skip("Set RUN_AWQ_FP16_HEURISTIC_BENCH=1 to enable this benchmark")

    tabulate = pytest.importorskip("tabulate").tabulate

    backend = None
    if gemm_awq.awq_ext is not None:
        backend = "awq_ext"
    elif gemm_awq.tritonv2.TRITON_AVAILABLE:
        backend = "triton"
    else:
        pytest.skip("No AWQ backend available for benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    group_size = 32

    x = torch.randn((batch, seq, in_features), device=device, dtype=torch.float16)
    qweight = torch.randint(0, 16, (in_features, out_features // 8), device=device, dtype=torch.int32)
    scales = torch.randn((in_features // group_size, out_features), device=device, dtype=torch.float16)
    qzeros = torch.zeros((in_features // group_size, out_features // 8), device=device, dtype=torch.int32)

    if backend == "triton":
        from gptqmodel.quantization.awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton

    def run_dequant_matmul():
        with torch.inference_mode():
            if backend == "awq_ext":
                weight = gemm_awq.awq_ext.dequantize_weights_cuda(qweight, scales, qzeros, 0, 0, 0, False)
            else:
                try:
                    weight = awq_dequantize_triton(qweight, scales, qzeros)
                except AttributeError as err:
                    pytest.skip(f"Triton backend is incompatible: {err}")
            return torch.matmul(x, weight.to(x.dtype))

    def run_fused_gemm():
        with torch.inference_mode():
            x2d = x.reshape(-1, x.shape[-1])
            if backend == "awq_ext":
                return gemm_awq.awq_ext.gemm_forward_cuda(x2d, qweight, scales, qzeros, 8)
            try:
                return awq_gemm_triton(x2d, qweight, scales, qzeros, split_k_iters=8)
            except AttributeError as err:
                pytest.skip(f"Triton backend is incompatible: {err}")

    def benchmark(fn, iters=3):
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e3

    dequant_ms = benchmark(run_dequant_matmul)
    fused_ms = benchmark(run_fused_gemm)

    meets_condition = batch * seq >= 1024
    rows = [
        [case_name, batch, seq, meets_condition, f"{in_features}->{out_features}", "condition=True (dequant+matmul)", f"{dequant_ms:.3f} ms"],
        [case_name, batch, seq, meets_condition, f"{in_features}->{out_features}", "condition=False (fused gemm)", f"{fused_ms:.3f} ms"],
    ]
    print(tabulate(rows, headers=["case", "batch", "seq", "meets >=1024", "matmul (in->out)", "path", "avg latency"]))
