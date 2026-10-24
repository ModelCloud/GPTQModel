import pytest
import torch
from parameterized import parameterized
from torch.utils._python_dispatch import TorchDispatchMode

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


_GRID_DEVICES = ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)]


def _sequential_clip_grid(qcfg, w, input_feat, n_grid=20, max_shrink=0.5):
    """Independent scalar-grid oracle, including low-precision arithmetic."""
    group_size = qcfg.group_size if qcfg.group_size > 0 else w.shape[1]
    x = input_feat.reshape(1, -1, w.shape[1] // group_size, group_size)
    w = w.reshape(w.shape[0], 1, -1, group_size)
    original_bound = w.abs().amax(dim=-1, keepdim=True)
    best_bound = original_bound.clone()
    best_error = torch.full_like(original_bound, float("inf"))
    original_output = (x * w).sum(-1)
    for step in range(int(n_grid * max_shrink)):
        bound = original_bound * (1 - step / n_grid)
        clipped = torch.clamp(w, -bound, bound)
        if qcfg.sym:
            scale = clipped.abs().amax(-1, keepdim=True).clamp(min=1e-5)
            scale = scale / (2 ** (qcfg.bits - 1) - 1)
            quantized = torch.round(clipped / scale).clamp(
                -(2 ** (qcfg.bits - 1)), 2 ** (qcfg.bits - 1) - 1,
            ) * scale
        else:
            low = clipped.amin(-1, keepdim=True)
            high = clipped.amax(-1, keepdim=True)
            maxq = 2 ** qcfg.bits - 1
            scale = (high - low).clamp(min=1e-5) / maxq
            zero = (-torch.round(low / scale)).clamp(0, maxq)
            quantized = (torch.clamp(torch.round(clipped / scale) + zero, 0, maxq) - zero) * scale
        output = (x * quantized).sum(-1)
        error = (output - original_output).pow(2).mean(1).reshape_as(best_error)
        better = error < best_error
        best_bound = torch.where(better, bound, best_bound)
        best_error = torch.where(better, error, best_error)
    return best_bound.squeeze(1)


@pytest.mark.parametrize("device", _GRID_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("group_size", [-1, 32])
@pytest.mark.parametrize("budget", [0, 1])
def test_clip_grid_low_precision_oracle(device, dtype, sym, group_size, budget):
    g = torch.Generator().manual_seed(42)
    w = torch.randn(64, 128, generator=g).to(device=device, dtype=dtype)
    x = torch.randn(19, 128, generator=g).to(device=device, dtype=dtype)
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=group_size, sym=sym,
        mse={"clip_grid_budget": budget},
    )
    processor = _ClipTestAWQProcessor(qcfg)
    expected = _sequential_clip_grid(qcfg, w, x)
    actual = processor._compute_best_clip(w, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", _GRID_DEVICES)
@pytest.mark.parametrize("kind", ["empty_grid", "ties", "overflow", "mixed_dtype"])
def test_clip_grid_edge_cases(device, kind):
    g = torch.Generator().manual_seed(42)
    w = torch.randn(128, 64, generator=g).to(device=device, dtype=torch.float16)
    x = torch.randn(7, 64, generator=g).to(device=device, dtype=torch.float16)
    shrink = 0.0 if kind == "empty_grid" else 0.5
    if kind == "ties":
        x.zero_()
    elif kind == "overflow":
        x.mul_(1e4)
    elif kind == "mixed_dtype":
        x = x.float()
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
        mse={"clip_grid_budget": 0},
    )
    processor = _ClipTestAWQProcessor(qcfg)
    expected = _sequential_clip_grid(qcfg, w, x, max_shrink=shrink)
    actual = processor._compute_best_clip(w, x, max_shrink=shrink)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_clip_grid_slices_reference_product():
    """The unclipped reference must obey the same token tiling as candidates."""
    products = []

    class TrackProducts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if func in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.out) and output.ndim >= 4:
                products.append(output.numel())
            return output

    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
        mse={"clip_grid_budget": 0},
    )
    processor = _ClipTestAWQProcessor(qcfg)
    w = torch.ones(64, 128)
    x = torch.ones(19, 128)
    with TrackProducts():
        processor._compute_best_clip(w, x)
    assert products
    assert max(products) <= w.numel()


class _ClipTestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: QuantizeConfig) -> None:
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            gptq_model=None,
            model=None,
            require_fwd=True,
            calculate_w_wq_diff=False,
            calibration_concat_separator=None,
        )

    def _module_forward(self, x, module, module_kwargs):
        return module(x)


def _legacy_clip(processor: AWQProcessor, w: torch.Tensor, input_feat: torch.Tensor):
    group_size = processor.qcfg.group_size if processor.qcfg.group_size > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    step_size = max(1, input_feat.shape[1] // 512)
    input_feat = input_feat[:, ::step_size]

    w = w.reshape(w.shape[0], 1, -1, group_size)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
    assert w.shape[0] % oc_batch_size == 0
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w_chunk = w[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]
        org_max_val = w_chunk.abs().amax(dim=-1, keepdim=True)
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w_chunk.device)
        org_out = (input_feat * w_chunk).sum(dim=-1)
        for i_s in range(int(0.5 * 20)):
            max_val = org_max_val * (1 - i_s / 20)
            min_val = -max_val
            cur_w = torch.clamp(w_chunk, min_val, max_val)
            q_w = processor.pseudo_quantize_tensor(cur_w)[0]
            cur_out = (input_feat * q_w).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)
    return torch.cat(best_max_val_all, dim=0).squeeze(1)


@parameterized.expand([
    ("cpu", "cpu"),
    ("cuda", "cuda:0"),
])
def test_awq_clip_consistency(device_name: str, device_str: str):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available for clip consistency test")

    dtype = torch.float32 if device_name == "cpu" else torch.float16
    processor = _ClipTestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))

    out_features = 256
    in_features = 3584
    w = torch.randn(out_features, in_features, dtype=dtype, device=device_str)
    tokens = 1024
    input_feat = torch.randn(tokens, in_features, dtype=dtype, device=device_str)

    # Compare the streaming implementation against the legacy tensor-per-iter path
    expected = _legacy_clip(processor, w.clone(), input_feat.clone())
    actual = processor._compute_best_clip(w, input_feat)

    tol = 1e-6 if dtype == torch.float32 else 1e-4
    assert torch.allclose(actual.cpu(), expected.cpu(), atol=tol, rtol=tol), \
        f"Inconsistent clip: max diff {(actual - expected).abs().max().item():.3e}"


@parameterized.expand([
    ("cpu", "cpu"),
    ("cuda", "cuda:0"),
])
def test_awq_clip_bound_fast_path_matches_derived(device_name: str, device_str: str):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available for the clip bound fast path test")

    dtype = torch.float32 if device_name == "cpu" else torch.float16
    processor = _ClipTestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))

    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=dtype, device=device_str).reshape(-1, 128)
    bound = w.abs().amax(dim=1, keepdim=True)

    # `_compute_best_clip` clamps the block itself and then hands the quantizer the
    # bound it already knows, instead of letting it re-derive `abs().amax()`. The
    # supplied bound must reproduce the derived one exactly, for every clip ratio.
    for shrink in (1.0, 0.5, 0.25):
        limit = bound * shrink
        src = w.clamp(-limit, limit)
        derived = torch.empty_like(src)
        processor._pseudo_quantize_tensor_into(src.clone(), derived)
        provided = torch.empty_like(src)
        processor._pseudo_quantize_tensor_into(src.clone(), provided, abs_max=limit)
        assert torch.equal(derived.cpu(), provided.cpu()), (
            f"Bound fast path diverged at shrink {shrink}: "
            f"max diff {(derived - provided).abs().max().item():.3e}"
        )
