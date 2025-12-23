# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

import torch
from logbar import LogBar
from transformers import AutoModelForCausalLM

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import (
    FailSafe,
    FailSafeStrategy,
    QuantizeConfig,
    SmoothLog,
    SmoothMAD,
    SmoothMSE,
    SmoothOutlier,
    SmoothPercentile,
    SmoothRowCol,
    SmoothSoftNorm,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module


MODEL_DIR = "/monster/data/model/Llama-3.2-1B-Instruct" # "/monster/data/model/llama3-8B" #

log = LogBar.shared()

DEVICE = torch.device("cuda:0")
ATOL_CHECKS = [0.001, 0.005, 0.01, 0.05, 0.1]


def _load_down_proj(dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    down_proj = model.model.layers[0].mlp.down_proj
    down_proj = down_proj.to(device=device, dtype=dtype)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return down_proj


def _quantize_to_torch_linear(
    layer: torch.nn.Module,
    failsafe: FailSafe,
    device: torch.device,
) -> TorchQuantLinear:
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        failsafe=failsafe,
    )

    gptq = GPTQ(layer, qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.failsafe = qcfg.failsafe

    wq, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=128)

    packed_linear = torch.nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        device="cpu",
        dtype=layer.weight.dtype,
    )
    packed_linear.weight.data = wq.detach().to(device="cpu", dtype=layer.weight.dtype)
    if layer.bias is not None:
        packed_linear.bias.data = layer.bias.detach().to(device="cpu", dtype=layer.bias.dtype)

    qlinear = TorchQuantLinear(
        bits=qcfg.bits,
        group_size=qcfg.group_size,
        sym=qcfg.sym,
        desc_act=qcfg.desc_act,
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        pack_dtype=qcfg.pack_dtype,
        adapter=None,
    )
    qlinear.pack(
        linear=packed_linear,
        scales=scales.to(device="cpu"),
        zeros=zeros.to(device="cpu"),
        g_idx=g_idx.to(device="cpu"),
    )
    convert_gptq_v1_to_v2_format_module(
        module=qlinear,
        bits=qcfg.bits,
        pack_dtype=qcfg.pack_dtype,
    )
    qlinear = qlinear.to(device=device)
    qlinear.post_init()
    qlinear.eval()
    return qlinear


def _clone_linear(layer: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    cloned = torch.nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        device=device,
        dtype=layer.weight.dtype,
    )
    cloned.weight.data.copy_(layer.weight.data)
    if layer.bias is not None:
        cloned.bias.data.copy_(layer.bias.data)
    return cloned


def _init_stats():
    return {
        "sum": 0.0,
        "count": 0,
        "max": None,
        "min": None,
        "passes": dict.fromkeys(ATOL_CHECKS, 0),
    }


def _update_stats(stats, diff: torch.Tensor):
    diff_sum = diff.sum().item()
    diff_max = diff.max().item()
    diff_min = diff.min().item()
    for atol in ATOL_CHECKS:
        stats["passes"][atol] += torch.count_nonzero(diff <= atol).item()
    stats["sum"] += diff_sum
    stats["count"] += diff.numel()
    stats["max"] = diff_max if stats["max"] is None else max(stats["max"], diff_max)
    stats["min"] = diff_min if stats["min"] is None else min(stats["min"], diff_min)


def _finalize_stats(stats):
    mean = stats["sum"] / max(stats["count"], 1)
    pass_rates = {
        atol: stats["passes"][atol] / max(stats["count"], 1)
        for atol in ATOL_CHECKS
    }
    return mean, stats["max"] or 0.0, stats["min"] or 0.0, pass_rates


def _parse_shapes(expr: str):
    shapes = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        dim_str, samples_str = part.split(":", 1)
        shapes.append((int(dim_str), int(samples_str)))
    return shapes


def _select_shapes():
    large_shapes = [(1, 256), (16, 128), (32, 64), (64, 32), (128, 16)]
    medium_shapes = [(1, 128), (16, 64), (32, 32), (64, 16)]
    small_shapes = [(1, 64), (8, 32), (16, 16)]

    env_shapes = os.getenv("GPTQMODEL_KERNEL_TEST_SHAPES")
    if env_shapes:
        return _parse_shapes(env_shapes)

    total_mem_gb = 0.0
    if torch.cuda.is_available():
        device_index = DEVICE.index if DEVICE.index is not None else 0
        try:
            if torch.cuda.device_count() > device_index:
                props = torch.cuda.get_device_properties(device_index)
                total_mem_gb = props.total_memory / (1024 ** 3)
        except Exception:
            total_mem_gb = 0.0

    if os.getenv("GPTQMODEL_FAST_TESTS", "0") == "1":
        return small_shapes
    if total_mem_gb >= 80:
        return large_shapes
    if total_mem_gb >= 48:
        return medium_shapes
    return small_shapes


def test_kernel_output_failsafe():
    if not os.path.isdir(MODEL_DIR):
        import pytest

        pytest.skip(f"Model path missing: {MODEL_DIR}")

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required for failsafe kernel output test")

    torch.manual_seed(0)

    device = DEVICE
    dtype = torch.float16
    down_proj = _load_down_proj(dtype=dtype, device=device)
    assert down_proj.weight.device.type == "cuda"

    shapes = _select_shapes()
    variants = [
        ("rtn", FailSafe(strategy=FailSafeStrategy.RTN, threshold=True)),
        ("midpoint", FailSafe(strategy=FailSafeStrategy.MIDPOINT, threshold=True)),
        ("mean", FailSafe(strategy=FailSafeStrategy.MEAN, threshold=True)),
        ("stdclip", FailSafe(strategy=FailSafeStrategy.STDCLIP, threshold=True)),
        ("rtn_p99", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothPercentile(percentile=99.0),
        )),
        ("rtn_mad", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothMAD(k=3.0),
        )),
        ("rtn_mse", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothMSE(steps=32, maxshrink=0.8),
        )),
        ("rtn_outlier", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothOutlier(pct=1.0),
        )),
        ("rtn_softnorm", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothSoftNorm(k=3.0),
        )),
        ("rtn_log", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothLog(percentile=99.0, mu=8.0),
        )),
        ("rtn_rowcol", FailSafe(
            strategy=FailSafeStrategy.RTN,
            threshold=True,
            smooth=SmoothRowCol(axis="row"),
        )),
    ]
    qlinears = {
        label: _quantize_to_torch_linear(_clone_linear(down_proj, device=device), failsafe, device=device)
        for label, failsafe in variants
    }
    for label, qlinear in qlinears.items():
        assert qlinear.list_buffers()[0].device.type == "cuda", f"{label} buffers not on CUDA"

    total_samples = sum(samples for _, samples in shapes)
    stats = {label: _init_stats() for label, _ in variants}
    with torch.inference_mode():
        for _ in log.pb(total_samples).title("Forward Pass on Random Input"):
            for dim_0, samples in shapes:
                for _ in range(samples):
                    x = torch.randn(
                        (dim_0, down_proj.in_features),
                        device=device,
                        dtype=dtype,
                    )
                    assert x.device.type == "cuda"
                    baseline = down_proj(x)
                    variant_out = {label: qlinears[label](x) for label, _ in variants}
                    assert baseline.device.type == "cuda"
                    for label, out in variant_out.items():
                        assert out.device.type == "cuda"
                        diff = torch.abs(baseline - out).float()
                        _update_stats(stats[label], diff)

    finalized = {}
    for label, _ in variants:
        mean_val, max_val, min_val, pass_rates = _finalize_stats(stats[label])
        finalized[label] = {
            "mean": mean_val,
            "max": max_val,
            "min": min_val,
            "pass": pass_rates,
            "atol": max_val,
        }

    pass_cols = [{"label": f"pass@{atol:g}", "width": "fit"} for atol in ATOL_CHECKS]
    cols = log.columns(
        cols=[
            {"label": "variant", "width": "fit"},
            {"label": "mean_diff", "width": "fit"},
            {"label": "max_diff", "width": "fit"},
            {"label": "min_diff", "width": "fit"},
            {"label": "atol_req", "width": "fit"},
        ] + pass_cols,
        padding=1,
    )
    cols.info.header()
    for label, _ in variants:
        metrics = finalized[label]
        cols.info(
            label,
            f"{metrics['mean']:.6f}",
            f"{metrics['max']:.6f}",
            f"{metrics['min']:.6f}",
            f"{metrics['atol']:.6f}",
            *[f"{metrics['pass'][atol]:.4f}" for atol in ATOL_CHECKS],
        )
    cols.info.header()

    for label, _ in variants:
        metrics = finalized[label]
        assert torch.isfinite(torch.tensor([metrics["mean"], metrics["max"], metrics["min"]])).all()
