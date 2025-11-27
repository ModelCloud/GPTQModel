# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqQuantLinear
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


CHECKPOINT_DIR = Path("/monster/data/model/deepseek-r1-distill-qwen-7b-awq")
CHECKPOINT_MODULE = os.environ.get(
    "GPTQMODEL_AWQ_TEST_MODULE", "model.layers.0.mlp.up_proj"
)


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache(maxsize=1)
def _load_awq_checkpoint_module():
    if not CHECKPOINT_DIR.exists():
        pytest.skip(f"AWQ checkpoint not available at {CHECKPOINT_DIR}")

    index_path = CHECKPOINT_DIR / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(f"Missing model index at {index_path}")

    with index_path.open("r", encoding="utf-8") as fh:
        index_data = json.load(fh)
    weight_map = index_data["weight_map"]

    config_path = CHECKPOINT_DIR / "config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    quant_cfg = config.get("quantization_config", {})
    bits = int(quant_cfg.get("bits", 4))
    group_size = int(quant_cfg.get("group_size", 128))

    suffixes = ["qweight", "qzeros", "scales", "bias"]
    tensors = {}
    file_to_keys = {}
    for suffix in suffixes:
        full_key = f"{CHECKPOINT_MODULE}.{suffix}"
        filename = weight_map.get(full_key)
        if filename is None:
            if suffix == "bias":
                continue
            raise KeyError(f"Missing tensor '{full_key}' in checkpoint index.")
        file_to_keys.setdefault(filename, []).append(full_key)

    for filename, keys in file_to_keys.items():
        tensor_path = CHECKPOINT_DIR / filename
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key).clone()

    qweight = tensors[f"{CHECKPOINT_MODULE}.qweight"].to(torch.int32).contiguous()
    qzeros = tensors[f"{CHECKPOINT_MODULE}.qzeros"].to(torch.int32).contiguous()
    scales = tensors[f"{CHECKPOINT_MODULE}.scales"].to(torch.float16).contiguous()
    bias_key = f"{CHECKPOINT_MODULE}.bias"
    bias = tensors.get(bias_key)
    if bias is not None:
        bias = bias.to(torch.float16).contiguous()

    pack_factor = 32 // bits
    in_features = qweight.shape[0]
    out_features = qweight.shape[1] * pack_factor

    return {
        "bits": bits,
        "group_size": group_size,
        "in_features": in_features,
        "out_features": out_features,
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "bias": bias,
    }


@pytest.mark.skipif(not TORCH_HAS_FUSED_OPS, reason="Torch fused ops require PyTorch>=2.8")
@pytest.mark.parametrize(
    "device_str",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "xpu:0",
            id="xpu",
            marks=pytest.mark.skipif(
                not _xpu_available(), reason="Torch fused AWQ XPU test requires Intel XPU runtime."
            ),
        ),
    ],
)
def test_torch_fused_awq_matches_checkpoint_module(device_str: str):
    module_data = _load_awq_checkpoint_module()
    bits = module_data["bits"]
    group_size = module_data["group_size"]
    in_features = module_data["in_features"]
    out_features = module_data["out_features"]
    qweight = module_data["qweight"]
    qzeros = module_data["qzeros"]
    scales = module_data["scales"]
    bias = module_data["bias"]

    device = torch.device(device_str)

    awq_module = AwqTorchQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
        register_buffers=True,
    )
    fused_module = TorchFusedAwqQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias is not None,
        register_buffers=True,
    )

    awq_module.qweight.copy_(qweight)
    awq_module.qzeros.copy_(qzeros)
    awq_module.scales.copy_(scales)
    if bias is not None:
        awq_module.bias.copy_(bias)
    awq_module.post_init()
    awq_module.eval()

    fused_module.register_buffer("qweight", qweight.clone(), persistent=True)
    fused_module.qzeros.copy_(qzeros)
    fused_module.scales.copy_(scales)
    if bias is not None:
        fused_module.bias.copy_(bias)
    fused_module.post_init()
    fused_module.eval()

    awq_module.to(device)
    fused_module.to(device)

    dtype = torch.float16
    batch = 4
    x = torch.randn(batch, in_features, dtype=dtype, device=device)
    baseline = awq_module(x)
    fused_out = fused_module(x)

    rtol = 5e-3
    atol = 5e-3
    abs_diff = (fused_out - baseline).abs()
    rel_diff = abs_diff / baseline.abs().clamp_min(1e-6)
    summary = tabulate(
        [
            [
                device_str,
                str(dtype),
                f"{rtol:.4g}",
                f"{atol:.4g}",
                f"{abs_diff.max().item():.4e}",
                f"{rel_diff.max().item():.4e}",
            ]
        ],
        headers=["Device", "DType", "RTol", "ATol", "AbsMaxDiff", "RelMaxDiff"],
        tablefmt="github",
    )
    print(summary)
    torch.testing.assert_close(fused_out, baseline, rtol=rtol, atol=atol)
