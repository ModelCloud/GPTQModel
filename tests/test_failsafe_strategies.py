import os
from glob import glob

import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer


def _quantize_rtn(weights: torch.Tensor, group_size: int, bits: int = 4) -> torch.Tensor:
    qcfg = QuantizeConfig(bits=bits, group_size=group_size, sym=False)
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)

    out = torch.empty_like(weights)

    for start in range(0, weights.shape[1], group_size):
        end = min(start + group_size, weights.shape[1])
        block = weights[:, start:end]
        quantizer.find_params(block, weight=True)
        q_block = quantizer.quantize(block)
        # Clip to valid range since quantizer.find_params uses per-channel layout.
        q_block = torch.clamp(q_block, block.min(), block.max())
        out[:, start:end] = q_block
    return out


def _quantize_midpoint(weights: torch.Tensor, group_size: int, bits: int = 4) -> torch.Tensor:
    maxq = 2 ** bits - 1
    out = torch.empty_like(weights)
    for start in range(0, weights.shape[1], group_size):
        end = min(start + group_size, weights.shape[1])
        block = weights[:, start:end]
        w_min = block.min(dim=1, keepdim=True).values
        w_max = block.max(dim=1, keepdim=True).values
        mid = (w_max + w_min) / 2.0
        scale = (w_max - w_min) / maxq
        zero = torch.full_like(scale, maxq / 2.0)
        q = torch.round((block - mid) / scale + zero)
        q = torch.clamp(q, 0, maxq)
        dequant = (q - zero) * scale + mid
        out[:, start:end] = dequant
    return out


def _quantize_mean_centered(weights: torch.Tensor, group_size: int, bits: int = 4) -> torch.Tensor:
    maxq = 2 ** bits - 1
    out = torch.empty_like(weights)
    for start in range(0, weights.shape[1], group_size):
        end = min(start + group_size, weights.shape[1])
        block = weights[:, start:end]
        mean = block.mean(dim=1, keepdim=True)
        max_dev = torch.max((block - mean).abs(), dim=1, keepdim=True).values
        max_dev = torch.clamp(max_dev, min=1e-6)
        scale = (2 * max_dev) / maxq
        zero = torch.full_like(scale, maxq / 2.0)
        q = torch.round((block - mean) / scale + zero)
        q = torch.clamp(q, 0, maxq)
        dequant = (q - zero) * scale + mean
        out[:, start:end] = dequant
    return out


def _quantize_std_clipped(weights: torch.Tensor, group_size: int, bits: int = 4, sigma: float = 3.0) -> torch.Tensor:
    maxq = 2 ** bits - 1
    out = torch.empty_like(weights)
    for start in range(0, weights.shape[1], group_size):
        end = min(start + group_size, weights.shape[1])
        block = weights[:, start:end]
        mean = block.mean(dim=1, keepdim=True)
        std = block.std(dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        lo = mean - sigma * std
        hi = mean + sigma * std
        scale = (hi - lo) / maxq
        zero = torch.round(-lo / scale)
        q = torch.round(block / scale + zero)
        q = torch.clamp(q, 0, maxq)
        dequant = (q - zero) * scale
        out[:, start:end] = dequant
    return out


def _scenarios():
    torch.manual_seed(0)
    base = torch.randn(8, 32)
    return {
        "centered_pos": base * 0.1 + 2.5,      # cluster away from zero
        "zero_centered": base * 0.1,           # symmetric around zero
        "pos_lean": base * 0.05 + 0.5,         # small positive leaning
        "neg_lean": base * 0.05 - 0.5,         # small negative leaning
        "wide_range": base * 2.5,              # wider spread / outliers
        "uniform": torch.linspace(-1, 1, 32).repeat(8, 1),  # evenly spread
    }


def test_midpoint_vs_rtn_across_distributions():
    rows = _collect_synthetic_rows()
    for scenario_name, group_size, rtn_err, midpoint_err, mean_err, std_err in rows:
        assert midpoint_err <= rtn_err, f"{scenario_name}, group={group_size}: midpoint_err={midpoint_err}, rtn_err={rtn_err}"
        assert mean_err <= rtn_err * 1.10, f"{scenario_name}, group={group_size}: mean_err={mean_err}, rtn_err={rtn_err}"
        # std-clip intentionally biases to avoid outliers; allow generous headroom while keeping it bounded.
        assert std_err <= rtn_err * 2.50, f"{scenario_name}, group={group_size}: std_err={std_err}, rtn_err={rtn_err}"


def _load_weight_slice(model_dir: str, tensor_name: str, *, max_rows: int = 256, max_cols: int = 256) -> torch.Tensor:
    from safetensors import safe_open

    shards = sorted(glob(os.path.join(model_dir, "model-*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensor shards found under {model_dir}")

    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                return tensor[:max_rows, :max_cols].clone()
    raise FileNotFoundError(f"Tensor `{tensor_name}` not found in {model_dir}")


def test_midpoint_vs_rtn_on_qwen3_real_weights():
    model_dir = "/monster/data/model/Qwen3-30B-A3B"
    if not os.path.isdir(model_dir):
        import pytest
        pytest.skip(f"Model path missing: {model_dir}")

    targets = [
        "model.layers.0.mlp.experts.10.up_proj.weight",
        "model.layers.0.mlp.experts.10.down_proj.weight",
        "model.layers.0.mlp.experts.10.gate_proj.weight",
    ]
    group_sizes = (32, 64, 128)
    rows = []

    for name in targets:
        try:
            w = _load_weight_slice(model_dir, name, max_rows=256, max_cols=256)
        except FileNotFoundError:
            import pytest
            pytest.skip(f"Tensor `{name}` not found in model shards at {model_dir}")

        for group_size in group_sizes:
            rtn = _quantize_rtn(w, group_size=group_size)
            mid = _quantize_midpoint(w, group_size=group_size)
            mean_c = _quantize_mean_centered(w, group_size=group_size)
            std_c = _quantize_std_clipped(w, group_size=group_size, sigma=3.0)

            rtn_err = torch.mean((w - rtn).abs()).item()
            mid_err = torch.mean((w - mid).abs()).item()
            mean_err = torch.mean((w - mean_c).abs()).item()
            std_err = torch.mean((w - std_c).abs()).item()
            rows.append((name, group_size, rtn_err, mid_err, mean_err, std_err))

    synthetic_rows = _collect_synthetic_rows()
    combined = [("synthetic:" + s, gs, re, me, mne, se) for s, gs, re, me, mne, se in synthetic_rows]
    combined += [("real:" + m, gs, re, me, mne, se) for m, gs, re, me, mne, se in rows]

    header = "+-------------------------------+------------+---------+--------------+--------------+--------------+"
    try:
        from logbar import LogBar

        cols = LogBar.shared().columns(
            cols=[
                {"label": "case", "width": "fit"},
                {"label": "group_size", "width": "fit"},
                {"label": "rtn_err", "width": "fit"},
                {"label": "mid_err", "width": "fit"},
                {"label": "mean_err", "width": "fit"},
                {"label": "stdclip_err", "width": "fit"},
            ],
            padding=1,
        )
        cols.info.header()
        for label, gs, rtn_err, mid_err, mean_err, std_err in combined:
            errors = {"rtn": rtn_err, "mid": mid_err, "mean": mean_err, "std": std_err}
            sorted_methods = sorted(errors.items(), key=lambda kv: kv[1])
            palette = ["\033[32m", "\033[33m", "\033[35m", "\033[31m"]
            color_map = {name: palette[min(idx, len(palette) - 1)] for idx, (name, _) in enumerate(sorted_methods)}
            reset = "\033[0m"
            cols.info(
                label,
                str(gs),
                f"{color_map['rtn']}{rtn_err:.5f}{reset}",
                f"{color_map['mid']}{mid_err:.5f}{reset}",
                f"{color_map['mean']}{mean_err:.5f}{reset}",
                f"{color_map['std']}{std_err:.5f}{reset}",
            )
        cols.info.header()
    except Exception:
        print(header)
        print("| case                          | group_size | rtn_err | midpoint_err | mean_err     | stdclip_err  |")
        print(header)
        for label, gs, rtn_err, mid_err, mean_err, std_err in combined:
            print(f"| {label:29} | {gs:10d} | {rtn_err:7.5f} | {mid_err:12.5f} | {mean_err:12.5f} | {std_err:12.5f} |")
        print(header)

    for module, group_size, rtn_err, mid_err, mean_err, std_err in rows:
        assert mid_err <= rtn_err, f"{module}, group={group_size}: midpoint_err={mid_err}, rtn_err={rtn_err}"
        assert mean_err <= rtn_err * 1.20, f"{module}, group={group_size}: mean_err={mean_err}, rtn_err={rtn_err}"
        assert std_err <= rtn_err * 2.50, f"{module}, group={group_size}: std_err={std_err}, rtn_err={rtn_err}"


def _collect_synthetic_rows():
    rows = []
    for scenario_name, weights in _scenarios().items():
        for group_size in (16, 32, 64, 128):
            rtn = _quantize_rtn(weights, group_size=group_size)
            midpoint = _quantize_midpoint(weights, group_size=group_size)
            mean_centered = _quantize_mean_centered(weights, group_size=group_size)
            std_clip = _quantize_std_clipped(weights, group_size=group_size, sigma=3.0)

            rtn_err = torch.mean((weights - rtn).abs()).item()
            midpoint_err = torch.mean((weights - midpoint).abs()).item()
            mean_err = torch.mean((weights - mean_centered).abs()).item()
            std_err = torch.mean((weights - std_clip).abs()).item()
            rows.append((scenario_name, group_size, rtn_err, midpoint_err, mean_err, std_err))
    return rows
