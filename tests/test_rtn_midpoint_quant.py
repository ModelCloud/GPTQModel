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

    header = "+----------------+------------+---------+--------------+--------------+--------------+"
    try:
        from logbar import LogBar

        cols = LogBar.shared().columns(
            cols=[
                {"label": "scenario", "width": "fit"},
                {"label": "group_size", "width": "fit"},
                {"label": "rtn_err", "width": "fit"},
                {"label": "midpoint_err", "width": "fit"},
                {"label": "mean_err", "width": "fit"},
                {"label": "stdclip_err", "width": "fit"},
            ],
            padding=1,
        )
        cols.info.header()
        for scenario_name, group_size, rtn_err, midpoint_err, mean_err, std_err in rows:
            errors = {
                "rtn": rtn_err,
                "mid": midpoint_err,
                "mean": mean_err,
                "std": std_err,
            }
            sorted_methods = sorted(errors.items(), key=lambda kv: kv[1])
            palette = ["\033[32m", "\033[33m", "\033[35m", "\033[31m"]
            color_map = {name: palette[min(idx, len(palette) - 1)] for idx, (name, _) in enumerate(sorted_methods)}
            reset = "\033[0m"
            cols.info(
                scenario_name,
                str(group_size),
                f"{color_map['rtn']}{rtn_err:.5f}{reset}",
                f"{color_map['mid']}{midpoint_err:.5f}{reset}",
                f"{color_map['mean']}{mean_err:.5f}{reset}",
                f"{color_map['std']}{std_err:.5f}{reset}",
            )
        cols.info.header()
    except Exception:
        # Fallback to plain ASCII print if LogBar is unavailable
        print(header)
        print("| scenario       | group_size | rtn_err | midpoint_err | mean_err     | stdclip_err  |")
        print(header)
        for scenario_name, group_size, rtn_err, midpoint_err, mean_err, std_err in rows:
            print(f"| {scenario_name:14} | {group_size:10d} | {rtn_err:7.5f} | {midpoint_err:12.5f} | {mean_err:12.5f} | {std_err:12.5f} |")
        print(header)

    for scenario_name, group_size, rtn_err, midpoint_err, mean_err, std_err in rows:
        assert midpoint_err <= rtn_err, f"{scenario_name}, group={group_size}: midpoint_err={midpoint_err}, rtn_err={rtn_err}"
        assert mean_err <= rtn_err * 1.10, f"{scenario_name}, group={group_size}: mean_err={mean_err}, rtn_err={rtn_err}"
        # std-clip intentionally biases to avoid outliers; allow generous headroom while keeping it bounded.
        assert std_err <= rtn_err * 2.50, f"{scenario_name}, group={group_size}: std_err={std_err}, rtn_err={rtn_err}"
