import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer


def _quantize_rtn(weights: torch.Tensor, group_size: int, bits: int = 4) -> torch.Tensor:
    qcfg = QuantizeConfig(bits=bits, group_size=group_size, sym=False)
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)

    out = torch.empty_like(weights)
    maxq = 2 ** bits - 1

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

            rtn_err = torch.mean((weights - rtn).abs()).item()
            midpoint_err = torch.mean((weights - midpoint).abs()).item()
            rows.append((scenario_name, group_size, rtn_err, midpoint_err))

    header = "+----------------+------------+---------+--------------+"
    print(header)
    print("| scenario       | group_size | rtn_err | midpoint_err |")
    print(header)
    for scenario_name, group_size, rtn_err, midpoint_err in rows:
        print(f"| {scenario_name:14} | {group_size:10d} | {rtn_err:7.5f} | {midpoint_err:12.5f} |")
    print(header)

    for scenario_name, group_size, rtn_err, midpoint_err in rows:
        assert midpoint_err <= rtn_err, f"{scenario_name}, group={group_size}: midpoint_err={midpoint_err}, rtn_err={rtn_err}"
