import math
import os
from glob import glob

import torch

from gptqmodel.quantization.config import (
    FailSafe,
    FailSafeStrategy,
    QuantizeConfig,
    SmoothMAD,
    SmoothPercentileAsymmetric,
)
from gptqmodel.quantization.gptq import GPTQ


def _failsafe_quantize(
    weights: torch.Tensor,
    group_size: int,
    strategy: FailSafeStrategy,
    *,
    bits: int = 4,
    sym: bool = False,
    smooth=None,
) -> torch.Tensor:
    module = torch.nn.Linear(weights.shape[1], weights.shape[0], bias=False)
    module = module.to(device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        module.weight.copy_(weights)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        failsafe=FailSafe(strategy=strategy, smooth=smooth),
        offload_to_disk=False,
    )
    gptq = GPTQ(module=module, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    dequant, *_ = gptq._failsafe_quantize(strategy, blocksize=group_size)
    return dequant


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


def _assert_failsafe_bounds(
    label: str,
    weights: torch.Tensor,
    group_size: int,
    rtn_err: float,
    midpoint_err: float,
    mean_err: float,
    median_err: float,
    std_err: float,
    asym_err: float,
) -> None:
    for name, err in (
        ("rtn", rtn_err),
        ("midpoint", midpoint_err),
        ("mean", mean_err),
        ("median", median_err),
        ("stdclip", std_err),
        ("asym", asym_err),
    ):
        assert math.isfinite(err), f"{label}, group={group_size}: {name}_err is not finite ({err})"

    min_val = weights.min().item()
    max_val = weights.max().item()
    one_sided = min_val > 0.0 or max_val < 0.0

    if one_sided:
        floor = rtn_err * 5.0
        assert midpoint_err >= floor, f"{label}, group={group_size}: midpoint_err={midpoint_err}, rtn_err={rtn_err}"
        assert mean_err >= floor, f"{label}, group={group_size}: mean_err={mean_err}, rtn_err={rtn_err}"
        assert median_err >= floor, f"{label}, group={group_size}: median_err={median_err}, rtn_err={rtn_err}"
        assert std_err >= floor, f"{label}, group={group_size}: std_err={std_err}, rtn_err={rtn_err}"
        assert asym_err >= floor, f"{label}, group={group_size}: asym_err={asym_err}, rtn_err={rtn_err}"
    else:
        ceiling = rtn_err * 3.0
        assert midpoint_err <= ceiling, f"{label}, group={group_size}: midpoint_err={midpoint_err}, rtn_err={rtn_err}"
        assert mean_err <= ceiling, f"{label}, group={group_size}: mean_err={mean_err}, rtn_err={rtn_err}"
        assert median_err <= ceiling, f"{label}, group={group_size}: median_err={median_err}, rtn_err={rtn_err}"
        assert std_err <= ceiling, f"{label}, group={group_size}: std_err={std_err}, rtn_err={rtn_err}"
        assert asym_err <= ceiling, f"{label}, group={group_size}: asym_err={asym_err}, rtn_err={rtn_err}"


def _assert_finite_errors(label: str, group_size: int, errors: dict) -> None:
    for name, err in errors.items():
        assert math.isfinite(err), f"{label}, group={group_size}: {name} err is not finite ({err})"


def test_midpoint_vs_rtn_across_distributions():
    scenarios = _scenarios()
    rows = _collect_synthetic_rows(scenarios)
    for scenario_name, group_size, rtn_err, midpoint_err, mean_err, median_err, std_err, asym_err in rows:
        weights = scenarios[scenario_name]
        _assert_failsafe_bounds(
            scenario_name,
            weights,
            group_size,
            rtn_err,
            midpoint_err,
            mean_err,
            median_err,
            std_err,
            asym_err,
        )


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
    rows_mad = []

    for name in targets:
        try:
            w = _load_weight_slice(model_dir, name, max_rows=256, max_cols=256)
        except FileNotFoundError:
            import pytest
            pytest.skip(f"Tensor `{name}` not found in model shards at {model_dir}")

        for group_size in group_sizes:
            rtn = _failsafe_quantize(w, group_size, FailSafeStrategy.RTN)
            mid = _failsafe_quantize(w, group_size, FailSafeStrategy.MIDPOINT)
            mean_c = _failsafe_quantize(w, group_size, FailSafeStrategy.MEAN)
            median_c = _failsafe_quantize(w, group_size, FailSafeStrategy.MEDIAN)
            std_c = _failsafe_quantize(w, group_size, FailSafeStrategy.STDCLIP)
            asym_c = _failsafe_quantize(
                w,
                group_size,
                FailSafeStrategy.MIDPOINT,
                smooth=SmoothPercentileAsymmetric(low=0.5, high=99.5),
            )

            rtn_err = torch.mean((w - rtn).abs()).item()
            mid_err = torch.mean((w - mid).abs()).item()
            mean_err = torch.mean((w - mean_c).abs()).item()
            median_err = torch.mean((w - median_c).abs()).item()
            std_err = torch.mean((w - std_c).abs()).item()
            asym_err = torch.mean((w - asym_c).abs()).item()
            _assert_failsafe_bounds(
                name,
                w,
                group_size,
                rtn_err,
                mid_err,
                mean_err,
                median_err,
                std_err,
                asym_err,
            )
            rows.append((name, group_size, rtn_err, mid_err, mean_err, median_err, std_err, asym_err))

            rtn_mad = _failsafe_quantize(w, group_size, FailSafeStrategy.RTN, smooth=SmoothMAD())
            mid_mad = _failsafe_quantize(w, group_size, FailSafeStrategy.MIDPOINT, smooth=SmoothMAD())
            mean_mad = _failsafe_quantize(w, group_size, FailSafeStrategy.MEAN, smooth=SmoothMAD())
            median_mad = _failsafe_quantize(w, group_size, FailSafeStrategy.MEDIAN, smooth=SmoothMAD())
            std_mad = _failsafe_quantize(w, group_size, FailSafeStrategy.STDCLIP, smooth=SmoothMAD())

            rtn_mad_err = torch.mean((w - rtn_mad).abs()).item()
            mid_mad_err = torch.mean((w - mid_mad).abs()).item()
            mean_mad_err = torch.mean((w - mean_mad).abs()).item()
            median_mad_err = torch.mean((w - median_mad).abs()).item()
            std_mad_err = torch.mean((w - std_mad).abs()).item()
            _assert_finite_errors(
                f"{name} (mad)",
                group_size,
                {
                    "rtn_mad": rtn_mad_err,
                    "mid_mad": mid_mad_err,
                    "mean_mad": mean_mad_err,
                    "median_mad": median_mad_err,
                    "stdclip_mad": std_mad_err,
                },
            )
            rows_mad.append(
                (
                    name,
                    group_size,
                    rtn_mad_err,
                    mid_mad_err,
                    mean_mad_err,
                    median_mad_err,
                    std_mad_err,
                )
            )

    scenarios = _scenarios()
    synthetic_rows = _collect_synthetic_rows(scenarios)
    combined = [("synthetic:" + s, gs, re, me, mne, mde, se, ae) for s, gs, re, me, mne, mde, se, ae in synthetic_rows]
    combined += [("real:" + m, gs, re, me, mne, mde, se, ae) for m, gs, re, me, mne, mde, se, ae in rows]
    native_map = {
        (name, group_size): {
            "rtn": rtn_err,
            "mid": mid_err,
            "mean": mean_err,
            "median": median_err,
            "std": std_err,
        }
        for name, group_size, rtn_err, mid_err, mean_err, median_err, std_err, _ in rows
    }

    header = "+-------------------------------+------------+---------+--------------+--------------+--------------+--------------+--------------+"
    try:
        from logbar import LogBar

        cols = LogBar.shared().columns(
            cols=[
                {"label": "case", "width": "fit"},
                {"label": "group_size", "width": "fit"},
                {"label": "rtn_err", "width": "fit"},
                {"label": "mid_err", "width": "fit"},
                {"label": "mean_err", "width": "fit"},
                {"label": "median_err", "width": "fit"},
                {"label": "stdclip_err", "width": "fit"},
                {"label": "asym_err", "width": "fit"},
            ],
            padding=1,
        )
        cols.info.header()
        for label, gs, rtn_err, mid_err, mean_err, median_err, std_err, asym_err in combined:
            errors = {
                "rtn": rtn_err,
                "mid": mid_err,
                "mean": mean_err,
                "median": median_err,
                "std": std_err,
                "asym": asym_err,
            }
            sorted_methods = sorted(errors.items(), key=lambda kv: kv[1])
            palette = ["\033[32m", "\033[33m", "\033[35m", "\033[34m", "\033[36m", "\033[31m"]
            color_map = {name: palette[min(idx, len(palette) - 1)] for idx, (name, _) in enumerate(sorted_methods)}
            reset = "\033[0m"
            cols.info(
                label,
                str(gs),
                f"{color_map['rtn']}{rtn_err:.5f}{reset}",
                f"{color_map['mid']}{mid_err:.5f}{reset}",
                f"{color_map['mean']}{mean_err:.5f}{reset}",
                f"{color_map['median']}{median_err:.5f}{reset}",
                f"{color_map['std']}{std_err:.5f}{reset}",
                f"{color_map['asym']}{asym_err:.5f}{reset}",
            )
        cols.info.header()
    except Exception:
        print(header)
        print("| case                          | group_size | rtn_err | midpoint_err | mean_err     | median_err   | stdclip_err  | asym_err     |")
        print(header)
        for label, gs, rtn_err, mid_err, mean_err, median_err, std_err, asym_err in combined:
            print(f"| {label:29} | {gs:10d} | {rtn_err:7.5f} | {mid_err:12.5f} | {mean_err:12.5f} | {median_err:12.5f} | {std_err:12.5f} | {asym_err:12.5f} |")
        print(header)

    if rows_mad:
        mad_header = (
            "+-------------------------------+------------+-------------+-------------+-------------+-------------+"
            "-------------+-------------+-------------+-------------+-------------+-------------+-------------+"
        )
        try:
            from logbar import LogBar

            cols = LogBar.shared().columns(
                cols=[
                    {"label": "case (mad)", "width": "fit"},
                    {"label": "group_size", "width": "fit"},
                    {"label": "rtn_mad", "width": "fit"},
                    {"label": "rtn_vs", "width": "fit"},
                    {"label": "mid_mad", "width": "fit"},
                    {"label": "mid_vs", "width": "fit"},
                    {"label": "mean_mad", "width": "fit"},
                    {"label": "mean_vs", "width": "fit"},
                    {"label": "median_mad", "width": "fit"},
                    {"label": "median_vs", "width": "fit"},
                    {"label": "stdclip_mad", "width": "fit"},
                    {"label": "stdclip_vs", "width": "fit"},
                ],
                padding=1,
            )
            cols.info.header()
            for label, gs, rtn_err, mid_err, mean_err, median_err, std_err in rows_mad:
                errors = {
                    "rtn": rtn_err,
                    "mid": mid_err,
                    "mean": mean_err,
                    "median": median_err,
                    "std": std_err,
                }
                sorted_methods = sorted(errors.items(), key=lambda kv: kv[1])
                palette = ["\033[32m", "\033[33m", "\033[35m", "\033[34m", "\033[36m", "\033[31m"]
                color_map = {name: palette[min(idx, len(palette) - 1)] for idx, (name, _) in enumerate(sorted_methods)}
                reset = "\033[0m"
                native = native_map.get((label, gs), {})
                deltas = {
                    "rtn": rtn_err - native.get("rtn", rtn_err),
                    "mid": mid_err - native.get("mid", mid_err),
                    "mean": mean_err - native.get("mean", mean_err),
                    "median": median_err - native.get("median", median_err),
                    "std": std_err - native.get("std", std_err),
                }
                def _color_delta(value: float) -> str:
                    if value > 0:
                        return f"\033[31m{value:+.5f}\033[0m"
                    if value < 0:
                        return f"\033[32m{value:+.5f}\033[0m"
                    return f"{value:+.5f}"

                cols.info(
                    f"mad:{label}",
                    str(gs),
                    f"{color_map['rtn']}{rtn_err:.5f}{reset}",
                    _color_delta(deltas["rtn"]),
                    f"{color_map['mid']}{mid_err:.5f}{reset}",
                    _color_delta(deltas["mid"]),
                    f"{color_map['mean']}{mean_err:.5f}{reset}",
                    _color_delta(deltas["mean"]),
                    f"{color_map['median']}{median_err:.5f}{reset}",
                    _color_delta(deltas["median"]),
                    f"{color_map['std']}{std_err:.5f}{reset}",
                    _color_delta(deltas["std"]),
                )
            cols.info.header()
        except Exception:
            print(mad_header)
            print(
                "| case (mad)                   | group_size | rtn_mad     | rtn_vs      | mid_mad     | mid_vs      | mean_mad    | mean_vs     |"
                " median_mad  | median_vs   | stdclip_mad | stdclip_vs  |"
            )
            print(mad_header)
            for label, gs, rtn_err, mid_err, mean_err, median_err, std_err in rows_mad:
                native = native_map.get((label, gs), {})
                deltas = {
                    "rtn": rtn_err - native.get("rtn", rtn_err),
                    "mid": mid_err - native.get("mid", mid_err),
                    "mean": mean_err - native.get("mean", mean_err),
                    "median": median_err - native.get("median", median_err),
                    "std": std_err - native.get("std", std_err),
                }
                def _color_delta_plain(value: float) -> str:
                    if value > 0:
                        return f"\033[31m{value:+.5f}\033[0m"
                    if value < 0:
                        return f"\033[32m{value:+.5f}\033[0m"
                    return f"{value:+.5f}"

                print(
                    f"| mad:{label:24} | {gs:10d} | {rtn_err:11.5f} | {_color_delta_plain(deltas['rtn']):11} |"
                    f" {mid_err:11.5f} | {_color_delta_plain(deltas['mid']):11} | {mean_err:11.5f} |"
                    f" {_color_delta_plain(deltas['mean']):11} | {median_err:11.5f} | {_color_delta_plain(deltas['median']):11} |"
                    f" {std_err:11.5f} | {_color_delta_plain(deltas['std']):11} |"
                )
            print(mad_header)


def _collect_synthetic_rows(scenarios=None):
    if scenarios is None:
        scenarios = _scenarios()
    rows = []
    for scenario_name, weights in scenarios.items():
        for group_size in (16, 32, 64, 128):
            rtn = _failsafe_quantize(weights, group_size, FailSafeStrategy.RTN)
            midpoint = _failsafe_quantize(weights, group_size, FailSafeStrategy.MIDPOINT)
            mean_centered = _failsafe_quantize(weights, group_size, FailSafeStrategy.MEAN)
            median_centered = _failsafe_quantize(weights, group_size, FailSafeStrategy.MEDIAN)
            std_clip = _failsafe_quantize(weights, group_size, FailSafeStrategy.STDCLIP)
            asym_clip = _failsafe_quantize(
                weights,
                group_size,
                FailSafeStrategy.MIDPOINT,
                smooth=SmoothPercentileAsymmetric(low=0.5, high=99.5),
            )

            rtn_err = torch.mean((weights - rtn).abs()).item()
            midpoint_err = torch.mean((weights - midpoint).abs()).item()
            mean_err = torch.mean((weights - mean_centered).abs()).item()
            median_err = torch.mean((weights - median_centered).abs()).item()
            std_err = torch.mean((weights - std_clip).abs()).item()
            asym_err = torch.mean((weights - asym_clip).abs()).item()
            rows.append((scenario_name, group_size, rtn_err, midpoint_err, mean_err, median_err, std_err, asym_err))
    return rows
