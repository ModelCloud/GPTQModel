import math
import os
import types
import unittest
from glob import glob
from types import SimpleNamespace

import torch
import torch.nn as nn
from module_tree.test_subset import _StubAWQProcessor

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    FailSafe,
    FailSafeStrategy,
    QuantizeConfig,
    SmoothMAD,
    SmoothPercentileAsymmetric,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.failsafe import should_use_failsafe
from gptqmodel.utils.pause_resume import PauseResumeController


class TestGPTQHessianSimilarity(unittest.TestCase):
    """
    This test verifies that Hessian-based GPTQ produces quantized weights
    that remain numerically close to RTN fallback, while still introducing
    minimal corrective differences.

    The test intentionally checks *similarity*, not equality.
    """

    def _run_test(self, device: str):
        torch.manual_seed(0)

        # Large dimensions are intentionally used to:
        # - avoid degenerate small-layer behavior
        # - amplify Hessian effects in a stable manner
        in_features = 1024
        out_features = 2048
        batch = 4
        seq = 16

        inp = torch.randn(batch, seq, in_features, device=device)
        linear = nn.Linear(in_features, out_features, bias=False).to(device)

        qcfg = QuantizeConfig(
            bits=4,
            group_size=128,
            failsafe={"strategy": "rtn", "threshold": False},
        )

        # ============================================================
        # Hessian-based GPTQ (use_hessian = True)
        # ============================================================
        gptq_h = GPTQ(linear, qcfg)
        gptq_h.quantizer.configure(perchannel=True)
        gptq_h.failsafe = False

        # Accumulate Hessian via the public API
        gptq_h.add_batch(inp, None)

        Q_h, scale_h, zero_h, gidx_h, *_ = gptq_h.quantize()

        # ============================================================
        # RTN fallback (use_hessian = False)
        # ============================================================
        qcfg.failsafe={"strategy": "rtn", "threshold": True}
        gptq_r = GPTQ(linear, qcfg)
        gptq_r.quantizer.configure(perchannel=True)
        gptq_r.failsafe = qcfg.failsafe

        # IMPORTANT:
        # We intentionally do NOT call add_batch here,
        # so nsamples == 0 and the code falls back to RTN-style quantization.
        Q_r, scale_r, zero_r, gidx_r, *_ = gptq_r.quantize()

        # ============================================================
        # Assertions
        # ============================================================

        # ------------------------------------------------------------
        # 1. Quantized weights should remain numerically close
        #
        # GPTQ tries to stay very close to the RTN baseline while reducing
        # global error using Hessian-based correction. The RTN scale tensor
        # stores the uniform quantizer step size for each group; its mean
        # stands in for a single-bin width when we decide what counts as "close".
        # ------------------------------------------------------------
        quant_step = torch.mean(scale_r).item()
        self.assertGreater(quant_step, 0.0, msg="RTN-derived quantization step must be positive")

        close_mask = torch.isclose(Q_h, Q_r, atol=quant_step)
        close_ratio = close_mask.float().mean().item()

        self.assertGreater(
            close_ratio,
            0.95,
            msg="At least 95% of quantized values should stay within one average RTN quantization step",
        )

        # ------------------------------------------------------------
        # 2. Quantized weights must NOT be exactly identical
        #
        # At least some discrete corrections are expected when
        # Hessian-based error propagation is active.
        # ------------------------------------------------------------
        self.assertFalse(
            torch.equal(Q_h, Q_r),
            msg="Quantized weights should not be exactly identical",
        )

        self.assertGreater(
            torch.count_nonzero(Q_h != Q_r).item(),
            0,
            msg="At least one quantized element should differ due to Hessian correction",
        )

        # ------------------------------------------------------------
        # 3. Group indices must be identical
        #
        # Group assignment depends only on group_size and ordering,
        # and must NOT be affected by Hessian usage.
        # ------------------------------------------------------------
        self.assertTrue(
            torch.equal(gidx_h, gidx_r),
            msg="Group indices (g_idx) must be identical regardless of Hessian usage",
        )

        # ------------------------------------------------------------
        # 4. Scale tensors: shape must match and values must remain stable
        #
        # Scale is allowed to change slightly due to redistribution
        # of weights, but should remain within a small relative bound.
        # ------------------------------------------------------------
        self.assertEqual(scale_h.shape, scale_r.shape)
        self.assertEqual(zero_h.shape, zero_r.shape)

        scale_rel_diff = torch.mean(
            torch.abs(scale_h - scale_r) / scale_r
        ).item()

        self.assertLess(
            scale_rel_diff,
            0.05,
            msg="Relative scale deviation should remain below 5%",
        )

        # ------------------------------------------------------------
        # 5. Zero-points may shift slightly, but the shift must be bounded
        #
        # A bounded zero-point shift corresponds to less than one
        # quantization bin and is expected behavior.
        # ------------------------------------------------------------
        zero_diff = torch.mean(torch.abs(zero_h - zero_r)).item()

        self.assertLess(
            zero_diff * quant_step,
            quant_step,
            msg="Zero-point shift should correspond to less than one quantization bin",
        )

    def test_cpu(self):
        self._run_test("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._run_test("cuda")


class TestFailsafeConfig(unittest.TestCase):
    def test_failsafe_none_round_trip(self):
        qcfg = QuantizeConfig(failsafe=None)
        payload = qcfg.to_dict()

        self.assertIn("failsafe", payload.get("meta", {}))
        self.assertIsNone(payload["meta"]["failsafe"])

        loaded = QuantizeConfig.from_quant_config(payload)
        self.assertIsNone(loaded.failsafe)

######## test_failsafe_awq.py ########

def _dummy_prepare_dataset(
        *,
        calibration_dataset,
        calibration_dataset_concat_size,
        calibration_dataset_sort,
        batch_size,
        calibration_concat_separator=None,
):
    return calibration_dataset


class _DummyProgressBar:
    def title(self, _):
        return self

    def subtitle(self, _):
        return self

    def draw(self):
        return None


def test_awq_failsafe_falls_back_to_rtn_when_no_activations(monkeypatch):
    model = nn.Module()
    model.linear = nn.Linear(8, 8, bias=False)

    gptq_model = SimpleNamespace(model=model, lm_head=None, quant_region_timer=None)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=-1,
        failsafe={"strategy": "rtn", "threshold": "1.0%"},
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
    )
    processor = _StubAWQProcessor(
        qcfg=qcfg,
    )
    processor._pause_controller = PauseResumeController()
    processor.pb = _DummyProgressBar()

    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)
    processor.preprocess(named, failsafe=qcfg.failsafe)

    calls = {}

    def fake_pack(self, nm):
        calls["called"] = True
        calls["name"] = nm.full_name
        nm.state["wq"] = nm.module.weight.detach().clone()

    processor.pack_module = types.MethodType(fake_pack, processor)

    processor.process(named)

    processor.submodule_finalize(named, gptq_model)

    layer_state = processor._get_layer_state(0)

    assert calls.get("called") is True
    assert calls.get("name") == "linear"
    assert layer_state.quantized is True
    assert "wq" in named.state

######### test_failsafe_strategies.py #############


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



######### test_failsafe_thresholds.py #############


def test_should_use_failsafe_parses_numeric_and_percent():
    assert should_use_failsafe(True, observed_samples=0, expected_total_samples=100)
    assert not should_use_failsafe(True, observed_samples=1, expected_total_samples=100)

    assert should_use_failsafe("10", observed_samples=5, expected_total_samples=100)
    assert not should_use_failsafe("10", observed_samples=11, expected_total_samples=100)

    assert should_use_failsafe("10%", observed_samples=8, expected_total_samples=90)
    assert should_use_failsafe("10%", observed_samples=10, expected_total_samples=200)


def test_gptq_failsafe_threshold_triggers_rtn_when_samples_below_percent():
    torch.manual_seed(0)
    layer = nn.Linear(8, 8, bias=False)

    qcfg = QuantizeConfig(bits=4, group_size=4, failsafe="75%")
    gptq = GPTQ(layer, qcfg)
    gptq.failsafe = qcfg.failsafe
    gptq.expected_nsamples = 4  # pretend we expected 4 token rows
    gptq.quantizer.configure(perchannel=True)

    # Capture only a single token worth of activations (< 75% of expected total)
    inp = torch.randn(1, 1, 8)
    gptq.add_batch(inp, None)

    _, _, _, _, _, avg_loss, _, nsamples = gptq.quantize(blocksize=4)

    assert nsamples == 1
    assert avg_loss == "failsafe(rtn): 0.0062505"
