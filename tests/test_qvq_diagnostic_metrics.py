# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.quantization.qvq import yaqa_sketch_b
from gptqmodel.quantization.qvq_yaqa import YaqaGramSketch, _sketch_b_gram_updates
from gptqmodel.utils.diagnostic_metrics import (
    greedy_trajectory_metrics,
    native_divergence_metrics_cuda,
    shared_prefix_top1_metrics,
)
from scripts.analyze_gptq_low_bit_grid import (
    _load_nm_calibration,
    _summary,
    aggregate_storage_rate,
    allocate_exl3_module_bits,
    capture_calibration_hessians,
    capture_forward,
    capture_yaqa_sketch_b,
    load_nm_calibration_batches,
    load_nm_evaluation_batch,
    normalize_requested_rates,
    quantize_module_weight,
    quantize_module_weight_exl3,
    quantize_module_weight_qvq,
    resolve_qvq_device,
    select_valid_token_rows,
    synchronize_benchmark_device,
    target_modules,
    tensor_metrics,
)
from scripts.analyze_gptq_low_bit_grid import main as diagnostic_main
from scripts.compare_qvq_codecs_llama_qkvo import (
    _divergence_metrics,
    _independent_greedy_divergence_metrics,
)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))

    def forward(self, hidden):
        return self.proj(hidden)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer()])

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBackbone()

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        ).last_hidden_state
        return SimpleNamespace(logits=torch.cat((hidden, -hidden), dim=-1))


class _NamedProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(2, 2, bias=False)

    def forward(self, hidden):
        return self.self_attn.q_proj(hidden)


class _TwoLayerNamedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_NamedProjectionLayer(), _NamedProjectionLayer()])

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=torch.cat((hidden, -hidden), dim=-1))


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        }


class _FlattenOutputLinear(nn.Linear):
    def forward(self, input):
        return super().forward(input).flatten(0, 1)


class _SliceOutputLinear(nn.Linear):
    def forward(self, input):
        return super().forward(input)[:, :-1]


class _DetachedOutputLinear(nn.Linear):
    def forward(self, input):
        return super().forward(input).detach()


class _MalformedLayer(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "gradient_rank2":
            self.proj = _FlattenOutputLinear(2, 2, bias=False)
        elif mode == "token_mismatch":
            self.proj = _SliceOutputLinear(2, 2, bias=False)
        elif mode == "output_no_grad":
            self.proj = _DetachedOutputLinear(2, 2, bias=False)
        else:
            self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, hidden):
        if self.mode == "reuse":
            return self.proj(self.proj(hidden))
        if self.mode == "rank2":
            shape = hidden.shape
            return self.proj(hidden.flatten(0, 1)).reshape(shape)
        if self.mode == "gradient_rank2":
            return self.proj(hidden).reshape_as(hidden)
        if self.mode == "token_mismatch":
            projected = self.proj(hidden)
            return F.pad(projected, (0, 0, 0, 1))
        if self.mode == "mask_mismatch":
            projected = self.proj(hidden[:, :-1])
            return F.pad(projected, (0, 0, 0, 1))
        if self.mode == "missing":
            return hidden * 1.0
        if self.mode == "disconnected":
            return hidden + self.proj(hidden).detach() * 0.0
        return self.proj(hidden)


class _MalformedCausalModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_MalformedLayer(mode)])

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        hidden = self.model.layers[0](hidden)
        logits = torch.cat((hidden, -hidden), dim=-1)
        if self.mode == "nonfinite_logits":
            logits = logits * torch.tensor(float("inf"))
        if self.mode == "logits_type":
            logits = "not-a-tensor"
        return SimpleNamespace(logits=logits)


class _KeywordLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, *, hidden_states):
        return self.proj(hidden_states)


class _KeywordCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_KeywordLayer()])

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        hidden = self.model.layers[0](hidden_states=hidden)
        return SimpleNamespace(logits=torch.cat((hidden, -hidden), dim=-1))


def test_qvq_diagnostic_exact_order_statistic_summary_matches_linear_quantiles():
    values = torch.tensor([30.0, 0.0, 20.0, 10.0])
    expected = torch.quantile(values, torch.tensor([0.50, 0.95, 0.99]))

    summary = _summary(values)

    assert summary["mean"] == 15.0
    assert summary["max"] == 30.0
    assert summary["p50"] == pytest.approx(expected[0].item(), rel=0, abs=1e-6)
    assert summary["p95"] == pytest.approx(expected[1].item(), rel=0, abs=1e-6)
    assert summary["p99"] == pytest.approx(expected[2].item(), rel=0, abs=1e-6)








@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")),
)
def test_qvq_diagnostic_auto_device_prefers_cuda_then_mps(cuda_available, mps_available, expected):
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("torch.backends.mps.is_available", return_value=mps_available),
    ):
        assert resolve_qvq_device("auto") == torch.device(expected)


@pytest.mark.parametrize(
    ("requested", "availability"),
    (("cuda", "torch.cuda"), ("mps", "torch.backends.mps")),
)
def test_qvq_diagnostic_explicit_accelerator_fails_closed_when_unavailable(requested, availability):
    with (
        patch(f"{availability}.is_available", return_value=False),
        pytest.raises(RuntimeError, match=f"qvq-device={requested}"),
    ):
        resolve_qvq_device(requested)


@pytest.mark.parametrize(
    ("requested", "availability"),
    (("cuda", "torch.cuda"), ("mps", "torch.backends.mps")),
)
def test_qvq_diagnostic_explicit_accelerator_resolves_when_available(requested, availability):
    with patch(f"{availability}.is_available", return_value=True):
        assert resolve_qvq_device(requested) == torch.device(requested)


def test_qvq_diagnostic_explicit_cpu_does_not_require_an_accelerator():
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_qvq_device("cpu") == torch.device("cpu")


def test_qvq_diagnostic_can_keep_the_full_model_while_targeting_only_early_layers():
    model = _TwoLayerNamedModel().eval()

    assert list(target_modules(model)) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
    ]
    selected = target_modules(model, layer_count=1)

    assert list(selected) == ["model.layers.0.self_attn.q_proj"]
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        target_modules(model, layer_count=3)


def test_qvq_diagnostic_benchmark_sync_closes_only_cuda_work():
    with patch("torch.cuda.synchronize") as synchronize:
        synchronize_benchmark_device(torch.device("cpu"))
        synchronize.assert_not_called()

        synchronize_benchmark_device(torch.device("cuda:0"))
        synchronize.assert_called_once_with(torch.device("cuda:0"))


def test_qvq_diagnostic_accepts_and_canonicalizes_every_half_step_for_qvq():
    requested = [value / 2 for value in range(2, 17)]

    assert normalize_requested_rates(requested, method="qvq") == [
        1,
        1.5,
        2,
        2.5,
        3,
        3.5,
        4,
        4.5,
        5,
        5.5,
        6,
        6.5,
        7,
        7.5,
        8,
    ]


@pytest.mark.parametrize("method", ("gptq", "both", "all"))
def test_qvq_diagnostic_rejects_half_steps_for_integer_only_comparison_arms(method):
    with pytest.raises(ValueError, match="without a GPTQ"):
        normalize_requested_rates([2.5], method=method)


@pytest.mark.parametrize("method", ("qvq", "exl3", "qvq-exl3"))
def test_qvq_diagnostic_accepts_half_steps_for_qvq_and_group_allocated_exl3(method):
    assert normalize_requested_rates([1.5, 2.5], method=method) == [1.5, 2.5]


def test_qvq_diagnostic_exl3_fractional_allocator_matches_production_group_priority_and_budget():
    module_numels = {
        "model.layers.0.self_attn.q_proj": 4,
        "model.layers.0.self_attn.k_proj": 1,
        "model.layers.0.self_attn.v_proj": 1,
        "model.layers.0.self_attn.o_proj": 4,
        "model.layers.0.mlp.gate_proj": 16,
        "model.layers.0.mlp.up_proj": 16,
        "model.layers.0.mlp.down_proj": 16,
        "model.layers.1.self_attn.q_proj": 4,
        "model.layers.1.self_attn.k_proj": 1,
        "model.layers.1.self_attn.v_proj": 1,
        "model.layers.1.self_attn.o_proj": 4,
        "model.layers.1.mlp.gate_proj": 16,
        "model.layers.1.mlp.up_proj": 16,
        "model.layers.1.mlp.down_proj": 16,
    }

    allocation = allocate_exl3_module_bits(module_numels, target_bpw=1.5)

    assert {allocation[f"model.layers.0.self_attn.{name}_proj"] for name in ("q", "k", "v")} == {3}
    assert allocation["model.layers.0.self_attn.o_proj"] == 2
    assert {allocation[f"model.layers.1.self_attn.{name}_proj"] for name in ("q", "k", "v", "o")} == {2}
    assert {allocation[f"model.layers.0.mlp.{name}_proj"] for name in ("gate", "up")} == {2}
    assert allocation["model.layers.0.mlp.down_proj"] == 1
    assert {allocation[f"model.layers.1.mlp.{name}_proj"] for name in ("gate", "up", "down")} == {1}
    assert sum(module_numels[name] * bits for name, bits in allocation.items()) == 1.5 * sum(module_numels.values())


@pytest.mark.parametrize(
    ("module_numels", "target_bpw", "error", "message"),
    (
        ({}, 1.5, ValueError, "at least one"),
        ({"x": 0}, 1.5, ValueError, "positive integers"),
        ({"x": 1}, True, TypeError, "real scalar"),
        ({"x": 1}, float("nan"), ValueError, "finite"),
        ({"x": 1}, 8.5, ValueError, r"\[1, 8\]"),
    ),
)
def test_qvq_diagnostic_exl3_fractional_allocator_rejects_invalid_inputs(
    module_numels, target_bpw, error, message
):
    with pytest.raises(error, match=message):
        allocate_exl3_module_bits(module_numels, target_bpw=target_bpw)


@pytest.mark.parametrize("rate", (0.5, 1.25, 8.5))
def test_qvq_diagnostic_rejects_rates_outside_the_qvq_half_step_grid(rate):
    with pytest.raises(ValueError, match="rate|half-integer"):
        normalize_requested_rates([rate], method="qvq")


def test_qvq_diagnostic_cli_accepts_fractional_qvq_rate_before_model_load():
    arguments = [
        "analyze_gptq_low_bit_grid.py",
        "--model",
        "/does/not/exist",
        "--method",
        "qvq",
        "--bits",
        "1.5",
        "--qvq-device",
        "cpu",
        "--json-out",
        "/tmp/must-not-be-written.json",
    ]
    with (
        patch.object(sys, "argv", arguments),
        patch(
            "scripts.analyze_gptq_low_bit_grid.AutoConfig.from_pretrained",
            side_effect=RuntimeError("fractional rate accepted"),
        ) as load_model_config,
        pytest.raises(RuntimeError, match="fractional rate accepted"),
    ):
        diagnostic_main()
    load_model_config.assert_called_once()


def test_qvq_diagnostic_cli_rejects_fractional_mixed_arm_before_model_load():
    arguments = [
        "analyze_gptq_low_bit_grid.py",
        "--model",
        "/does/not/exist",
        "--method",
        "both",
        "--bits",
        "2.5",
        "--json-out",
        "/tmp/must-not-be-written.json",
    ]
    with (
        patch.object(sys, "argv", arguments),
        patch("scripts.analyze_gptq_low_bit_grid.AutoConfig.from_pretrained") as load_model_config,
        pytest.raises(SystemExit, match="2"),
    ):
        diagnostic_main()
    load_model_config.assert_not_called()


def test_qvq_diagnostic_rejects_unavailable_cuda_before_loading_model():
    arguments = [
        "analyze_gptq_low_bit_grid.py",
        "--model",
        "/does/not/exist",
        "--method",
        "qvq",
        "--qvq-device",
        "cuda",
        "--qvq-output-channel-scales",
        "--qvq-viterbi-objective",
        "hessian_diagonal",
        "--qvq-viterbi-minimum-proxy-improvement",
        "0.001",
        "--json-out",
        "/tmp/must-not-be-written.json",
    ]
    with (
        patch.object(sys, "argv", arguments),
        patch("torch.cuda.is_available", return_value=False),
        patch("scripts.analyze_gptq_low_bit_grid.AutoConfig.from_pretrained") as load_model_config,
        pytest.raises(SystemExit, match="2"),
    ):
        diagnostic_main()
    load_model_config.assert_not_called()


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    (
        (("--qvq-output-channel-scales",), "cannot be combined"),
        (("--qvq-module-scale-search",), "cannot be combined"),
        (
            ("--qvq-viterbi-objective", "hessian_diagonal"),
            "requires --qvq-viterbi-objective=euclidean",
        ),
        (("--qvq-yaqa-regularization", "nan"), "must be finite and nonnegative"),
        (("--qvq-yaqa-regularization", "-0.01"), "must be finite and nonnegative"),
        (("--qvq-yaqa-minimum-sequences", "0"), "must be positive"),
    ),
)
def test_qvq_diagnostic_rejects_incompatible_yaqa_controls_before_loading_model(extra_arguments, message):
    arguments = [
        "analyze_gptq_low_bit_grid.py",
        "--model",
        "/does/not/exist",
        "--method",
        "qvq",
        "--qvq-rounding",
        "yaqa",
        *extra_arguments,
        "--json-out",
        "/tmp/must-not-be-written.json",
    ]
    with (
        patch.object(sys, "argv", arguments),
        patch("scripts.analyze_gptq_low_bit_grid.AutoConfig.from_pretrained") as load_model_config,
        pytest.raises(SystemExit, match="2"),
    ):
        diagnostic_main()
    load_model_config.assert_not_called()


def test_qvq_diagnostic_yaqa_keeps_full_model_and_wires_both_sketch_factors(tmp_path):
    model = _TwoLayerNamedModel().eval()
    tokenizer = _TinyTokenizer()
    config = SimpleNamespace(num_hidden_layers=2)
    calibration_batch = tokenizer()
    calibration_stats = {
        "valid_tokens": 3,
        "padded_tokens_excluded": 3,
    }
    activation_hessian = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    quantize_calls = []

    def fake_quantize(weight, H, **kwargs):
        reconstructed = weight.float().clone()
        quantize_calls.append(
            {
                "rounding": kwargs["rounding"],
                "tail_biting_candidates": kwargs["tail_biting_candidates"],
                "input_hessian": H.clone(),
                "output_hessian": None if kwargs["output_hessian"] is None else kwargs["output_hessian"].clone(),
            }
        )
        return reconstructed, {
            "method": "qvq",
            "codebook": kwargs["codebook_version"],
            "rounding": kwargs["rounding"],
            "weight": tensor_metrics(weight.float(), reconstructed, normalize_distribution=True),
        }

    output_path = tmp_path / "yaqa.json"
    arguments = [
        "analyze_gptq_low_bit_grid.py",
        "--model",
        "/tiny/model",
        "--layers",
        "1",
        "--bits",
        "2",
        "--method",
        "qvq",
        "--qvq-device",
        "cpu",
        "--qvq-rounding",
        "block_ldlq",
        "yaqa",
        "--qvq-yaqa-minimum-sequences",
        "2",
        "--qvq-tail-biting-candidates",
        "1",
        "4",
        "--json-out",
        str(output_path),
    ]
    with (
        patch.object(sys, "argv", arguments),
        patch("scripts.analyze_gptq_low_bit_grid.TARGET_SUFFIXES", ("self_attn.q_proj",)),
        patch(
            "scripts.analyze_gptq_low_bit_grid.AutoConfig.from_pretrained",
            return_value=config,
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load_model,
        patch(
            "scripts.analyze_gptq_low_bit_grid.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.load_nm_calibration_batches",
            return_value=([calibration_batch], calibration_stats),
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.load_nm_evaluation_batch",
            return_value=(calibration_batch, calibration_stats),
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.capture_calibration_hessians",
            return_value=(
                {"model.layers.0.self_attn.q_proj": activation_hessian},
                {"model.layers.0.self_attn.q_proj": 3},
            ),
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.quantize_module_weight_qvq",
            side_effect=fake_quantize,
        ),
        patch(
            "scripts.analyze_gptq_low_bit_grid.request_performance_qos",
            return_value=True,
        ),
        patch("scripts.analyze_gptq_low_bit_grid.torch.set_num_threads"),
        patch("scripts.analyze_gptq_low_bit_grid.torch.set_num_interop_threads"),
    ):
        diagnostic_main()

    assert config.num_hidden_layers == 2
    assert load_model.call_args.kwargs["config"] is config
    assert [(call["rounding"], call["tail_biting_candidates"]) for call in quantize_calls] == [
        ("block_ldlq", 1),
        ("block_ldlq", 4),
        ("yaqa", 1),
        ("yaqa", 4),
    ]
    torch.testing.assert_close(quantize_calls[0]["input_hessian"], activation_hessian)
    assert quantize_calls[0]["output_hessian"] is None
    assert quantize_calls[2]["output_hessian"] is not None
    assert quantize_calls[2]["input_hessian"].shape == (2, 2)
    assert quantize_calls[2]["output_hessian"].shape == (2, 2)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["settings"]["source_model_layers"] == 2
    assert report["settings"]["loaded_model_layers"] == 2
    assert report["settings"]["layers"] == 1
    assert report["settings"]["full_model_loaded_for_yaqa"] is True
    assert report["yaqa_sketch_b"]["independent_sequences"] == 2
    assert set(report["arms"]) == {
        "w2-qvq-pgc16-v1-candidates-1-block-ldlq",
        "w2-qvq-pgc16-v1-candidates-4-block-ldlq",
        "w2-qvq-pgc16-v1-candidates-1-yaqa",
        "w2-qvq-pgc16-v1-candidates-4-yaqa",
    }
    assert report["settings"]["qvq_tail_biting_candidates"] == [1, 4]
    assert {arm["tail_biting_candidates"] for arm in report["arms"].values()} == {1, 4}


def test_qvq_diagnostic_forwards_viterbi_margin_and_reports_candidate_improvement():
    result = SimpleNamespace(
        weight=torch.eye(2),
        rounding="block_ldlq",
        proxy_loss=torch.tensor(0.9),
        baseline_proxy_loss=torch.tensor(1.0),
        output_scale_optimized_channels=0,
        hessian_viterbi_selected=True,
        hessian_viterbi_candidate_relative_improvement=0.1,
        trellis=torch.zeros((1, 1), dtype=torch.int32),
        SU=torch.ones(2, dtype=torch.float32),
        SV=torch.ones(2, dtype=torch.float32),
        kronecker_proxy_loss=None,
    )
    with patch("scripts.analyze_gptq_low_bit_grid.quantize_qvq_linear", return_value=result) as quantize:
        reconstructed, report = quantize_module_weight_qvq(
            torch.eye(2),
            torch.eye(2),
            bits=2,
            bias=None,
            module_name="model.layers.0.proj",
            device=torch.device("cpu"),
            trellis_batch_size=1,
            tail_biting_candidates=4,
            module_scale_search=True,
            viterbi_objective="hessian_diagonal",
            viterbi_minimum_proxy_improvement=0.001,
        )

    assert torch.equal(reconstructed, torch.eye(2))
    assert quantize.call_args.kwargs["viterbi_minimum_proxy_improvement"] == 0.001
    assert quantize.call_args.kwargs["tail_biting_candidates"] == 4
    assert quantize.call_args.kwargs["module_scale_search"] is True
    assert report["module_scale_search_selected"] is False
    assert report["module_scale_multiplier"] == 1.0
    assert report["module_scale_reencoded"] is False
    assert report["hessian_viterbi_candidate_relative_improvement"] == 0.1
    assert report["tail_biting_candidates"] == 4
    assert report["payload_bytes"] == 4
    assert report["auxiliary_bytes"] == 16
    assert report["stored_bytes"] == 20
    assert report["payload_bits_per_weight"] == 8
    assert report["effective_bits_per_weight"] == 40


def test_qvq_diagnostic_storage_rate_uses_total_bytes_and_includes_auxiliary_data():
    modules = {
        "small": {"weight_numel": 100, "payload_bytes": 10, "stored_bytes": 20},
        "large": {"weight_numel": 300, "payload_bytes": 60, "stored_bytes": 90},
    }

    assert aggregate_storage_rate(modules, "payload_bytes") == pytest.approx(1.4)
    assert aggregate_storage_rate(modules, "stored_bytes") == pytest.approx(2.2)
    assert aggregate_storage_rate(modules, "stored_bytes", shared_auxiliary_bytes=10) == pytest.approx(2.4)
    assert aggregate_storage_rate({}, "stored_bytes") is None
    assert aggregate_storage_rate({"missing": {"weight_numel": 1}}, "stored_bytes") is None
    with pytest.raises(TypeError, match="integer byte count"):
        aggregate_storage_rate(modules, "stored_bytes", shared_auxiliary_bytes=True)
    with pytest.raises(ValueError, match="nonnegative"):
        aggregate_storage_rate(modules, "stored_bytes", shared_auxiliary_bytes=-1)
    with pytest.raises(ValueError, match="positive total weight count"):
        aggregate_storage_rate({"empty": {"weight_numel": 0, "stored_bytes": 1}}, "stored_bytes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 quantization requires CUDA/HIP")
@pytest.mark.parametrize(("bits", "max_relative_l2"), ((1, 0.8), (2, 0.4), (4, 0.12), (8, 0.02)))
def test_qvq_diagnostic_exl3_arm_returns_exact_reconstruction_and_storage_rate(bits, max_relative_l2):
    generator = torch.Generator().manual_seed(700 + bits)
    weight = torch.randn((128, 128), generator=generator)
    activations = torch.randn((256, 128), generator=generator)
    hessian = activations.T @ activations / activations.shape[0]

    reconstructed, report = quantize_module_weight_exl3(
        weight,
        hessian,
        bits=bits,
        device=torch.device("cuda"),
        codebook="mcg",
    )

    relative_l2 = ((reconstructed - weight).square().sum() / weight.square().sum()).sqrt()
    assert torch.isfinite(reconstructed).all()
    assert relative_l2 < max_relative_l2
    assert report["method"] == "exl3"
    assert report["codebook"] == "mcg"
    assert report["payload_bits_per_weight"] == bits
    assert report["auxiliary_bytes"] == 516
    assert report["effective_bits_per_weight"] == pytest.approx(bits + 0.251953125)
    assert report["trellis_shape"] == [8, 8, 16 * bits]


def test_qvq_diagnostic_exl3_arm_rejects_non_cuda_and_invalid_geometry():
    with pytest.raises(ValueError, match="CUDA/HIP"):
        quantize_module_weight_exl3(
            torch.zeros((128, 128)),
            torch.eye(128),
            bits=2,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="dimensions do not match"):
        quantize_module_weight_exl3(
            torch.zeros((128, 128)),
            torch.eye(64),
            bits=2,
            device=torch.device("cuda"),
        )
    with pytest.raises(ValueError, match="divisible by 128"):
        quantize_module_weight_exl3(
            torch.zeros((64, 128)),
            torch.eye(128),
            bits=2,
            device=torch.device("cuda"),
        )


def test_qvq_diagnostic_selects_only_valid_token_rows_for_left_and_right_padding():
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]])

    selected = select_valid_token_rows(tensor, mask)

    torch.testing.assert_close(selected, tensor[mask.bool()], rtol=0, atol=0)
    assert selected.shape == (4, 3)


@pytest.mark.parametrize(
    ("tensor", "mask", "message"),
    (
        (torch.zeros(2, 3), torch.ones(2, 3), "batch, sequence"),
        (torch.zeros(2, 3, 4), torch.ones(2, 1, 3), "rank-2"),
        (torch.zeros(2, 3, 4), torch.ones(2, 4), "does not match"),
        (torch.zeros(2, 3, 4), torch.zeros(2, 3), "no valid tokens"),
    ),
)
def test_qvq_diagnostic_valid_token_selection_rejects_invalid_geometry(tensor, mask, message):
    with pytest.raises(ValueError, match=message):
        select_valid_token_rows(tensor, mask)


def test_qvq_diagnostic_capture_excludes_padding_from_hessian_and_all_outputs():
    model = _TinyCausalModel().eval()
    modules = {"proj": model.model.layers[0].proj}
    encoded = {
        "input_ids": torch.tensor([[1, 2, 900], [3, 800, 700]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }
    second_batch = {
        "input_ids": torch.tensor([[4, 600]]),
        "attention_mask": torch.tensor([[1, 0]]),
    }
    first_inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    all_calibration_inputs = torch.cat((first_inputs, torch.tensor([[4.0, 5.0]])))
    expected_hessian = all_calibration_inputs.mT @ all_calibration_inputs / all_calibration_inputs.shape[0]

    hessians, counts = capture_calibration_hessians(
        model,
        [encoded, second_batch],
        modules,
        device=torch.device("cpu"),
    )
    logits, inputs, outputs = capture_forward(model, encoded, modules, capture_inputs=True)

    assert counts == {"proj": 4}
    torch.testing.assert_close(hessians["proj"], expected_hessian, rtol=0, atol=0)
    torch.testing.assert_close(inputs["proj"], first_inputs, rtol=0, atol=0)
    torch.testing.assert_close(outputs["proj"], first_inputs, rtol=0, atol=0)
    torch.testing.assert_close(outputs["layer.0.hidden"], first_inputs, rtol=0, atol=0)
    torch.testing.assert_close(logits, torch.cat((first_inputs, -first_inputs), dim=-1), rtol=0, atol=0)

    _, inputs_not_requested, outputs_only = capture_forward(model, encoded, modules, capture_inputs=False)
    assert inputs_not_requested == {}
    torch.testing.assert_close(outputs_only["proj"], first_inputs, rtol=0, atol=0)


def test_qvq_diagnostic_capture_requires_attention_masks_and_valid_samples():
    model = _TinyCausalModel().eval()
    modules = {"proj": model.model.layers[0].proj}
    input_ids = torch.tensor([[1, 2]])

    with pytest.raises(ValueError, match="attention_mask is required"):
        capture_forward(model, {"input_ids": input_ids}, modules, capture_inputs=True)
    with pytest.raises(ValueError, match="must include attention_mask"):
        capture_calibration_hessians(
            model,
            [{"input_ids": input_ids}],
            modules,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="capture layer count"):
        capture_forward(
            model,
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
            modules,
            capture_inputs=True,
            layer_count=2,
        )
    assert not modules["proj"]._forward_hooks


@pytest.mark.parametrize(
    ("batches", "message"),
    (
        (
            [
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.zeros(1, 2, dtype=torch.long),
                }
            ],
            "no valid tokens",
        ),
        (
            [
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.ones(1, 1, dtype=torch.long),
                }
            ],
            "does not match calibration activation geometry",
        ),
        ([], "observed no valid calibration tokens"),
    ),
)
def test_qvq_diagnostic_calibration_hessian_fails_closed_on_invalid_sample_state(batches, message):
    model = _TinyCausalModel().eval()
    modules = {"proj": model.model.layers[0].proj}

    with pytest.raises(ValueError, match=message):
        capture_calibration_hessians(model, batches, modules, device=torch.device("cpu"))

    assert not modules["proj"]._forward_hooks


@pytest.mark.parametrize("shape", ((3, 7, 8, 6), (4, 32, 64, 48), (2, 128, 128, 96)))
@pytest.mark.parametrize("strategy", ("flattened", "token_space"))
def test_yaqa_sketch_b_gram_strategies_preserve_fp32_factor_geometry(shape, strategy):
    batch, tokens, in_features, out_features = shape
    generator = torch.Generator().manual_seed(20260819)
    activation = torch.randn((batch, tokens, in_features), generator=generator, dtype=torch.float32)
    gradient = torch.randn((batch, tokens, out_features), generator=generator, dtype=torch.float32)

    expected = _sketch_b_gram_updates(activation, gradient, strategy="batched")
    actual = _sketch_b_gram_updates(activation, gradient, strategy=strategy)

    for actual_factor, expected_factor in zip(actual, expected, strict=True):
        relative_l2 = torch.linalg.vector_norm(actual_factor - expected_factor) / torch.linalg.vector_norm(
            expected_factor
        )
        assert float(relative_l2) <= 1e-6
        assert torch.equal(actual_factor, actual_factor.T)


@pytest.mark.parametrize("strategy", ("batched", "flattened", "projected", "token_space"))
def test_yaqa_all_sketch_b_gram_strategies_are_bit_exact_symmetric(strategy):
    generator = torch.Generator().manual_seed(1000)
    activation = torch.randn((5, 50, 97), generator=generator, dtype=torch.float32)
    gradient = torch.randn((5, 50, 71), generator=generator, dtype=torch.float32)
    kwargs = {"strategy": strategy}
    if strategy == "projected":
        kwargs["projections"] = (
            torch.randn((71, 32), generator=generator, dtype=torch.float32),
            torch.randn((97, 32), generator=generator, dtype=torch.float32),
        )

    for factor in _sketch_b_gram_updates(activation, gradient, **kwargs):
        assert torch.equal(factor, factor.T)


def test_yaqa_projected_sketch_b_gram_matches_explicit_projected_weight_scores():
    generator = torch.Generator().manual_seed(20260819)
    activation = torch.randn((3, 11, 13), generator=generator, dtype=torch.float32)
    gradient = torch.randn((3, 11, 17), generator=generator, dtype=torch.float32)
    output_projection = torch.randn((17, 5), generator=generator, dtype=torch.float32)
    input_projection = torch.randn((13, 5), generator=generator, dtype=torch.float32)
    weight_scores = torch.bmm(gradient.transpose(1, 2), activation)
    expected_input = torch.bmm(
        (weight_scores.transpose(1, 2) @ output_projection),
        (weight_scores.transpose(1, 2) @ output_projection).transpose(1, 2),
    ).sum(dim=0)
    expected_output = torch.bmm(
        weight_scores @ input_projection,
        (weight_scores @ input_projection).transpose(1, 2),
    ).sum(dim=0)

    actual_input, actual_output = _sketch_b_gram_updates(
        activation,
        gradient,
        strategy="projected",
        projections=(output_projection, input_projection),
    )

    for actual, expected in ((actual_input, expected_input), (actual_output, expected_output)):
        relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
        assert float(relative_l2) <= 1e-6


@pytest.mark.parametrize("strategy", ("batched", "flattened", "projected", "token_space"))
def test_yaqa_sketch_b_sequence_weights_scale_each_sequence_gram_once(strategy):
    generator = torch.Generator().manual_seed(20260830)
    activation = torch.randn((2, 7, 5), generator=generator, dtype=torch.float32)
    gradient = torch.randn((2, 7, 3), generator=generator, dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
    kwargs = {"strategy": strategy, "sequence_weights": weights}
    if strategy == "projected":
        kwargs["projections"] = (
            torch.randn((3, 4), generator=generator, dtype=torch.float32),
            torch.randn((5, 4), generator=generator, dtype=torch.float32),
        )

    actual = _sketch_b_gram_updates(activation, gradient, **kwargs)
    if strategy == "projected":
        expected_parts = [
            _sketch_b_gram_updates(
                activation[index : index + 1],
                gradient[index : index + 1],
                strategy=strategy,
                projections=kwargs["projections"],
            )
            for index in range(2)
        ]
    else:
        expected_parts = [
            _sketch_b_gram_updates(
                activation[index : index + 1], gradient[index : index + 1], strategy=strategy
            )
            for index in range(2)
        ]
    expected = tuple(expected_parts[0][axis] + 2.0 * expected_parts[1][axis] for axis in range(2))
    for actual_factor, expected_factor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_factor, expected_factor, rtol=2e-6, atol=2e-6)


def test_yaqa_diagnostic_sketch_b_matches_independent_per_sequence_autograd_oracle():
    model = _TinyCausalModel().eval()
    reference = copy.deepcopy(model)
    module = model.model.layers[0].proj
    original_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())
    batch = {
        "input_ids": torch.tensor([[1, 2, 900], [3, 800, 700]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "labels": torch.full((2, 3), -100),
    }
    second_batch = {
        "input_ids": torch.tensor([[4, 5, 600]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "labels": torch.full((1, 3), -100),
    }

    progress = []
    input_hessians, output_hessians, stats = capture_yaqa_sketch_b(
        model,
        [batch, second_batch],
        {"proj": module},
        device=torch.device("cpu"),
        seed=7,
        progress_callback=progress.append,
    )
    assert progress == [
        {"completed_batches": 1, "total_batches": 2, "completed_sequences": 2, "valid_tokens": 3},
        {"completed_batches": 2, "total_batches": 2, "completed_sequences": 3, "valid_tokens": 5},
    ]

    generator = torch.Generator(device="cpu").manual_seed(7)
    gradients = []
    for calibration_batch in (batch, second_batch):
        reference_batch = {name: value for name, value in calibration_batch.items() if name != "labels"}
        logits = reference(**reference_batch, use_cache=False).logits
        valid_logits = logits[calibration_batch["attention_mask"].bool()]
        sampled_tokens = torch.multinomial(
            valid_logits.detach().float().softmax(dim=-1),
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
        offset = 0
        for sequence_index in range(calibration_batch["input_ids"].shape[0]):
            keep = calibration_batch["attention_mask"][sequence_index].bool()
            sequence_logits = logits[sequence_index, keep]
            sequence_samples = sampled_tokens[offset : offset + sequence_logits.shape[0]]
            offset += sequence_logits.shape[0]
            sequence_loss = F.cross_entropy(sequence_logits.float(), sequence_samples, reduction="sum")
            gradients.append(
                torch.autograd.grad(
                    sequence_loss,
                    reference.model.layers[0].proj.weight,
                    retain_graph=True,
                )[0]
            )
    expected_input, expected_output = yaqa_sketch_b(torch.stack(gradients))

    torch.testing.assert_close(input_hessians["proj"], expected_input.float(), rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(output_hessians["proj"], expected_output.float(), rtol=1e-6, atol=2e-7)
    # Sketch-B factors are deliberately spilled to host memory while each
    # sequence is processed so MPS does not retain quadratic Gram matrices.
    assert input_hessians["proj"].device == torch.device("cpu")
    assert output_hessians["proj"].device == torch.device("cpu")
    assert stats.pop("accumulator_device") == "cpu"
    assert stats.pop("accumulator_bytes") == 32
    assert stats.pop("capture_wall_seconds") >= 0
    assert stats.pop("capture_cuda_ms") is None
    assert stats.pop("final_host_transfer_seconds") >= 0
    phase_wall_seconds = stats.pop("phase_wall_seconds")
    assert set(phase_wall_seconds) == {
        "forward",
        "loss",
        "backward_and_sketch",
        "python_gc",
        "mps_synchronize",
        "mps_empty_cache",
    }
    assert all(seconds >= 0 for seconds in phase_wall_seconds.values())
    assert stats == {
        "method": "YAQA-v3 Sketch B real Fisher",
        "full_model_backward": True,
        "independent_sequences": 3,
        "unique_sequences": 3,
        "valid_output_samples": 5,
        "raw_valid_tokens": 5,
        "effective_weighted_sequences": 3.0,
        "effective_weighted_tokens": 5.0,
        "monte_carlo_samples_per_output": 1,
        "sequence_loss_reduction": "per_sequence_token_sum",
        "activation_checkpointing": False,
        "checkpointed_modules": 0,
            "packed_symmetric_accumulators": False,
            "gram_strategy": "batched",
            "gram_projection_rank": None,
            "gram_projection_distribution": None,
            "factor_approximate": False,
            "mps_cleanup_interval": 8,
        "mps_cleanup_count": 0,
        "minimum_sequences": 1,
        "factor_dtype": "float32",
        "input_factor_elements": 4,
        "output_factor_elements": 4,
        "factor_storage_bytes": 32,
        "dense_factor_storage_bytes": 32,
        "factor_compression_ratio": 1.0,
        "tf32": False,
        "seed": 7,
        "activation_quantization_error": None,
    }
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == original_requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not module._forward_hooks

    invalid_average_input, invalid_average_output = yaqa_sketch_b(torch.stack(gradients).mean(dim=0, keepdim=True))
    assert not torch.allclose(input_hessians["proj"], invalid_average_input.float())
    assert not torch.allclose(output_hessians["proj"], invalid_average_output.float())


def test_yaqa_streaming_projected_factor_is_compact_deterministic_and_materializable():
    model = _TinyCausalModel().eval()
    batches = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[5, 6]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
    ]
    modules = {"proj": model.model.layers[0].proj}

    exact_input, exact_output, _ = capture_yaqa_sketch_b(
        model,
        batches,
        modules,
        device=torch.device("cpu"),
        seed=91,
    )
    captures = [
        capture_yaqa_sketch_b(
            model,
            batches,
            modules,
            device=torch.device("cpu"),
            seed=91,
            gram_strategy="streaming_projected",
            gram_projection_rank=4096,
        )
        for _ in range(2)
    ]

    first_input, first_output, stats = captures[0]
    assert isinstance(first_input["proj"], YaqaGramSketch)
    assert isinstance(first_output["proj"], YaqaGramSketch)
    assert first_input["proj"].source.shape == (2, 4096)
    assert first_output["proj"].source.shape == (2, 4096)
    assert stats["gram_strategy"] == "streaming_projected"
    assert stats["factor_storage_bytes"] == 2 * 2 * 4096 * 4
    assert stats["dense_factor_storage_bytes"] == 2 * 2 * 2 * 4
    torch.testing.assert_close(first_input["proj"].source, captures[1][0]["proj"].source)
    torch.testing.assert_close(first_output["proj"].source, captures[1][1]["proj"].source)
    torch.testing.assert_close(
        first_input["proj"].materialize(device=torch.device("cpu")),
        exact_input["proj"],
        rtol=0.05,
        atol=0.05,
    )
    torch.testing.assert_close(
        first_output["proj"].materialize(device=torch.device("cpu")),
        exact_output["proj"],
        rtol=0.05,
        atol=0.05,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_yaqa_streaming_projected_materialization_forces_ieee_fp32_and_restores_backend():
    source = torch.randn((32, 64), dtype=torch.float32)
    sketch = YaqaGramSketch(source=source, normalizer=64.0, seed=17)
    expected = (source @ source.T) / sketch.normalizer
    cuda_matmul = torch.backends.cuda.matmul
    previous_precision = cuda_matmul.fp32_precision
    try:
        cuda_matmul.fp32_precision = "tf32"
        actual = sketch.materialize(device=torch.device("cuda")).cpu()
        assert cuda_matmul.fp32_precision == "tf32"
    finally:
        cuda_matmul.fp32_precision = previous_precision

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_yaqa_weighted_capture_tracks_effective_coverage_without_duplication():
    model = _TinyCausalModel().eval()
    reference = copy.deepcopy(model).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "fisher_sequence_weight": torch.tensor([2.0], dtype=torch.float64),
    }
    unweighted_batch = {name: value for name, value in batch.items() if name != "fisher_sequence_weight"}

    weighted_input, weighted_output, stats = capture_yaqa_sketch_b(
        model,
        [batch],
        {"proj": model.model.layers[0].proj},
        device=torch.device("cpu"),
        seed=17,
    )
    baseline_input, baseline_output, _ = capture_yaqa_sketch_b(
        reference,
        [unweighted_batch],
        {"proj": reference.model.layers[0].proj},
        device=torch.device("cpu"),
        seed=17,
    )

    torch.testing.assert_close(weighted_input["proj"], baseline_input["proj"])
    torch.testing.assert_close(weighted_output["proj"], baseline_output["proj"])
    assert stats["unique_sequences"] == 1
    assert stats["raw_valid_tokens"] == 2
    assert stats["effective_weighted_sequences"] == 2.0
    assert stats["effective_weighted_tokens"] == 4.0
    assert not model.model.layers[0]._forward_pre_hooks

def test_yaqa_activation_checkpointing_preserves_exact_factors_and_restores_module_forwards():
    baseline_model = _TinyCausalModel().eval()
    checkpointed_model = copy.deepcopy(baseline_model).eval()
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        },
        {
            "input_ids": torch.tensor([[6, 0]]),
            "attention_mask": torch.tensor([[1, 0]]),
        },
    ]
    baseline_module = baseline_model.model.layers[0].proj
    checkpointed_layer = checkpointed_model.model.layers[0]
    checkpointed_module = checkpointed_layer.proj
    assert "forward" not in checkpointed_layer.__dict__

    baseline_input, baseline_output, baseline_stats = capture_yaqa_sketch_b(
        baseline_model,
        batches,
        {"proj": baseline_module},
        device=torch.device("cpu"),
        seed=787,
    )
    checkpointed_input, checkpointed_output, checkpointed_stats = capture_yaqa_sketch_b(
        checkpointed_model,
        batches,
        {"proj": checkpointed_module},
        device=torch.device("cpu"),
        seed=787,
        checkpoint_modules=(checkpointed_layer,),
    )

    torch.testing.assert_close(checkpointed_input["proj"], baseline_input["proj"], rtol=0, atol=0)
    torch.testing.assert_close(checkpointed_output["proj"], baseline_output["proj"], rtol=0, atol=0)
    assert baseline_stats["activation_checkpointing"] is False
    assert baseline_stats["checkpointed_modules"] == 0
    assert checkpointed_stats["activation_checkpointing"] is True
    assert checkpointed_stats["checkpointed_modules"] == 1
    assert checkpointed_stats["independent_sequences"] == 3
    assert "forward" not in checkpointed_layer.__dict__
    assert not checkpointed_module._forward_hooks
    assert not checkpointed_layer._forward_pre_hooks


def test_yaqa_activation_checkpointing_preserves_rng_for_stochastic_eval_modules():
    class StochasticLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)

        def forward(self, hidden):
            return self.proj(hidden) * torch.rand_like(hidden)

    baseline_model = _TinyCausalModel().eval()
    baseline_model.model.layers[0] = StochasticLayer().eval()
    checkpointed_model = copy.deepcopy(baseline_model).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 0]]),
        "attention_mask": torch.tensor([[1, 1], [1, 0]]),
    }

    torch.manual_seed(20260812)
    baseline_input, baseline_output, _ = capture_yaqa_sketch_b(
        baseline_model,
        [batch],
        {"proj": baseline_model.model.layers[0].proj},
        device=torch.device("cpu"),
        seed=787,
    )
    torch.manual_seed(20260812)
    checkpointed_layer = checkpointed_model.model.layers[0]
    checkpointed_input, checkpointed_output, _ = capture_yaqa_sketch_b(
        checkpointed_model,
        [batch],
        {"proj": checkpointed_layer.proj},
        device=torch.device("cpu"),
        seed=787,
        checkpoint_modules=(checkpointed_layer,),
    )

    torch.testing.assert_close(checkpointed_input["proj"], baseline_input["proj"], rtol=0, atol=0)
    torch.testing.assert_close(checkpointed_output["proj"], baseline_output["proj"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("checkpoint_modules", "exception", "message"),
    (
        (lambda layer: (layer, layer), ValueError, "must be unique"),
        (lambda _layer: (object(),), TypeError, "must be modules"),
    ),
)
def test_yaqa_activation_checkpointing_rejects_invalid_module_sets_and_cleans_hooks(
    checkpoint_modules,
    exception,
    message,
):
    model = _TinyCausalModel().eval()
    layer = model.model.layers[0]
    module = layer.proj
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    with pytest.raises(exception, match=message):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": module},
            device=torch.device("cpu"),
            checkpoint_modules=checkpoint_modules(layer),
        )

    assert "forward" not in layer.__dict__
    assert not module._forward_hooks
    assert not layer._forward_pre_hooks


def test_yaqa_diagnostic_sketch_b_rejects_an_undersized_independent_sequence_population_before_forward():
    model = _TinyCausalModel().eval()
    module = model.model.layers[0].proj
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="2 independent sequences, below the configured minimum of 3"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": module},
            device=torch.device("cpu"),
            minimum_sequences=3,
        )

    assert not module._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


def test_yaqa_diagnostic_sketch_b_is_invariant_to_batch_grouping_and_variable_valid_lengths():
    grouped_model = _TinyCausalModel().eval()
    split_model = copy.deepcopy(grouped_model)
    grouped = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
    }
    split = [
        {name: value[index : index + 1] for name, value in grouped.items()}
        for index in range(grouped["input_ids"].shape[0])
    ]

    def deterministic_sample(probabilities, *, num_samples, generator):
        del generator
        assert num_samples == 1
        return probabilities.argmax(dim=-1, keepdim=True)

    with patch("gptqmodel.quantization.qvq_yaqa.torch.multinomial", side_effect=deterministic_sample):
        grouped_input, grouped_output, _ = capture_yaqa_sketch_b(
            grouped_model,
            [grouped],
            {"proj": grouped_model.model.layers[0].proj},
            device=torch.device("cpu"),
            minimum_sequences=2,
        )
        split_input, split_output, _ = capture_yaqa_sketch_b(
            split_model,
            split,
            {"proj": split_model.model.layers[0].proj},
            device=torch.device("cpu"),
            minimum_sequences=2,
        )

    torch.testing.assert_close(grouped_input["proj"], split_input["proj"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(grouped_output["proj"], split_output["proj"], rtol=1e-6, atol=1e-7)


def test_yaqa_diagnostic_sketch_b_includes_downstream_source_model_layers():
    model = _TwoLayerNamedModel().eval()
    blocked = copy.deepcopy(model)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.copy_(torch.eye(2))
        model.model.layers[1].self_attn.q_proj.weight.copy_(torch.eye(2))
        blocked.model.layers[0].self_attn.q_proj.weight.copy_(torch.eye(2))
        blocked.model.layers[1].self_attn.q_proj.weight.zero_()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    full_input, full_output, _ = capture_yaqa_sketch_b(
        model,
        [batch],
        {"first": model.model.layers[0].self_attn.q_proj},
        device=torch.device("cpu"),
        seed=3,
    )
    blocked_input, blocked_output, _ = capture_yaqa_sketch_b(
        blocked,
        [batch],
        {"first": blocked.model.layers[0].self_attn.q_proj},
        device=torch.device("cpu"),
        seed=3,
    )

    assert full_input["first"].norm() > 0
    assert full_output["first"].norm() > 0
    torch.testing.assert_close(blocked_input["first"], torch.zeros_like(blocked_input["first"]), rtol=0, atol=0)
    torch.testing.assert_close(
        blocked_output["first"],
        torch.zeros_like(blocked_output["first"]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("batches", "message"),
    (
        ([], "observed no independent"),
        ([{"input_ids": torch.tensor([[1, 2]])}], "must include attention_mask"),
        (
            [
                {
                    "input_ids": torch.tensor([[1, 2], [3, 4]]),
                    "attention_mask": torch.tensor([[1, 1], [0, 0]]),
                }
            ],
            "every YAQA calibration sequence",
        ),
    ),
)
def test_yaqa_diagnostic_sketch_b_fails_closed_and_restores_model_state(batches, message):
    model = _TinyCausalModel().eval()
    modules = {"proj": model.model.layers[0].proj}
    original_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())

    with pytest.raises(ValueError, match=message):
        capture_yaqa_sketch_b(model, batches, modules, device=torch.device("cpu"))

    assert tuple(parameter.requires_grad for parameter in model.parameters()) == original_requires_grad
    assert not modules["proj"]._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


def test_yaqa_diagnostic_sketch_b_requires_eval_mode_and_target_modules():
    model = _TinyCausalModel()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="eval mode"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": model.model.layers[0].proj},
            device=torch.device("cpu"),
        )
    model.eval()
    with pytest.raises(ValueError, match="at least one target"):
        capture_yaqa_sketch_b(model, [batch], {}, device=torch.device("cpu"))


def test_yaqa_diagnostic_sketch_b_rejects_invalid_accumulator_devices():
    model = _TinyCausalModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    modules = {"proj": model.model.layers[0].proj}

    with pytest.raises(ValueError, match="CPU, CUDA, or MPS"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            modules,
            device=torch.device("cpu"),
            accumulator_device=torch.device("meta"),
        )
    with pytest.raises(ValueError, match="require CUDA collection"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            modules,
            device=torch.device("cpu"),
            accumulator_device=torch.device("cuda"),
        )


@pytest.mark.parametrize(
    ("modules", "seed", "exception", "message"),
    (
        ({"bad": nn.ReLU()}, 0, TypeError, "linear modules"),
        (
            {"first": nn.Linear(2, 2), "second": nn.Linear(2, 2)},
            True,
            TypeError,
            "seed must be an integer",
        ),
    ),
)
def test_yaqa_diagnostic_sketch_b_rejects_invalid_targets_and_seed(modules, seed, exception, message):
    model = _TinyCausalModel().eval()
    with pytest.raises(exception, match=message):
        capture_yaqa_sketch_b(model, [], modules, device=torch.device("cpu"), seed=seed)

    duplicate = model.model.layers[0].proj
    with pytest.raises(ValueError, match="must be unique"):
        capture_yaqa_sketch_b(
            model,
            [],
            {"first": duplicate, "second": duplicate},
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("mode", "exception", "message"),
    (
        ("reuse", ValueError, "was reused"),
        ("rank2", ValueError, "input must have"),
        ("gradient_rank2", ValueError, "output gradient must have"),
        ("token_mismatch", ValueError, "input/output token geometry"),
        ("mask_mismatch", ValueError, "attention-mask geometry"),
        ("output_no_grad", RuntimeError, "not connected to the full-model loss"),
        ("nonfinite_logits", ValueError, "logits must contain only finite"),
        ("logits_type", TypeError, "must return tensor logits"),
        ("missing", ValueError, "did not execute target modules"),
        ("disconnected", ValueError, "did not produce one gradient"),
    ),
)
def test_yaqa_diagnostic_sketch_b_rejects_invalid_full_model_gradient_geometry(mode, exception, message):
    model = _MalformedCausalModel(mode).eval()
    module = model.model.layers[0].proj
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    with pytest.raises(exception, match=message):
        capture_yaqa_sketch_b(model, [batch], {"proj": module}, device=torch.device("cpu"))

    assert not module._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


def test_yaqa_diagnostic_sketch_b_supports_keyword_decoder_hidden_states():
    model = _KeywordCausalModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    input_hessians, output_hessians, stats = capture_yaqa_sketch_b(
        model,
        [batch],
        {"proj": model.model.layers[0].proj},
        device=torch.device("cpu"),
    )

    assert input_hessians["proj"].shape == (2, 2)
    assert output_hessians["proj"].shape == (2, 2)
    assert stats["independent_sequences"] == 1


def test_yaqa_diagnostic_sketch_b_rejects_nonfinite_full_model_gradients():
    model = _TinyCausalModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    def nonfinite_loss(logits, _attention_mask, *, generator, token_weights=None):
        del generator, token_weights
        return logits.sum() * torch.tensor(float("nan")), 2

    with (
        patch(
            "gptqmodel.quantization.qvq_yaqa.yaqa_real_fisher_loss",
            side_effect=nonfinite_loss,
        ),
        pytest.raises(ValueError, match="non-finite full-model weight gradient"),
    ):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": model.model.layers[0].proj},
            device=torch.device("cpu"),
        )


def test_yaqa_diagnostic_sketch_b_rejects_missing_decoder_and_hidden_state_contracts():
    class EmptyDecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()

    empty = EmptyDecoderModel().eval()
    with pytest.raises(ValueError, match="at least one decoder layer"):
        capture_yaqa_sketch_b(empty, [], {"proj": empty.proj}, device=torch.device("cpu"))

    class SeedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def forward(self, states):
            return self.proj(states)

    class BadSeedModel(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([SeedLayer()])

        def forward(self, input_ids, attention_mask, use_cache=False):
            del input_ids, attention_mask, use_cache
            if self.mode == "positional":
                hidden = self.model.layers[0]("not-a-tensor")
            else:
                hidden = self.model.layers[0](states=torch.ones(1, 2, 2))
            return SimpleNamespace(logits=hidden)

    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    for mode, message in (
        ("positional", "must be a tensor"),
        ("keyword", "could not locate"),
    ):
        model = BadSeedModel(mode).eval()
        with pytest.raises((TypeError, ValueError), match=message):
            capture_yaqa_sketch_b(
                model,
                [batch],
                {"proj": model.model.layers[0].proj},
                device=torch.device("cpu"),
            )


def test_yaqa_diagnostic_sketch_b_rejects_non_matrix_attention_masks():
    model = _TinyCausalModel().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 1, 2, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="rank-2"):
        capture_yaqa_sketch_b(
            model,
            [batch],
            {"proj": model.model.layers[0].proj},
            device=torch.device("cpu"),
        )


def test_yaqa_diagnostic_sketch_b_disables_and_restores_cuda_tf32_without_a_gpu():
    model = _TinyCausalModel().eval()
    original_precision = torch.backends.cuda.matmul.fp32_precision
    with (
        patch(
            "scripts.analyze_gptq_low_bit_grid.torch.Generator",
            return_value=torch.Generator(device="cpu"),
        ),
        pytest.raises(ValueError, match="observed no independent"),
    ):
        capture_yaqa_sketch_b(
            model,
            [],
            {"proj": model.model.layers[0].proj},
            device=torch.device("cuda"),
        )
    assert torch.backends.cuda.matmul.fp32_precision == original_precision


def test_qvq_diagnostic_gptq_rejects_hessian_with_wrong_feature_geometry():
    with pytest.raises(ValueError, match="does not match weight columns 4"):
        quantize_module_weight(
            torch.eye(4),
            torch.eye(3),
            bits=4,
            group_size=2,
            sym=True,
            adjacent_zero_search=True,
        )


@pytest.mark.parametrize("rounding", ("block_ldlq", "yaqa"))
def test_qvq_diagnostic_quantizer_forwards_rounding_and_sketch_factors(rounding):
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    input_hessian = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    output_hessian = torch.tensor([[1.0, 0.1], [0.1, 0.5]]) if rounding == "yaqa" else None
    fake_result = SimpleNamespace(
        weight=weight.clone(),
        rounding=rounding,
        proxy_loss=torch.tensor(1.0),
        baseline_proxy_loss=torch.tensor(1.5),
        output_scale_optimized_channels=0,
        hessian_viterbi_selected=False,
        hessian_viterbi_candidate_relative_improvement=None,
        trellis=torch.ones((1, 1), dtype=torch.int32),
        SU=torch.ones(2, dtype=torch.float32),
        SV=torch.ones(2, dtype=torch.float32),
        kronecker_proxy_loss=torch.tensor(0.75) if rounding == "yaqa" else None,
    )

    with patch(
        "scripts.analyze_gptq_low_bit_grid.quantize_qvq_linear",
        return_value=fake_result,
    ) as quantize:
        reconstructed, metrics = quantize_module_weight_qvq(
            weight,
            input_hessian,
            bits=2,
            output_hessian=output_hessian,
            bias=None,
            module_name="model.layers.0.self_attn.q_proj",
            device=torch.device("cpu"),
            trellis_batch_size=3,
            rounding=rounding,
        )

    torch.testing.assert_close(reconstructed, weight)
    assert quantize.call_args.kwargs["rounding"] == rounding
    if output_hessian is None:
        assert quantize.call_args.kwargs["output_hessian"] is None
        assert "kronecker_proxy_loss" not in metrics
    else:
        torch.testing.assert_close(quantize.call_args.kwargs["output_hessian"], output_hessian)
        assert metrics["kronecker_proxy_loss"] == pytest.approx(0.75)
    assert metrics["rounding"] == rounding
    assert metrics["weight"]["relative_l2"] == 0.0












def test_qvq_diagnostic_loads_first_256_nm_rows_without_concatenation():
    dataset = MagicMock()
    dataset.__len__.return_value = 300
    selected = object()
    dataset.select.return_value = selected
    prepared = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        for _ in range(256)
    ]

    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset) as load,
        patch(
            "scripts.analyze_gptq_low_bit_grid.prepare_calibration_dataset",
            return_value=prepared,
        ) as prepare,
    ):
        batches, stats = load_nm_calibration_batches(
            tokenizer=object(),
            config=object(),
            dataset_path=Path("/nm/dataset"),
            rows=256,
            concat_size=0,
            batch_size=1,
        )

    load.assert_called_once_with(path="/nm/dataset", name="LLM", split="train")
    dataset.select.assert_called_once_with(range(256))
    assert prepare.call_args.kwargs["calibration_dataset"] is selected
    assert prepare.call_args.kwargs["calibration_dataset_concat_size"] is None
    assert prepare.call_args.kwargs["calibration_dataset_sort"] == "desc"
    assert batches == prepared
    assert stats["source_rows"] == 256
    assert stats["concat_size"] is None
    assert stats["prepared_batches"] == 256
    assert stats["prepared_sequences"] == 256
    assert stats["valid_tokens"] == 768
    assert stats["padded_tokens_excluded"] == 0


def test_qvq_diagnostic_unpacked_calibration_rejects_lost_source_rows():
    dataset = MagicMock()
    dataset.__len__.return_value = 300
    dataset.select.return_value = object()
    prepared = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        for _ in range(255)
    ]

    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        patch("scripts.analyze_gptq_low_bit_grid.prepare_calibration_dataset", return_value=prepared),
        pytest.raises(ValueError, match="selected 256 rows but prepared 255 sequences"),
    ):
        load_nm_calibration_batches(
            tokenizer=object(),
            config=object(),
            dataset_path=Path("/nm/dataset"),
            rows=256,
            concat_size=None,
            batch_size=1,
        )


def test_qvq_diagnostic_nm_calibration_rejects_invalid_row_requests():
    with pytest.raises(ValueError, match="positive"):
        load_nm_calibration_batches(
            tokenizer=object(),
            config=object(),
            dataset_path=Path("/nm/dataset"),
            rows=0,
            concat_size=2048,
            batch_size=1,
        )
    dataset = MagicMock()
    dataset.__len__.return_value = 2
    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        pytest.raises(ValueError, match="contains only 2"),
    ):
        load_nm_calibration_batches(
            tokenizer=object(),
            config=object(),
            dataset_path=Path("/nm/dataset"),
            rows=3,
            concat_size=2048,
            batch_size=1,
        )


@pytest.mark.parametrize(
    ("concat_size", "batch_size", "message"),
    ((-1, 1, "concat size must be nonnegative"), (None, 0, "batch size must be positive")),
)
def test_qvq_diagnostic_nm_calibration_rejects_invalid_preparation_geometry(concat_size, batch_size, message):
    with pytest.raises(ValueError, match=message):
        load_nm_calibration_batches(
            tokenizer=object(),
            config=object(),
            dataset_path=Path("/nm/dataset"),
            rows=256,
            concat_size=concat_size,
            batch_size=batch_size,
        )


def test_qvq_diagnostic_loads_disjoint_nm_rows_for_held_out_evaluation():
    dataset = MagicMock()
    dataset.__len__.return_value = 300
    selected = {"text": ["first held-out row", "second held-out row"]}
    dataset.select.return_value = selected
    tokenizer = MagicMock(
        return_value={
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
    )

    with patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset) as load:
        evaluation, stats = load_nm_evaluation_batch(
            tokenizer,
            dataset_path=Path("/nm/dataset"),
            row_offset=128,
            rows=2,
            max_length=48,
        )

    load.assert_called_once_with(path="/nm/dataset", name="LLM", split="train")
    dataset.select.assert_called_once_with(range(128, 130))
    tokenizer.assert_called_once_with(
        selected["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=48,
    )
    assert evaluation["input_ids"].shape == (2, 3)
    assert stats == {
        "dataset_path": "/nm/dataset",
        "dataset_config": "LLM",
        "row_start": 128,
        "row_end_exclusive": 130,
        "source_rows": 2,
        "max_length": 48,
        "valid_tokens": 5,
        "padded_tokens_excluded": 1,
    }


def test_qvq_diagnostic_loads_direct_parquet_without_flattening_structured_rows():
    dataset = object()
    path = Path("/nm/llm.parquet")
    with (
        patch("scripts.analyze_gptq_low_bit_grid.os.path.isfile", return_value=True),
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset) as load,
    ):
        result = _load_nm_calibration(path)

    load.assert_called_once_with("parquet", data_files={"train": str(path)}, split="train")
    assert result is dataset


def test_qvq_diagnostic_evaluation_none_preserves_full_rows_without_truncation():
    dataset = MagicMock()
    dataset.__len__.return_value = 4
    selected = {"text": ["one", "two"]}
    dataset.select.return_value = selected
    tokenizer = MagicMock(
        return_value={
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }
    )
    with patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset):
        _, stats = load_nm_evaluation_batch(
            tokenizer,
            dataset_path=Path("/nm/dataset"),
            row_offset=2,
            rows=2,
            max_length=None,
        )

    tokenizer.assert_called_once_with(
        selected["text"],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    assert stats["max_length"] == "full_row"


@pytest.mark.parametrize(
    ("row_offset", "rows", "max_length", "message"),
    ((-1, 1, 48, "nonnegative"), (0, 0, 48, "positive"), (0, 1, 0, "positive")),
)
def test_qvq_diagnostic_nm_evaluation_rejects_invalid_geometry(row_offset, rows, max_length, message):
    with pytest.raises(ValueError, match=message):
        load_nm_evaluation_batch(
            tokenizer=object(),
            dataset_path=Path("/nm/dataset"),
            row_offset=row_offset,
            rows=rows,
            max_length=max_length,
        )


def test_qvq_diagnostic_nm_evaluation_rejects_missing_rows_and_invalid_text_or_mask():
    dataset = MagicMock()
    dataset.__len__.return_value = 129
    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        pytest.raises(ValueError, match=r"\[128, 130\).*contains only 129"),
    ):
        load_nm_evaluation_batch(
            tokenizer=object(),
            dataset_path=Path("/nm/dataset"),
            row_offset=128,
            rows=2,
            max_length=48,
        )

    dataset.__len__.return_value = 300
    dataset.select.return_value = {"text": ["", "valid"]}
    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        pytest.raises(ValueError, match="nonempty"),
    ):
        load_nm_evaluation_batch(
            tokenizer=object(),
            dataset_path=Path("/nm/dataset"),
            row_offset=128,
            rows=2,
            max_length=48,
        )

    dataset.select.return_value = {"text": ["valid"]}
    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        pytest.raises(ValueError, match="attention mask"),
    ):
        load_nm_evaluation_batch(
            tokenizer=MagicMock(return_value={"input_ids": torch.ones((1, 1), dtype=torch.long)}),
            dataset_path=Path("/nm/dataset"),
            row_offset=128,
            rows=1,
            max_length=48,
        )

    with (
        patch("scripts.analyze_gptq_low_bit_grid.load_dataset", return_value=dataset),
        pytest.raises(ValueError, match="no valid tokens"),
    ):
        load_nm_evaluation_batch(
            tokenizer=MagicMock(
                return_value={
                    "input_ids": torch.zeros((1, 1), dtype=torch.long),
                    "attention_mask": torch.zeros((1, 1), dtype=torch.long),
                }
            ),
            dataset_path=Path("/nm/dataset"),
            row_offset=128,
            rows=1,
            max_length=48,
        )


def test_qvq_diagnostic_reports_divergence_cross_entropy_and_topk_metrics():
    dense = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            [6.0, 5.0, 3.0, 2.0, 1.0, -2.0],
        ]
    )
    quantized = torch.tensor(
        [
            [3.0, 4.0, 2.0, 1.0, 0.0, -1.0],
            [5.0, 6.0, 3.0, 2.0, 1.0, -2.0],
        ]
    )

    metrics = tensor_metrics(dense, quantized, normalize_distribution=False)

    assert metrics["kl_forward"]["mean"] > 0
    assert metrics["kl_reverse"]["mean"] > 0
    assert metrics["jensen_shannon"]["mean"] > 0
    assert metrics["total_variation"]["mean"] > 0
    assert metrics["hellinger"]["mean"] > 0
    assert (metrics["dense_to_quantized_cross_entropy"]["mean"] - metrics["dense_entropy"]["mean"]) == pytest.approx(
        metrics["kl_forward"]["mean"], abs=1e-6
    )
    assert metrics["top1_agreement"] == 0.0
    assert metrics["top5_overlap"]["mean"] == 1.0
    assert metrics["top5_exact_agreement"] == 1.0
    assert metrics["dense_top1_in_quantized_top5"] == 1.0
    assert metrics["quantized_top1_in_dense_top5"] == 1.0


def test_qvq_diagnostic_reports_requested_final_logit_top10_metrics():
    dense = torch.arange(12, 0, -1, dtype=torch.float32).unsqueeze(0)
    quantized = dense.clone()
    quantized[0, 9] = -1.0
    quantized[0, 10] = 3.5

    metrics = tensor_metrics(
        dense,
        quantized,
        normalize_distribution=False,
        include_top10=True,
    )

    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_overlap"]["mean"] == 1.0
    assert metrics["top10_overlap"]["mean"] == pytest.approx(0.9)
    assert metrics["top10_exact_agreement"] == 0.0
    assert metrics["dense_top1_in_quantized_top10"] == 1.0
    assert metrics["quantized_top1_in_dense_top10"] == 1.0


def test_divergence_metrics_report_token_top1_exact_sequence_and_first_mismatch():
    dense = torch.zeros((32, 4), dtype=torch.float32)
    candidate = dense.clone()
    dense[:, 0] = 2.0
    candidate[:, 0] = 2.0
    candidate[5, 1] = 3.0
    candidate[20, 2] = 4.0

    metrics = _divergence_metrics(dense, candidate, token_count=32)

    assert metrics is not None
    assert metrics["token_top1_agreement"] == pytest.approx(30 / 32)
    assert metrics["exact_sequence_agreement"] == 0.0
    assert metrics["first_divergence_token"] == 6.0
    assert metrics["divergent_sequence_fraction"] == 1.0


def test_divergence_metrics_excludes_short_rows_and_uses_horizon_sentinel():
    dense = torch.zeros((31, 3), dtype=torch.float32)
    candidate = dense.clone()
    assert _divergence_metrics(dense, candidate, token_count=32) is None

    dense = torch.zeros((32, 3), dtype=torch.float32)
    candidate = dense.clone()
    metrics = _divergence_metrics(dense, candidate, token_count=32)
    assert metrics is not None
    assert metrics["token_top1_agreement"] == 1.0
    assert metrics["exact_sequence_agreement"] == 1.0
    assert metrics["first_divergence_token"] == 33.0
    assert metrics["divergent_sequence_fraction"] == 0.0


def test_greedy_trajectory_metrics_report_survival_and_first_divergence():
    dense = torch.arange(32)
    quantized = dense.clone()
    quantized[5:] += 100

    metrics = greedy_trajectory_metrics(dense, quantized, token_count=32)

    assert metrics["trajectory_survival"] == 0.0
    assert metrics["exact_sequence_agreement"] == 0.0
    assert metrics["aligned_token_agreement"] == pytest.approx(5 / 32)
    assert metrics["aligned_token_matches"].tolist() == [1.0] * 5 + [0.0] * 27
    assert metrics["first_divergence_token"] == 6.0
    assert metrics["prefix_survival"].tolist() == [1.0] * 5 + [0.0] * 27


def test_shared_prefix_top1_metrics_reports_same_context_agreement():
    dense = torch.zeros((32, 4), dtype=torch.float32)
    candidate = dense.clone()
    dense[:, 0] = 2.0
    candidate[:, 0] = 2.0
    candidate[5, 1] = 3.0
    candidate[20, 2] = 4.0

    metrics = shared_prefix_top1_metrics(dense, candidate, token_count=32)

    assert metrics is not None
    assert metrics["top1_agreement"] == pytest.approx(30 / 32)
    assert metrics["exact_sequence_agreement"] == 0.0
    assert metrics["first_mismatch_token"] == 6.0


def test_shared_prefix_top1_metrics_excludes_short_rows():
    logits = torch.zeros((31, 4), dtype=torch.float32)

    assert shared_prefix_top1_metrics(logits, logits.clone(), token_count=32) is None


def test_shared_prefix_top1_metrics_honors_warmup_start_index():
    dense = torch.zeros((64, 3), dtype=torch.float32)
    candidate = dense.clone()
    dense[:, 0] = 2.0
    candidate[:, 0] = 2.0
    candidate[:32, 1] = 3.0

    warm = shared_prefix_top1_metrics(dense, candidate, token_count=32, start_index=32)
    cold = shared_prefix_top1_metrics(dense, candidate, token_count=32)

    assert warm is not None and warm["top1_agreement"] == 1.0
    assert cold is not None and cold["top1_agreement"] == 0.0


def test_greedy_trajectory_metrics_identical_horizon_survives():
    tokens = torch.arange(32)

    metrics = greedy_trajectory_metrics(tokens, tokens.clone(), token_count=32)

    assert metrics["trajectory_survival"] == 1.0
    assert metrics["aligned_token_agreement"] == 1.0
    assert metrics["first_divergence_token"] == 33.0
    assert metrics["prefix_survival"].tolist() == [1.0] * 32


def test_independent_greedy_divergence_uses_each_models_own_rollout():
    class FixedForward(nn.Module):
        def __init__(self, continuation):
            super().__init__()
            self.continuation = list(continuation)
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            logits = torch.full((1, kwargs["input_ids"].shape[1], 16), -100.0)
            logits[:, -1, self.continuation[self.calls - 1]] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=None)

    row = {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    metrics = _independent_greedy_divergence_metrics(
        FixedForward([1, 2, 3, 4]),
        FixedForward([1, 2, 9, 8]),
        row,
        token_count=4,
    )

    assert metrics["trajectory_survival"] == 0.0
    assert metrics["aligned_token_agreement"] == 0.5
    assert metrics["first_divergence_token"] == 3.0


@pytest.mark.parametrize("bad_count", [True, 0, -1, 1.5])
def test_greedy_trajectory_metrics_reject_invalid_horizon(bad_count):
    with pytest.raises(ValueError, match="positive integer"):
        greedy_trajectory_metrics(torch.arange(2), torch.arange(2), token_count=bad_count)


def test_greedy_trajectory_metrics_requires_exact_matching_horizons():
    with pytest.raises(ValueError, match="exactly 32 tokens"):
        greedy_trajectory_metrics(torch.arange(31), torch.arange(32), token_count=32)


def test_greedy_trajectory_metrics_rejects_multi_sequence_batches():
    with pytest.raises(ValueError, match="batch size 1"):
        greedy_trajectory_metrics(torch.arange(32).reshape(2, 16), torch.arange(32), token_count=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused divergence operator")
def test_fused_cuda_divergence_matches_reference_and_tie_order():
    dense = torch.tensor(
        [[1.0, 3.0, 3.0, 0.0], [4.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 0.0]],
        device="cuda",
    )
    candidate = dense.clone()
    candidate[1, 2] = 5.0
    candidate[2, 0] = 1.0

    result = native_divergence_metrics_cuda(dense, candidate, token_count=3)

    assert result is not None
    assert result.cpu().tolist() == [1, 0, 2, 3]
    reference = _divergence_metrics(dense.cpu(), candidate.cpu(), token_count=3)
    assert reference is not None
    assert result[0].item() == int(reference["token_top1_agreement"].item() * 3)
    assert result[1].item() == int(reference["exact_sequence_agreement"].item())
    assert result[2].item() == int(reference["first_divergence_token"].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the fused divergence operator")
def test_fused_cuda_divergence_reduces_winners_across_warps():
    dense = torch.zeros((32, 512), device="cuda")
    candidate = torch.zeros_like(dense)
    dense[:, 0] = 1.0
    candidate[:, 1] = 2.0
    dense[:, 40] = 10.0
    candidate[:, 40] = 10.0

    result = native_divergence_metrics_cuda(dense, candidate, token_count=32)

    assert result is not None
    assert result.cpu().tolist() == [32, 1, 33, 32]


def test_qvq_diagnostic_identical_standardized_channels_are_exact():
    dense = torch.tensor([[[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]]])

    metrics = tensor_metrics(dense, dense.clone(), normalize_distribution=True)

    assert metrics["rmse"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["kl_forward"]["max"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_overlap"]["mean"] == 1.0
    assert metrics["top5_exact_agreement"] == 1.0


def test_qvq_diagnostic_zero_signal_exact_match_has_zero_relative_error():
    dense = torch.zeros((2, 6), dtype=torch.float32)

    metrics = tensor_metrics(dense, dense.clone(), normalize_distribution=False)

    assert metrics["rmse"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["sqnr_db"] == 0.0
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_overlap"]["mean"] == 1.0
