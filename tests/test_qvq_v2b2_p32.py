# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.config import FORMAT, QVQConfig, YaqaConfig
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    QVQQuantizationTelemetry,
    block_ldlq_inner,
    fixed_boundary_v2b2_p32_segment_quantize,
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    prepare_qvq_input_hessian,
    quantize_qvq_linear,
    reconstruct_qvq_inner_weight,
    tail_biting_v2b2_p32_quantize,
    tail_biting_viterbi_quantize,
    unpack_qvq_binary_bank_ids,
    yaqa_inner_v2b2_p32,
    yaqa_localized_spectral_refine_v2b2_p32,
    yaqa_output_spectral_refine_v2b2_p32,
    yaqa_spectral_push_v2b2_p32,
)
from gptqmodel.quantization.qvq_codecs import pgc16_codebook, pgc16_codebook_v2_bank
from scripts.analyze_gptq_low_bit_grid import (
    capture_calibration_hessians,
    tensor_metrics,
)
from scripts.compare_qvq_codecs_llama_qkvo import (
    ARM_CONFIG,
    DEFAULT_ARMS,
    _acceptance_logit_metrics,
    _aggregate_qvq_telemetry,
    _all_linear_dependency_stages,
    _install_qvq_prefix_artifact,
    _load_qvq_prefix_artifact,
    _load_yaqa_factor_cache,
    _mlp_layer_groups,
    _padded_batch_chunks,
    _parse_target_rate_ladders,
    _parser,
    _passes_mlp_acceptance,
    _quantized_linear_modules,
    _resolve_mlp_rate_geometry,
    _save_qvq_prefix_artifact,
    _save_yaqa_factor_cache,
    _select_mlp_layer_candidates,
    _selected_storage_metrics,
    _shared_input_hessian_groups,
    _streaming_compare_models,
    _WeightedMetricAccumulator,
    _yaqa_cache_metadata,
)
from scripts.compare_qvq_p4_arc import _arm_summary, _paired_summary, _prompt_ids
from scripts.compare_qvq_p4_gsm8k import (
    _extract_answers as _gsm8k_extract_answers,
)
from scripts.compare_qvq_p4_gsm8k import (
    _gold_answer as _gsm8k_gold_answer,
)
from scripts.compare_qvq_p4_gsm8k import (
    _paired_summary as _gsm8k_paired_summary,
)
from scripts.compare_qvq_p4_gsm8k import (
    _progress_summary as _gsm8k_progress_summary,
)
from scripts.compare_qvq_p4_gsm8k import (
    _task_prompt as _gsm8k_task_prompt,
)
from scripts.compare_qvq_p4_prefixes import _compare_locked_rows, _metric_value
from scripts.validate_qvq_p4_live_prefix import (
    _capture_target_inputs,
    _install_prefix_artifacts,
    _localized_summary,
    _minimax_relative_replay_score,
    _passes_confirmation,
    _validate_disjoint_splits,
)
from scripts.validate_qvq_p4_live_prefix import (
    _parser as _p4_parser,
)


class _SharedInputHarness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4, bias=False)
        self.k = torch.nn.Linear(4, 4, bias=False)
        self.v = torch.nn.Linear(4, 4, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
        return self.q(hidden) + self.k(hidden) + self.v(hidden)


class _EarlyStopHarness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.before = torch.nn.Linear(4, 4, bias=False)
        self.target = torch.nn.Linear(4, 4, bias=False)
        self.after_calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = self.before(input_ids.float())
        hidden = self.target(hidden)
        self.after_calls += 1
        return hidden


class _LockedLogitHarness(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, input_ids, attention_mask):
        del attention_mask
        values = input_ids.float()
        logits = torch.stack((values, -values, values * 0.5, values * -0.25 + self.offset), dim=-1)
        return type("Output", (), {"logits": logits})()


def test_qvq_p4_locked_prefix_comparison_streams_matched_rows():
    rows = (
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[4, 5]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
    )
    teacher = _LockedLogitHarness(0.0)
    baseline = _LockedLogitHarness(0.5)
    candidate = _LockedLogitHarness(0.0)

    baseline_metrics, candidate_metrics = _compare_locked_rows(teacher, baseline, candidate, rows)

    assert candidate_metrics["kl_forward"]["mean"] == pytest.approx(0.0, abs=1e-7)
    assert _metric_value(candidate_metrics, "kl_forward") < _metric_value(baseline_metrics, "kl_forward")
    assert _metric_value(candidate_metrics, "top1_agreement") == 1.0
    assert _metric_value(candidate_metrics, "top5_overlap") == 1.0
    assert _metric_value(candidate_metrics, "top10_overlap") == 1.0


def test_qvq_p4_baseline_only_is_explicit_and_default_off():
    required = (
        "--model",
        "model",
        "--dataset",
        "dataset",
        "--prefix-artifact",
        "prefix.safetensors",
        "--yaqa-factor-cache",
        "factors.pt",
        "--yaqa-metadata",
        "metadata.json",
        "--output",
        "report.json",
    )
    defaults = _p4_parser().parse_args(required)
    assert not defaults.baseline_only
    assert defaults.replay_candidates == 0
    assert defaults.replay_folds == 1
    replay = _p4_parser().parse_args((*required, "--replay-candidates", "4"))
    assert replay.replay_candidates == 4
    assert _p4_parser().parse_args((*required, "--baseline-only")).baseline_only


def test_qvq_propagation_gate_keeps_search_scorer_and_owner_lifecycle():
    processor = QVQProcessor.__new__(QVQProcessor)
    processor._propagation_gates = {}
    processor._propagation_gates_lock = threading.RLock()
    acceptance = lambda proposal, baseline: True
    scorer = lambda candidate: 0.0
    processor.set_propagation_gate(
        "layer.q_proj",
        torch.eye(2),
        torch.zeros((2, 2)),
        acceptance,
        scorer,
    )

    gate = processor._require_propagation_gate("layer.q_proj", torch.device("cpu"))
    assert gate[2] is acceptance
    assert gate[3] is scorer
    owner = gate[4]
    processor._pop_propagation_gate("layer.q_proj", owner)
    assert processor._propagation_gates == {}


class _SwiGLUHessianHarness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = input_ids.float()
        gated = torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(gated)


def test_qvq_p4_arc_summary_preserves_paired_flip_directions():
    samples = [
        {
            "gold_index": 0,
            "baseline": {
                "raw_prediction": 1,
                "normalized_prediction": 0,
                "raw_gold_margin": -0.5,
                "normalized_gold_margin": 0.25,
            },
            "candidate": {
                "raw_prediction": 0,
                "normalized_prediction": 1,
                "raw_gold_margin": 0.5,
                "normalized_gold_margin": -0.25,
            },
        },
        {
            "gold_index": 1,
            "baseline": {
                "raw_prediction": 1,
                "normalized_prediction": 1,
                "raw_gold_margin": 0.75,
                "normalized_gold_margin": 0.5,
            },
            "candidate": {
                "raw_prediction": 1,
                "normalized_prediction": 1,
                "raw_gold_margin": 1.0,
                "normalized_gold_margin": 0.75,
            },
        },
    ]

    assert _arm_summary(samples, "candidate") == {
        "rows": 2,
        "accuracy": 1.0,
        "accuracy_normalized": 0.5,
        "raw_correct": 2,
        "normalized_correct": 1,
        "mean_gold_margin": 0.75,
        "mean_gold_margin_normalized": 0.25,
    }
    assert _paired_summary(samples, "baseline", "candidate") == {
        "raw": {"prediction_changes": 1, "wrong_to_correct": 1, "correct_to_wrong": 0, "net_correct": 1},
        "normalized": {"prediction_changes": 1, "wrong_to_correct": 0, "correct_to_wrong": 1, "net_correct": -1},
    }


def test_qvq_p4_arc_prompt_matches_evalution_render_then_tokenize_contract():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Question: Why?\nAnswer:"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered prompt"

        def __call__(self, prompt, *, add_special_tokens):
            assert prompt == "rendered prompt"
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[1, 2, 3])

    assert _prompt_ids(Tokenizer(), "Why?", apply_chat_template=True) == [1, 2, 3]


def test_qvq_p4_gsm8k_contract_reuses_task_prompt_and_paired_numeric_flips():
    prompt = _gsm8k_task_prompt("What is 2 + 2?", [{"question": "What is 1 + 1?", "target": "The answer is 2."}])
    assert prompt == "Q: What is 1 + 1?\n\nA: The answer is 2.\n\nQ: What is 2 + 2?\n\nA:"
    assert _gsm8k_gold_answer("work\n#### $1,024") == "1024"
    assert _gsm8k_extract_answers("work 3. The answer is $1,024.") == ("1024", "1024")

    samples = [
        {
            "gold_answer": "4",
            "baseline": {"strict_answer": "3", "flexible_answer": "3"},
            "candidate": {"strict_answer": "4", "flexible_answer": "4"},
        },
        {
            "gold_answer": "7",
            "baseline": {"strict_answer": "7", "flexible_answer": "7"},
            "candidate": {"strict_answer": None, "flexible_answer": "8"},
        },
    ]
    assert _gsm8k_paired_summary(samples, "baseline", "candidate") == {
        "strict": {"answer_changes": 2, "wrong_to_correct": 1, "correct_to_wrong": 1, "net_correct": 0},
        "flexible": {"answer_changes": 2, "wrong_to_correct": 1, "correct_to_wrong": 1, "net_correct": 0},
    }
    assert _gsm8k_progress_summary(samples, ("baseline", "candidate")) == (
        "baseline=1/2(invalid=0) candidate=1/2(invalid=0)"
    )


def test_qvq_v2b2_p32_is_the_default_matched_model_comparison():
    assert DEFAULT_ARMS == ("v2", "v2b2-p32")
    args = _parser().parse_args(("--model", "model", "--dataset", "dataset", "--output", "report.json"))
    assert args.layers == 4
    assert args.module_scope == "qkvo"
    assert args.all_linear_hessian_mode == "staged"
    assert args.calibration_rows == 64
    assert args.evaluation_rows == 64
    assert args.evaluation_row_offset == 64
    assert args.max_length is None
    assert args.qvq_telemetry is False
    assert ARM_CONFIG["v2b2-p32"] == {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
    }
    assert ARM_CONFIG["v2b2-p32-yaqa-spectral"]["yaqa_spectral_refinement"] is True
    assert ARM_CONFIG["v2b2-p32-yaqa-spectral-fixed"]["yaqa_v2b2_family_mode"] == "fixed_block_ldlq"
    assert ARM_CONFIG["v2b2-p32-yaqa-spectral-push-fixed"]["yaqa_spectral_push"] is True
    assert ARM_CONFIG["v2b2-p32-yaqa-spectral-push"]["yaqa_v2b2_family_mode"] == "reselect"


def test_qvq_comparison_harness_can_select_every_decoder_linear_projection():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, torch.nn.Linear(4, 4, bias=False))
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj = torch.nn.Linear(4, 8, bias=False)
        layer.mlp.up_proj = torch.nn.Linear(4, 8, bias=False)
        layer.mlp.down_proj = torch.nn.Linear(8, 4, bias=False)
        model.model.layers.append(layer)
    model.lm_head = torch.nn.Linear(4, 16, bias=False)

    qkvo = _quantized_linear_modules(model, layer_count=2, module_scope="qkvo")
    all_linear = _quantized_linear_modules(model, layer_count=2, module_scope="all-linear")

    assert len(qkvo) == 8
    assert len(all_linear) == 14
    assert "model.layers.0.mlp.gate_proj" in all_linear
    assert "model.layers.1.mlp.up_proj" in all_linear
    assert "model.layers.1.mlp.down_proj" in all_linear
    assert "lm_head" not in all_linear
    with pytest.raises(ValueError, match="unsupported module scope"):
        _quantized_linear_modules(model, layer_count=2, module_scope="everything")

    stages = _all_linear_dependency_stages(model, all_linear)
    assert stages == (
        tuple(f"model.layers.{layer}.self_attn.{role}" for layer in range(2) for role in ("q_proj", "k_proj", "v_proj")),
        tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in range(2)),
        tuple(f"model.layers.{layer}.mlp.{role}" for layer in range(2) for role in ("gate_proj", "up_proj")),
        tuple(f"model.layers.{layer}.mlp.down_proj" for layer in range(2)),
    )
    assert _all_linear_dependency_stages(model, all_linear, layerwise=True) == tuple(
        stage
        for layer in range(2)
        for stage in (
            tuple(f"model.layers.{layer}.self_attn.{role}" for role in ("q_proj", "k_proj", "v_proj")),
            (f"model.layers.{layer}.self_attn.o_proj",),
            tuple(f"model.layers.{layer}.mlp.{role}" for role in ("gate_proj", "up_proj")),
            (f"model.layers.{layer}.mlp.down_proj",),
        )
    )


def test_qvq_comparison_shared_input_groups_cover_qkv_and_gate_up_only():
    model = torch.nn.Module()
    model.attention = torch.nn.Module()
    model.attention.q_proj = torch.nn.Linear(4, 4, bias=False)
    model.attention.k_proj = torch.nn.Linear(4, 2, bias=False)
    model.attention.v_proj = torch.nn.Linear(4, 2, bias=False)
    model.attention.o_proj = torch.nn.Linear(4, 4, bias=False)
    model.feed_forward = torch.nn.Module()
    model.feed_forward.gate_proj = torch.nn.Linear(4, 8, bias=False)
    model.feed_forward.up_proj = torch.nn.Linear(4, 8, bias=False)
    model.feed_forward.down_proj = torch.nn.Linear(8, 4, bias=False)
    modules = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

    assert _shared_input_hessian_groups(model, modules) == (
        ("attention.q_proj", "attention.k_proj", "attention.v_proj"),
        ("feed_forward.gate_proj", "feed_forward.up_proj"),
    )


def test_shared_hessian_capture_matches_independent_and_aliases_storage():
    model = _SharedInputHarness()
    modules = {"q": model.q, "k": model.k, "v": model.v}
    batches = [{"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}]
    independent, independent_counts = capture_calibration_hessians(
        model, batches, modules, device=torch.device("cpu")
    )
    shared, shared_counts = capture_calibration_hessians(
        model,
        batches,
        modules,
        device=torch.device("cpu"),
        shared_input_groups=(("q", "k", "v"),),
    )
    assert shared_counts == independent_counts == {"q": 2, "k": 2, "v": 2}
    assert all(torch.equal(shared[name], independent[name]) for name in modules)
    assert shared["q"].data_ptr() == shared["k"].data_ptr() == shared["v"].data_ptr()


def test_hessian_capture_can_stop_after_last_required_module():
    model = _EarlyStopHarness().eval()
    batches = [
        {
            "input_ids": torch.randn((1, 5, 4), generator=torch.Generator().manual_seed(17)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        }
    ]
    expected, expected_counts = capture_calibration_hessians(
        model,
        batches,
        {"target": model.target},
        device=torch.device("cpu"),
    )
    assert model.after_calls == 1
    model.after_calls = 0
    actual, actual_counts = capture_calibration_hessians(
        model,
        batches,
        {"target": model.target},
        device=torch.device("cpu"),
        stop_after_module=model.target,
    )

    assert actual_counts == expected_counts == {"target": 4}
    torch.testing.assert_close(actual["target"], expected["target"], rtol=0, atol=0)
    assert model.after_calls == 0
    assert not model.target._forward_hooks

    with pytest.raises(ValueError, match="must be one of"):
        capture_calibration_hessians(
            model,
            batches,
            {"target": model.target},
            device=torch.device("cpu"),
            stop_after_module=model.before,
        )


def test_swiglu_down_hessian_must_be_recaptured_after_gate_up_change():
    torch.manual_seed(19)
    model = _SwiGLUHessianHarness()
    modules = {"down": model.down_proj}
    batches = [
        {
            "input_ids": torch.randn((2, 3, 4)),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
    ]
    dense_hessians, counts = capture_calibration_hessians(
        model,
        batches,
        modules,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        model.gate_proj.weight.zero_()
    replayed_hessians, replayed_counts = capture_calibration_hessians(
        model,
        batches,
        modules,
        device=torch.device("cpu"),
    )

    assert counts == replayed_counts == {"down": 5}
    assert dense_hessians["down"].abs().max() > 0
    assert torch.equal(replayed_hessians["down"], torch.zeros_like(replayed_hessians["down"]))


def test_qvq_shared_input_factorization_is_exact_reused_and_provenance_checked():
    generator = torch.Generator().manual_seed(91)
    weight = torch.randn((16, 16), generator=generator)
    basis = torch.randn((16, 16), generator=generator)
    hessian = basis.mT @ basis + torch.eye(16)
    expected = quantize_qvq_linear(weight, hessian, bits=4, seed=17, trellis_batch_size=1)
    preparation = prepare_qvq_input_hessian(hessian, seed=17)

    with patch("gptqmodel.quantization.qvq.block_ldl_factor", side_effect=AssertionError("unexpected refactor")):
        actual = quantize_qvq_linear(
            weight,
            hessian,
            bits=4,
            seed=17,
            trellis_batch_size=1,
            input_hessian_preparation=preparation,
        )
    assert torch.equal(actual.trellis, expected.trellis)
    assert torch.equal(actual.weight, expected.weight)
    assert torch.equal(actual.inner_weight, expected.inner_weight)

    with pytest.raises(ValueError, match="does not match the source geometry"):
        quantize_qvq_linear(
            weight,
            hessian,
            bits=4,
            seed=18,
            trellis_batch_size=1,
            input_hessian_preparation=preparation,
        )
    hessian.diagonal().add_(1)
    with pytest.raises(ValueError, match="does not match the source geometry"):
        quantize_qvq_linear(
            weight,
            hessian,
            bits=4,
            seed=17,
            trellis_batch_size=1,
            input_hessian_preparation=preparation,
        )


def test_qvq_all_linear_scope_has_distinct_yaqa_cache_metadata():
    args = _parser().parse_args(
        (
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--output",
            "report.json",
            "--module-scope",
            "all-linear",
        )
    )
    metadata = _yaqa_cache_metadata(args, {"model.layers.0.mlp.down_proj": [4, 8]}, row_offset=128)

    assert metadata["version"] == 2
    assert metadata["module_scope"] == "all-linear"


def test_qvq_all_linear_streamed_metrics_use_generic_schema():
    module_name = "model.layers.0.mlp.gate_proj"
    dense_module = torch.nn.Linear(2, 2, bias=False)
    quantized_module = torch.nn.Linear(2, 2, bias=False)
    dense_inputs = {module_name: torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    dense_outputs = {
        module_name: torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
        "layer.0.hidden": torch.tensor([[0.25, 0.75], [1.25, 1.75]]),
    }
    live_outputs = {
        module_name: dense_outputs[module_name] + 0.01,
        "layer.0.hidden": dense_outputs["layer.0.hidden"] + 0.01,
    }
    dense_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    quantized_logits = dense_logits + 0.01
    captures = (
        (dense_logits, dense_inputs, dense_outputs),
        (quantized_logits, {}, live_outputs),
    )

    with patch("scripts.compare_qvq_codecs_llama_qkvo.capture_forward", side_effect=captures):
        metrics = _streaming_compare_models(
            torch.nn.Module(),
            torch.nn.Module(),
            ({"attention_mask": torch.ones((1, 2), dtype=torch.long)},),
            {module_name: dense_module},
            {module_name: quantized_module},
            {module_name: dense_module.weight.detach().float()},
            layer_count=1,
            progress_label="test",
            module_scope="all-linear",
        )

    assert "local_modules" in metrics
    assert "live_modules" in metrics
    assert "local_qkvo" not in metrics
    assert "live_qkvo" not in metrics
    assert metrics["logits"]["streamed_rows"] == 2


def _acceptance_metrics(kl: float, topn: float = 0.9, *, finite: bool = True) -> dict:
    return {
        "finite": finite,
        "kl_forward": {"mean": kl},
        "top1_agreement": topn,
        "top5_overlap": {"mean": topn},
        "top10_overlap": {"mean": topn},
    }


def test_qvq_mlp_acceptance_device_reduction_matches_full_reference():
    generator = torch.Generator().manual_seed(190)
    dense = torch.randn((2, 3, 17), generator=generator)
    candidate = dense + 0.2 * torch.randn((2, 3, 17), generator=generator)
    expected = tensor_metrics(
        dense.flatten(0, -2),
        candidate.flatten(0, -2),
        normalize_distribution=False,
        include_top10=True,
    )
    actual = _acceptance_logit_metrics(dense, candidate)

    assert actual["shape"] == [6, 17]
    assert actual["finite"] is True
    assert actual["kl_forward"]["mean"] == pytest.approx(expected["kl_forward"]["mean"], abs=1e-7)
    assert actual["top1_agreement"] == expected["top1_agreement"]
    assert actual["top5_overlap"]["mean"] == pytest.approx(expected["top5_overlap"]["mean"], abs=1e-7)
    assert actual["top10_overlap"]["mean"] == pytest.approx(expected["top10_overlap"]["mean"], abs=1e-7)


def test_qvq_mlp_acceptance_is_atomic_tree_derived_and_fail_closed():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    modules = {}
    candidates = {}
    originals = {}
    for layer_index in range(2):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        root.model.layers.append(layer)
        for role in ("gate_proj", "up_proj", "down_proj"):
            module = torch.nn.Linear(2, 2, bias=False)
            setattr(layer.mlp, role, module)
            name = f"model.layers.{layer_index}.mlp.{role}"
            modules[name] = module
            originals[name] = torch.ones_like(module.weight)
            candidates[name] = torch.full_like(module.weight, 2.0 + layer_index)
            with torch.no_grad():
                module.weight.copy_(candidates[name])

    assert _mlp_layer_groups(root, modules) == (
        (
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ),
        (
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.mlp.down_proj",
        ),
    )

    observed = []

    def evaluate(label):
        observed.append((label, tuple(float(module.weight.detach()[0, 0]) for module in modules.values())))
        return {
            "qkvo_dense_mlp_baseline": _acceptance_metrics(1.0),
            "mlp_layer_0_gate_w2": _acceptance_metrics(0.85),
            "mlp_layer_0_up_w2": _acceptance_metrics(0.95),
            "mlp_layer_0_gate_up_w2": _acceptance_metrics(0.7),
            "mlp_layer_0_down_w2": _acceptance_metrics(0.9),
            "mlp_layer_0_gate_down_w2": _acceptance_metrics(0.82),
            "mlp_layer_0_up_down_w2": _acceptance_metrics(0.88),
            "mlp_layer_0_full_w2": _acceptance_metrics(0.8),
            "mlp_layer_1_gate_w2": _acceptance_metrics(1.01),
            "mlp_layer_1_up_w2": _acceptance_metrics(1.02),
            "mlp_layer_1_gate_up_w2": _acceptance_metrics(1.1),
            "mlp_layer_1_down_w2": _acceptance_metrics(1.2),
            "mlp_layer_1_gate_down_w2": _acceptance_metrics(1.15),
            "mlp_layer_1_up_down_w2": _acceptance_metrics(1.25),
            "mlp_layer_1_full_w2": _acceptance_metrics(1.3),
        }[label]

    selected, report = _select_mlp_layer_candidates(
        root,
        modules,
        {2.0: candidates},
        originals,
        evaluate=evaluate,
        kl_regression_limit=0.0,
        topn_regression_limit=0.0,
    )

    assert [decision["accepted"] for decision in report["decisions"]] == [True, False]
    assert report["accepted_layers"] == 1
    assert report["fully_quantized_layers"] == 0
    assert report["decisions"][0]["selected_variant"] == "gate_up"
    assert {candidate["variant"] for candidate in report["decisions"][0]["candidates"]} == {
        "gate", "up", "gate_up", "down", "gate_down", "up_down", "full",
    }
    assert all(value == 1.0 for value in observed[0][1])
    for name in report["decisions"][0]["selected_modules"]:
        assert torch.equal(selected[name], candidates[name])
        assert torch.equal(modules[name].weight, candidates[name])
    assert torch.equal(selected["model.layers.0.mlp.down_proj"], originals["model.layers.0.mlp.down_proj"])
    for name in report["decisions"][1]["modules"]:
        assert torch.equal(selected[name], originals[name])
        assert torch.equal(modules[name].weight, originals[name])

    baseline = _acceptance_metrics(1.0)
    assert _passes_mlp_acceptance(
        baseline, _acceptance_metrics(0.9), kl_regression_limit=0.0, topn_regression_limit=0.0
    )
    assert _passes_mlp_acceptance(
        baseline, _acceptance_metrics(1.01), kl_regression_limit=0.01, topn_regression_limit=1.0
    )
    assert not _passes_mlp_acceptance(
        baseline, _acceptance_metrics(1.1), kl_regression_limit=0.01, topn_regression_limit=1.0
    )
    assert not _passes_mlp_acceptance(
        baseline, _acceptance_metrics(float("nan")), kl_regression_limit=1.0, topn_regression_limit=1.0
    )
    assert not _passes_mlp_acceptance(
        baseline,
        _acceptance_metrics(0.9, finite=False),
        kl_regression_limit=1.0,
        topn_regression_limit=1.0,
    )
    assert not _passes_mlp_acceptance(
        baseline,
        _acceptance_metrics(0.9, topn=0.8),
        kl_regression_limit=1.0,
        topn_regression_limit=0.05,
    )

    storage = _selected_storage_metrics(originals, report, codec_bpw=2.03125)
    assert storage["target_parameters"] == 24
    assert storage["quantized_parameters"] == 8
    assert storage["dense_fallback_parameters"] == 16
    assert storage["quantized_parameter_fraction"] == pytest.approx(1 / 3)
    assert storage["selected_effective_bpw"] == pytest.approx((2.03125 + 2 * 16.0) / 3)
    per_module_bpw = {name: 3.03125 for name in originals}
    mapped_storage = _selected_storage_metrics(originals, report, codec_bpw=per_module_bpw)
    assert mapped_storage["selected_effective_bpw"] == pytest.approx((3.03125 + 2 * 16.0) / 3)

    for name, module in modules.items():
        with torch.no_grad():
            module.weight.copy_(candidates[name])
    with pytest.raises(RuntimeError, match="confirmation failed"):
        _select_mlp_layer_candidates(
            root,
            modules,
            {2.0: candidates},
            originals,
            evaluate=lambda label: (
                _acceptance_metrics(1.0)
                if label == "qkvo_dense_mlp_baseline"
                else (_ for _ in ()).throw(RuntimeError("confirmation failed"))
            ),
            kl_regression_limit=0.0,
            topn_regression_limit=0.0,
        )
    for name in report["decisions"][0]["modules"]:
        assert torch.equal(modules[name].weight, originals[name])


def test_qvq_mlp_rate_ladder_promotes_to_cheapest_safe_complete_subset():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList((layer,))
    modules = {}
    originals = {}
    candidates_by_rate = {2.0: {}, 3.0: {}, 3.5: {}, 4.0: {}}
    fixed_name = "model.layers.0.self_attn.q_proj"
    fixed_module = torch.nn.Linear(2, 2, bias=False)
    modules[fixed_name] = fixed_module
    originals[fixed_name] = torch.ones_like(fixed_module.weight)
    for rate in candidates_by_rate:
        candidates_by_rate[rate][fixed_name] = torch.full_like(fixed_module.weight, 9.0)
    with torch.no_grad():
        fixed_module.weight.copy_(candidates_by_rate[2.0][fixed_name])
    for role in ("gate_proj", "up_proj", "down_proj"):
        module = torch.nn.Linear(2, 2, bias=False)
        setattr(layer.mlp, role, module)
        name = f"model.layers.0.mlp.{role}"
        modules[name] = module
        originals[name] = torch.ones_like(module.weight)
        for rate in candidates_by_rate:
            candidates_by_rate[rate][name] = torch.full_like(module.weight, rate)

    def evaluate(label):
        if label == "qkvo_dense_mlp_baseline":
            return _acceptance_metrics(1.0)
        variant, rate_text = label.removeprefix("mlp_layer_0_").rsplit("_w", 1)
        rate = float(rate_text)
        if rate == 2.0 and variant == "up":
            return _acceptance_metrics(0.8)
        if rate >= 3.0 and variant == "full":
            return _acceptance_metrics(0.9 + 0.01 * rate)
        return _acceptance_metrics(1.1)

    selected, report = _select_mlp_layer_candidates(
        root,
        modules,
        candidates_by_rate,
        originals,
        evaluate=evaluate,
        kl_regression_limit=0.0,
        topn_regression_limit=0.0,
        candidate_codecs={2.0: "v2b2-p32", 3.0: "v2b2-p32", 3.5: "v2b2-p32", 4.0: "v2"},
        candidate_roundings={2.0: "yaqa", 3.0: "yaqa", 3.5: "yaqa", 4.0: "yaqa"},
        candidate_bpw={2.0: 2.03125, 3.0: 3.03125, 3.5: 3.53125, 4.0: 4.0},
    )

    decision = report["decisions"][0]
    assert decision["selected_variant"] == "full"
    assert decision["selected_rate"] == 3.0
    assert decision["selected_codec"] == "v2b2-p32"
    assert decision["selected_rounding"] == "yaqa"
    assert decision["selected_effective_bpw"] == pytest.approx(3.03125)
    assert len(decision["candidates"]) == 28
    assert set(decision["selected_rates"].values()) == {3.0}
    assert set(decision["selected_codecs"].values()) == {"v2b2-p32"}
    assert set(decision["selected_roundings"].values()) == {"yaqa"}
    assert all(torch.equal(selected[name], candidates_by_rate[3.0][name]) for name in modules)
    assert all(torch.equal(module.weight, candidates_by_rate[3.0][name]) for name, module in modules.items())
    assert torch.equal(selected[fixed_name], candidates_by_rate[2.0][fixed_name])


def test_qvq_target_rate_ladder_uses_canonical_v2_for_unsupported_banked_rates():
    target_ladders = _parse_target_rate_ladders([["mlp", "4", "4.5", "5.5", "6", "6.5"]])
    assert target_ladders == {"mlp": (4, 4.5, 5.5, 6, 6.5)}
    banked = ARM_CONFIG["v2b2-p32"]
    for rate in target_ladders["mlp"]:
        geometry, codec, effective_bpw = _resolve_mlp_rate_geometry(
            banked,
            rate=rate,
            codec_policy="same",
        )
        assert codec == "v2"
        assert effective_bpw == rate
        assert geometry["vector_size"] == 2
        assert geometry["trellis_window"] == 16
        assert geometry["bank_count"] == 1
        assert "v2b2_p32" not in geometry

    geometry, codec, effective_bpw = _resolve_mlp_rate_geometry(
        banked,
        rate=3.5,
        codec_policy="same",
    )
    assert codec == "v2b2-p32"
    assert effective_bpw == 3.53125
    assert geometry["v2b2_p32"] is True

    # The ladder changes only the codec. YAQA and all generic controls remain
    # active when an unsupported segmented rate falls back to canonical V2.
    banked_yaqa = {
        **ARM_CONFIG["v2b2-p32-yaqa"],
        "damp_percent": 0.025,
        "tail_biting_candidates": 3,
        "codebook_version": "test-version",
    }
    original = dict(banked_yaqa)
    for rate, codec_policy in ((4, "same"), (2, "v2")):
        geometry, codec, effective_bpw = _resolve_mlp_rate_geometry(
            banked_yaqa,
            rate=rate,
            codec_policy=codec_policy,
        )
        assert codec == "v2"
        assert effective_bpw == rate
        assert geometry["rounding"] == "yaqa"
        assert geometry["damp_percent"] == 0.025
        assert geometry["tail_biting_candidates"] == 3
        assert geometry["codebook_version"] == "test-version"
        assert "yaqa_v2b2_family_mode" not in geometry
        assert geometry["bank_count"] == 1
    assert banked_yaqa == original

    with pytest.raises(ValueError, match="unsupported target"):
        _parse_target_rate_ladders([["attention", "4"]])
    with pytest.raises(ValueError, match="strictly low-to-high"):
        _parse_target_rate_ladders([["mlp", "4", "3.5"]])
    with pytest.raises(ValueError, match="duplicate target"):
        _parse_target_rate_ladders([["mlp", "4"], ["mlp", "5"]])


def test_qvq_banked_yaqa_sweep_arms_and_disjoint_batch_contract():
    assert ARM_CONFIG["v2-yaqa"]["rounding"] == "yaqa"
    assert ARM_CONFIG["v2b2-p32-yaqa-fixed"]["yaqa_v2b2_family_mode"] == "fixed_block_ldlq"
    assert ARM_CONFIG["v2b2-p32-yaqa"]["yaqa_v2b2_family_mode"] == "reselect"
    args = _parser().parse_args(
        (
            "--model", "model", "--dataset", "dataset", "--output", "report.json",
            "--arms", "v2", "v2-yaqa", "v2b2-p32-yaqa",
            "--calibration-rows", "512", "--evaluation-rows", "512", "--evaluation-row-offset", "512",
            "--yaqa-rows", "512", "--yaqa-row-offset", "1024",
        )
    )
    assert args.yaqa_batch_size == 8
    mixed_mlp = _parser().parse_args(
        (
            "--model", "model", "--dataset", "dataset", "--output", "report.json",
            "--module-scope", "all-linear", "--mlp-rate", "4", "--mlp-gate-up-rate", "6",
            "--mlp-down-rate", "3.5", "--mlp-codec", "v2",
        )
    )
    assert mixed_mlp.mlp_rate == 4
    assert mixed_mlp.mlp_gate_up_rate == 6
    assert mixed_mlp.mlp_down_rate == 3.5
    assert mixed_mlp.mlp_codec == "v2"
    ladder = _parser().parse_args(
        (
            "--model", "model", "--dataset", "dataset", "--output", "report.json",
            "--module-scope", "all-linear", "--mlp-rate-ladder", "2", "3", "3.5", "4",
        )
    )
    assert ladder.mlp_rate_ladder == [2.0, 3.0, 3.5, 4.0]
    targeted = _parser().parse_args(
        (
            "--model", "model", "--dataset", "dataset", "--output", "report.json",
            "--module-scope", "all-linear", "--target-rate-ladder", "mlp", "4", "4.5", "5.5", "6", "6.5",
        )
    )
    assert targeted.target_rate_ladder == [["mlp", "4", "4.5", "5.5", "6", "6.5"]]
    assert (args.calibration_rows, args.evaluation_row_offset, args.yaqa_row_offset) == (512, 512, 1024)

    encoded = {
        "input_ids": torch.tensor([[0, 0, 11, 12, 13], [0, 21, 22, 23, 24], [0, 0, 0, 31, 32]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1]]),
    }
    batches = _padded_batch_chunks(encoded, batch_size=2)
    assert [tuple(batch["attention_mask"].shape) for batch in batches] == [(2, 4), (1, 2)]
    assert sum(int(batch["attention_mask"].sum()) for batch in batches) == 9


def test_qvq_banked_yaqa_factor_cache_is_atomic_and_validated(tmp_path):
    path = tmp_path / "sketch_b.pt"
    metadata = {"version": 1, "module_shapes": {"proj": [3, 2]}}
    input_hessians = {"proj": torch.eye(2)}
    output_hessians = {"proj": torch.eye(3)}
    stats = {"independent_sequences": 512}
    _save_yaqa_factor_cache(
        path,
        metadata=metadata,
        input_hessians=input_hessians,
        output_hessians=output_hessians,
        stats=stats,
    )

    loaded_input, loaded_output, loaded_stats = _load_yaqa_factor_cache(path, expected_metadata=metadata)
    assert torch.equal(loaded_input["proj"], input_hessians["proj"])
    assert torch.equal(loaded_output["proj"], output_hessians["proj"])
    assert loaded_stats == stats
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(ValueError, match="metadata does not match"):
        _load_yaqa_factor_cache(path, expected_metadata={**metadata, "version": 2})


def test_qvq_v2b2_prefix_artifact_round_trips_packed_modules_without_dense_weights(tmp_path):
    generator = torch.Generator().manual_seed(20260816)
    source = torch.randn((16, 16), generator=generator) * 0.1
    result = quantize_qvq_linear(
        source,
        torch.eye(16),
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    path = tmp_path / "prefix.safetensors"
    provenance = {"model": "tiny", "yaqa_seed": 1, "rows": [1024, 1536]}
    manifest = _save_qvq_prefix_artifact(
        path,
        module_results={"layer.proj": result},
        bits=2,
        provenance=provenance,
    )

    with safe_open(path, framework="pt", device="cpu") as handle:
        storage_names = list(handle.keys())
        assert not any(key.endswith((".weight", ".inner_weight")) for key in storage_names)
    loaded_manifest, payload = _load_qvq_prefix_artifact(path, expected_provenance=provenance)
    assert loaded_manifest == manifest
    assert set(payload["layer.proj"]) == {"trellis", "SU", "SV", "bank_ids", "bank_alt_id"}
    for name, expected in result.serialized_tensors().items():
        torch.testing.assert_close(payload["layer.proj"][name], expected, rtol=0, atol=0)
    assert _localized_summary(result, {"baseline": {}, "proposal": {}, "accepted": True}) == {
        "proposed": True,
        "accepted": True,
        "selector_churn": result.yaqa_spectral_selector_churn,
        "family_changed": result.yaqa_spectral_family_changed,
        "selected_changes": 0,
        "selected_candidates": [],
    }

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.proj = torch.nn.Linear(16, 16, bias=False)
    replacements = _install_qvq_prefix_artifact(model, manifest=loaded_manifest, module_tensors=payload)
    assert model.layer.proj is replacements["layer.proj"]
    assert isinstance(model.layer.proj, QVQLinear)
    inputs = torch.randn((7, 16), generator=generator)
    torch.testing.assert_close(model.layer.proj(inputs), inputs @ result.weight.T, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="provenance"):
        _load_qvq_prefix_artifact(path, expected_provenance={**provenance, "yaqa_seed": 2})

    tampered_path = tmp_path / "tampered-prefix.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensor_names = list(handle.keys())
        tampered = {name: handle.get_tensor(name) for name in tensor_names}
    selector_name = "layer.proj.bank_ids"
    tampered[selector_name] = tampered[selector_name].clone()
    tampered[selector_name].view(-1)[0] ^= 1
    save_file(tampered, tampered_path, metadata=metadata)
    with pytest.raises(ValueError, match="checksum"):
        _load_qvq_prefix_artifact(tampered_path)


def test_qvq_v2b2_prefix_artifact_install_is_transactional_on_geometry_error(tmp_path):
    generator = torch.Generator().manual_seed(20260817)
    results = {
        name: quantize_qvq_linear(
            torch.randn((16, 16), generator=generator) * 0.1,
            torch.eye(16),
            bits=2,
            bank_count=2,
            v2b2_p32=True,
            trellis_batch_size=1,
        )
        for name in ("first", "second")
    }
    path = tmp_path / "prefix.safetensors"
    _save_qvq_prefix_artifact(path, module_results=results, bits=2, provenance={"model": "tiny"})
    manifest, payload = _load_qvq_prefix_artifact(path)
    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 16, bias=False)
    model.second = torch.nn.Linear(32, 16, bias=False)
    first = model.first

    with pytest.raises(ValueError, match="does not match the serialized geometry"):
        _install_qvq_prefix_artifact(model, manifest=manifest, module_tensors=payload)
    assert model.first is first
    assert isinstance(model.first, torch.nn.Linear)


def test_qvq_p4_live_prefix_splits_reject_overlap_and_capture_stops_at_target():
    _validate_disjoint_splits({"search": (10, 2), "confirmation": (12, 3), "evaluation": (15, 1)})
    with pytest.raises(ValueError, match="must be disjoint"):
        _validate_disjoint_splits({"search": (10, 3), "confirmation": (12, 2), "evaluation": (20, 1)})

    model = _EarlyStopHarness().eval()
    rows = tuple(
        {
            "input_ids": torch.randn((1, tokens, 4), generator=torch.Generator().manual_seed(tokens)),
            "attention_mask": torch.ones((1, tokens), dtype=torch.int64),
        }
        for tokens in (3, 5)
    )
    with torch.no_grad():
        expected = torch.cat([model.before(row["input_ids"].float()).reshape(-1, 4) for row in rows])
    actual = _capture_target_inputs(model, model.target, rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.after_calls == 0
    assert not model.target._forward_pre_hooks


def test_qvq_p4_multiple_prefixes_install_atomically_and_reject_duplicate_or_mixed_contract(tmp_path):
    generator = torch.Generator().manual_seed(20260818)

    def quantized(bits):
        return quantize_qvq_linear(
            torch.randn((16, 16), generator=generator) * 0.1,
            torch.eye(16),
            bits=bits,
            bank_count=2,
            v2b2_p32=True,
            trellis_batch_size=1,
        )

    model_path = tmp_path / "model-snapshot"
    model_path.mkdir()
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    mixed_path = tmp_path / "mixed.safetensors"
    provenance = {"model": str(model_path.resolve())}
    _save_qvq_prefix_artifact(first_path, module_results={"first": quantized(2)}, bits=2, provenance=provenance)
    _save_qvq_prefix_artifact(second_path, module_results={"second": quantized(2)}, bits=2, provenance=provenance)
    _save_qvq_prefix_artifact(mixed_path, module_results={"second": quantized(1)}, bits=1, provenance=provenance)

    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 16, bias=False)
    model.second = torch.nn.Linear(16, 16, bias=False)
    manifests, replacements = _install_prefix_artifacts(
        model,
        paths=(first_path, second_path),
        model_path=model_path,
    )
    assert len(manifests) == 2
    assert set(replacements) == {"first", "second"}
    assert isinstance(model.first, QVQLinear)
    assert isinstance(model.second, QVQLinear)

    duplicate_model = torch.nn.Module()
    duplicate_model.first = torch.nn.Linear(16, 16, bias=False)
    duplicate_first = duplicate_model.first
    with pytest.raises(ValueError, match="duplicate modules"):
        _install_prefix_artifacts(
            duplicate_model,
            paths=(first_path, first_path),
            model_path=model_path,
        )
    assert duplicate_model.first is duplicate_first

    mixed_model = torch.nn.Module()
    mixed_model.first = torch.nn.Linear(16, 16, bias=False)
    mixed_model.second = torch.nn.Linear(16, 16, bias=False)
    mixed_first = mixed_model.first
    with pytest.raises(ValueError, match="one exact codec contract"):
        _install_prefix_artifacts(
            mixed_model,
            paths=(first_path, mixed_path),
            model_path=model_path,
        )
    assert mixed_model.first is mixed_first


def test_qvq_p4_confirmation_requires_kl_improvement_and_bounded_topn():
    baseline = {
        "finite": True,
        "kl_forward": {"mean": 0.1},
        "top1_agreement": 0.8,
        "top5_overlap": {"mean": 0.85},
        "top10_overlap": {"mean": 0.9},
    }
    proposal = {
        "finite": True,
        "kl_forward": {"mean": 0.09},
        "top1_agreement": 0.799,
        "top5_overlap": {"mean": 0.848},
        "top10_overlap": {"mean": 0.899},
    }
    assert _passes_confirmation(baseline, proposal, topn_regression_limit=0.0025)
    assert not _passes_confirmation(
        baseline,
        {**proposal, "kl_forward": {"mean": 0.09995}},
        topn_regression_limit=0.0025,
        minimum_relative_kl_improvement=0.001,
    )
    assert _passes_confirmation(
        baseline,
        {**proposal, "kl_forward": {"mean": 0.0998}},
        topn_regression_limit=0.0025,
        minimum_relative_kl_improvement=0.001,
    )
    assert not _passes_confirmation(
        baseline,
        {**proposal, "kl_forward": {"mean": 0.1}},
        topn_regression_limit=0.0025,
    )
    assert not _passes_confirmation(
        baseline,
        {**proposal, "top5_overlap": {"mean": 0.84}},
        topn_regression_limit=0.0025,
    )
    assert not _passes_confirmation(
        baseline,
        {**proposal, "finite": False},
        topn_regression_limit=0.0025,
    )


def test_qvq_full_horizon_replay_requires_every_search_fold_to_improve():
    assert _minimax_relative_replay_score((0.8, 1.8), (1.0, 2.0)) == pytest.approx(0.9)
    # The pooled mean improves from 1.5 to 1.45, but fold zero regresses.
    assert _minimax_relative_replay_score((1.1, 1.8), (1.0, 2.0)) == pytest.approx(1.1)
    assert math.isinf(_minimax_relative_replay_score((float("nan"), 1.8), (1.0, 2.0)))
    with pytest.raises(ValueError, match="non-empty and aligned"):
        _minimax_relative_replay_score((0.8,), (1.0, 2.0))


def test_qvq_streamed_metrics_match_monolithic_kl_and_topn_means():
    generator = torch.Generator().manual_seed(20260820)
    dense = torch.randn((11, 37), generator=generator)
    quantized = dense + torch.randn((11, 37), generator=generator) * 0.1
    expected = tensor_metrics(dense, quantized, normalize_distribution=False, include_top10=True)
    accumulator = _WeightedMetricAccumulator()
    for start, stop in ((0, 3), (3, 8), (8, 11)):
        accumulator.add(
            tensor_metrics(
                dense[start:stop],
                quantized[start:stop],
                normalize_distribution=False,
                include_top10=True,
            ),
            rows=stop - start,
        )
    actual = accumulator.result()

    assert actual["shape"] == [11, 37]
    for path in (
        ("kl_forward", "mean"),
        ("kl_reverse", "mean"),
        ("jensen_shannon", "mean"),
        ("top5_overlap", "mean"),
        ("top10_overlap", "mean"),
        ("top1_agreement",),
    ):
        expected_value = expected
        actual_value = actual
        for name in path:
            expected_value = expected_value[name]
            actual_value = actual_value[name]
        assert actual_value == pytest.approx(expected_value, abs=1e-7, rel=1e-7)


def test_qvq_v2b2_p32_native_mlx_conversion_preserves_selector_payload():
    pytest.importorskip("mlx.core")
    from gptqmodel.utils.mlx import _qvq_mlx_linear_from_torch
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    layer = QVQLinear(bits=2, in_features=16, out_features=16, bank_count=2, v2b2_p32=True)
    converted = _qvq_mlx_linear_from_torch(layer)

    assert isinstance(converted, QVQMLXLinear)
    assert converted.v2b2_p32 is True
    assert converted.v2b4_p64 is False
    assert converted.bank_ids.shape == (1,)
    assert converted.bank_alt_id.shape == (1,)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5))
def test_qvq_v2b2_p32_config_round_trip(bits):
    config = QVQConfig(bits=bits, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    assert config.vector_size == 2
    assert config.trellis_window == 16
    assert config.bank_count == 2
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.format == FORMAT.QVQ_V2B2_P32
    assert reloaded.quant_linear_init_kwargs()["v2b2_p32"] is True


def test_qvq_v2b2_p32_config_accepts_yaqa_and_weighted_block_ldlq():
    with pytest.raises(ValueError, match="W1 through W3.5"):
        QVQConfig(bits=4, format=FORMAT.QVQ_V2B2_P32, offload_to_disk=False)
    config = QVQConfig(bits=2, format=FORMAT.QVQ_V2B2_P32, rounding="yaqa", offload_to_disk=False)
    assert config.rounding == "yaqa"
    weighted = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        viterbi_objective="hessian_diagonal",
        offload_to_disk=False,
    )
    assert QVQConfig.from_quant_config(weighted.to_dict()).viterbi_objective == "hessian_diagonal"
    with pytest.raises(ValueError, match="one tail-biting candidate"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B2_P32,
            tail_biting_candidates=2,
            offload_to_disk=False,
        )
    with pytest.raises(ValueError, match="propagation replay"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B2_P32,
            propagated_bank_selection=True,
            offload_to_disk=False,
        )


@pytest.mark.parametrize("family_mode", ("fixed_block_ldlq", "reselect"))
def test_qvq_v2b2_p32_yaqa_family_mode_config_round_trip(family_mode):
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(v2b2_family_mode=family_mode),
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.v2b2_family_mode == family_mode


def test_qvq_v2b2_p32_yaqa_spectral_config_round_trip():
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(
            spectral_refinement=True,
            spectral_ranks=[4, 8, 4],
            spectral_lambdas=[0.25, 0.5, 0.25],
        ),
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.spectral_refinement is True
    assert reloaded.yaqa.spectral_ranks == (4, 8)
    assert reloaded.yaqa.spectral_lambdas == (0.25, 0.5)

    with pytest.raises(ValueError, match="requires `format=qvq_v2b2_p32`"):
        QVQConfig(
            bits=2,
            rounding="yaqa",
            yaqa=YaqaConfig(spectral_refinement=True),
            offload_to_disk=False,
        )


def test_qvq_v2b2_p32_yaqa_spectral_push_config_round_trip():
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(
            spectral_push=True,
            spectral_ranks=[4, 8, 4],
            spectral_push_alphas=[0.25, 0.5, 0.25],
        ),
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.spectral_push is True
    assert reloaded.yaqa.spectral_ranks == (4, 8)
    assert reloaded.yaqa.spectral_push_alphas == (0.25, 0.5)

    with pytest.raises(ValueError, match="mutually exclusive"):
        YaqaConfig(spectral_refinement=True, spectral_push=True)
    with pytest.raises(ValueError, match="spectral experiment requires"):
        QVQConfig(
            bits=2,
            rounding="yaqa",
            yaqa=YaqaConfig(spectral_push=True),
            offload_to_disk=False,
        )


def test_qvq_v2b2_p32_yaqa_localized_spectral_config_round_trip():
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(
            spectral_localized=True,
            spectral_ranks=[4, 8, 4],
            spectral_localized_alphas=[0.25, 0.5, 0.25],
            spectral_localized_max_segments=12,
            spectral_localized_max_changes=3,
            spectral_localized_replay_candidates=4,
        ),
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.yaqa.spectral_localized is True
    assert reloaded.yaqa.spectral_ranks == (4, 8)
    assert reloaded.yaqa.spectral_localized_alphas == (0.25, 0.5)
    assert reloaded.yaqa.spectral_localized_max_segments == 12
    assert reloaded.yaqa.spectral_localized_max_changes == 3
    assert reloaded.yaqa.spectral_localized_replay_candidates == 4

    with pytest.raises(ValueError, match="mutually exclusive"):
        YaqaConfig(spectral_push=True, spectral_localized=True)
    with pytest.raises(ValueError, match="positive integer"):
        YaqaConfig(spectral_localized_max_segments=0)
    with pytest.raises(ValueError, match="between 1"):
        YaqaConfig(spectral_localized_max_segments=2, spectral_localized_max_changes=3)
    with pytest.raises(ValueError, match="nonnegative integer"):
        YaqaConfig(spectral_localized_replay_candidates=-1)


def test_qvq_v2b2_p32_localized_propagation_requires_the_exact_yaqa_mode():
    config = QVQConfig(
        bits=2,
        format=FORMAT.QVQ_V2B2_P32,
        rounding="yaqa",
        yaqa=YaqaConfig(spectral_localized=True),
        propagated_bank_selection=True,
        offload_to_disk=False,
    )
    reloaded = QVQConfig.from_quant_config(config.to_dict())
    assert reloaded.propagated_bank_selection is True
    assert reloaded.yaqa.spectral_localized is True

    with pytest.raises(ValueError, match="propagation replay"):
        QVQConfig(
            bits=2,
            format=FORMAT.QVQ_V2B2_P32,
            propagated_bank_selection=True,
            offload_to_disk=False,
        )


@pytest.mark.parametrize("family_mode", (None, "unknown"))
def test_qvq_v2b2_p32_rejects_invalid_yaqa_family_mode(family_mode):
    error = TypeError if family_mode is None else ValueError
    with pytest.raises(error, match="v2b2_family_mode"):
        YaqaConfig(v2b2_family_mode=family_mode)


def test_qvq_v2b2_p32_binary_selector_round_trip_and_validation():
    selectors = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.uint8)
    packed = pack_qvq_binary_bank_ids(selectors)
    assert packed.tolist() == [0b10010110, 0b00000001]
    assert torch.equal(unpack_qvq_binary_bank_ids(packed, selectors.numel()), selectors)
    with pytest.raises(ValueError, match="binary"):
        pack_qvq_binary_bank_ids(torch.tensor([0, 2], dtype=torch.uint8))
    with pytest.raises(ValueError, match="invalid selector count"):
        unpack_qvq_binary_bank_ids(torch.zeros(2, dtype=torch.uint8), 8)


def test_qvq_v2b2_p32_identical_banks_reproduce_canonical_v2_path():
    generator = torch.Generator().manual_seed(20260815)
    sequences = torch.randn((1, 128, 2), generator=generator)
    canonical = pgc16_codebook(dtype=torch.float32)
    v2 = tail_biting_viterbi_quantize(sequences, canonical, bits=2.5, candidate_count=1)
    v2b2 = tail_biting_v2b2_p32_quantize(
        sequences,
        torch.stack((canonical, canonical)),
        bits=2.5,
    )
    assert torch.equal(v2b2.states, v2.states)
    assert torch.equal(v2b2.values, v2.values)
    assert torch.equal(v2b2.squared_error, v2.squared_error)
    assert torch.count_nonzero(v2b2.segment_bank_ids) == 0


def test_qvq_v2b2_p32_selector_round_trip_reconstructs_selected_alternative():
    generator = torch.Generator().manual_seed(47)
    sequence = torch.randn((1, 128, 2), generator=generator)
    alt_id = 3
    banks = torch.stack(
        (
            pgc16_codebook_v2_bank(0, bits=2),
            pgc16_codebook_v2_bank(alt_id, bits=2),
        )
    )
    result = tail_biting_v2b2_p32_quantize(sequence, banks, bits=2)
    trellis = pack_trellis_states(result.states, bits=2)
    packed_selectors = pack_qvq_binary_bank_ids(result.segment_bank_ids.reshape(-1))
    actual = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=packed_selectors,
        v2b2_p32=True,
        bank_alt_id=torch.tensor([alt_id], dtype=torch.uint8),
    )
    assert packed_selectors.numel() == 1
    assert torch.equal(
        unpack_qvq_binary_bank_ids(packed_selectors, QVQ_V2B2_P32_SEGMENTS_PER_TILE),
        result.segment_bank_ids.reshape(-1),
    )
    torch.testing.assert_close(actual, result.values.reshape(16, 16), rtol=0, atol=0)


@pytest.mark.parametrize("bits", (1, 2, 3.5))
@pytest.mark.parametrize("segment_id", (0, 3, 7))
def test_qvq_v2b2_p32_fixed_boundary_segment_recovers_exact_baseline(bits, segment_id):
    generator = torch.Generator().manual_seed(20260816)
    sequence = torch.randn((1, 128, 2), generator=generator)
    banks = torch.stack(
        (
            pgc16_codebook_v2_bank(0, bits=bits),
            pgc16_codebook_v2_bank(2, bits=bits),
        )
    )
    baseline = tail_biting_v2b2_p32_quantize(sequence, banks, bits=bits)
    start = segment_id * 16
    stop = start + 16
    entry = baseline.states[:, start - 1] if start else baseline.states[:, -1]
    localized = fixed_boundary_v2b2_p32_segment_quantize(
        baseline.values[:, start:stop],
        banks,
        bits=bits,
        entry_states=entry,
        exit_states=baseline.states[:, stop - 1],
    )

    assert torch.equal(localized.states, baseline.states[:, start:stop])
    assert torch.equal(localized.values, baseline.values[:, start:stop])
    assert torch.equal(localized.segment_bank_ids[:, 0], baseline.segment_bank_ids[:, segment_id])
    assert torch.equal(localized.squared_error, torch.zeros_like(localized.squared_error))


def test_qvq_v2b2_p32_fixed_boundary_segment_changes_only_interior_states():
    generator = torch.Generator().manual_seed(20260817)
    sequence = torch.randn((1, 128, 2), generator=generator)
    banks = torch.stack(
        (
            pgc16_codebook_v2_bank(0, bits=2),
            pgc16_codebook_v2_bank(3, bits=2),
        )
    )
    baseline = tail_biting_v2b2_p32_quantize(sequence, banks, bits=2)
    segment_id = 4
    start = segment_id * 16
    stop = start + 16
    target = banks[1, baseline.states[:, start:stop]].clone()
    localized = fixed_boundary_v2b2_p32_segment_quantize(
        target,
        banks,
        bits=2,
        entry_states=baseline.states[:, start - 1],
        exit_states=baseline.states[:, stop - 1],
    )

    assert torch.equal(localized.states[:, -1], baseline.states[:, stop - 1])
    assert torch.equal(localized.states[:, 0] >> 4, baseline.states[:, start - 1] & ((1 << 12) - 1))
    assert localized.states.shape == (1, 16)
    assert localized.segment_bank_ids.shape == (1, 1)
    assert torch.isfinite(localized.squared_error).all()


def test_qvq_v2b2_p32_localized_spectral_refinement_recovers_one_segment_without_path_avalanche():
    generator = torch.Generator().manual_seed(20260818)
    library = tuple(pgc16_codebook_v2_bank(bank, bits=2) for bank in range(4))
    pair = torch.stack((library[0], library[2]))
    baseline_path = tail_biting_v2b2_p32_quantize(
        torch.randn((1, 128, 2), generator=generator),
        pair,
        bits=2,
    )
    segment_id = 3
    start = segment_id * 16
    stop = start + 16
    alternate = fixed_boundary_v2b2_p32_segment_quantize(
        torch.randn((1, 16, 2), generator=generator),
        pair,
        bits=2,
        entry_states=baseline_path.states[:, start - 1],
        exit_states=baseline_path.states[:, stop - 1],
    )
    assert not torch.equal(alternate.states, baseline_path.states[:, start:stop])

    baseline_weight = baseline_path.values.reshape(16, 16)
    source = baseline_weight.clone()
    source[segment_id * 2 : segment_id * 2 + 2] = alternate.values.reshape(2, 16)
    baseline = (
        baseline_weight,
        baseline_path.states,
        baseline_path.segment_bank_ids.reshape(-1),
        torch.tensor([2], dtype=torch.uint8),
    )
    diagnostics = {}
    with patch(
        "gptqmodel.eora.eora._eora_compute_svd",
        side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
    ):
        refined = yaqa_localized_spectral_refine_v2b2_p32(
            source,
            torch.eye(16),
            torch.eye(16),
            library,
            baseline,
            ranks=(2,),
            alphas=(1.0,),
            max_segments=1,
            diagnostics=diagnostics,
            bits=2,
        )

    torch.testing.assert_close(refined[0], source, rtol=0, atol=0)
    assert torch.equal(refined[1][:, :start], baseline_path.states[:, :start])
    assert torch.equal(refined[1][:, stop:], baseline_path.states[:, stop:])
    assert torch.equal(refined[1][:, stop - 1], baseline_path.states[:, stop - 1])
    assert torch.count_nonzero(refined[2] != baseline[2]) <= 1
    assert torch.equal(refined[3], baseline[3])
    assert diagnostics["spectral_method"] == "localized_p32"
    assert diagnostics["spectral_selected"] is True
    assert diagnostics["localized_boundary_preserved"] is True
    assert diagnostics["spectral_selected_loss"] == pytest.approx(0.0, abs=1e-6)
    assert sum(candidate["selected"] for candidate in diagnostics["spectral_candidates"].values()) == 1


def test_qvq_v2b2_p32_localized_spectral_refinement_composes_two_fixed_boundary_segments():
    generator = torch.Generator().manual_seed(20260827)
    library = tuple(pgc16_codebook_v2_bank(bank, bits=2) for bank in range(4))
    pair = torch.stack((library[0], library[2]))
    baseline_path = tail_biting_v2b2_p32_quantize(
        torch.randn((1, 128, 2), generator=generator),
        pair,
        bits=2,
    )
    segment_ids = (1, 6)
    source = baseline_path.values.reshape(16, 16).clone()
    expected_states = baseline_path.states.clone()
    expected_selectors = baseline_path.segment_bank_ids.clone()
    for segment_id in segment_ids:
        start = segment_id * 16
        stop = start + 16
        alternate = fixed_boundary_v2b2_p32_segment_quantize(
            torch.randn((1, 16, 2), generator=generator),
            pair,
            bits=2,
            entry_states=baseline_path.states[:, start - 1],
            exit_states=baseline_path.states[:, stop - 1],
        )
        assert not torch.equal(alternate.states, baseline_path.states[:, start:stop])
        source[segment_id * 2 : segment_id * 2 + 2] = alternate.values.reshape(2, 16)
        expected_states[:, start:stop] = alternate.states
        expected_selectors[:, segment_id] = alternate.segment_bank_ids[:, 0]

    baseline = (
        baseline_path.values.reshape(16, 16),
        baseline_path.states,
        baseline_path.segment_bank_ids.reshape(-1),
        torch.tensor([2], dtype=torch.uint8),
    )
    diagnostics = {}
    with patch(
        "gptqmodel.eora.eora._eora_compute_svd",
        side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
    ):
        refined = yaqa_localized_spectral_refine_v2b2_p32(
            source,
            torch.eye(16),
            torch.eye(16),
            library,
            baseline,
            ranks=(4,),
            alphas=(1.0,),
            max_segments=2,
            max_changes=2,
            diagnostics=diagnostics,
            bits=2,
        )

    torch.testing.assert_close(refined[0], source, rtol=0, atol=0)
    assert torch.equal(refined[1], expected_states)
    assert torch.equal(refined[2], expected_selectors.reshape(-1))
    assert diagnostics["localized_selected_changes"] == 2
    assert len(diagnostics["localized_selected_candidates"]) == 2
    assert sum(candidate["selected"] for candidate in diagnostics["spectral_candidates"].values()) == 2
    assert diagnostics["spectral_selected_loss"] == pytest.approx(0.0, abs=1e-6)

    # The second segment is locally useful for the dense source above but is
    # harmful to this live-output target. Conditional recomputation must stop
    # after the first replacement instead of summing stale standalone gains.
    one_change_target = baseline[0].clone()
    selected_segment = segment_ids[0]
    selected_rows = slice(selected_segment * 2, selected_segment * 2 + 2)
    one_change_target[selected_rows] = source[selected_rows]
    conditional_diagnostics = {}
    with patch(
        "gptqmodel.eora.eora._eora_compute_svd",
        side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
    ):
        conditional = yaqa_localized_spectral_refine_v2b2_p32(
            source,
            torch.eye(16),
            torch.eye(16),
            library,
            baseline,
            ranks=(4,),
            alphas=(1.0,),
            max_segments=2,
            max_changes=2,
            search_inputs=torch.eye(16),
            search_target=one_change_target,
            diagnostics=conditional_diagnostics,
            bits=2,
        )

    torch.testing.assert_close(conditional[0], one_change_target, rtol=0, atol=0)
    assert conditional_diagnostics["localized_selected_changes"] == 1
    assert sum(candidate["selected"] for candidate in conditional_diagnostics["spectral_candidates"].values()) == 1

    replay_target = one_change_target
    replay_calls = []

    def replay_score(candidate_weight, candidate_states, candidate_selectors, candidate_alt_id):
        replay_calls.append(
            (
                candidate_weight.clone(),
                candidate_states.clone(),
                candidate_selectors.clone(),
                candidate_alt_id.clone(),
            )
        )
        return float((candidate_weight - replay_target).square().sum().item())

    replay_diagnostics = {}
    with patch(
        "gptqmodel.eora.eora._eora_compute_svd",
        side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
    ):
        replay_refined = yaqa_localized_spectral_refine_v2b2_p32(
            source,
            torch.eye(16),
            torch.eye(16),
            library,
            baseline,
            ranks=(4,),
            alphas=(1.0,),
            max_segments=2,
            max_changes=2,
            replay_candidates=2,
            candidate_score=replay_score,
            diagnostics=replay_diagnostics,
            bits=2,
        )

    torch.testing.assert_close(replay_refined[0], replay_target, rtol=0, atol=0)
    assert replay_diagnostics["localized_selected_changes"] == 1
    assert replay_diagnostics["localized_replay_candidates"] == 2
    assert replay_diagnostics["localized_replay_callback_error"] is False
    assert len(replay_calls) == 4


def test_qvq_v2b2_p32_localized_full_horizon_baseline_error_fails_closed():
    generator = torch.Generator().manual_seed(20260901)
    library = tuple(pgc16_codebook_v2_bank(bank, bits=2) for bank in range(4))
    pair = torch.stack((library[0], library[1]))
    source = torch.randn((16, 16), generator=generator)
    baseline_path = tail_biting_v2b2_p32_quantize(source.reshape(1, 128, 2), pair, bits=2)
    baseline = (
        baseline_path.values.reshape(16, 16),
        baseline_path.states,
        baseline_path.segment_bank_ids.reshape(-1),
        torch.tensor([1], dtype=torch.uint8),
    )
    diagnostics = {}

    with patch(
        "gptqmodel.eora.eora._eora_compute_svd",
        side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
    ):
        result = yaqa_localized_spectral_refine_v2b2_p32(
            source,
            torch.eye(16),
            torch.eye(16),
            library,
            baseline,
            ranks=(4,),
            alphas=(1.0,),
            max_segments=2,
            replay_candidates=2,
            candidate_score=lambda *_: (_ for _ in ()).throw(RuntimeError("replay failed")),
            diagnostics=diagnostics,
            bits=2,
        )

    for actual, expected in zip(result, baseline, strict=True):
        assert torch.equal(actual, expected)
    assert diagnostics["localized_replay_callback_error"] is True
    assert diagnostics["localized_selected_changes"] == 0


@pytest.mark.parametrize("bits", (2, 3, 3.5))
def test_qvq_v2b2_p32_block_ldlq_pack_reload_and_torch_forward(bits):
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    result = quantize_qvq_linear(
        weight,
        torch.eye(16),
        bits=bits,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    tensors = result.serialized_tensors()
    assert result.bank_ids is not None and result.bank_ids.numel() == 8
    assert tensors["bank_ids"].dtype == torch.uint8
    assert tensors["bank_ids"].numel() == 1
    assert tensors["bank_alt_id"].shape == (1,)
    assert 1 <= int(tensors["bank_alt_id"].item()) <= 3
    decoded = reconstruct_qvq_inner_weight(
        result.trellis,
        bits=bits,
        in_features=16,
        out_features=16,
        bank_ids=tensors["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=tensors["bank_alt_id"],
    )
    torch.testing.assert_close(decoded, result.inner_weight, rtol=0, atol=0)

    layer = QVQLinear(
        bits=bits,
        in_features=16,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        tensors=tensors,
    ).eval()
    shell = QVQLinear(bits=bits, in_features=16, out_features=16, bank_count=2, v2b2_p32=True)
    shell.load_state_dict(layer.state_dict(), strict=True)
    x = torch.randn((3, 16), generator=generator)
    torch.testing.assert_close(layer(x), x @ result.weight.T, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(shell(x), layer(x), rtol=0, atol=0)


@pytest.mark.parametrize("viterbi_objective", ("euclidean", "hessian_diagonal"))
def test_qvq_v2b2_p32_full_proxy_cannot_regress_independent_v2_oracle(viterbi_objective):
    generator = torch.Generator().manual_seed(20260815)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    calibration = torch.randn((97, 16), generator=generator)
    hessian = calibration.T @ calibration / calibration.shape[0]
    canonical = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        trellis_batch_size=1,
        viterbi_objective=viterbi_objective,
    )
    banked = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
        viterbi_objective=viterbi_objective,
    )
    assert torch.isfinite(banked.proxy_loss)
    assert banked.proxy_loss <= canonical.proxy_loss
    assert banked.bank_ids is not None
    assert banked.bank_alt_id is not None
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=banked.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=banked.bank_alt_id,
    )
    torch.testing.assert_close(decoded, banked.inner_weight, rtol=0, atol=0)


def test_qvq_v2b2_p32_yaqa_pack_reload_and_full_proxy_cannot_regress_v2_yaqa():
    generator = torch.Generator().manual_seed(20260818)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_hessian = torch.eye(16)
    output_hessian = torch.eye(16)
    canonical = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        trellis_batch_size=1,
    )
    banked = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    block_control = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    assert banked.kronecker_proxy_loss <= canonical.kronecker_proxy_loss
    assert banked.bank_ids is not None and banked.bank_ids.numel() == 8
    assert banked.bank_alt_id is not None
    assert isinstance(banked.yaqa_bank_fallback_to_v2, bool)
    assert banked.yaqa_selector_churn is not None and 0.0 <= banked.yaqa_selector_churn <= 1.0
    assert isinstance(banked.yaqa_family_changed, bool)
    assert banked.yaqa_block_family_id in {1, 2, 3}
    decoded = reconstruct_qvq_inner_weight(
        banked.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=banked.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=banked.bank_alt_id,
    )
    torch.testing.assert_close(decoded, banked.inner_weight, rtol=0, atol=0)
    assert torch.equal(banked.trellis, block_control.trellis)
    assert torch.equal(banked.bank_ids, block_control.bank_ids)
    assert torch.equal(banked.bank_alt_id, block_control.bank_alt_id)


def test_qvq_v2b2_p32_yaqa_output_spectral_candidate_uses_original_proxy_and_keeps_baseline():
    source = torch.zeros((8, 8))
    input_hessian = torch.eye(8)
    output_hessian = torch.eye(8)
    baseline_weight = torch.eye(8)
    states = torch.zeros((1, 32), dtype=torch.long)
    selectors = torch.zeros(8, dtype=torch.uint8)
    alt_id = torch.tensor([1], dtype=torch.uint8)
    baseline = baseline_weight, states, selectors, alt_id
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4))
    boosted_factors = []

    def fake_candidate(_weight, _input, candidate_output, *_args, **_kwargs):
        boosted_factors.append(candidate_output.clone())
        value = 0.5 if len(boosted_factors) == 1 else 2.0
        return (
            torch.eye(8) * value,
            states,
            torch.ones_like(selectors),
            torch.tensor([2], dtype=torch.uint8),
        )

    diagnostics = {}
    with (
        patch(
            "gptqmodel.eora.eora._eora_compute_svd",
            side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix),
        ),
        patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", side_effect=fake_candidate),
    ):
        actual = yaqa_output_spectral_refine_v2b2_p32(
            source,
            input_hessian,
            output_hessian,
            codebooks,
            baseline,
            ranks=(1,),
            lambdas=(0.25, 1.0),
            bits=2,
            diagnostics=diagnostics,
        )

    assert torch.equal(actual[0], torch.eye(8) * 0.5)
    assert diagnostics["spectral_selected"] is True
    assert diagnostics["spectral_rank"] == 1
    assert diagnostics["spectral_lambda"] == 0.25
    assert diagnostics["spectral_selected_loss"] < diagnostics["spectral_original_loss"]
    assert diagnostics["spectral_selector_churn"] == 1.0
    assert diagnostics["spectral_family_changed"] is True
    assert len(boosted_factors) == 2
    for strength, boosted in zip((0.25, 1.0), boosted_factors, strict=True):
        assert float(boosted.trace()) == pytest.approx(8.0 * (1.0 + strength))


def test_qvq_v2b2_p32_yaqa_spectral_push_uses_low_rank_target_and_original_proxy():
    source = torch.zeros((8, 8))
    input_hessian = torch.eye(8)
    output_hessian = torch.eye(8)
    baseline_weight = torch.eye(8)
    states = torch.zeros((1, 32), dtype=torch.long)
    selectors = torch.zeros(8, dtype=torch.uint8)
    alt_id = torch.tensor([1], dtype=torch.uint8)
    baseline = baseline_weight, states, selectors, alt_id
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4))
    rounding_biases = []

    def fake_candidate(_weight, _input, _output, *_args, **kwargs):
        rounding_biases.append(kwargs["_rounding_bias"].clone())
        kwargs["diagnostics"].update(
            {
                "fallback_to_v2": False,
                "family_candidates": {"1": {"fallback_to_bank0": False}},
            }
        )
        value = 0.5 if len(rounding_biases) == 1 else 2.0
        return (
            torch.eye(8) * value,
            states,
            torch.ones_like(selectors),
            torch.tensor([2], dtype=torch.uint8),
        )

    diagnostics = {}
    with (
        patch(
            "gptqmodel.eora.eora._eora_compute_svd",
            side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
        ),
        patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", side_effect=fake_candidate),
    ):
        actual = yaqa_spectral_push_v2b2_p32(
            source,
            input_hessian,
            output_hessian,
            codebooks,
            baseline,
            ranks=(8,),
            alphas=(0.5, 1.0),
            bits=2,
            diagnostics=diagnostics,
        )

    assert torch.equal(actual[0], torch.eye(8) * 0.5)
    assert diagnostics["spectral_method"] == "push"
    assert diagnostics["spectral_selected"] is True
    assert diagnostics["spectral_rank"] == 8
    assert diagnostics["spectral_lambda"] is None
    assert diagnostics["spectral_alpha"] == 0.5
    assert diagnostics["spectral_oracle_losses"]["8"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["spectral_selected_loss"] < diagnostics["spectral_original_loss"]
    assert diagnostics["spectral_selector_churn"] == 1.0
    assert diagnostics["spectral_family_changed"] is True
    assert set(diagnostics["spectral_candidates"]) == {"r8_a0.5", "r8_a1"}
    selected = diagnostics["spectral_candidates"]["r8_a0.5"]
    assert selected["selected"] is True
    assert selected["relative_improvement"] > 0
    assert selected["state_churn"] == 0.0
    assert selected["selector_churn"] == 1.0
    assert selected["family_changed"] is True
    assert selected["fallback_to_v2"] is False
    assert selected["family_candidates"]["1"]["fallback_to_bank0"] is False
    assert diagnostics["spectral_candidates"]["r8_a1"]["selected"] is False
    torch.testing.assert_close(rounding_biases[0], -0.5 * torch.eye(8), rtol=0, atol=1e-6)
    torch.testing.assert_close(rounding_biases[1], -torch.eye(8), rtol=0, atol=1e-6)


def test_qvq_v2b2_p32_yaqa_spectral_push_preserves_rejected_candidate_diagnostics():
    source = torch.zeros((8, 8))
    baseline_weight = torch.eye(8)
    baseline_states = torch.zeros((1, 32), dtype=torch.long)
    baseline_selectors = torch.zeros(8, dtype=torch.uint8)
    baseline_alt_id = torch.tensor([1], dtype=torch.uint8)
    baseline = baseline_weight, baseline_states, baseline_selectors, baseline_alt_id
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4))
    calls = 0

    def rejected_candidate(_weight, _input, _output, *_args, **kwargs):
        nonlocal calls
        calls += 1
        internally_rolled_back = calls == 2
        kwargs["diagnostics"].update(
            {
                "fallback_to_v2": internally_rolled_back,
                "family_candidates": {
                    "1": {
                        "fallback_to_bank0": internally_rolled_back,
                        "pre_fallback_selector_churn": 0.75,
                        "pre_fallback_state_churn": 0.5,
                    }
                },
            }
        )
        if internally_rolled_back:
            return baseline
        return (
            torch.eye(8) * 2,
            torch.ones_like(baseline_states),
            torch.ones_like(baseline_selectors),
            torch.tensor([2], dtype=torch.uint8),
        )

    diagnostics = {}
    with (
        patch(
            "gptqmodel.eora.eora._eora_compute_svd",
            side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
        ),
        patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", side_effect=rejected_candidate),
    ):
        actual = yaqa_spectral_push_v2b2_p32(
            source,
            torch.eye(8),
            torch.eye(8),
            codebooks,
            baseline,
            ranks=(8,),
            alphas=(0.5, 1.0),
            bits=2,
            diagnostics=diagnostics,
        )

    assert all(torch.equal(left, right) for left, right in zip(actual, baseline, strict=True))
    assert diagnostics["spectral_selected"] is False
    assert diagnostics["spectral_absorption_efficiency"] == 0.0
    changed = diagnostics["spectral_candidates"]["r8_a0.5"]
    assert changed["relative_improvement"] < 0
    assert changed["state_churn"] == 1.0
    assert changed["selector_churn"] == 1.0
    assert changed["family_changed"] is True
    assert changed["fallback_to_v2"] is False
    rolled_back = diagnostics["spectral_candidates"]["r8_a1"]
    assert rolled_back["relative_improvement"] == 0.0
    assert rolled_back["state_churn"] == 0.0
    assert rolled_back["selector_churn"] == 0.0
    assert rolled_back["fallback_to_v2"] is True
    assert rolled_back["family_candidates"]["1"]["pre_fallback_selector_churn"] == 0.75


def test_qvq_v2b2_p32_yaqa_spectral_push_rejects_invalid_controls():
    source = torch.zeros((8, 8))
    baseline = (
        torch.zeros_like(source),
        torch.zeros((1, 32), dtype=torch.long),
        torch.zeros(8, dtype=torch.uint8),
        torch.tensor([1], dtype=torch.uint8),
    )
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4))
    with pytest.raises(ValueError, match="ranks"):
        yaqa_spectral_push_v2b2_p32(
            source, torch.eye(8), torch.eye(8), codebooks, baseline, ranks=(0,), alphas=(0.5,), bits=2
        )
    with pytest.raises(ValueError, match="alphas"):
        yaqa_spectral_push_v2b2_p32(
            source, torch.eye(8), torch.eye(8), codebooks, baseline, ranks=(4,), alphas=(float("nan"),), bits=2
        )


def test_qvq_v2b2_p32_yaqa_output_spectral_quantization_is_serialization_neutral_and_nonregressive():
    generator = torch.Generator().manual_seed(20260821)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((37, 16), generator=generator)
    output_samples = torch.randn((39, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    baseline = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    refined = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_spectral_refinement=True,
        yaqa_spectral_ranks=(1,),
        yaqa_spectral_lambdas=(0.25,),
        trellis_batch_size=1,
    )

    assert refined.kronecker_proxy_loss <= baseline.kronecker_proxy_loss
    assert isinstance(refined.yaqa_spectral_selected, bool)
    assert set(refined.yaqa_spectral_concentration) == {"1"}
    assert 0.0 <= refined.yaqa_spectral_concentration["1"] <= 1.0 + 1e-5
    assert set(refined.serialized_tensors()) == set(baseline.serialized_tensors())
    decoded = reconstruct_qvq_inner_weight(
        refined.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=refined.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=refined.bank_alt_id,
    )
    torch.testing.assert_close(decoded, refined.inner_weight, rtol=0, atol=0)

    live = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        tensors=refined.serialized_tensors(),
    ).eval()
    reloaded = QVQLinear(bits=2, in_features=16, out_features=16, bank_count=2, v2b2_p32=True).eval()
    reloaded.load_state_dict(live.state_dict(), strict=True)
    inputs = torch.randn((7, 16), generator=generator)
    torch.testing.assert_close(reloaded(inputs), live(inputs), rtol=0, atol=0)
    torch.testing.assert_close(live(inputs), inputs @ refined.weight.T, rtol=1e-5, atol=1e-6)


def test_qvq_v2b2_p32_yaqa_spectral_push_is_serialization_neutral_and_nonregressive():
    generator = torch.Generator().manual_seed(20260824)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((37, 16), generator=generator)
    output_samples = torch.randn((39, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    baseline = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    pushed = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_spectral_push=True,
        yaqa_spectral_ranks=(1,),
        yaqa_spectral_push_alphas=(0.25, 0.5),
        trellis_batch_size=1,
    )

    assert pushed.kronecker_proxy_loss <= baseline.kronecker_proxy_loss
    assert pushed.yaqa_spectral_method == "push"
    assert pushed.yaqa_spectral_lambda is None
    assert pushed.yaqa_spectral_alpha is None or pushed.yaqa_spectral_alpha in (0.25, 0.5)
    assert set(pushed.yaqa_spectral_concentration) == {"1"}
    assert set(pushed.yaqa_spectral_oracle_losses) == {"1"}
    assert set(pushed.yaqa_spectral_candidates) == {"r1_a0.25", "r1_a0.5"}
    for candidate in pushed.yaqa_spectral_candidates.values():
        assert set(candidate) >= {
            "loss",
            "relative_improvement",
            "state_churn",
            "selector_churn",
            "spectral_alignment",
            "fallback_to_v2",
            "family_candidates",
            "selected",
        }
        assert candidate["family_candidates"] is not None
    assert set(pushed.serialized_tensors()) == set(baseline.serialized_tensors())
    decoded = reconstruct_qvq_inner_weight(
        pushed.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=pushed.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=pushed.bank_alt_id,
    )
    torch.testing.assert_close(decoded, pushed.inner_weight, rtol=0, atol=0)


def test_qvq_v2b2_p32_yaqa_localized_spectral_is_serialization_neutral_and_nonregressive():
    generator = torch.Generator().manual_seed(20260825)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((37, 16), generator=generator)
    output_samples = torch.randn((39, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    baseline = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    refined = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_spectral_localized=True,
        yaqa_spectral_ranks=(1,),
        yaqa_spectral_localized_alphas=(0.5, 1.0),
        yaqa_spectral_localized_max_segments=4,
        trellis_batch_size=1,
    )

    assert refined.kronecker_proxy_loss <= baseline.kronecker_proxy_loss
    assert refined.yaqa_spectral_method == "localized_p32"
    assert refined.yaqa_spectral_family_changed is False
    assert refined.yaqa_spectral_selector_churn <= 1 / QVQ_V2B2_P32_SEGMENTS_PER_TILE
    assert set(refined.serialized_tensors()) == set(baseline.serialized_tensors())
    decoded = reconstruct_qvq_inner_weight(
        refined.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=refined.serialized_tensors()["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=refined.bank_alt_id,
    )
    torch.testing.assert_close(decoded, refined.inner_weight, rtol=0, atol=0)


def test_qvq_v2b2_p32_localized_propagation_accepts_or_atomically_rolls_back():
    generator = torch.Generator().manual_seed(20260831)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((37, 16), generator=generator)
    output_samples = torch.randn((39, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    search_inputs = torch.randn((23, 16), generator=generator)
    search_targets = search_inputs @ weight.T
    common = {
        "bits": 2,
        "rounding": "yaqa",
        "output_hessian": output_hessian,
        "bank_count": 2,
        "v2b2_p32": True,
        "trellis_batch_size": 1,
    }
    baseline = quantize_qvq_linear(weight, input_hessian, **common)
    callback_pairs = []

    def reject(proposal, rollback):
        callback_pairs.append((proposal.clone(), rollback.clone()))
        return False

    localized = {
        "yaqa_spectral_localized": True,
        "yaqa_spectral_ranks": (8, 16),
        "yaqa_spectral_localized_alphas": (1.0, 2.0, 4.0, 8.0),
        "yaqa_spectral_localized_max_segments": 8,
        "propagated_inputs": search_inputs,
        "propagated_target_output": search_targets,
    }
    with pytest.raises(ValueError, match="candidate scorer and a positive shortlist"):
        quantize_qvq_linear(
            weight,
            input_hessian,
            propagated_acceptance=lambda proposal, rollback: True,
            yaqa_spectral_localized_replay_candidates=1,
            **common,
            **localized,
        )
    with pytest.raises(ValueError, match="candidate scorer and a positive shortlist"):
        quantize_qvq_linear(
            weight,
            input_hessian,
            propagated_acceptance=lambda proposal, rollback: True,
            propagated_candidate_score=lambda candidate: 0.0,
            **common,
            **localized,
        )
    rejected = quantize_qvq_linear(
        weight,
        input_hessian,
        propagated_acceptance=reject,
        **common,
        **localized,
    )
    accepted = quantize_qvq_linear(
        weight,
        input_hessian,
        propagated_acceptance=lambda proposal, rollback: True,
        **common,
        **localized,
    )
    replayed_weights = []

    def replay_score(candidate):
        replayed_weights.append(candidate.clone())
        return -float(len(replayed_weights))

    replayed = quantize_qvq_linear(
        weight,
        input_hessian,
        propagated_acceptance=lambda proposal, rollback: True,
        propagated_candidate_score=replay_score,
        yaqa_spectral_localized_replay_candidates=2,
        **common,
        **localized,
    )
    callback_error = quantize_qvq_linear(
        weight,
        input_hessian,
        propagated_acceptance=lambda proposal, rollback: (_ for _ in ()).throw(RuntimeError("gate failed")),
        **common,
        **localized,
    )

    assert len(callback_pairs) == 1
    assert rejected.yaqa_spectral_selected is False
    assert callback_error.yaqa_spectral_selected is False
    assert accepted.yaqa_spectral_selected is True
    assert replayed.yaqa_spectral_selected is True
    assert len(replayed_weights) >= 2
    assert any(torch.equal(candidate, replayed.weight) for candidate in replayed_weights)
    torch.testing.assert_close(callback_pairs[0][1], baseline.weight, rtol=0, atol=0)
    torch.testing.assert_close(callback_pairs[0][0], accepted.weight, rtol=0, atol=0)
    for name in ("trellis", "bank_ids", "bank_alt_id"):
        assert torch.equal(rejected.serialized_tensors()[name], baseline.serialized_tensors()[name])
        assert torch.equal(callback_error.serialized_tensors()[name], baseline.serialized_tensors()[name])
    torch.testing.assert_close(rejected.weight, baseline.weight, rtol=0, atol=0)
    assert not torch.equal(accepted.trellis, baseline.trellis)
    decoded = reconstruct_qvq_inner_weight(
        accepted.trellis,
        bits=2,
        in_features=16,
        out_features=16,
        bank_ids=accepted.bank_ids,
        v2b2_p32=True,
        bank_alt_id=accepted.bank_alt_id,
    )
    torch.testing.assert_close(decoded, accepted.inner_weight, rtol=0, atol=0)


def test_qvq_v2b2_p32_yaqa_randomized_spectral_subspace_matches_exact_control():
    generator = torch.Generator().manual_seed(20260822)
    left = torch.randn((16, 4), generator=generator)
    right = torch.randn((4, 16), generator=generator)
    source = left @ right + 1e-3 * torch.randn((16, 16), generator=generator)
    baseline = (
        torch.zeros_like(source),
        torch.zeros((1, 128), dtype=torch.long),
        torch.zeros(8, dtype=torch.uint8),
        torch.tensor([1], dtype=torch.uint8),
    )
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32) for index in range(4))

    def run(*, exact: bool):
        boosted = []

        def keep_baseline(_weight, _input, candidate_output, *_args, **_kwargs):
            boosted.append(candidate_output.clone())
            return baseline

        diagnostics = {}
        svd_patch = (
            patch(
                "gptqmodel.eora.eora._eora_compute_svd",
                side_effect=lambda matrix, rank, algo: torch.linalg.svd(matrix, full_matrices=False),
            )
            if exact
            else patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", side_effect=keep_baseline)
        )
        contexts = (
            (
                patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", side_effect=keep_baseline),
                svd_patch,
            )
            if exact
            else (svd_patch,)
        )
        with contexts[0]:
            if len(contexts) == 2:
                with contexts[1]:
                    result = yaqa_output_spectral_refine_v2b2_p32(
                        source,
                        torch.eye(16),
                        torch.eye(16),
                        codebooks,
                        baseline,
                        ranks=(4,),
                        lambdas=(0.25,),
                        bits=2,
                        diagnostics=diagnostics,
                    )
            else:
                result = yaqa_output_spectral_refine_v2b2_p32(
                    source,
                    torch.eye(16),
                    torch.eye(16),
                    codebooks,
                    baseline,
                    ranks=(4,),
                    lambdas=(0.25,),
                    bits=2,
                    diagnostics=diagnostics,
                )
        return result, diagnostics, boosted

    approximate, approximate_diagnostics, approximate_boosted = run(exact=False)
    exact, exact_diagnostics, exact_boosted = run(exact=True)
    assert torch.equal(approximate[0], baseline[0])
    assert torch.equal(exact[0], baseline[0])
    assert approximate_diagnostics["spectral_concentration"]["4"] == pytest.approx(
        exact_diagnostics["spectral_concentration"]["4"], rel=1e-4, abs=1e-6
    )
    torch.testing.assert_close(approximate_boosted[0], exact_boosted[0], rtol=2e-3, atol=2e-3)


def test_qvq_v2b2_p32_spectral_candidates_reuse_the_baseline_block_family():
    generator = torch.Generator().manual_seed(20260823)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((29, 16), generator=generator)
    output_samples = torch.randn((31, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]

    from gptqmodel.quantization import qvq as qvq_module

    with patch.object(
        qvq_module,
        "block_ldlq_inner_v2b2_p32",
        wraps=qvq_module.block_ldlq_inner_v2b2_p32,
    ) as block_family_search:
        result = quantize_qvq_linear(
            weight,
            input_hessian,
            bits=2,
            rounding="yaqa",
            output_hessian=output_hessian,
            bank_count=2,
            v2b2_p32=True,
            yaqa_spectral_refinement=True,
            yaqa_spectral_ranks=(1,),
            yaqa_spectral_lambdas=(0.25,),
            trellis_batch_size=1,
        )

    assert block_family_search.call_count == 1
    assert result.yaqa_block_family_id in (1, 2, 3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")
def test_qvq_v2b2_p32_spectral_analysis_uses_cpu_svd_on_mps():
    source = torch.randn((16, 16), device="mps")
    states = torch.zeros((1, 128), dtype=torch.long, device="mps")
    selectors = torch.zeros(8, dtype=torch.uint8, device="mps")
    baseline = torch.zeros_like(source), states, selectors, torch.tensor([1], dtype=torch.uint8, device="mps")
    codebooks = tuple(torch.full((1,), index, dtype=torch.float32, device="mps") for index in range(4))
    observed_svd_devices = []

    def tracked_svd(matrix, rank, algo):
        observed_svd_devices.append(matrix.device.type)
        return torch.linalg.svd(matrix, full_matrices=False)

    diagnostics = {}
    with (
        patch("gptqmodel.eora.eora._eora_compute_svd", side_effect=tracked_svd),
        patch("gptqmodel.quantization.qvq.yaqa_inner_v2b2_p32", return_value=baseline),
    ):
        result = yaqa_output_spectral_refine_v2b2_p32(
            source,
            torch.eye(16, device="mps"),
            torch.eye(16, device="mps"),
            codebooks,
            baseline,
            ranks=(4,),
            lambdas=(0.25,),
            bits=2,
            diagnostics=diagnostics,
        )

    assert observed_svd_devices == ["cpu"]
    assert diagnostics["spectral_svd_device"] == "cpu"
    assert torch.equal(result[0], baseline[0])


def test_qvq_v2b2_p32_fixed_yaqa_uses_matched_block_ldlq_damping_for_family():
    generator = torch.Generator().manual_seed(20260819)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    input_samples = torch.randn((41, 16), generator=generator)
    output_samples = torch.randn((43, 16), generator=generator)
    input_hessian = input_samples.T @ input_samples / input_samples.shape[0]
    output_hessian = output_samples.T @ output_samples / output_samples.shape[0]
    block = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        trellis_batch_size=1,
    )
    fixed = quantize_qvq_linear(
        weight,
        input_hessian,
        bits=2,
        rounding="yaqa",
        output_hessian=output_hessian,
        bank_count=2,
        v2b2_p32=True,
        yaqa_v2b2_family_mode="fixed_block_ldlq",
        trellis_batch_size=1,
    )

    expected_family = int(block.bank_alt_id.item())
    assert fixed.yaqa_block_family_id == expected_family
    assert int(fixed.bank_alt_id.item()) == expected_family


@pytest.mark.parametrize(
    ("family_mode", "expected_family", "expected_changed"),
    (("fixed_block_ldlq", 2, False), ("reselect", 3, True)),
)
def test_qvq_v2b2_p32_yaqa_fixed_and_reselected_family_objectives(
    family_mode,
    expected_family,
    expected_changed,
):
    weight = torch.zeros((16, 16))
    hessian = torch.eye(16)
    states = torch.zeros((1, 128), dtype=torch.long)
    block_selectors = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    codebooks = tuple(torch.full((1,), family_id, dtype=torch.float32) for family_id in range(4))

    def fake_block(*args, **kwargs):
        del args, kwargs
        return weight, states, block_selectors, torch.tensor([2], dtype=torch.uint8)

    def fake_yaqa(*args, bank_codebooks=None, **kwargs):
        del args, kwargs
        if bank_codebooks is None:
            return torch.ones_like(weight), states
        family_id = int(bank_codebooks[1].item())
        candidate = torch.full_like(weight, {1: 0.3, 2: 0.2, 3: 0.1}[family_id])
        selectors = torch.full((8,), family_id & 1, dtype=torch.uint8)
        return candidate, states, selectors

    diagnostics = {}
    telemetry = QVQQuantizationTelemetry()
    with (
        patch("gptqmodel.quantization.qvq.block_ldlq_inner_v2b2_p32", side_effect=fake_block),
        patch("gptqmodel.quantization.qvq.yaqa_inner", side_effect=fake_yaqa),
    ):
        _, _, selectors, family = yaqa_inner_v2b2_p32(
            weight,
            hessian,
            hessian,
            codebooks,
            bits=2,
            family_mode=family_mode,
            diagnostics=diagnostics,
            telemetry=telemetry,
        )

    measured = telemetry.finalize()

    assert int(family.item()) == expected_family
    family_candidates = diagnostics.pop("family_candidates")
    assert diagnostics == {
        "fallback_to_v2": False,
        "selector_churn": float((selectors != block_selectors).to(torch.float32).mean()),
        "family_changed": expected_changed,
        "block_family_id": 2,
    }
    expected_ids = {"2"} if family_mode == "fixed_block_ldlq" else {"1", "2", "3"}
    expected_candidate_count = len(expected_ids)
    assert measured["counters"] == {
        "yaqa_v2b2_modules": 1,
        "yaqa_v2b2_reselect_modules": int(family_mode == "reselect"),
        "yaqa_v2b2_family_candidates": expected_candidate_count,
    }
    assert {name: values["calls"] for name, values in measured["phases"].items()} == {
        "yaqa_v2b2_block_family_selection": 1,
        "yaqa_v2b2_canonical": 1,
        "yaqa_v2b2_full_proxy": expected_candidate_count + 1,
        "yaqa_v2b2_family_candidate": expected_candidate_count,
    }
    assert all(values["gpu_ms"] is None for values in measured["phases"].values())
    assert set(family_candidates) == expected_ids
    for candidate in family_candidates.values():
        assert candidate == {
            "fallback_to_bank0": False,
            "mixed_loss_before_fallback": None,
            "bank0_loss": None,
            "pre_fallback_selector_churn": None,
            "pre_fallback_state_churn": None,
        }


def test_qvq_sweep_telemetry_aggregate_preserves_shape_attribution():
    modules = {
        "q_proj": {
            "qvq_telemetry": {
                "phases": {"yaqa_feedback": {"calls": 2, "host_dispatch_ms": 3.0, "gpu_ms": 4.0}},
                "counters": {"yaqa_tiles": 8, "yaqa_anti_diagonals": 3},
            }
        },
        "gate_proj": {
            "qvq_telemetry": {
                "phases": {"yaqa_feedback": {"calls": 5, "host_dispatch_ms": 7.0, "gpu_ms": 11.0}},
                "counters": {"yaqa_tiles": 32, "yaqa_anti_diagonals": 6},
            }
        },
    }

    aggregate = _aggregate_qvq_telemetry(modules)

    assert aggregate == {
        "phases": {"yaqa_feedback": {"calls": 7, "host_dispatch_ms": 10.0, "gpu_ms": 15.0}},
        "counters": {"yaqa_tiles": 40, "yaqa_anti_diagonals": 9},
    }


def test_qvq_v2b2_p32_reuses_one_canonical_block_ldlq_oracle():
    generator = torch.Generator().manual_seed(20260816)
    weight = torch.randn((16, 16), generator=generator) * 0.1
    hessian = torch.eye(16)
    with patch("gptqmodel.quantization.qvq.block_ldlq_inner", wraps=block_ldlq_inner) as canonical:
        quantize_qvq_linear(
            weight,
            hessian,
            bits=2,
            bank_count=2,
            v2b2_p32=True,
            trellis_batch_size=1,
        )
    assert canonical.call_count == 1


@pytest.mark.parametrize("alt_id", (0, 4))
def test_qvq_v2b2_p32_rejects_invalid_alternative_bank(alt_id):
    trellis = torch.zeros((1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match=r"in \[1, 3\]"):
        reconstruct_qvq_inner_weight(
            trellis,
            bits=2,
            in_features=16,
            out_features=16,
            bank_ids=torch.zeros(1, dtype=torch.uint8),
            v2b2_p32=True,
            bank_alt_id=torch.tensor([alt_id], dtype=torch.uint8),
        )
