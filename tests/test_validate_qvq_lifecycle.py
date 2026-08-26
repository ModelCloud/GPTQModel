# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import QVQConfig
from gptqmodel.utils.qvq_validation import (
    assert_qvq_dense_accuracy,
    assert_qvq_reload_parity,
    qvq_accuracy_metrics,
    validate_qvq_lifecycle_args,
)
from scripts.eval_qvq_checkpoint import TASKS, _select_tasks
from scripts.validate_qvq_lifecycle import (
    _calibration_controls,
    _install_semantic_attention_bits,
)


class _TinyLlamaTree(nn.Module):
    """Provide two real decoder slots for semantic mixed-rate unit tests."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.config = SimpleNamespace()


def _semantic_model(*, dynamic=None, flags=LlamaQModel.get_module_tree_flags):
    qcfg = QVQConfig(bits=1.5, rounding="block_ldlq", dynamic=dynamic, offload_to_disk=False)
    return SimpleNamespace(
        model=_TinyLlamaTree(),
        quantize_config=qcfg,
        simple_layer_modules=LlamaQModel.simple_layer_modules,
        get_module_tree_flags=flags,
        extract_layers_node=LlamaQModel.extract_layers_node,
    )


def test_qvq_lifecycle_semantic_attention_override_covers_qkvo_in_every_layer():
    model = _semantic_model()
    targets = _install_semantic_attention_bits(model, 3, layers=2)

    assert len(targets) == 8
    assert targets == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
    )
    assert all(model.quantize_config.dynamic_get(name, "bits") == 3 for name in targets)
    assert model.quantize_config.dynamic_get("model.layers.0.mlp.gate_proj", "bits", 1.5) == 1.5


def test_qvq_lifecycle_semantic_attention_override_fails_closed_on_missing_role_or_conflict():
    def missing_output(path):
        return frozenset() if path == "self_attn.o_proj" else LlamaQModel.get_module_tree_flags(path)

    with pytest.raises(ValueError, match="missing attention roles.*o"):
        _install_semantic_attention_bits(_semantic_model(flags=missing_output), 3, layers=2)

    conflict = {r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 2}}
    with pytest.raises(ValueError, match="conflicts with existing dynamic rules"):
        _install_semantic_attention_bits(_semantic_model(dynamic=conflict), 3, layers=2)

    with pytest.raises(ValueError, match="requested 3 layers.*has 2"):
        _install_semantic_attention_bits(_semantic_model(), 3, layers=3)


def test_qvq_checkpoint_evaluator_runs_all_full_row_llama_task_gates():
    by_label = {label: (task, chat, kwargs) for label, task, chat, kwargs in TASKS}
    assert by_label == {
        "arc_challenge": ("arc_challenge", True, {}),
        "gsm8k_platinum_cot": ("gsm8k_platinum_cot", True, {}),
        "mmlu_stem": ("mmlu_stem", False, {}),
        "mmlu_humanities": ("mmlu", False, {"subsets": "humanities"}),
        "mmlu_history": (
            "mmlu",
            False,
            {
                "subsets": (
                    "humanities.high_school_european_history",
                    "humanities.high_school_us_history",
                    "humanities.high_school_world_history",
                    "humanities.prehistory",
                )
            },
        ),
    }
    assert _select_tasks(None) == TASKS
    assert tuple(task[0] for task in _select_tasks(["mmlu_history", "arc_challenge"])) == (
        "arc_challenge",
        "mmlu_history",
    )
    with pytest.raises(ValueError, match="must be unique"):
        _select_tasks(["arc_challenge", "arc_challenge"])
    with pytest.raises(ValueError, match="Unknown.*not_a_task"):
        _select_tasks(["not_a_task"])


def test_qvq_lifecycle_script_preserves_natural_rows_and_exact_exclusions():
    natural = Namespace(concat_size=0, calibration_sort="none", exclude_module=[])
    assert _calibration_controls(natural) == (None, None, None)

    module_name = "model.layers.0.self_attn.v_proj"
    controlled = Namespace(concat_size=2048, calibration_sort="desc", exclude_module=[module_name])
    concat_size, calibration_sort, dynamic = _calibration_controls(controlled)
    assert concat_size == 2048
    assert calibration_sort == "desc"
    assert dynamic == {r"-:^model\.layers\.0\.self_attn\.v_proj$": {}}


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ({"concat_size": -1, "calibration_sort": "none", "exclude_module": []}, "concat-size"),
        ({"concat_size": 0, "calibration_sort": "none", "exclude_module": [""]}, "nonempty and unique"),
        (
            {"concat_size": 0, "calibration_sort": "none", "exclude_module": ["proj", "proj"]},
            "nonempty and unique",
        ),
    ),
)
def test_qvq_lifecycle_script_rejects_ambiguous_calibration_controls(values, message):
    with pytest.raises(ValueError, match=message):
        _calibration_controls(Namespace(**values))


def test_qvq_lifecycle_metrics_cover_exact_and_deviating_logits():
    reference = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]]])

    exact = qvq_accuracy_metrics(reference, reference.clone())
    assert exact["finite"] is True
    assert exact["mae"] == exact["mse"] == exact["rmse"] == exact["relative_l2"] == 0
    assert exact["forward_kld"] == exact["reverse_kld"] == exact["jensen_shannon"] == 0
    assert exact["cosine"] == pytest.approx(1.0)
    assert exact["top1_agreement"] == exact["top5_overlap"] == exact["top10_overlap"] == 1.0
    assert exact["max_abs"] == 0

    candidate = reference.clone()
    candidate[0, 0] = torch.tensor([-1.0, 0.0, 2.0])
    changed = qvq_accuracy_metrics(reference, candidate)
    assert changed["mae"] > 0
    assert changed["rmse"] > 0
    assert changed["relative_l2"] > 0
    assert changed["forward_kld"] > 0
    assert changed["reverse_kld"] > 0
    assert changed["jensen_shannon"] > 0
    assert changed["top1_agreement"] == 0.5
    assert changed["top5_overlap"] == 1.0
    assert changed["top10_overlap"] == 1.0


def test_qvq_lifecycle_metrics_fail_closed_on_shape_and_nonfinite_values():
    reference = torch.zeros((1, 2, 3))
    with pytest.raises(AssertionError, match="shape mismatch"):
        qvq_accuracy_metrics(reference, torch.zeros((1, 3, 3)))
    with pytest.raises(AssertionError, match="must be finite"):
        qvq_accuracy_metrics(reference, torch.full_like(reference, float("nan")))
    with pytest.raises(AssertionError, match="must be finite"):
        qvq_accuracy_metrics(torch.full_like(reference, float("inf")), reference)


def test_qvq_lifecycle_dense_accuracy_gate_checks_kld_and_top1():
    metrics = {"forward_kld": 0.03, "top1_agreement": 0.9}
    assert_qvq_dense_accuracy(metrics, Namespace(max_forward_kld=0.04, min_top1_agreement=0.88))

    with pytest.raises(AssertionError, match="Forward KLD"):
        assert_qvq_dense_accuracy(metrics, Namespace(max_forward_kld=0.02, min_top1_agreement=0.88))
    with pytest.raises(AssertionError, match="Top-1"):
        assert_qvq_dense_accuracy(metrics, Namespace(max_forward_kld=0.04, min_top1_agreement=0.95))


def test_qvq_lifecycle_reload_gate_compares_live_and_reloaded_logits_directly():
    live = torch.tensor([[[2.0, 0.0, -1.0]]])
    args = Namespace(reload_rtol=0.0, reload_atol=0.0)
    metrics = assert_qvq_reload_parity(live, live.clone(), args)
    assert metrics["max_abs"] == 0

    with pytest.raises(AssertionError, match="live and reloaded logits diverged"):
        assert_qvq_reload_parity(live, live + 0.01, args)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"rows": 0}, "rows and --layers"),
        ({"layers": 0}, "rows and --layers"),
        ({"max_forward_kld": -1.0}, "max-forward-kld"),
        ({"min_top1_agreement": -0.1}, "min-top1-agreement"),
        ({"min_top1_agreement": 1.1}, "min-top1-agreement"),
        ({"reload_rtol": -1.0}, "reload tolerances"),
        ({"reload_atol": -1.0}, "reload tolerances"),
    ),
)
def test_qvq_lifecycle_argument_gates_fail_closed(override, message):
    values = {
        "rows": 128,
        "layers": 2,
        "max_forward_kld": 0.04,
        "min_top1_agreement": 0.88,
        "reload_rtol": 0.0,
        "reload_atol": 0.0,
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        validate_qvq_lifecycle_args(Namespace(**values))


def test_qvq_lifecycle_argument_gates_accept_valid_thresholds():
    validate_qvq_lifecycle_args(
        Namespace(
            rows=128,
            layers=2,
            max_forward_kld=0.04,
            min_top1_agreement=0.88,
            reload_rtol=0.0,
            reload_atol=0.0,
        )
    )


@pytest.mark.parametrize("field", ("max_forward_kld", "min_top1_agreement", "reload_rtol", "reload_atol"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_qvq_lifecycle_argument_gates_reject_every_nonfinite_threshold(field, value):
    values = {
        "rows": 128,
        "layers": 2,
        "max_forward_kld": 0.04,
        "min_top1_agreement": 0.88,
        "reload_rtol": 0.0,
        "reload_atol": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="must be finite"):
        validate_qvq_lifecycle_args(Namespace(**values))


def test_qvq_accuracy_helpers_reject_nonfinite_thresholds_without_argument_validator():
    metrics = {"forward_kld": 0.03, "top1_agreement": 0.9}
    with pytest.raises(ValueError, match="max-forward-kld.*finite"):
        assert_qvq_dense_accuracy(metrics, Namespace(max_forward_kld=float("nan"), min_top1_agreement=0.88))

    logits = torch.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="reload-atol.*finite"):
        assert_qvq_reload_parity(logits, logits, Namespace(reload_rtol=0.0, reload_atol=float("inf")))
