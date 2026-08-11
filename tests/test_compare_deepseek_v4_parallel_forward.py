import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/compare_deepseek_v4_parallel_forward.py"
SPEC = importlib.util.spec_from_file_location("compare_deepseek_v4_parallel_forward", SCRIPT_PATH)
COMPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPARE)


def test_tensor_sha256_includes_exact_values_dtype_and_shape():
    baseline = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    assert COMPARE.tensor_sha256(baseline) == COMPARE.tensor_sha256(baseline.clone())
    assert COMPARE.tensor_sha256(baseline) != COMPARE.tensor_sha256(baseline + 1)
    assert COMPARE.tensor_sha256(baseline) != COMPARE.tensor_sha256(baseline.to(torch.float64))
    assert COMPARE.tensor_sha256(baseline) != COMPARE.tensor_sha256(baseline.reshape(2, 1))


def test_tensor_sample_sha256_is_exact_and_shape_aware_for_bounded_samples():
    baseline = torch.arange(128 * 129, dtype=torch.float32).reshape(128, 129)
    changed_sample = baseline.clone()
    changed_sample[0, 0] += 1

    assert COMPARE.tensor_sample_sha256(baseline) == COMPARE.tensor_sample_sha256(baseline.clone())
    assert COMPARE.tensor_sample_sha256(baseline) != COMPARE.tensor_sample_sha256(changed_sample)
    assert COMPARE.tensor_sample_sha256(baseline) != COMPARE.tensor_sample_sha256(baseline.T)


def _artifact(mode, modules):
    return {"mode": mode, "modules": modules}


def _module_record(prefix, *, relative_l2=0.25):
    return {
        "weight_sha256": f"{prefix}-weight",
        "scale_sha256": f"{prefix}-scale",
        "zero_sha256": f"{prefix}-zero",
        "g_idx_sha256": f"{prefix}-g-idx",
        "code_sha256": f"{prefix}-code",
        "output_error": {
            "mean_absolute_error": 0.1,
            "rmse": 0.2,
            "relative_l2_error": relative_l2,
            "softmax_kld_mean": 0.01,
            "top1_agreement": 0.75,
        },
    }


def test_compare_artifacts_reports_exact_tensor_and_metric_differences():
    reference = _artifact(
        "serial",
        {
            "same": _module_record("same"),
            "changed": _module_record("reference", relative_l2=0.2),
            "reference_only": _module_record("reference-only"),
        },
    )
    candidate_changed = _module_record("reference", relative_l2=0.3)
    candidate_changed["weight_sha256"] = "candidate-weight"
    candidate_changed["code_sha256"] = "candidate-code"
    candidate = _artifact(
        "parallel",
        {
            "same": _module_record("same"),
            "changed": candidate_changed,
            "candidate_only": _module_record("candidate-only"),
        },
    )

    comparison = COMPARE.compare_artifacts(reference, candidate)

    assert comparison["shared_module_count"] == 2
    assert comparison["missing_from_candidate"] == ["reference_only"]
    assert comparison["missing_from_reference"] == ["candidate_only"]
    assert comparison["tensor_mismatch_counts"] == {
        "hessian_sha256": 0,
        "hessian_inverse_sha256": 0,
        "weight_sha256": 1,
        "scale_sha256": 0,
        "zero_sha256": 0,
        "g_idx_sha256": 0,
        "code_sha256": 1,
    }
    assert comparison["changed_module_count"] == 1
    assert comparison["changed_modules"][0]["module"] == "changed"
    assert comparison["changed_modules"][0]["tensor_mismatches"] == ["weight_sha256", "code_sha256"]
    assert comparison["maximum_absolute_metric_delta"]["relative_l2_error"] == pytest.approx(0.1)
