import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/compare_llama_calibration_batch.py"
SPEC = importlib.util.spec_from_file_location("compare_llama_calibration_batch", SCRIPT_PATH)
COMPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPARE)


def _module_record(prefix: str) -> dict[str, str]:
    return {
        "hessian_sha256": f"{prefix}-hessian",
        "hessian_inverse_sha256": f"{prefix}-inverse",
        "weight_sha256": f"{prefix}-weight",
        "scale_sha256": f"{prefix}-scale",
        "zero_sha256": f"{prefix}-zero",
        "g_idx_sha256": f"{prefix}-g-idx",
        "code_sha256": f"{prefix}-code",
    }


def test_tensor_hashes_include_exact_bytes_dtype_shape_and_sampling_policy():
    baseline = torch.arange(129 * 130, dtype=torch.float32).reshape(129, 130)
    changed = baseline.clone()
    changed[0, 0] += 1

    assert COMPARE._tensor_sha256(baseline) == COMPARE._tensor_sha256(baseline.clone())
    assert COMPARE._tensor_sha256(baseline) != COMPARE._tensor_sha256(changed)
    assert COMPARE._tensor_sha256(baseline) != COMPARE._tensor_sha256(baseline.to(torch.float64))
    assert COMPARE._tensor_sample_sha256(baseline) != COMPARE._tensor_sample_sha256(changed)
    assert COMPARE._tensor_sample_sha256(baseline) != COMPARE._tensor_sample_sha256(baseline.T)

    full_digest, full_kind = COMPARE._hessian_sha256(torch.eye(4))
    sampled_digest, sampled_kind = COMPARE._hessian_sha256(torch.empty(4097, 4097, dtype=torch.uint8))
    assert full_digest == COMPARE._tensor_sha256(torch.eye(4))
    assert full_kind == "full"
    assert isinstance(sampled_digest, str)
    assert sampled_kind == "sampled"


def test_gptq_module_name_prefers_layer_qualified_named_module():
    qualified = SimpleNamespace(name="self_attn.q_proj", _named_module=SimpleNamespace(full_name="model.layers.3.self_attn.q_proj"))
    suffix_only = SimpleNamespace(name="self_attn.q_proj", _named_module=None)

    assert COMPARE._gptq_module_name(qualified) == "model.layers.3.self_attn.q_proj"
    assert COMPARE._gptq_module_name(suffix_only) == "self_attn.q_proj"


def test_compare_payloads_requires_identical_calibration_modules_and_tensor_fields():
    reference = {
        "calibration_sha256": "same-calibration",
        "modules": {
            "same": _module_record("same"),
            "changed": _module_record("reference"),
            "reference_only": _module_record("reference-only"),
        },
    }
    candidate_changed = _module_record("reference")
    candidate_changed["weight_sha256"] = "candidate-weight"
    candidate_changed["code_sha256"] = "candidate-code"
    candidate = {
        "calibration_sha256": "same-calibration",
        "modules": {
            "same": _module_record("same"),
            "changed": candidate_changed,
            "candidate_only": _module_record("candidate-only"),
        },
    }

    comparison = COMPARE.compare_payloads(reference, candidate)

    assert comparison["exact"] is False
    assert comparison["shared_module_count"] == 2
    assert comparison["missing_from_candidate"] == ["reference_only"]
    assert comparison["missing_from_reference"] == ["candidate_only"]
    assert comparison["changed_modules"] == [
        {"module": "changed", "tensor_mismatches": ["weight_sha256", "code_sha256"]}
    ]
    assert comparison["tensor_mismatch_counts"] == {
        "hessian_sha256": 0,
        "hessian_inverse_sha256": 0,
        "weight_sha256": 1,
        "scale_sha256": 0,
        "zero_sha256": 0,
        "g_idx_sha256": 0,
        "code_sha256": 1,
    }

    exact = COMPARE.compare_payloads(reference, reference)
    assert exact["exact"] is True

    different_calibration = dict(reference, calibration_sha256="different")
    assert COMPARE.compare_payloads(reference, different_calibration)["exact"] is False


def test_build_calibration_has_fixed_or_deterministic_variable_geometry():
    class Tokenizer:
        pad_token_id = 99
        padding_side = "right"

        def __call__(self, text, *, add_special_tokens, truncation, max_length=None):
            assert add_special_tokens
            base = [1] + [2 + (ord(char) % 17) for char in text]
            return {"input_ids": base if not truncation else base[:max_length]}

    frame = pd.DataFrame({"text": ["short", "also short", "third row", "fourth row"]})
    natural = COMPARE._build_calibration(frame, Tokenizer(), rows=4, seq_len=64, geometry="natural")
    fixed = COMPARE._build_calibration(frame, Tokenizer(), rows=4, seq_len=64, geometry="fixed")
    variable = COMPARE._build_calibration(frame, Tokenizer(), rows=4, seq_len=64, geometry="variable")

    assert [item["input_ids"].numel() for item in natural] == [6, 11, 10, 11]
    assert all(torch.equal(item["attention_mask"], torch.ones_like(item["input_ids"])) for item in natural)
    assert [item["input_ids"].numel() for item in fixed] == [64, 64, 64, 64]
    assert [item["attention_mask"].sum().item() for item in fixed] == [6, 11, 10, 11]
    assert [item["input_ids"][-1].item() for item in fixed] == [99, 99, 99, 99]
    variable_lengths = [item["input_ids"].numel() for item in variable]
    assert variable_lengths == [16, 20, 24, 28]
    assert all(torch.equal(item["attention_mask"], torch.ones_like(item["input_ids"])) for item in variable)


def test_build_fixed_calibration_respects_left_padding_and_requires_a_pad_id():
    class LeftTokenizer:
        pad_token_id = None
        eos_token_id = 77
        padding_side = "left"

        def __call__(self, text, *, add_special_tokens, truncation, max_length):
            assert add_special_tokens and truncation
            return {"input_ids": [1, 2, 3][:max_length]}

    frame = pd.DataFrame({"text": ["row"]})
    fixed = COMPARE._build_calibration(frame, LeftTokenizer(), rows=1, seq_len=8, geometry="fixed")
    assert fixed[0]["input_ids"].tolist() == [77, 77, 77, 77, 77, 1, 2, 3]
    assert fixed[0]["attention_mask"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]

    left = LeftTokenizer()
    left.eos_token_id = None
    with pytest.raises(ValueError, match="requires an integer"):
        COMPARE._build_calibration(frame, left, rows=1, seq_len=8, geometry="fixed")
