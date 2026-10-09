# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from tests.models.w4a_calibration_data import file_digest
from tests.models.w4a_heldout_trace import compare_heldout, hidden_statistics, logit_statistics, probe_positions


@pytest.mark.parametrize("length,limit", [(2, 64), (17, 64), (2048, 64), (5, 1)])
def test_uniform_positions_are_unique_and_have_next_tokens(length, limit):
    positions = probe_positions(length, limit)
    assert len(positions) == min(length - 1, limit)
    assert len(positions.unique()) == len(positions)
    assert positions[0] == 0 and int(positions.max()) < length - 1
    if len(positions) > 1:
        assert positions[-1] == length - 2


def test_hidden_metrics_do_not_hide_small_token_errors_behind_outliers():
    reference = torch.tensor([[100., 0.], [1., 0.]])
    actual = torch.tensor([[100., 0.], [0., 0.]])
    stats = hidden_statistics(reference, actual)
    assert stats["sse"] == 1 and stats["energy"] == 10001
    assert stats["token_relative_sse"] / stats["relative_tokens"] == .5
    assert stats["radial_sum"] / stats["relative_tokens"] == -.5
    assert stats["cosine_sum"] / stats["relative_tokens"] == .5


def test_zero_hidden_vectors_are_reported_without_division_by_zero():
    stats = hidden_statistics(torch.zeros(3, 2), torch.zeros(3, 2))
    assert stats["relative_tokens"] == 0 and stats["tokens"] == 3
    assert stats["sse"] == 0 and stats["token_relative_sse"] == 0


def test_full_vocabulary_kl_and_nll_match_scalar_oracle():
    reference = torch.tensor([[math.log(.75), math.log(.25)]], dtype=torch.float64)
    actual = reference.flip(-1)
    stats = logit_statistics(reference, actual, torch.tensor([0]))
    assert stats["kl_sum"] == pytest.approx(.5 * math.log(3), abs=1e-12)
    assert stats["reference_nll_sum"] == pytest.approx(-math.log(.75), abs=1e-12)
    assert stats["actual_nll_sum"] == pytest.approx(-math.log(.25), abs=1e-12)
    assert stats["argmax_agreement"] == 0


@pytest.mark.parametrize("target", [torch.tensor([-1]), torch.tensor([2]), torch.tensor([0.])])
def test_logit_metrics_reject_invalid_next_token_targets(target):
    with pytest.raises(ValueError, match="vocabulary"):
        logit_statistics(torch.zeros(1, 2), torch.zeros(1, 2), target)


def _trace(directory, variant, *, modify=None):
    directory.mkdir()
    tensors = {"input_ids": torch.tensor([0, 1, 0]), "probe_positions": torch.tensor([0, 1]),
               "next_tokens": torch.tensor([1, 0]), "logits": torch.tensor([[.1, .2], [.3, .4]]),
               "layer_0": torch.tensor([[100., 0.], [1., 0.]])}
    if modify:
        modify(tensors)
    path = directory / "sample.safetensors"
    save_file(tensors, str(path))
    manifest = {"version": 1, "variant": variant, "weight_file_sha256": "same-weights",
                "quantize_config_sha256": "same-policy",
                "data_manifest_sha256": "same-disjoint-data", "partition": "selection",
                "runtime_versions": {"torch": "fixture", "transformers": "fixture"},
                "sequence_length": 2048, "probe_limit": 64,
                "probe_policy": "uniform_including_first_excluding_last_v1", "decoder_layers": 1,
                "samples": [{"article_id": "article-a", "file": path.name, "sha256": file_digest(path),
                             "tokens": 3, "probes": 2}]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_paired_comparison_reports_identical_inputs_and_outputs(tmp_path):
    reference = _trace(tmp_path / "reference", "w4a16")
    actual = _trace(tmp_path / "actual", "w4a_nvfp4")
    report = compare_heldout(reference, actual)
    assert report["rows"] == 1
    assert report["logits"]["mean_kl"] == 0
    assert report["logits"]["argmax_agreement_fraction"] == 1
    assert report["layers"]["layer_0"]["equal_token_relative_rmse"] == 0


@pytest.mark.parametrize("key", ["weight_file_sha256", "data_manifest_sha256", "probe_policy"])
def test_paired_comparison_rejects_protocol_changes(tmp_path, key):
    reference = _trace(tmp_path / "reference", "w4a16")
    actual = _trace(tmp_path / "actual", "w4a_nvfp4")
    path = actual / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_heldout(reference, actual)


def test_paired_comparison_rejects_different_input_ids(tmp_path):
    reference = _trace(tmp_path / "reference", "w4a16")
    actual = _trace(tmp_path / "actual", "w4a_nvfp4", modify=lambda data: data["input_ids"].fill_(1))
    with pytest.raises(ValueError, match="inputs differ"):
        compare_heldout(reference, actual)


def test_paired_comparison_rejects_cherry_picked_positions_even_when_both_match(tmp_path):
    def modify(data):
        data["probe_positions"] = torch.tensor([1, 1])
        data["next_tokens"] = torch.tensor([0, 0])
    reference = _trace(tmp_path / "reference", "w4a16", modify=modify)
    actual = _trace(tmp_path / "actual", "w4a_nvfp4", modify=modify)
    with pytest.raises(ValueError, match="declared protocol"):
        compare_heldout(reference, actual)


def test_paired_comparison_rejects_missing_decoder_tensor(tmp_path):
    reference = _trace(tmp_path / "reference", "w4a16")
    actual = _trace(tmp_path / "actual", "w4a_nvfp4", modify=lambda data: data.pop("layer_0"))
    with pytest.raises(ValueError, match="Incomplete"):
        compare_heldout(reference, actual)


def test_paired_comparison_rejects_corrupted_trace(tmp_path):
    reference = _trace(tmp_path / "reference", "w4a16")
    actual = _trace(tmp_path / "actual", "w4a_nvfp4")
    with (actual / "sample.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_heldout(reference, actual)


def _replay_trace(directory, *, modify=None):
    result = _trace(directory, "w4a_nvfp4", modify=modify)
    path = result / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["execution"] = "weight_qad_replay"
    path.write_text(json.dumps(manifest))
    return result


def test_replay_fidelity_accepts_identical_outputs(tmp_path):
    reference = _trace(tmp_path / "runtime", "w4a_nvfp4")
    actual = _replay_trace(tmp_path / "replay")
    report = compare_heldout(reference, actual, training_replay=True)
    assert report["replay_validation"]["passed"]
    assert report["replay_validation"]["tensors"]["layer_0"]["elements"] == 4


def test_replay_fidelity_reports_float_failure_without_hiding_it_in_mean_error(tmp_path):
    reference = _trace(tmp_path / "runtime", "w4a_nvfp4")
    actual = _replay_trace(tmp_path / "replay", modify=lambda x: x["layer_0"].__setitem__((1, 1), .003))
    report = compare_heldout(reference, actual, training_replay=True)
    fidelity = report["replay_validation"]
    assert not fidelity["passed"]
    assert fidelity["tensors"]["layer_0"]["outside_tolerance"] == 1
    assert fidelity["argmax_mismatches"] == 0


def test_replay_fidelity_requires_exact_selected_tokens_even_with_close_logits(tmp_path):
    def base(x):
        x["logits"][0] = torch.tensor([.1001, .1])
    def changed(x):
        x["logits"][0] = torch.tensor([.1, .1001])
    reference = _trace(tmp_path / "runtime", "w4a_nvfp4", modify=base)
    actual = _replay_trace(tmp_path / "replay", modify=changed)
    fidelity = compare_heldout(reference, actual, training_replay=True)["replay_validation"]
    assert not fidelity["passed"]
    assert fidelity["tensors"]["logits"]["outside_tolerance"] == 0
    assert fidelity["argmax_mismatches"] == 1


def test_replay_cannot_be_used_as_the_runtime_quality_lane(tmp_path):
    reference = _trace(tmp_path / "runtime", "w4a16")
    actual = _replay_trace(tmp_path / "replay")
    with pytest.raises(ValueError, match="execution modes"):
        compare_heldout(reference, actual)


@pytest.mark.parametrize("edit,match", [
    ({"quantize_config_sha256": "changed"}, "identical activation policies"),
    ({"experimental_token_global": True}, "unmodified runtime traces"),
])
def test_replay_fidelity_rejects_different_policies(tmp_path, edit, match):
    reference = _trace(tmp_path / "runtime", "w4a_nvfp4")
    actual = _replay_trace(tmp_path / "replay")
    path = actual / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(edit)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        compare_heldout(reference, actual, training_replay=True)
