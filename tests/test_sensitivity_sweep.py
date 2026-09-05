# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections import UserDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from optimize.calibration_coverage import find_target_groups
from optimize.sensitivity import SensitivitySweep, _clone, _noise, shared_input_noise
from optimize.sensitivity_metrics import ErrorStats, logit_metrics, measured_gain, uncorrelated_prediction
from optimize.sweep_sensitivity import main, render_report


class PairBlock(nn.Module):
    def __init__(self, input_mode="shared"):
        super().__init__()
        self.q_proj = nn.Linear(3, 3, bias=False)
        self.k_proj = nn.Linear(3, 3, bias=False)
        self.v_proj = nn.Linear(3, 3, bias=False)
        self.input_mode = input_mode
        for module in (self.q_proj, self.k_proj, self.v_proj):
            module.weight.data.copy_(torch.eye(3))

    def forward(self, x):
        q = self.q_proj(x)
        if self.input_mode == "clone":
            k = self.k_proj(x.clone())
        elif self.input_mode == "view":
            k = self.k_proj(x.view_as(x))
        elif self.input_mode == "mutation":
            x.add_(1)
            k = self.k_proj(x)
        else:
            k = self.k_proj(x)
        return q - k + self.v_proj(x)


class TinyModel(nn.Module):
    def __init__(self, input_mode="shared", depth=2):
        super().__init__()
        self.layers = nn.ModuleList([PairBlock(input_mode) for _ in range(depth)])
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight.data.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 1.0, 1.0]])
        )

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, inputs_embeds, attention_mask=None, use_cache=True):
        assert use_cache is False
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return {"logits": self.lm_head(x)}


def batches():
    return [
        {"inputs_embeds": torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]), "attention_mask": torch.tensor([[1, 1]])}
    ]


def sweep(model=None, **kwargs):
    model = TinyModel() if model is None else model
    return SensitivitySweep(model, [f"layers.{i}" for i in range(len(model.layers))], **kwargs)


def run(scanner, **kwargs):
    return scanner.sweep(batches(), amplitudes=[0.01], top_k=1, **kwargs)


def test_metrics_match_independent_scalar_reference_and_chunking():
    reference = torch.arange(70000, dtype=torch.float64).reshape(350, 200).t()
    candidate = reference + 0.125
    stats = ErrorStats()
    stats.update(reference, candidate)
    result = stats.report()
    assert result["count"] == 70000
    assert result["max_abs"] == result["mean_abs"] == result["rmse"] == 0.125
    expected = math.sqrt(70000 * 0.125**2 / sum(i * i for i in range(70000)))
    assert result["relative_l2"] == pytest.approx(expected)
    logits = torch.tensor([[0.0, math.log(2)], [math.log(3), 0.0]], dtype=torch.float64)
    got = logits.flip(-1)
    metrics = logit_metrics(logits, got)
    expected_kl = ((1 / 3) * math.log(1 / 2) + (2 / 3) * math.log(2) + 0.75 * math.log(3) + 0.25 * math.log(1 / 3)) / 2
    assert metrics["kl_mean"] == pytest.approx(expected_kl)
    assert metrics["kl_max"] == pytest.approx(0.5 * math.log(3))
    assert metrics["top1_agreement"] == 0
    assert metrics["reference_margin_min"] == pytest.approx(math.log(2))
    many = logits.repeat(20, 1)
    assert logit_metrics(many, many)["kl_mean"] == 0


def test_metrics_undefined_empty_nonfinite_and_invalid_shapes():
    assert ErrorStats().report()["relative_l2"] is None
    zero = ErrorStats()
    zero.update(torch.zeros(2), torch.ones(2))
    assert zero.report()["relative_l2"] is None
    assert zero.report()["max_abs"] == 1
    with pytest.raises(ValueError, match="shapes"):
        zero.update(torch.zeros(2), torch.ones(3))
    bad = logit_metrics(torch.ones(2, 3), torch.full((2, 3), float("nan")))
    assert not bad["finite"] and bad["kl_mean"] is None
    for shape in ((0, 3), (2, 1), (3,)):
        with pytest.raises(ValueError, match="vocabulary"):
            logit_metrics(torch.ones(shape), torch.ones(shape))
    assert measured_gain(None, [1.0], 0)[0] is None
    assert measured_gain(1.0, [], 0)[0] is None
    assert measured_gain(1.0, [None], 0)[0] is None
    assert measured_gain(1.0, [0.0], 0)[1] == "no_resolved_local_error"
    assert measured_gain(0.1, [0.1], 0.2)[1] == "baseline_noise"
    assert measured_gain(0.5, [0.3, 0.4], 0) == (1.0, "measured")


def test_shared_inputs_expose_joint_cancellation_and_pairs():
    report = run(sweep(candidate=shared_input_noise))
    pair = next(
        r for r in report["rows"] if r["stage"] == "subset" and r["members"] == ["layers.0.k_proj", "layers.0.q_proj"]
    )
    assert pair["final"]["max_abs"] == 0
    assert pair["uncorrelated_prediction"] > 0
    assert pair["joint_over_prediction"] == 0
    assert pair["sharing"]["observations"] == 1
    assert {r["stage"] for r in report["rows"]} == {"layer", "module", "subset", "combined"}
    assert not any("lm_head" in name for name in report["targets"])
    assert len([r for r in report["rows"] if r["stage"] == "layer"]) == 2
    assert len([r for r in report["rows"] if r["stage"] == "module"]) == 3
    assert len([r for r in report["rows"] if r["stage"] == "subset"]) == 4


@pytest.mark.parametrize(
    "mode,expect_shared", [("shared", True), ("view", True), ("clone", False), ("mutation", False)]
)
def test_runtime_sharing_is_not_inferred_from_projection_names(mode, expect_shared):
    report = run(sweep(TinyModel(mode, depth=1)))
    shared_pair = any(
        r["stage"] == "subset" and set(r["members"]) == {"layers.0.q_proj", "layers.0.k_proj"} for r in report["rows"]
    )
    assert shared_pair is expect_shared


def test_declared_subset_preserves_evidence_when_inputs_are_distinct():
    report = run(sweep(TinyModel("clone", depth=1), subsets={"definition_qk": ["layers.0.q_proj", "layers.0.k_proj"]}))
    row = next(r for r in report["rows"] if r["target"] == "definition_qk")
    assert row["sharing"]["source"] == "declared_subset"
    assert row["sharing"]["observations"] == 0
    assert row["final"]["relative_l2"] is not None


def test_layer_ranking_drives_module_scope_and_gain_uses_measured_error():
    def candidate(ctx, module, args, kwargs, ref):
        return ref * (1 + ctx.amplitude) if ctx.name.startswith("layers.1") else ref

    report = run(sweep(candidate=candidate))
    assert report["selected_layers"] == ["layers.1"]
    assert all(r["target"].startswith("layers.1") for r in report["rows"] if r["stage"] == "module")
    row = next(r for r in report["rows"] if r["target"] == "layers.1.v_proj" and r["stage"] == "module")
    assert row["g_effective"] == pytest.approx(1.0, abs=1e-5)
    assert row["local"]["layers.1.v_proj"]["relative_l2"] == pytest.approx(0.01, abs=1e-6)
    assert not row["within_kernel_output_tolerance"]


def test_multiple_batches_and_masks_have_consistent_gain_normalization():
    scanner = sweep(TinyModel(depth=1), candidate=lambda c, m, a, k, r: r * (1 + c.amplitude))
    data = batches() + [{"inputs_embeds": batches()[0]["inputs_embeds"] * 7, "attention_mask": torch.tensor([[1, 0]])}]
    result = scanner.sweep(data, amplitudes=[0.001], top_k=1)
    row = next(r for r in result["rows"] if r["target"] == "layers.0.v_proj" and r["stage"] == "module")
    assert row["local"]["layers.0.v_proj"]["count"] == 9
    assert row["local_all_positions"]["layers.0.v_proj"]["count"] == 12
    assert row["final"]["count"] == 12
    assert row["g_effective"] == pytest.approx(1.0, abs=1e-4)


def test_padding_error_is_excluded_from_epsilon_but_not_absolute_kernel_gate():
    def candidate(c, m, a, k, r):
        r[:, -1] += 0.1
        return r

    data = batches()
    data[0]["attention_mask"][0, 1] = 0
    report = sweep(TinyModel(depth=1), candidate=candidate).sweep(data, amplitudes=[1], top_k=1)
    row = next(r for r in report["rows"] if r["target"] == "layers.0.v_proj" and r["stage"] == "module")
    assert row["local"]["layers.0.v_proj"]["relative_l2"] == 0
    assert row["final"]["relative_l2"] == 0
    assert row["g_effective"] is None
    assert not row["within_kernel_output_tolerance"]


def test_noop_restores_modes_rng_weights_and_hooks():
    model = TinyModel()
    model.train()
    model.layers[0].k_proj.eval()
    modes = [m.training for m in model.modules()]
    weights = {n: p.clone() for n, p in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    report = run(sweep(model, candidate=lambda c, m, a, k, r: r))
    assert all(r["final"]["max_abs"] == 0 for r in report["rows"])
    assert all(r["g_effective"] is None for r in report["rows"])
    assert all(r["within_kernel_output_tolerance"] for r in report["rows"])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_failure_cleanup_and_candidate_contract():
    model = TinyModel()

    def crash(*args):
        raise RuntimeError("candidate exploded")

    with pytest.raises(RuntimeError, match="candidate exploded"):
        run(sweep(model, candidate=crash))
    assert model.training
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    for callback in (lambda c, m, a, k, r: r.double(), lambda c, m, a, k, r: r[..., :1], lambda *args: None):
        with pytest.raises(ValueError, match="preserve output"):
            run(sweep(model, candidate=callback))


def test_nonfinite_unobserved_and_zero_reference_do_not_look_safe():
    bad = run(sweep(candidate=lambda c, m, a, k, r: torch.full_like(r, float("inf"))))
    assert all(not r["within_kernel_output_tolerance"] and r["g_effective"] is None for r in bad["rows"])
    json.dumps(bad, allow_nan=False)
    model = TinyModel(depth=1)
    model.layers[0].unused = nn.Linear(3, 3)
    report = run(sweep(model))
    unobserved = next(r for r in report["rows"] if r["target"] == "layers.0.unused")
    assert not unobserved["observed"] and unobserved["g_effective"] is None
    assert not unobserved["within_kernel_output_tolerance"]
    model.lm_head.weight.data.zero_()
    report = run(sweep(model))
    assert all(r["g_effective"] is None for r in report["rows"])


def test_group_limits_pairs_control_and_reproducibility():
    scanner = sweep(TinyModel(depth=1))
    first = scanner.sweep(batches(), amplitudes=[0.0, 0.001, 0.002], top_k=1, pairwise=False)
    second = scanner.sweep(batches(), amplitudes=[0.0, 0.001, 0.002], top_k=1, pairwise=False)
    assert first == second
    assert len([r for r in first["rows"] if r["stage"] == "subset"]) == 3
    limited = run(scanner, max_group_size=2)
    assert limited["skipped_groups"]
    assert not any(r["stage"] == "subset" for r in limited["rows"])
    no_detail = scanner.sweep(batches(), amplitudes=[0.001], top_k=0)
    assert {r["stage"] for r in no_detail["rows"]} == {"layer", "combined"}


def test_invalid_scopes_subsets_batches_and_amplitudes():
    model = TinyModel()
    for layers in ([], ["layers.0", "layers.0"], ["layers", "layers.0"], ["lm_head"]):
        with pytest.raises(ValueError):
            SensitivitySweep(model, layers)
    for subset in (
        ["layers.0.q_proj"],
        ["missing", "layers.0.q_proj"],
        ["layers.0.q_proj", "layers.1.q_proj"],
        ["layers.0.q_proj"] * 2,
    ):
        with pytest.raises(ValueError):
            sweep(model, subsets={"bad": subset})
    model.alias = model.layers[0].q_proj
    with pytest.raises(ValueError, match="Aliased"):
        sweep(model)
    del model.alias
    scanner = sweep(model)
    for amplitudes in ([], [1.0, 1.0], [-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="Amplitudes"):
            scanner.sweep(batches(), amplitudes=amplitudes)
    with pytest.raises(ValueError):
        scanner.sweep([])
    with pytest.raises(ValueError, match="input caches"):
        scanner.sweep([{**batches()[0], "past_key_values": None}])
    with pytest.raises(ValueError, match="only valid"):
        scanner.sweep(
            [{**batches()[0], "attention_mask": torch.tensor([[1, 0]]), "sensitivity_mask": torch.tensor([[1, 1]])}]
        )


def test_existing_census_covers_nonstandard_linears_without_false_suffix_matches():
    block = nn.Module()
    block.q_proj = nn.Linear(3, 2)
    block.k_proj = nn.Linear(4, 2)
    block.strange_q_proj = nn.Linear(3, 2)
    normal = find_target_groups(block)
    assert {tuple(n for n, _ in g.members) for g in normal} == {("q_proj",), ("k_proj",)}
    assert len(find_target_groups(block, include_all_linear=True)) == 3


def test_tiny_huggingface_layer_inference_on_cpu():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    data = [
        {"input_ids": torch.tensor([[1, 2, 3], [3, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
    ]
    report = SensitivitySweep(model, ["model.layers.0", "model.layers.1"]).sweep(
        data, amplitudes=[0.001, 0.002], top_k=1
    )
    assert len(report["targets"]) == 14
    assert len(report["rows"]) == 30
    assert all(r["finite"] for r in report["rows"])
    assert all(r["per_batch"][0]["count"] == 5 * 32 for r in report["rows"])
    assert any(
        r["stage"] == "subset" and len(r["members"]) == 2 and all("mlp" in n for n in r["members"])
        for r in report["rows"]
    )
    assert "undefined" not in render_report(report)


def test_cli_runs_tiny_model_and_serializes_report(tmp_path):
    assert (
        main(
            [
                "--tiny",
                "--output",
                str(tmp_path),
                "--max-batches",
                "1",
                "--max-length",
                "4",
                "--amplitudes",
                "0.001",
                "--top-layers",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "sensitivity.json").read_text())
    assert report["run"]["device"] == "cpu"
    assert report["run"]["tiny_random_fixture"]
    assert len(report["run"]["token_sha256"]) == 64
    assert "| subset |" in (tmp_path / "sensitivity.md").read_text()


def test_noise_casts_and_clone_helpers():
    value = torch.ones(4, dtype=torch.float16)
    assert torch.equal(value, _noise(value, 1e-12, "same", 0))
    assert _noise(torch.empty(0), 0.1, "empty", 0).numel() == 0
    assert _clone((1, [torch.ones(1)]))[0] == 1


def test_fp64_metric_overflow_fails_closed():
    stats = ErrorStats()
    stats.update(torch.tensor([1e308], dtype=torch.float64), torch.tensor([-1e308], dtype=torch.float64))
    assert stats.report()["relative_l2"] is None
    assert not stats.report()["finite"]
    finite_extreme = ErrorStats()
    finite_extreme.update(torch.tensor([1e-161], dtype=torch.float64), torch.tensor([1e154], dtype=torch.float64))
    assert finite_extreme.report()["finite"]
    assert finite_extreme.report()["relative_l2"] is None
    assert measured_gain(1e308, [1e-308], 0)[1] == "unresolved_ratio"
    assert measured_gain(1.0, [1e308] * 4, 0)[1] == "unresolved_ratio"
    assert uncorrelated_prediction([]) is None
    assert uncorrelated_prediction([None]) is None
    assert uncorrelated_prediction([1e308] * 4) is None
    assert uncorrelated_prediction([3.0, 4.0]) == 5.0


def test_baseline_noise_and_explicit_state_reset():
    class Stateful(TinyModel):
        tick = 0

        def forward(self, **kwargs):
            self.tick += 1
            out = super().forward(**kwargs)["logits"]
            return SimpleNamespace(logits=out + (100 if self.tick % 2 == 0 else 0))

    model = Stateful(depth=1)
    report = run(sweep(model))
    assert report["baseline_repeat"][0]["relative_l2"] > 0
    assert report["rows"][0]["g_status"] == "baseline_noise"
    report = run(sweep(model, reset_state=lambda m: setattr(m, "tick", 0)))
    assert report["baseline_repeat"][0]["relative_l2"] == 0
    assert report["rows"][0]["g_status"] == "measured"


def test_tensor_outputs_keyword_inputs_and_private_operator_adapter():
    class Operator(nn.Module):
        def forward(self, *, input):
            return input * 2

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = Operator()

        def forward(self, x):
            return self.projection(input=x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

        def forward(self, inputs_embeds, attention_mask):
            return self.layers[0](inputs_embeds)

    model = Model()
    scanner = SensitivitySweep(model, ["layers.0"], module_names=["layers.0.projection"])
    report = run(scanner)
    assert report["rows"][0]["final"]["relative_l2"] == pytest.approx(0.01, rel=1e-5)
    assert report["rows"][0]["g_effective"] == pytest.approx(1.0, rel=1e-5)
    with pytest.raises(TypeError, match="nn.Linear"):
        run(SensitivitySweep(model, ["layers.0"], module_names=["layers.0.projection"], candidate=shared_input_noise))
    for names in ([], ["layers.0.projection"] * 2, ["outside"], ["layers.0", "layers.0.projection"]):
        with pytest.raises(ValueError):
            SensitivitySweep(model, ["layers.0"], module_names=names)


def test_reference_failure_masks_and_non_tensor_targets():
    model = TinyModel(depth=1)
    model.lm_head.weight.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="Reference logits"):
        run(sweep(model))
    model = TinyModel(depth=1)
    with pytest.raises(ValueError, match="evaluation mask"):
        sweep(model).sweep([{**batches()[0], "attention_mask": torch.ones(1, 7)}])
    masked = {**batches()[0], "sensitivity_mask": torch.tensor([[1, 0]])}
    report = sweep(model).sweep([masked], amplitudes=[0.001], top_k=0)
    assert report["rows"][0]["final"]["count"] == 4

    class BadLinear(nn.Linear):
        def forward(self, x):
            return (super().forward(x),)

    model.layers[0].q_proj = BadLinear(3, 3)
    with pytest.raises(TypeError, match="floating tensor"):
        run(sweep(model))


def test_cli_raw_tokens_masks_and_shared_input_mode(tmp_path):
    path = tmp_path / "tokens.jsonl"
    path.write_text(
        "\n" + json.dumps({"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0], "sensitivity_mask": [0, 1, 0]}) + "\n"
    )
    subsets = tmp_path / "subsets.json"
    subsets.write_text(json.dumps({"qk": ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]}))
    assert (
        main(
            [
                "--tiny",
                "--data",
                str(path),
                "--output",
                str(tmp_path / "out"),
                "--subsets",
                str(subsets),
                "--probe",
                "shared-input",
                "--amplitudes",
                ".001",
                "--top-layers",
                "3",
                "--max-batches",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "out/sensitivity.json").read_text())
    assert all(r["final"]["count"] == 64 for r in report["rows"])
    assert report["run"]["token_sha256"] != report["run"]["input_sha256"]


def test_cli_rejects_missing_and_empty_data(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        main(["--tiny", "--output", str(tmp_path), "--max-batches", "0"])
    with pytest.raises(ValueError, match="held-out"):
        main(["--model", "unused", "--output", str(tmp_path)])
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps({"input_ids": []}) + "\n")
    with pytest.raises(ValueError, match="Empty token"):
        main(["--tiny", "--data", str(path), "--output", str(tmp_path)])
    path.write_text(json.dumps({"input_ids": [1, 2], "attention_mask": [1]}) + "\n")
    with pytest.raises(ValueError, match="must align"):
        main(["--tiny", "--data", str(path), "--output", str(tmp_path)])


def test_mapping_batches_are_copied_and_repeated_invocations_get_distinct_contexts():
    class Repeated(TinyModel):
        def forward(self, **kwargs):
            x = kwargs["inputs_embeds"]
            return self.lm_head(self.layers[0](self.layers[0](x)))

    seen = []

    def candidate(context, module, args, kwargs, reference):
        if context.name == "layers.0.q_proj":
            seen.append((context.invocation, context.input_group))
        return reference

    data = UserDict(batches()[0])
    data["sensitivity_mask"] = torch.tensor([[1, 0]])
    original = data["inputs_embeds"].clone()
    report = sweep(Repeated(depth=1), candidate=candidate).sweep([data], amplitudes=[1.0], top_k=0)
    assert seen[:2][0][0] == 1 and seen[:2][1][0] == 2
    assert seen[0][1] != seen[1][1]
    assert report["baseline_calls"]["layers.0.q_proj"] == 2
    assert "sensitivity_mask" in data and "use_cache" not in data
    assert torch.equal(original, data["inputs_embeds"])


def test_cli_local_checkpoint_load_and_text_tokenization(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    checkpoint = tmp_path / "checkpoint"
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        checkpoint
    )
    data = tmp_path / "text.jsonl"
    data.write_text(json.dumps({"text": "hello world"}) + "\n")
    assert (
        main(
            [
                "--model",
                str(checkpoint),
                "--data",
                str(data),
                "--output",
                str(tmp_path / "out"),
                "--amplitudes",
                ".001",
                "--top-layers",
                "1",
                "--max-batches",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "out/sensitivity.json").read_text())
    assert not report["run"]["tiny_random_fixture"]
    assert report["run"]["model_class"] == "LlamaForCausalLM"
