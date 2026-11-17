# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import MatchedQuantizeConfig, Matcher, QuantizeConfig, formats, solvers
from gptqmodel.models.definitions.qwen2 import Qwen2QModel


@pytest.fixture
def model():
    torch.manual_seed(17)
    config = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    dense = Qwen2ForCausalLM(config).eval()
    return Qwen2QModel(dense, quantized=False, quantize_config=None)


def target(matcher):
    return MatchedQuantizeConfig(matcher=matcher, solver=solvers.GPTQ(), format=formats.MXFP4())


def calibration():
    generator = torch.Generator().manual_seed(12)
    return [{"input_ids": torch.randint(3, 64, (1, 16), generator=generator)} for _ in range(3)]


def test_matched_config_inherits_and_keeps_legacy_factory():
    config = target(Matcher.attention(projections=("q", "k", "v")))
    assert isinstance(config, QuantizeConfig)
    assert config.solver == solvers.GPTQ()
    assert config.format == formats.MXFP4()
    assert QuantizeConfig(bits=4, group_size=32).bits == 4
    assert "matcher" in repr(config)
    assert config == copy.deepcopy(config)


def test_plan_prints_resolved_configs_without_mutation(model, capsys):
    original = {name: value.clone() for name, value in model.model.state_dict().items()}
    plan = model.plan(config=[target(Matcher.attention(projections=("q", "k", "v")))])
    assert plan.success and bool(plan)
    assert len(plan.assignments) == 6
    output = capsys.readouterr().out
    assert "model.layers.0.self_attn.q_proj" in output
    assert "mxfp4" in output and "gptq" in output
    for name, value in model.model.state_dict().items():
        assert torch.equal(value, original[name])


@pytest.mark.parametrize("configs", [
    [target(Matcher.moe_experts())],
    [target(Matcher.glob("*q_proj")), target(Matcher.attention(projections=("q",)))],
    [MatchedQuantizeConfig(matcher=Matcher.glob("*q_proj"), solver=solvers.AWQ(), format=formats.MXFP4())],
])
def test_unsupported_plan_blocks_before_calibration(model, configs):
    before = {name: value.clone() for name, value in model.model.state_dict().items()}
    plan = model.plan(config=configs)
    assert not plan.success and plan.errors

    def forbidden_calibration():
        pytest.fail("invalid plan consumed calibration")
        yield

    with pytest.raises(ValueError, match="plan"):
        model.quantize(forbidden_calibration(), config=configs)
    for name, value in model.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_quantize_automatically_plans_and_preserves_unmatched(model, capsys):
    from gptqmodel.nn_modules.qlinear.mxfp4 import MXFP4Linear

    config = [target(Matcher.attention(projections=("q", "k", "v")))]
    config_copy = copy.deepcopy(config)
    mlp = model.model.model.layers[0].mlp.gate_proj.weight.detach().clone()
    model.quantize(calibration(), config=config)
    assert "model.layers.1.self_attn.v_proj" in capsys.readouterr().out
    packed = [module for module in model.model.modules() if isinstance(module, MXFP4Linear)]
    assert len(packed) == 6
    assert torch.equal(mlp, model.model.model.layers[0].mlp.gate_proj.weight)
    assert [cfg.to_dict() for cfg in config] == [cfg.to_dict() for cfg in config_copy]
    inputs = calibration()[0]
    with torch.inference_mode():
        for _ in range(2):
            assert torch.isfinite(model.model(**inputs).logits).all()
    for module in packed:
        assert set(module.state_dict()) <= {"qweight", "scales", "bias"}
        assert module.qweight.dtype == module.scales.dtype == torch.uint8
        assert module.qweight.numel() * 2 == module.in_features * module.out_features
        assert not hasattr(module, "weight")


def test_public_save_load_resave_exact_payloads(model, tmp_path):
    from gptqmodel import GPTQModel

    model.quantize(calibration(), config=QuantizeConfig(solver=solvers.GPTQ(), format=formats.MXFP4()))
    inputs = calibration()[0]
    with torch.inference_mode():
        expected = model.model(**inputs).logits.clone()
    original = {name: value.clone() for name, value in model.model.state_dict().items()}
    for index in range(2):
        path = tmp_path / str(index)
        model.save(str(path), max_shard_size="30KB")
        model = GPTQModel.load(str(path), device="cpu")
        assert model.quantized
        with torch.inference_mode():
            torch.testing.assert_close(model.model(**inputs).logits, expected, rtol=0, atol=0)
        restored = model.model.state_dict()
        assert restored.keys() == original.keys()
        for name, value in original.items():
            assert restored[name].dtype == value.dtype
            assert torch.equal(restored[name], value), name


def test_plan_has_independent_config_snapshots_and_per_module_policies(model):
    first = target(Matcher.glob("*.q_proj"))
    second = MatchedQuantizeConfig(matcher=Matcher.glob("*.k_proj"),
                                  solver=solvers.GPTQ(damp_percent=0.1, desc_act=True), format=formats.MXFP4())
    plan = model.plan(config=[first, second])
    assert plan.success
    assert len(plan.assignments) == 4
    first.solver = solvers.GPTQ(damp_percent=0.2)
    assert all(entry.config.solver.damp_percent == (0.01 if entry.name.endswith("q_proj") else 0.1)
               for entry in plan.assignments)
    assert all(not hasattr(entry.config, "matcher") for entry in plan.assignments)
    model.quantize(calibration(), config=[first, second])
    assert all(spec["solver"]["damp_percent"] == (0.2 if name.endswith("q_proj") else 0.1)
               for name, spec in model._format_solutions.items())


@pytest.mark.parametrize("configs", [[], [QuantizeConfig(solver=solvers.GPTQ(), format=formats.MXFP4())]])
def test_invalid_target_lists(model, configs):
    assert not model.plan(config=configs)


def test_model_support_is_explicit_not_inherited(model):
    class UnqualifiedQwen(Qwen2QModel):
        pass

    wrapper = UnqualifiedQwen(model.model, quantized=False, quantize_config=None)
    plan = wrapper.plan(config=target(Matcher.glob("*q_proj")))
    assert not plan and any("not qualified" in error for error in plan.errors)


def test_partial_failure_removes_hooks_and_blocks_retry_save(model, tmp_path, monkeypatch):
    import gptqmodel.quantization.solver_format as runner

    original = runner.solve_gptq_hessian
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected solve failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "solve_gptq_hessian", fail_second)
    cfg = target(Matcher.attention())
    with pytest.raises(RuntimeError, match="injected"):
        model.quantize(calibration(), config=cfg)
    assert calls == 2 and model._solver_format_partial
    assert all(not module._forward_pre_hooks for module in model.model.modules())
    with pytest.raises(ValueError, match="incomplete"):
        model.save(str(tmp_path / "partial"))
    assert not model.plan(config=cfg)
    with torch.inference_mode():
        assert torch.isfinite(model.model(**calibration()[0]).logits).all()


def test_invalid_calibration_leaves_model_untouched(model):
    with pytest.raises(ValueError, match="mask"):
        model.quantize([{"input_ids": [[1, 2]], "attention_mask": [[0, 0]]}],
                       config=target(Matcher.glob("*q_proj")))
    assert not model.quantized
    assert not getattr(model, "_solver_format_partial", False)
    assert all(not module._forward_pre_hooks for module in model.model.modules())


def test_save_failure_does_not_publish_and_can_retry(model, tmp_path, monkeypatch):
    from gptqmodel.models import solver_format_checkpoint as checkpoint

    model.quantize(calibration(), config=target(Matcher.glob("*q_proj")))
    original = checkpoint.save_file

    def fail(*args, **kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(checkpoint, "save_file", fail)
    path = tmp_path / "checkpoint"
    with pytest.raises(OSError, match="injected"):
        model.save(str(path))
    assert not path.exists() and not list(tmp_path.iterdir())
    monkeypatch.setattr(checkpoint, "save_file", original)
    model.save(str(path))
    before = (path / "config.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        model.save(str(path))
    assert (path / "config.json").read_bytes() == before


def test_load_rejects_manifest_layout_mismatch(model, tmp_path):
    from gptqmodel import GPTQModel

    model.quantize(calibration(), config=target(Matcher.glob("*q_proj")))
    path = tmp_path / "checkpoint"
    model.save(str(path))
    config = json.loads((path / "config.json").read_text())
    first = next(iter(config["quantization_config"]["modules"].values()))
    first["format"]["layout"] = "vendor_swizzled"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format/layout"):
        GPTQModel.load(str(path), device="cpu")


def test_sdpa_and_padded_calibration(model):
    model.model.config._attn_implementation = "sdpa"
    rows = calibration()
    rows[0]["attention_mask"] = torch.tensor([[0] * 3 + [1] * 13])
    records = model.quantize(rows, config=target(Matcher.attention()))
    assert all(row["tokens"] == 45 for row in records)
    with torch.inference_mode():
        assert torch.isfinite(model.model(**rows[0]).logits).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_model_dtype_roundtrip(model, tmp_path, dtype):
    from transformers import AutoModelForCausalLM

    from gptqmodel import GPTQModel

    model.model = AutoModelForCausalLM.from_config(copy.deepcopy(model.model.config), dtype=dtype).eval()
    model.quantize(calibration(), config=target(Matcher.attention()))
    with torch.inference_mode():
        expected = model.model(**calibration()[0], use_cache=False).logits.clone()
    path = tmp_path / "checkpoint"
    model.save(str(path))
    restored = GPTQModel.load(str(path), device="cpu")
    with torch.inference_mode():
        actual = restored.model(**calibration()[0], use_cache=False).logits
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_plan_rejects_cast_rope_buffers(model):
    model.model.to(torch.bfloat16)
    plan = model.plan(config=target(Matcher.attention()))
    assert not plan and any("nonpersistent" in error for error in plan.errors)
