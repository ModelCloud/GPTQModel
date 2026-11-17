# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import CalibrationConfig, MatchedQuantizeConfig, Matcher, QuantizeConfig, formats, solvers
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


def test_calibration_config_object_reaches_matched_sink_statistics(model):
    rows = calibration()
    for row in rows:
        row["input_ids"][0, 0] = 1
    model.tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0)
    other = Qwen2QModel(copy.deepcopy(model.model), quantized=False, quantize_config=None)
    other.tokenizer = model.tokenizer
    config = CalibrationConfig(dataset=rows, exclude_attention_sinks=True)
    records = model.quantize(config, config=target(Matcher.attention()))
    repeated = other.quantize(config, config=target(Matcher.attention()))
    assert records == repeated
    assert all(torch.equal(value, other.model.state_dict()[name]) for name, value in model.model.state_dict().items())
    assert all(row["tokens"] == 45 and row["excluded_attention_sink_tokens"] == 3 for row in records)
    assert config.prepare_dataset is None and config.dataset is rows
    assert all(row["input_ids"][0, 0] == 1 for row in rows)


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
        model.quantize(CalibrationConfig(dataset=forbidden_calibration()), config=configs)
    for name, value in model.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_quantize_automatically_plans_and_preserves_unmatched(model, capsys):
    from gptqmodel.nn_modules.qlinear.mxfp4 import MXFP4Linear

    config = [target(Matcher.attention(projections=("q", "k", "v")))]
    config_copy = copy.deepcopy(config)
    mlp = model.model.model.layers[0].mlp.gate_proj.weight.detach().clone()
    model.quantize(CalibrationConfig(dataset=calibration()), config=config)
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

    model.quantize(CalibrationConfig(dataset=calibration()),
                   config=QuantizeConfig(solver=solvers.GPTQ(), format=formats.MXFP4()))
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
    model.quantize(CalibrationConfig(dataset=calibration()), config=[first, second])
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
        model.quantize(CalibrationConfig(dataset=calibration()), config=cfg)
    assert calls == 2 and model._solver_format_partial
    assert all(not module._forward_pre_hooks for module in model.model.modules())
    with pytest.raises(ValueError, match="incomplete"):
        model.save(str(tmp_path / "partial"))
    assert not model.plan(config=cfg)
    with torch.inference_mode():
        assert torch.isfinite(model.model(**calibration()[0]).logits).all()


def test_invalid_calibration_leaves_model_untouched(model):
    with pytest.raises(ValueError, match="empty"):
        model.quantize(CalibrationConfig(dataset=[{"input_ids": [[1, 2]], "attention_mask": [[0, 0]]}]),
                       config=target(Matcher.glob("*q_proj")))
    assert not model.quantized
    assert not getattr(model, "_solver_format_partial", False)
    assert all(not module._forward_pre_hooks for module in model.model.modules())


def test_save_failure_does_not_publish_and_can_retry(model, tmp_path, monkeypatch):
    from gptqmodel.models import solver_format_checkpoint as checkpoint

    model.quantize(CalibrationConfig(dataset=calibration()), config=target(Matcher.glob("*q_proj")))
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

    model.quantize(CalibrationConfig(dataset=calibration()), config=target(Matcher.glob("*q_proj")))
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
    records = model.quantize(CalibrationConfig(dataset=rows, min_length=0), config=target(Matcher.attention()))
    assert all(row["tokens"] == 45 for row in records)
    with torch.inference_mode():
        assert torch.isfinite(model.model(**rows[0]).logits).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_model_dtype_roundtrip(model, tmp_path, dtype):
    from transformers import AutoModelForCausalLM

    from gptqmodel import GPTQModel

    model.model = AutoModelForCausalLM.from_config(copy.deepcopy(model.model.config), dtype=dtype).eval()
    model.quantize(CalibrationConfig(dataset=calibration()), config=target(Matcher.attention()))
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


def test_attention_sink_config_reaches_hessian_without_changing_forward_context(model, monkeypatch):
    import gptqmodel.quantization.solver_format as runner

    rows = [{"input_ids": torch.tensor([[1, 3, 1, 4, 1, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]])},
            {"input_ids": torch.tensor([[5, 1, 7, 8, 9, 1]]), "attention_mask": torch.ones(1, 6, dtype=torch.long)}]
    before = copy.deepcopy(rows)
    inputs = []
    module = model.model.model.layers[0].self_attn.q_proj
    handle = module.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().double().clone()))
    with torch.inference_mode():
        for row in rows:
            model.model(**row, use_cache=False)
    handle.remove()
    selected = torch.cat([value[(row["input_ids"] != 1) & row["attention_mask"].bool()]
                          for value, row in zip(inputs, rows)])
    expected = selected.T @ selected / 6
    original = runner.solve_gptq_hessian
    calls = []

    def check_hessian(weight, hessian, solver):
        torch.testing.assert_close(hessian, expected, rtol=1e-14, atol=1e-14)
        calls.append(hessian.clone())
        return original(weight, hessian, solver)

    monkeypatch.setattr(runner, "solve_gptq_hessian", check_hessian)
    model.tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0)
    records = model.quantize(CalibrationConfig(dataset=rows, min_length=0, exclude_attention_sinks=True),
                             config=target(Matcher.glob("model.layers.0.self_attn.q_proj")))
    assert len(calls) == 1
    assert records[0]["tokens"] == 6 and records[0]["excluded_attention_sink_tokens"] == 4
    for row, old in zip(rows, before):
        assert all(torch.equal(row[name], value) for name, value in old.items())


def test_attention_sink_option_without_tokenizer_bos_is_bitwise_noop(model):
    other = Qwen2QModel(copy.deepcopy(model.model), quantized=False, quantize_config=None)
    rows = calibration()
    rows[0]["input_ids"][0, 0] = 1
    model.model.config.bos_token_id = other.model.config.bos_token_id = 1
    cfg = target(Matcher.attention())
    other.tokenizer = SimpleNamespace(bos_token_id=None, pad_token_id=0)
    baseline = model.quantize(CalibrationConfig(dataset=rows), config=cfg)
    filtered = other.quantize(CalibrationConfig(dataset=rows, exclude_attention_sinks=True), config=cfg)
    assert baseline == filtered
    assert all(row["tokens"] == 48 and row["excluded_attention_sink_tokens"] == 0 for row in filtered)
    assert all(torch.equal(value, other.model.state_dict()[name]) for name, value in model.model.state_dict().items())


def test_all_sink_tokens_fail_before_replay_or_weight_changes(model):
    before = {name: tensor.clone() for name, tensor in model.model.state_dict().items()}
    model.tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0)
    with pytest.raises(ValueError, match="No calibration tokens"):
        model.quantize(CalibrationConfig(dataset=[[1, 1, 1]], min_length=0, exclude_attention_sinks=True),
                       config=target(Matcher.attention()))
    assert all(torch.equal(value, model.model.state_dict()[name]) for name, value in before.items())
    assert all(not module._forward_pre_hooks for module in model.model.modules())


def test_attention_sink_flag_requires_boolean_before_reading_calibration(model):
    def unreadable():
        raise AssertionError("calibration was consumed")
        yield

    with pytest.raises(TypeError, match="exclude_attention_sinks"):
        model.quantize(CalibrationConfig(dataset=unreadable(), exclude_attention_sinks=1),
                       config=target(Matcher.attention()))


@pytest.mark.parametrize("form", ["rows", "mappings", "columns", "tensors", "generator"])
@pytest.mark.parametrize("sort", [None, "asc", "desc"])
def test_shared_preparation_reaches_forward_and_statistics(model, form, sort):
    rows = [[1, 3, 4, 1, 5, 6], [1, 7], [8, 1, 9, 10]]
    dataset = rows
    if form == "mappings":
        dataset = [{"input_ids": row} for row in rows]
    elif form == "columns":
        dataset = {"input_ids": rows}
    elif form == "tensors":
        dataset = [torch.tensor(row) for row in rows]
    elif form == "generator":
        dataset = (row for row in rows)
    model.tokenizer = SimpleNamespace(bos_token_id=1, pad_token_id=0)
    calibration_config = CalibrationConfig(dataset=dataset, min_length=2, sort=sort, exclude_attention_sinks=True)
    seen = []
    handle = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["input_ids"].clone()), with_kwargs=True)
    try:
        records = model.quantize(calibration_config, config=target(Matcher.attention()))
    finally:
        handle.remove()
    expected = [rows[2], rows[0]] if sort == "asc" else [rows[0], rows[2]]
    assert [row.tolist()[0] for row in seen] == expected
    assert len(records) == 6
    assert all(row["tokens"] == 7 and row["excluded_attention_sink_tokens"] == 3 for row in records)
    assert calibration_config.dataset is dataset and calibration_config.prepare_dataset is None


def test_text_concatenation_tokenizer_override_and_sink_roundtrip(model, tmp_path):
    from transformers import Qwen2TokenizerFast

    from gptqmodel import GPTQModel

    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 63}
    vocab.update({chr(97 + i): i + 3 for i in range(26)})
    vocab.update({f"unused{i}": i for i in range(29, 63)})
    tokenizer = Qwen2TokenizerFast(vocab=vocab, merges=[], bos_token="[BOS]", eos_token="[EOS]",
                                   pad_token="[PAD]", unk_token="[UNK]", model_max_length=128)
    model.model.config.bos_token_id = 1
    model.model.config.eos_token_id = 2
    model.model.config.pad_token_id = 0
    config = CalibrationConfig(dataset=["abc", "defg"], min_length=0, sort=None,
                               concat_size=6, concat_separator="[BOS]", exclude_attention_sinks=True)
    seen = []
    handle = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((kwargs["input_ids"].clone(), kwargs["attention_mask"].clone())),
        with_kwargs=True)
    try:
        records = model.quantize(config, config=target(Matcher.attention()), tokenizer=tokenizer)
    finally:
        handle.remove()
    assert [ids.tolist()[0] for ids, mask in seen] == [[3, 4, 5, 1, 6, 7], [8, 9, 0, 0, 0, 0]]
    assert [mask.tolist()[0] for ids, mask in seen] == [[1] * 6, [1, 1, 0, 0, 0, 0]]
    assert all(row["tokens"] == 7 and row["excluded_attention_sink_tokens"] == 1 for row in records)
    inputs = model.tokenizer("ab", return_tensors="pt")
    with torch.inference_mode():
        expected = model.model(**inputs).logits
    path = tmp_path / "packed"
    model.save(str(path))
    restored = GPTQModel.load(str(path), device="cpu")
    assert restored.tokenizer.bos_token_id == 1
    assert restored.tokenizer("ab")["input_ids"] == [3, 4]
    with torch.inference_mode():
        assert torch.equal(restored.model(**inputs).logits, expected)


@pytest.mark.parametrize("options,match", [({"batch_size": 2}, "batch_size"),
                                        ({"adapter_dataset": [[1, 2]]}, "options")])
def test_unsupported_calibration_options_fail_before_consumption(model, options, match):
    def unreadable():
        pytest.fail("unsupported calibration policy consumed data")
        yield

    with pytest.raises(ValueError, match=match):
        model.quantize(CalibrationConfig(dataset=unreadable(), **options), config=target(Matcher.attention()))
    assert not model.quantized
    assert all(not module._forward_pre_hooks for module in model.model.modules())


def test_matched_calibration_rejects_old_api_and_bound_config(model):
    config = target(Matcher.attention())
    with pytest.raises(TypeError, match="CalibrationConfig"):
        model.quantize(calibration(), config=config)
    with pytest.raises(TypeError, match="calibration_exclude_attention_sinks"):
        model.quantize(CalibrationConfig(dataset=calibration()), config=config, calibration_exclude_attention_sinks=True)
    with pytest.raises(ValueError, match="unbound"):
        model.quantize(CalibrationConfig(dataset=calibration()).bind(model.prepare_dataset), config=config)
