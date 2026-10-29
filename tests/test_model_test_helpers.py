from types import SimpleNamespace

import model_test as model_test_module
import torch
from model_test import ModelTest

from gptqmodel import BACKEND


class FakeBatchEncoding(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 7

    def __init__(self):
        self.decode_calls = []
        self.batch_decode_calls = []

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        assert prompt == "hello"
        return FakeBatchEncoding(torch.tensor([[101, 102]]))

    def decode(self, tokens, skip_special_tokens=True):
        self.decode_calls.append(
            {
                "tokens": tokens.tolist(),
                "skip_special_tokens": skip_special_tokens,
            }
        )
        return f"decoded:{tokens.tolist()}"

    def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.batch_decode_calls.append(
            {
                "sequences": [seq.tolist() for seq in sequences],
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return [f"batch:{[seq.tolist() for seq in sequences]}"]


class FakeProcessor(FakeTokenizer):
    pass


class FakeModel:
    def __init__(self, generated):
        self.device = "cpu"
        self.generated = generated
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.generated


def test_generate_stable_with_limit_for_prompt_uses_deterministic_kwargs():
    tokenizer = FakeTokenizer()
    model = FakeModel(torch.tensor([[101, 102, 103, 104]]))

    output = ModelTest.generate_stable_with_limit(
        model,
        tokenizer,
        "hello",
        min_new_tokens=2,
        max_new_tokens=4,
        skip_special_tokens=False,
    )

    assert output == "decoded:[101, 102, 103, 104]"
    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert model.calls[0]["min_new_tokens"] == 2
    assert model.calls[0]["max_new_tokens"] == 4
    assert model.calls[0]["pad_token_id"] == tokenizer.eos_token_id
    assert model.calls[0]["eos_token_id"] == tokenizer.eos_token_id
    assert tokenizer.decode_calls == [
        {
            "tokens": [101, 102, 103, 104],
            "skip_special_tokens": False,
        }
    ]


def test_generate_stable_with_limit_for_prepared_inputs_batch_decodes_suffix():
    processor = FakeProcessor()
    prepared_inputs = FakeBatchEncoding(torch.tensor([[10, 11]]))
    model = FakeModel(torch.tensor([[10, 11, 21, 22]]))

    output = ModelTest.generate_stable_with_limit(
        model,
        processor,
        inputs=prepared_inputs,
        prompt=None,
        batch_decode=True,
        max_new_tokens=2,
        clean_up_tokenization_spaces=False,
    )

    assert output == "batch:[[21, 22]]"
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].tolist() == [[10, 11]]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert processor.batch_decode_calls == [
        {
            "sequences": [[21, 22]],
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
    ]


def test_load_dataset_falls_back_when_datasets_import_failed(monkeypatch):
    fallback_dataset = ModelTest._LocalCalibrationDataset([{"text": "a"}, {"text": "b"}])

    monkeypatch.setattr(model_test_module, "hf_load_dataset", None)
    monkeypatch.setattr(model_test_module, "DATASETS_IMPORT_ERROR", SyntaxError("source code string cannot contain null bytes"))
    monkeypatch.setattr(ModelTest, "_load_calibration_parquet", staticmethod(lambda **_kwargs: fallback_dataset))

    dataset = ModelTest.load_dataset(rows=1)

    assert list(dataset) == [{"text": "a"}]


def test_load_dataset_falls_back_when_hf_loader_raises(monkeypatch):
    fallback_dataset = ModelTest._LocalCalibrationDataset([{"text": "x"}, {"text": "y"}])

    def _broken_load_dataset(*args, **kwargs):
        raise RuntimeError("broken datasets install")

    monkeypatch.setattr(model_test_module, "hf_load_dataset", _broken_load_dataset)
    monkeypatch.setattr(ModelTest, "_load_calibration_parquet", staticmethod(lambda **_kwargs: fallback_dataset))

    dataset = ModelTest.load_dataset(rows=5)

    assert list(dataset) == [{"text": "x"}, {"text": "y"}]


def test_detect_gpu_profile_from_cuda0_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: "NVIDIA A100-SXM4-80GB")
    assert ModelTest._detect_gpu_profile() == "A100"

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: "NVIDIA GeForce RTX 4090")
    assert ModelTest._detect_gpu_profile() == "RTX4090"


def test_resolve_metric_baseline_value_uses_gpu_profile(monkeypatch):
    helper = ModelTest(methodName="runTest")
    monkeypatch.setattr(ModelTest, "_detect_gpu_profile", classmethod(lambda cls: "A100"))
    selected = helper._resolve_metric_baseline_value({"A100": 0.6, "RTX4090": 0.5})
    assert selected == 0.6

    monkeypatch.setattr(ModelTest, "_detect_gpu_profile", classmethod(lambda cls: "unknown"))
    selected_a100_fallback = helper._resolve_metric_baseline_value({"A100": 0.6, "RTX4090": 0.5})
    assert selected_a100_fallback == 0.6

    monkeypatch.setattr(ModelTest, "_detect_gpu_profile", classmethod(lambda cls: "A100"))
    selected_a100_fallback = helper._resolve_metric_baseline_value(0.6)
    assert selected_a100_fallback == 0.6

    monkeypatch.setattr(ModelTest, "_detect_gpu_profile", classmethod(lambda cls: "RTX 4090"))
    selected_a100_fallback = helper._resolve_metric_baseline_value(0.6)
    assert selected_a100_fallback == 0.6

    monkeypatch.setattr(ModelTest, "_detect_gpu_profile", classmethod(lambda cls: "unknown"))
    selected_a100_fallback = helper._resolve_metric_baseline_value(0.6)
    assert selected_a100_fallback == 0.6


def test_mode_specific_baseline_value_supports_gpu_mapping(monkeypatch):
    class _ModeSpecificHarness(ModelTest):
        NATIVE_ARC_CHALLENGE_ACC_FAST = {"A100": 0.55, "RTX4090": 0.53}

    helper = _ModeSpecificHarness(methodName="runTest")
    monkeypatch.setattr(_ModeSpecificHarness, "_is_fast_model_test_mode", lambda self: True)
    monkeypatch.setattr(_ModeSpecificHarness, "_detect_gpu_profile", classmethod(lambda cls: "RTX4090"))

    assert helper._mode_specific_baseline_value("NATIVE_ARC_CHALLENGE_ACC") == 0.53


def test_derive_fast_eval_tasks_skips_evalution_metadata_keys():
    eval_tasks = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            # `"auto"` is a valid batch-size sentinel and is not numeric.
            "evalution_batch_size": "auto",
            "evalution_model_args": {"dtype": "bfloat16", "device": "cuda:0"},
            "evalution_suite_kwargs": {"batch_size": 32, "max_new_tokens": 256},
            "acc,num": {"value": 0.47, "floor_pct": 0.04, "ceil_pct": 1.0},
        },
    }

    fast_tasks = ModelTest.derive_fast_eval_tasks(eval_tasks, min_ceil_pct=1.0)
    metrics = fast_tasks["gsm8k_platinum_cot"]

    # Metadata is preserved verbatim, not coerced into metric dicts.
    assert metrics["chat_template"] is True
    assert metrics["evalution_use_model_path"] is True
    assert metrics["evalution_batch_size"] == "auto"
    assert metrics["evalution_model_args"] == {"dtype": "bfloat16", "device": "cuda:0"}
    assert metrics["evalution_suite_kwargs"] == {"batch_size": 32, "max_new_tokens": 256}

    # Only real metrics are normalized/ceil-adjusted.
    assert metrics["acc,num"] == {"value": 0.47, "floor_pct": 0.04, "ceil_pct": 1.0}

    # Input config is not mutated in place.
    assert eval_tasks["gsm8k_platinum_cot"]["acc,num"] == {"value": 0.47, "floor_pct": 0.04, "ceil_pct": 1.0}


def test_derive_fast_eval_tasks_preserves_numeric_evalution_batch_size():
    # `evalution_batch_size` accepts concrete integers as well as the "auto"
    # sentinel. A numeric value must stay metadata rather than being silently
    # coerced into a metric dict by `float()`.
    eval_tasks = {
        "task": {
            "chat_template": True,
            "evalution_batch_size": 8,
            "acc": {"value": 0.5, "floor_pct": 0.05, "ceil_pct": 1.0},
        },
    }

    metrics = ModelTest.derive_fast_eval_tasks(eval_tasks)["task"]

    assert metrics["evalution_batch_size"] == 8
    assert not isinstance(metrics["evalution_batch_size"], dict)
    assert metrics["acc"] == {"value": 0.5, "floor_pct": 0.05, "ceil_pct": 1.0}
    assert set(metrics) == {"chat_template", "evalution_batch_size", "acc"}


def test_derive_fast_eval_tasks_preserves_all_evalution_batch_size_forms():
    for batch_size in (8, 4, 0, "auto", None):
        eval_tasks = {"task": {"evalution_batch_size": batch_size, "acc": 0.5}}
        metrics = ModelTest.derive_fast_eval_tasks(eval_tasks)["task"]

        assert metrics["evalution_batch_size"] == batch_size, batch_size
        assert not isinstance(metrics["evalution_batch_size"], dict), batch_size
        assert isinstance(metrics["acc"], dict), batch_size


def test_derive_fast_eval_tasks_wraps_scalar_metrics_and_raises_calibrated_ceil():
    eval_tasks = {
        "arc_challenge": {
            "chat_template": False,
            "acc": 0.30,
            "acc_norm": {"value": 0.35, "ceil_pct": 0.01},
        },
    }

    fast_tasks = ModelTest.derive_fast_eval_tasks(eval_tasks, min_ceil_pct=1.0)
    metrics = fast_tasks["arc_challenge"]

    assert metrics["acc"] == {
        "value": 0.30,
        "floor_pct": model_test_module.DEFAULT_FLOOR_PCT,
        "ceil_pct": 1.0,
    }
    # Existing metric dicts keep their floor and get the raised ceil.
    assert metrics["acc_norm"] == {"value": 0.35, "ceil_pct": 1.0}


def test_derive_fast_eval_tasks_skips_unknown_evalution_prefixed_keys():
    eval_tasks = {
        "task": {
            # A future metadata key that is not yet in the explicit set.
            "evalution_future_option": "auto",
            "acc": 0.5,
        },
    }

    metrics = ModelTest.derive_fast_eval_tasks(eval_tasks, min_ceil_pct=0.10)["task"]

    assert metrics["evalution_future_option"] == "auto"
    assert metrics["acc"] == {
        "value": 0.5,
        "floor_pct": model_test_module.DEFAULT_FLOOR_PCT,
        "ceil_pct": model_test_module.DEFAULT_CEIL_PCT,
    }


def test_derive_fast_eval_tasks_handles_none_and_non_dict_task_payloads():
    assert ModelTest.derive_fast_eval_tasks(None) is None

    eval_tasks = {"arc_challenge": None, "gsm8k": ("accuracy",)}

    assert ModelTest.derive_fast_eval_tasks(eval_tasks) == eval_tasks


def test_get_eval_tasks_consumes_metadata_and_keeps_metric_baselines():
    class _Harness(ModelTest):
        EVAL_TASKS = {
            "gsm8k_platinum_cot": {
                "chat_template": True,
                "evalution_use_model_path": True,
                "evalution_batch_size": "auto",
                "evalution_model_args": {"dtype": "bfloat16"},
                "evalution_suite_kwargs": {"batch_size": 32},
                "acc,num": {"value": 0.47, "floor_pct": 0.04, "ceil_pct": 1.0},
            },
        }

    helper = _Harness(methodName="runTest")
    baselines = helper.get_eval_tasks()

    assert baselines == {
        "gsm8k_platinum_cot": {
            "acc,num": {
                "value": 0.47,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
                "metric_key": None,
            }
        }
    }
    assert helper._task_evalution_batch_size == {"gsm8k_platinum_cot": "auto"}
    assert helper._task_evalution_use_model_path == {"gsm8k_platinum_cot": True}
    assert helper._task_chat_template == {"gsm8k_platinum_cot": True}


def test_eval_task_metadata_keys_match_get_eval_tasks_consumed_keys():
    class _Harness(ModelTest):
        EVAL_TASKS = {
            "task": {
                "chat_template": True,
                "evalution_use_model_path": True,
                "evalution_batch_size": "auto",
                "evalution_model_args": {"dtype": "bfloat16"},
                "evalution_suite_kwargs": {"batch_size": 1},
                "acc": {"value": 0.5},
            },
        }

    # If `get_eval_tasks` starts consuming a new metadata key, the shared set used
    # by `derive_fast_eval_tasks` must be updated with it.
    helper = _Harness(methodName="runTest")
    normalized = helper.get_eval_tasks()["task"]
    assert set(normalized) == {"acc"}
    assert model_test_module.EVAL_TASK_METADATA_KEYS == {
        "chat_template",
        "evalution_suite_kwargs",
        "evalution_model_args",
        "evalution_use_model_path",
        "evalution_batch_size",
    }


def test_evalution_threads_seed_and_explicit_greedy_gen_kwargs(monkeypatch):
    captured = {}

    class _Harness(ModelTest):
        EVAL_TASKS = ("arc_challenge",)
        LOAD_BACKEND = BACKEND.AUTO
        QUANT_BACKEND = BACKEND.AUTO

    def _fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"tests": [{"name": "arc_challenge", "metrics": {"accuracy,loglikelihood": 1.0}}]}

    helper = _Harness(methodName="runTest")
    monkeypatch.setattr(model_test_module, "evaluate", _fake_evaluate)
    monkeypatch.setattr(_Harness, "_cleanup_quantized_model", lambda self, model, enabled=False: None)
    monkeypatch.setattr(_Harness, "_normalize_task_list", lambda self: ["arc_challenge"])
    monkeypatch.setattr(
        _Harness,
        "_current_load_backend",
        lambda self: SimpleNamespace(name="AUTO"),
    )

    results = helper.evaluate_model("/tmp/model", extra_args={"device": "cuda:0"})

    assert results == {"arc_challenge": {"accuracy,loglikelihood": 1.0}}
    assert captured["model_args"]["device"] == "cuda:0"
    assert captured["model_args"]["seed"] == model_test_module.RAND_SEED
    assert "random_seed" not in captured["model_args"]
    assert captured["gen_kwargs"] == "do_sample=false,temperature=0.0,top_p=1.0,top_k=50"


def test_quantize_and_evaluate_runs_evalution_for_prequantized_model_without_cached_post_quant_results(monkeypatch):
    class _PrequantizedModel:
        quantized = True

    class _Harness(ModelTest):
        NATIVE_MODEL_ID = "/tmp/prequantized-model"
        LOAD_BACKEND = BACKEND.TORCH_FUSED
        DELETE_QUANTIZED_MODEL = False
        NATIVE_ARC_CHALLENGE_ACC = 0.30
        NATIVE_ARC_CHALLENGE_ACC_NORM = 0.31

        def __init__(self):
            super().__init__(methodName="runTest")
            self.evaluate_model_calls = 0

        def quantModel(self, *args, **kwargs):
            self._post_quant_eval_records = {}
            self._loaded_model_was_prequantized = True
            return _PrequantizedModel(), None, None

        def evaluate_model(self, model, trust_remote_code=False, delete_quantized_model=False, extra_args=None):
            self.evaluate_model_calls += 1
            assert model is self.model
            assert delete_quantized_model is False
            return {
                "arc_challenge": {
                    "accuracy,loglikelihood": 0.30,
                    "accuracy,loglikelihood_norm": 0.31,
                }
            }

    helper = _Harness()

    monkeypatch.setattr(_Harness, "check_kernel", lambda self, model, expected: None)
    monkeypatch.setattr(_Harness, "_cleanup_quantized_model", lambda self, model, enabled=False: None)

    helper.quantize_and_evaluate()

    assert helper.evaluate_model_calls == 1
    assert helper._post_quant_eval_records[BACKEND.TORCH_FUSED] == {
        "arc_challenge": {
            "accuracy,loglikelihood": 0.30,
            "accuracy,loglikelihood_norm": 0.31,
        }
    }


def test_quant_model_for_prequantized_vl_forwards_trust_remote_code_to_processor(monkeypatch):
    tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=2)
    prequantized_model = SimpleNamespace(
        config=SimpleNamespace(model_type="mage_vl", pad_token_id=1, eos_token_id=2),
        modality=[model_test_module.MODALITY.IMAGE_TO_TEXT],
        quantized=True,
        tokenizer=tokenizer,
    )
    processor = object()
    processor_calls = []

    class _Harness(ModelTest):
        USE_FLASH_ATTN = False

    helper = _Harness(methodName="runTest")
    quantize_config = SimpleNamespace(requires_calibration_dataset=lambda: False)

    monkeypatch.setattr(helper, "_build_quantize_config", lambda: quantize_config)
    monkeypatch.setattr(helper, "_apply_model_compat_quant_overrides", lambda _model: None)
    monkeypatch.setattr(model_test_module.GPTQModel, "load", lambda *args, **kwargs: prequantized_model)

    def _load_processor(*args, **kwargs):
        processor_calls.append((args, kwargs))
        return processor

    monkeypatch.setattr(model_test_module.AutoProcessor, "from_pretrained", _load_processor)

    loaded_model, loaded_tokenizer, loaded_processor = helper.quantModel(
        "/tmp/prequantized-vl",
        trust_remote_code=True,
        need_eval=False,
    )

    assert loaded_model is prequantized_model
    assert loaded_tokenizer is tokenizer
    assert loaded_processor is processor
    assert processor_calls == [(("/tmp/prequantized-vl",), {"trust_remote_code": True})]


def test_perform_post_quant_validation_retries_without_forced_cuda_map_after_oom(monkeypatch):
    class _LoadedModel:
        def __init__(self):
            self.tokenizer = object()

    class _Harness(ModelTest):
        LOAD_BACKEND = BACKEND.AUTO

    helper = _Harness(methodName="runTest")
    load_calls = []

    def _fake_load_quant_model(*args, **kwargs):
        load_calls.append(kwargs.copy())
        if kwargs.get("device_map") == {"": "cuda:0"}:
            raise torch.OutOfMemoryError("simulated OOM")
        return _LoadedModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(helper, "loadQuantModel", _fake_load_quant_model)
    monkeypatch.setattr(helper, "run_eval_tasks", lambda model, backend, trust_remote_code=False: {"ok": 1.0})
    monkeypatch.setattr(helper, "render_inference_summary", lambda records: None)
    monkeypatch.setattr(helper, "render_eval_summary", lambda records: None)
    monkeypatch.setattr(model_test_module, "torch_empty_cache", lambda: None)

    reuse_candidates, eval_records = helper.perform_post_quant_validation("/tmp/model", trust_remote_code=False)

    assert load_calls == [
        {"trust_remote_code": False, "backend": BACKEND.AUTO, "device_map": {"": "cuda:0"}},
        {"trust_remote_code": False, "backend": BACKEND.AUTO},
    ]
    assert reuse_candidates == {}
    assert eval_records == {BACKEND.AUTO: {"ok": 1.0}}
