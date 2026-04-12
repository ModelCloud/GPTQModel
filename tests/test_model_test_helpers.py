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
    monkeypatch.setattr(ModelTest, "_load_calibration_parquet", staticmethod(lambda: fallback_dataset))

    dataset = ModelTest.load_dataset(rows=1)

    assert list(dataset) == [{"text": "a"}]


def test_load_dataset_falls_back_when_hf_loader_raises(monkeypatch):
    fallback_dataset = ModelTest._LocalCalibrationDataset([{"text": "x"}, {"text": "y"}])

    def _broken_load_dataset(*args, **kwargs):
        raise RuntimeError("broken datasets install")

    monkeypatch.setattr(model_test_module, "hf_load_dataset", _broken_load_dataset)
    monkeypatch.setattr(ModelTest, "_load_calibration_parquet", staticmethod(lambda: fallback_dataset))

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
    assert captured["model_args"]["random_seed"] == model_test_module.RAND_SEED
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
