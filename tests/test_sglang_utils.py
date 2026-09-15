# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import loader
from gptqmodel.utils import sglang as sglang_utils


class _FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.result = {"text": "engine result"}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeRuntime:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSglang:
    Engine = _FakeEngine
    Runtime = _FakeRuntime

    def __init__(self):
        self.default_backend = None

    def set_default_backend(self, runtime):
        self.default_backend = runtime


@pytest.fixture
def fake_sglang(monkeypatch):
    fake = _FakeSglang()
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(sglang_utils, "sgl", fake)
    return fake


def test_engine_aliases_dtype_device_and_config_reuse(fake_sglang, monkeypatch):
    config = object()
    monkeypatch.setattr(
        sglang_utils,
        "AutoConfig",
        SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: pytest.fail("config was reloaded")
        ),
    )

    runtime, returned_config = sglang_utils.load_model_by_sglang(
        "/tmp/model",
        trust_remote_code=True,
        config=config,
        dtype=torch.bfloat16,
        device="cuda:2",
        tensor_parallel_size="2",
        gpu_memory_utilization="0.75",
        max_model_len="4096",
        seed="7",
        enforce_eager=True,
    )

    assert returned_config is config
    assert runtime.kwargs["dtype"] == "bfloat16"
    assert runtime.kwargs["device"] == "cuda"
    assert runtime.kwargs["base_gpu_id"] == 2
    assert runtime.kwargs["tp_size"] == 2
    assert runtime.kwargs["mem_fraction_static"] == 0.75
    assert runtime.kwargs["context_length"] == 4096
    assert runtime.kwargs["random_seed"] == 7
    assert runtime.kwargs["disable_cuda_graph"] is True
    assert runtime.kwargs["trust_remote_code"] is True
    assert getattr(runtime, sglang_utils._ENGINE_MARKER) is True
    assert fake_sglang.default_backend is None
    assert (
        sglang_utils._normalize_sglang_engine_kwargs({"dtype": "torch.float16"}, False)[
            "dtype"
        ]
        == "float16"
    )


def test_legacy_runtime_is_selected_when_engine_is_absent(monkeypatch):
    fake = SimpleNamespace(
        Runtime=_FakeRuntime,
        set_default_backend=lambda runtime: setattr(fake, "backend", runtime),
    )
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(sglang_utils, "sgl", fake)
    monkeypatch.setattr(
        sglang_utils,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: "config"),
    )

    runtime, config = sglang_utils.load_model_by_sglang(
        "model", trust_remote_code=False
    )

    assert config == "config"
    assert runtime.kwargs["trust_remote_code"] is False
    assert getattr(runtime, sglang_utils._ENGINE_MARKER) is False
    assert fake.backend is runtime


def test_engine_initialization_failure_does_not_fall_back_to_legacy(monkeypatch):
    class BrokenEngine:
        def __init__(self, **_kwargs):
            raise ModuleNotFoundError("No module named 'flashinfer'")

    class UnexpectedRuntime:
        def __init__(self, **_kwargs):
            pytest.fail("legacy Runtime must not hide a broken Engine installation")

    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        sglang_utils,
        "sgl",
        SimpleNamespace(Engine=BrokenEngine, Runtime=UnexpectedRuntime),
    )

    with pytest.raises(
        RuntimeError, match="Failed to initialize SGLang Engine.*flashinfer"
    ):
        sglang_utils.load_model_by_sglang(
            "model",
            trust_remote_code=False,
            config=object(),
        )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("tensor_parallel_size", "tp_size"),
        ("gpu_memory_utilization", "mem_fraction_static"),
        ("max_model_len", "context_length"),
        ("seed", "random_seed"),
    ],
)
def test_engine_alias_conflicts_are_rejected(source, target):
    with pytest.raises(ValueError, match="Pass only one"):
        sglang_utils._normalize_sglang_engine_kwargs(
            {source: 1, target: 2}, trust_remote_code=False
        )


def test_enforce_eager_conflict_and_rocm_normalization():
    with pytest.raises(ValueError, match="enforce_eager"):
        sglang_utils._normalize_sglang_engine_kwargs(
            {"enforce_eager": True, "disable_cuda_graph": False},
            trust_remote_code=False,
        )
    assert (
        sglang_utils._normalize_sglang_engine_kwargs({"device": "rocm"}, False)[
            "device"
        ]
        == "cuda"
    )


def test_loader_sglang_kwargs_pass_only_resolved_dtype():
    result = loader._build_sglang_runtime_kwargs(
        kwargs={"dtype": "wrong", "foo": "bar"},
        device=torch.device("cuda:1"),
        requested_device_map=None,
        resolved_dtype=torch.float32,
    )
    assert result["dtype"] is torch.float32
    assert result["device"] == "cuda"
    assert result["foo"] == "bar"


def test_loader_sglang_explicit_string_dtype_precedes_auto_config_dtype():
    assert loader._normalize_sglang_load_dtype("bfloat16") is torch.bfloat16
    assert loader._normalize_sglang_load_dtype("torch.float16") is torch.float16
    assert loader._normalize_sglang_load_dtype(torch.float32) is torch.float32
    assert loader._normalize_sglang_load_dtype("auto") == "auto"

    with pytest.raises(ValueError, match="Invalid SGLang dtype"):
        loader._normalize_sglang_load_dtype("not_a_dtype")


def test_sglang_format_allowlist_is_not_expanded():
    for format_code in (
        loader.FORMAT.GPTQ,
        loader.FORMAT.GPTQ_V2,
        loader.FORMAT.GEMM,
        loader.FORMAT.MARLIN,
    ):
        loader._validate_external_backend_format(loader.BACKEND.SGLANG, format_code)

    with pytest.raises(ValueError, match="only supports"):
        loader._validate_external_backend_format(
            loader.BACKEND.SGLANG,
            loader.FORMAT.GEMV_FAST,
        )


def test_input_normalization_supports_text_and_tensor_batches():
    assert sglang_utils._normalize_sglang_inputs("hello", None, None) == ("hello", None)
    assert sglang_utils._normalize_sglang_inputs(["a", "b"], None, None) == (
        ["a", "b"],
        None,
    )
    assert sglang_utils._normalize_sglang_inputs(None, torch.tensor([1, 2]), None) == (
        None,
        [1, 2],
    )
    assert sglang_utils._normalize_sglang_inputs(None, ((1, 2), [3, 4]), None) == (
        None,
        [[1, 2], [3, 4]],
    )


def test_input_ids_mask_removes_padding_and_validates_shape_and_empty():
    assert sglang_utils._normalize_sglang_inputs(
        None, [[0, 1, 2], [3, 4, 5]], [[0, 1, 1], [1, 1, 0]]
    ) == (None, [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="same batch size"):
        sglang_utils._normalize_sglang_inputs(None, [[1, 2], [3, 4]], [[1, 1]])
    with pytest.raises(ValueError, match="same length"):
        sglang_utils._normalize_sglang_inputs(None, [1, 2], [1])
    with pytest.raises(ValueError, match="every token"):
        sglang_utils._normalize_sglang_inputs(None, [1, 2], [0, 0])
    with pytest.raises(TypeError, match="0/1"):
        sglang_utils._normalize_sglang_inputs(None, [1, 2], [1, 2])
    with pytest.raises(ValueError, match="left or right padding"):
        sglang_utils._normalize_sglang_inputs(None, [1, 2, 3], [1, 0, 1])


def test_prompt_input_and_prompt_mask_conflicts():
    with pytest.raises(ValueError, match="only one"):
        sglang_utils._normalize_sglang_inputs("hello", [1, 2], None)
    with pytest.raises(ValueError, match="only supported"):
        sglang_utils._normalize_sglang_inputs("hello", None, [1])
    with pytest.raises(ValueError, match="only supported"):
        sglang_utils._normalize_sglang_inputs(["hello", "world"], None, [[1], [1]])


def test_sampling_params_merge_stop_ids_and_force_determinism():
    params = sglang_utils._build_sglang_sampling_params(
        {"stop_token_ids": (2, 3, 2), "temperature": 0.8},
        {
            "stop_token_ids": torch.tensor([3, 4]),
            "eos_token_id": [4, 5],
            "do_sample": False,
            "max_new_tokens": 12,
        },
    )
    assert params["stop_token_ids"] == [2, 3, 4, 5]
    assert params["temperature"] == 0.0
    assert params["max_new_tokens"] == 12
    assert (
        sglang_utils._build_sglang_sampling_params({"max_tokens": 4}, {})[
            "max_new_tokens"
        ]
        == 4
    )
    assert (
        sglang_utils._build_sglang_sampling_params({"min_tokens": 2}, {})[
            "min_new_tokens"
        ]
        == 2
    )


def test_sampling_params_reject_bad_aliases_and_lengths():
    with pytest.raises(ValueError, match="max_new_tokens"):
        sglang_utils._build_sglang_sampling_params(
            None, {"max_new_tokens": 1, "max_tokens": 2}
        )
    with pytest.raises(ValueError, match="max_length"):
        sglang_utils._build_sglang_sampling_params(None, {"max_length": 1})
    with pytest.raises(TypeError, match="eos_token_id"):
        sglang_utils._build_sglang_sampling_params(None, {"eos_token_id": [1, "2"]})
    with pytest.raises(TypeError, match="stop_token_ids"):
        sglang_utils._build_sglang_sampling_params(None, {"stop_token_ids": [1, "2"]})


def test_engine_generation_passes_dict_sampling_and_parses_single_batch_results(
    fake_sglang,
):
    runtime = _FakeEngine()
    setattr(runtime, sglang_utils._ENGINE_MARKER, True)
    runtime.result = [{"text": "one"}, {"text": "two"}]
    output = sglang_utils.sglang_generate(
        runtime,
        prompts=["a", "b"],
        do_sample=False,
        temperature=0.9,
        max_new_tokens=3,
    )
    assert output == ["one", "two"]
    call = runtime.calls[0]
    assert call["prompt"] == ["a", "b"]
    assert call["input_ids"] is None
    assert call["sampling_params"] == {"temperature": 0.0, "max_new_tokens": 3}


def test_engine_generation_input_ids_and_result_validation(fake_sglang):
    runtime = _FakeEngine()
    setattr(runtime, sglang_utils._ENGINE_MARKER, True)
    runtime.result = {"text": "ok"}
    assert (
        sglang_utils.sglang_generate(
            runtime, input_ids=torch.tensor([0, 1, 2]), attention_mask=[0, 1, 1]
        )
        == "ok"
    )
    assert runtime.calls[-1]["input_ids"] == [1, 2]
    with pytest.raises(RuntimeError, match="missing.*text"):
        runtime.result = {"value": "bad"}
        sglang_utils.sglang_generate(runtime, prompts="x")
    with pytest.raises(TypeError, match="must be a string"):
        runtime.result = {"text": None}
        sglang_utils.sglang_generate(runtime, prompts="x")
    with pytest.raises(TypeError, match="Unexpected"):
        runtime.result = "bad"
        sglang_utils.sglang_generate(runtime, prompts="x")


def test_legacy_generation_is_flattened_and_supports_batch_text(monkeypatch):
    class FakeSgl:
        @staticmethod
        def gen(name, *, temperature=None, max_tokens=None, stop_token_ids=None):
            return None

    calls = []

    class FakeProgramState:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            if key != "result":
                raise KeyError(key)
            return self.value

    class FakeGenerate:
        @staticmethod
        def run(**kwargs):
            calls.append(kwargs)
            return FakeProgramState(kwargs["prompt"].upper())

    monkeypatch.setattr(sglang_utils, "sgl", FakeSgl)
    monkeypatch.setattr(sglang_utils, "_legacy_generate", FakeGenerate)
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)

    assert sglang_utils.sglang_generate(
        object(), prompts=["a", "b"], temperature=0.5, max_new_tokens=3
    ) == ["A", "B"]
    assert calls == [
        {"prompt": "a", "temperature": 0.5, "max_tokens": 3},
        {"prompt": "b", "temperature": 0.5, "max_tokens": 3},
    ]


def test_legacy_generation_rejects_input_ids_and_unsupported_or_bad_state(monkeypatch):
    class FakeSgl:
        @staticmethod
        def gen(name, *, temperature=None):
            return None

    class FakeGenerate:
        @staticmethod
        def run(**_kwargs):
            return {"wrong": "field"}

    monkeypatch.setattr(sglang_utils, "sgl", FakeSgl)
    monkeypatch.setattr(sglang_utils, "_legacy_generate", FakeGenerate)
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    with pytest.raises(ValueError, match="input_ids"):
        sglang_utils.sglang_generate(object(), input_ids=[1, 2])
    with pytest.raises(ValueError, match="does not support"):
        sglang_utils.sglang_generate(object(), prompts="x", top_p=0.8)
    with pytest.raises(ValueError, match="request parameters: stream"):
        sglang_utils.sglang_generate(object(), prompts="x", stream=True)
    with pytest.raises(RuntimeError, match="missing.*result"):
        sglang_utils.sglang_generate(object(), prompts="x")


def test_sglang_unavailable_messages_distinguish_missing_and_import_error(monkeypatch):
    monkeypatch.setattr(sglang_utils, "SGLANG_VERSION", None)
    monkeypatch.setattr(
        sglang_utils,
        "SGLANG_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'sglang'"),
    )
    assert "not installed" in sglang_utils._sglang_unavailable_message()

    monkeypatch.setattr(sglang_utils, "SGLANG_VERSION", "1.0")
    monkeypatch.setattr(
        sglang_utils, "SGLANG_IMPORT_ERROR", RuntimeError("broken dependency")
    )
    message = sglang_utils._sglang_unavailable_message()
    assert "failed to import" in message
    assert "broken dependency" in message

    monkeypatch.setattr(sglang_utils, "SGLANG_VERSION", None)
    monkeypatch.setattr(
        sglang_utils,
        "SGLANG_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'flashinfer'"),
    )
    assert "failed to import" in sglang_utils._sglang_unavailable_message()


def test_require_sglang_reports_install_hint(monkeypatch):
    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", False)
    monkeypatch.setattr(
        sglang_utils,
        "SGLANG_INSTALL_HINT",
        "SGLang is not installed; install the sglang extra",
    )
    with pytest.raises(ValueError, match="not installed"):
        sglang_utils._require_sglang()
