# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import importlib.util  # noqa: E402
import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import gptqmodel.models.loader as loader_module  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.models.loader import ModelLoader, _validate_sglang_quantization  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.quantization.config import resolve_quant_format  # noqa: E402
from gptqmodel.utils import sglang as sglang_utils  # noqa: E402
from gptqmodel.utils.sglang import SGLANG_AVAILABLE, SGLANG_INSTALL_HINT  # noqa: E402


_PROMPT = "The capital city of France is named"
_GENERATION_KWARGS = {
    "ignore_eos": True,
    "max_new_tokens": 8,
    "temperature": 0.0,
    "top_k": 1,
}
_SGLANG_RUNTIME_KWARGS = {
    "context_length": 256,
    "disable_cuda_graph": True,
    "launch_timeout": 900.0,
    "max_total_tokens": 512,
    "mem_fraction_static": 0.1,
}
_INTEGRATION_CASES = (
    pytest.param(
        METHOD.GPTQ,
        FORMAT.GPTQ,
        "GPTQMODEL_SGLANG_GPTQ_MODEL",
        "/monster/data/model/Qwen2.5-0.5B-Instruct-gptq-4bit",
        id="gptq",
    ),
    pytest.param(
        METHOD.AWQ,
        FORMAT.GEMM,
        "GPTQMODEL_SGLANG_AWQ_MODEL",
        "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-bit4-g128-symFasle-descFalse",
        id="awq-gemm",
    ),
)


@pytest.mark.parametrize(
    ("method", "format_code"),
    (
        pytest.param(METHOD.GPTQ, FORMAT.GPTQ, id="gptq"),
        pytest.param(METHOD.GPTQ, FORMAT.GPTQ_V2, id="gptq-v2"),
        pytest.param(METHOD.GPTQ, FORMAT.MARLIN, id="gptq-marlin"),
        pytest.param(METHOD.AWQ, FORMAT.GEMM, id="awq-gemm"),
        pytest.param(METHOD.AWQ, FORMAT.MARLIN, id="awq-marlin"),
    ),
)
def test_sglang_quantization_contract_accepts_verified_formats(method, format_code):
    _validate_sglang_quantization(method, format_code)


@pytest.mark.parametrize(
    ("method", "format_code"),
    (
        pytest.param(METHOD.GPTQ, FORMAT.GEMM, id="gptq-gemm"),
        pytest.param(METHOD.AWQ, FORMAT.GPTQ, id="awq-gptq"),
        pytest.param(METHOD.AWQ, FORMAT.GPTQ_V2, id="awq-gptq-v2"),
        pytest.param(METHOD.AWQ, FORMAT.GEMV, id="awq-gemv"),
        pytest.param(METHOD.AWQ, FORMAT.GEMV_FAST, id="awq-gemv-fast"),
        pytest.param(METHOD.AWQ, FORMAT.LLM_AWQ, id="llm-awq"),
    ),
)
def test_sglang_quantization_contract_rejects_unverified_formats(method, format_code):
    with pytest.raises(
        ValueError,
        match="SGLANG backend only supports",
    ):
        _validate_sglang_quantization(method, format_code)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    (
        pytest.param(None, None, id="none"),
        pytest.param("auto", "auto", id="auto"),
        pytest.param("half", "half", id="half"),
        pytest.param("bf16", "bf16", id="bf16"),
        pytest.param("torch.float16", "float16", id="torch-string-float16"),
        pytest.param(torch.float16, "float16", id="torch-float16"),
        pytest.param(torch.bfloat16, "bfloat16", id="torch-bfloat16"),
        pytest.param(torch.float32, "float32", id="torch-float32"),
    ),
)
def test_normalize_sglang_dtype(dtype, expected):
    assert sglang_utils._normalize_dtype(dtype) == expected


def test_load_model_by_sglang_is_repeatable_without_changing_process_start_method(monkeypatch):
    runtimes = []
    default_backends = []
    config_calls = []

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_calls = 0
            runtimes.append(self)

        def shutdown(self):
            self.shutdown_calls += 1

    fake_sglang = SimpleNamespace(
        Runtime=FakeRuntime,
        set_default_backend=default_backends.append,
    )

    def fail_if_start_method_is_changed(*args, **kwargs):
        raise AssertionError(f"multiprocessing start method changed: args={args}, kwargs={kwargs}")

    def fake_config_load(model, **kwargs):
        config_calls.append((model, kwargs))
        return SimpleNamespace(model_type="llama")

    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(sglang_utils, "sgl", fake_sglang)
    monkeypatch.setattr(sglang_utils.AutoConfig, "from_pretrained", fake_config_load)
    monkeypatch.setattr(multiprocessing, "set_start_method", fail_if_start_method_is_changed)

    first, _ = sglang_utils.load_model_by_sglang(
        "first-model",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    first.shutdown()
    second, _ = sglang_utils.load_model_by_sglang(
        "second-model",
        trust_remote_code=False,
        dtype="auto",
    )
    second.shutdown()

    assert first is runtimes[0]
    assert second is runtimes[1]
    assert default_backends == runtimes
    assert first.kwargs == {
        "model_path": "first-model",
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    assert second.kwargs == {
        "model_path": "second-model",
        "trust_remote_code": False,
        "dtype": "auto",
    }
    assert config_calls == [
        ("first-model", {"trust_remote_code": True}),
        ("second-model", {"trust_remote_code": False}),
    ]
    assert [runtime.shutdown_calls for runtime in runtimes] == [1, 1]


def test_load_model_by_sglang_loads_config_before_starting_runtime(monkeypatch):
    runtime_calls = []
    runtime = SimpleNamespace(shutdown_calls=0)

    def shutdown():
        runtime.shutdown_calls += 1

    runtime.shutdown = shutdown

    def fake_runtime(**kwargs):
        runtime_calls.append(kwargs)
        return runtime

    fake_sglang = SimpleNamespace(Runtime=fake_runtime, set_default_backend=lambda backend: None)

    def fail_config_load(*args, **kwargs):
        raise RuntimeError("broken config")

    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(sglang_utils, "sgl", fake_sglang)
    monkeypatch.setattr(sglang_utils.AutoConfig, "from_pretrained", fail_config_load)

    with pytest.raises(RuntimeError, match="broken config"):
        sglang_utils.load_model_by_sglang("model", trust_remote_code=False)

    assert runtime_calls == []
    assert runtime.shutdown_calls == 0


def test_sglang_generate_uses_loaded_runtime_and_drops_hf_pad_token(monkeypatch):
    captured = {}

    class FakeEngine:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"text": "Paris"}

    runtime = FakeEngine()
    setattr(runtime, sglang_utils._ENGINE_MARKER, True)

    monkeypatch.setattr(sglang_utils, "SGLANG_AVAILABLE", True)

    output = sglang_utils.sglang_generate(
        runtime,
        prompts=_PROMPT,
        max_new_tokens=8,
        pad_token_id=0,
        temperature=0.0,
    )

    assert output == "Paris"
    assert captured == {
        "prompt": _PROMPT,
        "input_ids": None,
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0.0,
        },
    }


@pytest.mark.parametrize(
    ("method", "format_code", "dtype", "expected_runtime_dtype"),
    (
        pytest.param(METHOD.GPTQ, FORMAT.GPTQ, "auto", torch.float32, id="gptq-auto"),
        pytest.param(METHOD.GPTQ, FORMAT.GPTQ, torch.bfloat16, torch.bfloat16, id="gptq-bfloat16"),
        pytest.param(METHOD.AWQ, FORMAT.GEMM, "auto", torch.float32, id="awq-auto"),
        pytest.param(METHOD.AWQ, FORMAT.GEMM, torch.float16, torch.float16, id="awq-float16"),
    ),
)
def test_quantized_loader_passes_requested_dtype_to_sglang(
    monkeypatch,
    method,
    format_code,
    dtype,
    expected_runtime_dtype,
):
    captured = {}
    config = SimpleNamespace(model_type="llama", dtype=torch.float16)
    runtime = SimpleNamespace(config=None)

    class FakeQuantizeConfig:
        def __init__(self):
            self.adapter = None
            self.device = None
            self.format = format_code
            self.method = method
            self.offload_to_disk = True
            self.quant_method = method

        def calculate_bits_per_weight(self):
            captured["calculated_bits_per_weight"] = True

        def export_quant_method(self):
            return method

    qcfg = FakeQuantizeConfig()

    class FakeModel:
        require_dtype = None
        require_pkgs = ()
        require_trust_remote_code = False

        def __init__(self, model, **kwargs):
            self.model = model
            self.init_kwargs = kwargs

    def fake_sglang_load(*, model, trust_remote_code, dtype, **kwargs):
        captured.update(
            model=model,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            runtime_kwargs=kwargs,
        )
        return runtime, config

    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    monkeypatch.setattr(loader_module, "normalize_model_id_or_path_for_hf_gguf", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(loader_module, "normalize_device_device_map", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader_module, "auto_select_device", lambda *args, **kwargs: loader_module.DEVICE.CUDA)
    monkeypatch.setattr(loader_module, "get_model_local_path", lambda value, **kwargs: value)
    monkeypatch.setattr(loader_module, "resolve_trust_remote_code", lambda path, trust_remote_code: trust_remote_code)
    monkeypatch.setattr(loader_module, "has_native_transformers_causallm_support", lambda path: True)
    monkeypatch.setattr(loader_module, "check_versions", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(loader_module.defuser, "replace_fused_blocks", lambda model_type: None)
    monkeypatch.setattr(loader_module, "normalize_hf_config_compat", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader_module, "prepare_remote_model_init_compat", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader_module, "auto_dtype", lambda **kwargs: torch.float32)
    monkeypatch.setattr(
        loader_module,
        "_resolve_native_quantized_gguf_checkpoint",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(loader_module.QuantizeConfig, "from_pretrained", lambda *args, **kwargs: qcfg)
    monkeypatch.setattr(loader_module, "load_hf_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_utils, "load_model_by_sglang", fake_sglang_load)
    monkeypatch.setattr(sglang_utils, "sglang_generate", lambda *args, **kwargs: "generated")

    loaded_cls = ModelLoader(FakeModel)
    loaded = loaded_cls.from_quantized(
        "quantized-model",
        backend=BACKEND.SGLANG,
        dtype=dtype,
    )

    assert loaded.model is runtime
    assert loaded._runtime_generate is sglang_utils.sglang_generate
    assert captured == {
        "calculated_bits_per_weight": True,
        "model": "quantized-model",
        "trust_remote_code": False,
        "dtype": expected_runtime_dtype,
        "runtime_kwargs": {"device": "cuda"},
    }


def _require_sglang_integration(model_path):
    if importlib.util.find_spec("sglang") is None or not SGLANG_AVAILABLE:
        pytest.skip(SGLANG_INSTALL_HINT)
    if not torch.cuda.is_available():
        pytest.skip("SGLang integration test requires CUDA")
    if not model_path.is_dir():
        pytest.skip(f"missing local SGLang integration model: {model_path}")


def _direct_sglang_generate(model_path):
    runtime_kwargs = sglang_utils._normalize_sglang_engine_kwargs(
        {"dtype": "auto", **_SGLANG_RUNTIME_KWARGS},
        trust_remote_code=False,
    )
    engine_factory = getattr(sglang_utils.sgl, "Engine", None)
    if engine_factory is not None:
        runtime = engine_factory(model_path=str(model_path), **runtime_kwargs)
    else:
        runtime = sglang_utils.sgl.Runtime(model_path=str(model_path), **runtime_kwargs)
        sglang_utils.sgl.set_default_backend(runtime)

    sampling_params = sglang_utils._build_sglang_sampling_params(None, _GENERATION_KWARGS)
    try:
        if engine_factory is not None:
            result = runtime.generate(prompt=_PROMPT, sampling_params=sampling_params)
            return sglang_utils._extract_sglang_text(result)

        state = sglang_utils._legacy_generate.run(
            prompt=_PROMPT,
            **sglang_utils._legacy_sglang_sampling_params(sampling_params),
        )
        return state["result"]
    finally:
        runtime.shutdown()


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "format_code", "model_env", "default_model_path"),
    _INTEGRATION_CASES,
)
def test_sglang_checkpoint_matches_direct_runtime_across_restarts(
    method,
    format_code,
    model_env,
    default_model_path,
):
    model_path = Path(os.environ.get(model_env, default_model_path))
    _require_sglang_integration(model_path)

    direct_output = _direct_sglang_generate(model_path)
    wrapped_outputs = []

    for _ in range(2):
        model = GPTQModel.load(
            str(model_path),
            backend=BACKEND.SGLANG,
            dtype="auto",
            **_SGLANG_RUNTIME_KWARGS,
        )
        runtime = model.model
        try:
            assert model.quantize_config.export_quant_method() == method
            assert resolve_quant_format(model.quantize_config.format, model.quantize_config.method) == format_code
            wrapped_outputs.append(
                model.generate(
                    prompts=_PROMPT,
                    **_GENERATION_KWARGS,
                )
            )
        finally:
            model.shutdown()

        if hasattr(runtime, "pid"):
            assert runtime.pid is None

    assert direct_output
    assert wrapped_outputs == [direct_output, direct_output]
