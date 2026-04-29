# GPU=-1
from types import SimpleNamespace

from gptqmodel.adapter.adapter import Lora
from gptqmodel.models import auto


class _FakeNativeModel:
    def __init__(self):
        self.generate_calls = []

    def _eora_generate(self, **kwargs):
        self.generate_calls.append(kwargs)


def _run_adapter_generate(tmp_path, monkeypatch, *, device):
    load_calls = []
    find_modules_calls = []

    quantized_model = SimpleNamespace(quantize_config="qcfg", model="quantized-model")
    native_model = _FakeNativeModel()

    def fake_load(cls, model_id_or_path, *args, **kwargs):
        load_calls.append((model_id_or_path, kwargs.copy()))
        if model_id_or_path == "quantized":
            return quantized_model
        if model_id_or_path == "native":
            return native_model
        raise AssertionError(f"unexpected load target: {model_id_or_path}")

    monkeypatch.setattr(auto.GPTQModel, "load", classmethod(fake_load))
    monkeypatch.setattr(
        auto,
        "find_modules",
        lambda module, layers: find_modules_calls.append((module, layers)) or {"module": object()},
    )
    monkeypatch.setattr(auto, "torch_empty_cache", lambda: None)

    adapter = Lora(path=str(tmp_path / "adapter"), rank=8)
    kwargs = {
        "adapter": adapter,
        "model_id_or_path": "native",
        "quantized_model_id_or_path": "quantized",
        "calibration_dataset": ["sample"],
    }
    if device is not None:
        kwargs["device"] = device

    auto.GPTQModel.adapter.generate(**kwargs)

    return load_calls, find_modules_calls, native_model.generate_calls


def test_adapter_generate_defaults_to_loader_device_selection(tmp_path, monkeypatch):
    load_calls, find_modules_calls, generate_calls = _run_adapter_generate(
        tmp_path,
        monkeypatch,
        device=None,
    )

    assert [kwargs["device"] for _, kwargs in load_calls] == [None, None]
    assert find_modules_calls == [("quantized-model", [auto.TorchLinear])]
    assert generate_calls[0]["quantized_modules"].keys() == {"module"}


def test_adapter_generate_forwards_explicit_device(tmp_path, monkeypatch):
    load_calls, _, _ = _run_adapter_generate(
        tmp_path,
        monkeypatch,
        device="cuda:2",
    )

    assert [kwargs["device"] for _, kwargs in load_calls] == ["cuda:2", "cuda:2"]
