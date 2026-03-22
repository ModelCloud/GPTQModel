from types import SimpleNamespace

import torch

from gptqmodel.models import GPTQModel, auto, loader
from gptqmodel.quantization import QuantizeConfig


def test_load_treats_missing_absolute_path_as_local(monkeypatch):
    model_path = "/monster/data/model/Qwen3.5-35B-A3B"
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()

    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "list_repo_files", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected remote lookup")))

    def fake_from_pretrained(cls, model_id_or_path, quantize_config, **kwargs):
        assert model_id_or_path == model_path
        assert quantize_config is None
        return sentinel

    def fake_from_quantized(cls, *args, **kwargs):
        raise AssertionError("unexpected quantized load")

    monkeypatch.setattr(GPTQModel, "from_pretrained", classmethod(fake_from_pretrained))
    monkeypatch.setattr(GPTQModel, "from_quantized", classmethod(fake_from_quantized))

    result = GPTQModel.load(model_path)

    assert result is sentinel


def test_get_model_local_path_skips_snapshot_download_for_absolute_path(monkeypatch):
    model_path = "/monster/data/model/Qwen3.5-35B-A3B"

    monkeypatch.setattr(
        loader,
        "snapshot_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected snapshot download")),
    )

    assert loader.get_model_local_path(model_path) == model_path


def test_model_loader_isolates_shell_config_from_turtle_load(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "llama"
            self.sub_configs = {}
            self.dtype = None

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.seqlen = None

        def eval(self):
            return self

    base_config = FakeConfig()
    shell_configs = []
    turtle_configs = []

    def fake_build_shell_model(_loader, config, **_kwargs):
        shell_configs.append(config)
        return FakeModel(config)

    def fake_convert_model(model, cleanup_original=False):
        assert cleanup_original is False
        model.config._experts_implementation = "linear_loop"
        return model

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(_path, config=None, **_kwargs):
            turtle_configs.append(config)
            assert getattr(config, "_experts_implementation", None) is None
            return FakeModel(config)

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = True
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.turtle_model = kwargs.get("turtle_model")
            self.config = model.config

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.float16)
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: base_config)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("gptqmodel.utils.hf.build_shell_model", fake_build_shell_model)
    monkeypatch.setattr(loader.defuser, "convert_model", fake_convert_model)

    instance = DummyQModel.from_pretrained(
        "/tmp/fake-model",
        quantize_config=QuantizeConfig(offload_to_disk=True),
        trust_remote_code=False,
    )

    assert shell_configs
    assert turtle_configs
    assert shell_configs[0] is not base_config
    assert turtle_configs[0] is base_config
    assert shell_configs[0]._experts_implementation == "linear_loop"
    assert turtle_configs[0]._experts_implementation is None
    assert instance.model.config._experts_implementation == "linear_loop"
    assert instance.turtle_model.config._experts_implementation == "linear_loop"
    assert instance.turtle_model.config is not base_config
    assert instance.turtle_model is not None


def test_model_loader_falls_back_when_meta_shell_build_hits_meta_tensor_item(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "llama"
            self.sub_configs = {}
            self.dtype = None

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.seqlen = None

        def eval(self):
            return self

    base_config = FakeConfig()
    load_calls = []

    def fake_build_shell_model(_loader, config, **_kwargs):
        raise RuntimeError("Tensor.item() cannot be called on meta tensors")

    def fake_convert_model(model, cleanup_original=False):
        assert cleanup_original is False
        model.config._experts_implementation = "linear_loop"
        return model

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(_path, config=None, **kwargs):
            load_calls.append(kwargs)
            return FakeModel(config)

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = True
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.turtle_model = kwargs.get("turtle_model")
            self.quantize_config = kwargs.get("quantize_config")
            self.config = model.config

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.float16)
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: base_config)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("gptqmodel.utils.hf.build_shell_model", fake_build_shell_model)
    monkeypatch.setattr(loader.defuser, "convert_model", fake_convert_model)

    instance = DummyQModel.from_pretrained(
        "/tmp/fake-model",
        quantize_config=QuantizeConfig(offload_to_disk=True),
        trust_remote_code=False,
    )

    assert len(load_calls) == 1
    assert "device_map" not in load_calls[0]
    assert load_calls[0]["low_cpu_mem_usage"] is False
    assert instance.turtle_model is None
    assert instance.quantize_config.offload_to_disk is True
    assert instance.model.config._experts_implementation == "linear_loop"
    assert instance.model.config is not base_config
