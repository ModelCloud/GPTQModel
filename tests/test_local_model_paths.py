import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import GenerationConfig

from gptqmodel.models import GPTQModel, auto, loader
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils import BACKEND, PROFILE
from gptqmodel.utils import model as model_utils
from gptqmodel.utils.hf import INTERNAL_HF_GGUF_FILE_KWARG
from gptqmodel.utils.structure import LazyTurtle


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


def test_copy_py_files_uses_remote_metadata_downloads(monkeypatch, tmp_path):
    downloads = []

    monkeypatch.setattr(
        model_utils,
        "model_info",
        lambda *args, **kwargs: SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename="model.py"),
                SimpleNamespace(rfilename="README.md"),
                SimpleNamespace(rfilename="subdir/tokenizer.py"),
            ]
        ),
    )
    monkeypatch.setattr(
        model_utils,
        "hf_hub_download",
        lambda **kwargs: downloads.append(kwargs) or str(tmp_path / kwargs["filename"].replace("/", "_")),
    )

    model_utils.copy_py_files(str(tmp_path), model_id_or_path="org/repo")

    assert downloads == [
        {"repo_id": "org/repo", "filename": "model.py", "local_dir": str(tmp_path)},
        {"repo_id": "org/repo", "filename": "subdir/tokenizer.py", "local_dir": str(tmp_path)},
    ]


def test_get_model_files_size_uses_remote_metadata_sizes(monkeypatch):
    monkeypatch.setattr(
        model_utils,
        "model_info",
        lambda *args, **kwargs: SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename="model.safetensors", size=10),
                SimpleNamespace(rfilename="weights.bin", size=20),
                SimpleNamespace(rfilename="README.md", size=99),
                SimpleNamespace(rfilename="adapter.pt", size=None),
            ]
        ),
    )

    result = model_utils.get_model_files_size("org/repo")

    assert result == (30 / (1024 * 1024))


def test_gptqmodel_from_pretrained_forwards_dtype_kwarg(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    captured = {}

    class FakeModelDefinition:
        @classmethod
        def from_pretrained(cls, pretrained_model_id_or_path, quantize_config, **kwargs):
            captured["path"] = pretrained_model_id_or_path
            captured["quantize_config"] = quantize_config
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: FakeModelDefinition)

    result = GPTQModel.from_pretrained(
        "/tmp/fake-model",
        quantize_config=None,
        dtype=torch.float16,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["quantize_config"] is None
    assert captured["kwargs"]["dtype"] is torch.float16
    assert "torch_dtype" not in captured["kwargs"]


def test_gptqmodel_from_pretrained_forwards_backend_and_profile(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    captured = {}

    class FakeModelDefinition:
        @classmethod
        def from_pretrained(cls, pretrained_model_id_or_path, quantize_config, **kwargs):
            captured["path"] = pretrained_model_id_or_path
            captured["quantize_config"] = quantize_config
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: FakeModelDefinition)

    result = GPTQModel.from_pretrained(
        "/tmp/fake-model",
        quantize_config=None,
        backend="gguf_torch",
        profile=2,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["quantize_config"] is None
    assert captured["kwargs"]["backend"] == BACKEND.GGUF_TORCH
    assert captured["kwargs"]["profile"] == PROFILE.LOW_MEMORY


def test_gptqmodel_load_rejects_public_gguf_file_kwarg():
    with pytest.raises(TypeError, match="does not accept `gguf_file`"):
        GPTQModel.load("/tmp/fake-model", gguf_file="bonsai.gguf")


def test_gptqmodel_load_normalizes_direct_gguf_path_for_config_probe(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    config_calls = []
    captured = {}

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(auto, "isdir", lambda path: path == "/tmp/fake-model")
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: config_calls.append(kwargs) or fake_config,
    )
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(GPTQModel, "from_pretrained", classmethod(
        lambda cls, model_id_or_path, quantize_config, **kwargs: captured.update(
            {"path": model_id_or_path, "kwargs": kwargs}
        ) or sentinel
    ))
    monkeypatch.setattr(
        GPTQModel,
        "from_quantized",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected quantized load"))),
    )

    result = GPTQModel.load("/tmp/fake-model/bonsai.gguf")

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["kwargs"][INTERNAL_HF_GGUF_FILE_KWARG] == "bonsai.gguf"
    assert "gguf_file" not in captured["kwargs"]
    assert config_calls[0]["gguf_file"] == "bonsai.gguf"


def test_gptqmodel_load_forwards_backend_and_profile_to_from_pretrained(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    captured = {}

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "list_repo_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        GPTQModel,
        "from_pretrained",
        classmethod(
            lambda cls, model_id_or_path, quantize_config, **kwargs: captured.update(
                path=model_id_or_path,
                quantize_config=quantize_config,
                kwargs=kwargs,
            )
            or sentinel
        ),
    )
    monkeypatch.setattr(
        GPTQModel,
        "from_quantized",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected quantized load"))),
    )

    result = GPTQModel.load(
        "/tmp/fake-model",
        backend="gguf_torch",
        profile=PROFILE.LOW_MEMORY,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["quantize_config"] is None
    assert captured["kwargs"]["backend"] == BACKEND.GGUF_TORCH
    assert captured["kwargs"]["profile"] == PROFILE.LOW_MEMORY


def test_gptqmodel_load_forwards_profile_without_explicit_backend(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    captured = {}

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "list_repo_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        GPTQModel,
        "from_pretrained",
        classmethod(
            lambda cls, model_id_or_path, quantize_config, **kwargs: captured.update(
                path=model_id_or_path,
                quantize_config=quantize_config,
                kwargs=kwargs,
            )
            or sentinel
        ),
    )
    monkeypatch.setattr(
        GPTQModel,
        "from_quantized",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected quantized load"))),
    )

    result = GPTQModel.load(
        "/tmp/fake-model",
        profile=PROFILE.LOW_MEMORY,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["quantize_config"] is None
    assert captured["kwargs"]["backend"] == BACKEND.AUTO
    assert captured["kwargs"]["profile"] == PROFILE.LOW_MEMORY


def test_gptqmodel_from_pretrained_normalizes_direct_gguf_path(monkeypatch):
    fake_config = SimpleNamespace(quantization_config=None)
    sentinel = object()
    captured = {}
    config_calls = []

    class FakeModelDefinition:
        @classmethod
        def from_pretrained(cls, pretrained_model_id_or_path, quantize_config, **kwargs):
            captured["path"] = pretrained_model_id_or_path
            captured["kwargs"] = kwargs
            return sentinel

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: config_calls.append(kwargs) or fake_config,
    )
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda config: False)
    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: FakeModelDefinition)

    result = GPTQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert config_calls[0]["gguf_file"] == "bonsai.gguf"
    assert captured["kwargs"][INTERNAL_HF_GGUF_FILE_KWARG] == "bonsai.gguf"
    assert "gguf_file" not in captured["kwargs"]


def test_gptqmodel_from_quantized_forwards_dtype_kwarg(monkeypatch):
    sentinel = object()
    captured = {}

    class FakeModelDefinition:
        @classmethod
        def from_quantized(cls, model_id_or_path, **kwargs):
            captured["path"] = model_id_or_path
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "normalize_adapter", lambda adapter: adapter)
    monkeypatch.setattr(auto, "normalize_backend", lambda backend: backend)
    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: FakeModelDefinition)

    result = GPTQModel.from_quantized(
        "/tmp/fake-model",
        dtype=torch.bfloat16,
    )

    assert result is sentinel
    assert captured["path"] == "/tmp/fake-model"
    assert captured["kwargs"]["dtype"] is torch.bfloat16
    assert "torch_dtype" not in captured["kwargs"]


def test_model_loader_requires_lazy_turtle_for_offload_to_disk(monkeypatch):
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
            raise AssertionError("legacy eager turtle load should not run")

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = True
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        @staticmethod
        def resolve_hf_conversion_map_reversed(*_args, **_kwargs):
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
    monkeypatch.setattr(loader.LazyTurtle, "maybe_create", classmethod(lambda cls, **_kwargs: None))

    with pytest.raises(RuntimeError, match="can't open model path"):
        DummyQModel.from_pretrained(
            "/tmp/fake-model",
            quantize_config=QuantizeConfig(offload_to_disk=True),
            trust_remote_code=False,
        )

    assert shell_configs
    assert shell_configs[0] is not base_config
    assert shell_configs[0]._experts_implementation == "linear_loop"


def test_model_loader_uses_lazy_turtle_for_local_safetensors(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    tensors = {
        "model.layers.0.linear.weight": torch.arange(16, dtype=torch.float16).view(4, 4),
    }
    save_file(tensors, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard_name)})
    )

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
    load_calls = []

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
            load_calls.append(config)
            return FakeModel(config)

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = True
        HF_CONVERSION_MAP_REVERSED = (
            SimpleNamespace(source_patterns=["shell_model"], target_patterns=["model"]),
        )
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        @classmethod
        def resolve_hf_conversion_map_reversed(cls, *_args, **_kwargs):
            return copy.deepcopy(cls.HF_CONVERSION_MAP_REVERSED)

        def __init__(self, model, **kwargs):
            self.model = model
            self.turtle_model = kwargs.get("turtle_model")
            self.config = model.config

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: str(model_dir))
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.float16)
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: base_config)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("gptqmodel.utils.hf.build_shell_model", fake_build_shell_model)
    monkeypatch.setattr(loader.defuser, "convert_model", fake_convert_model)

    instance = DummyQModel.from_pretrained(
        str(model_dir),
        quantize_config=QuantizeConfig(offload_to_disk=True),
        trust_remote_code=False,
    )

    assert shell_configs
    assert load_calls == []
    assert isinstance(instance.turtle_model, LazyTurtle)
    assert [
        (entry.source_patterns[0], entry.target_patterns[0])
        for entry in instance.turtle_model._runtime_to_checkpoint_renamings
    ] == [
        (entry.source_patterns[0], entry.target_patterns[0])
        for entry in DummyQModel.HF_CONVERSION_MAP_REVERSED
    ]
    assert instance.turtle_model.config._experts_implementation == "linear_loop"
    assert instance.turtle_model.config is not instance.model.config


def test_model_loader_from_pretrained_forwards_dtype_kwarg(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "llama"
            self.sub_configs = {}
            self.dtype = None

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def eval(self):
            return self

    load_calls = []

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(_path, config=None, **kwargs):
            load_calls.append(kwargs)
            return FakeModel(config)

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.config = model.config
            self.quantized = kwargs.get("quantized")

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: FakeConfig())
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)

    instance = DummyQModel.from_pretrained(
        "/tmp/fake-model",
        quantize_config=None,
        dtype=torch.float16,
    )

    assert instance.quantized is False
    assert len(load_calls) == 1
    assert load_calls[0]["dtype"] is torch.float16
    assert "torch_dtype" not in load_calls[0]


def test_model_loader_from_pretrained_normalizes_direct_gguf_path(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "llama"
            self.sub_configs = {}
            self.dtype = None

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def eval(self):
            return self

    config_calls = []
    model_calls = []
    tokenizer_calls = []

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(path, config=None, **kwargs):
            model_calls.append({"path": path, "kwargs": kwargs})
            return FakeModel(config)

    @loader.ModelLoader
    class DummyQModel:
        loader = FakeInnerLoader
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.config = model.config
            self.quantized = kwargs.get("quantized")

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.float16)
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **kwargs: config_calls.append(kwargs) or FakeConfig(),
    )
    monkeypatch.setattr(
        loader.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **kwargs: tokenizer_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)

    DummyQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
    )

    assert config_calls[0]["gguf_file"] == "bonsai.gguf"
    assert tokenizer_calls[0]["gguf_file"] == "bonsai.gguf"
    assert model_calls[0]["path"] == "/tmp/fake-model"
    assert model_calls[0]["kwargs"]["gguf_file"] == "bonsai.gguf"


@pytest.mark.parametrize("profile", [PROFILE.AUTO, PROFILE.FAST])
def test_model_loader_from_pretrained_native_bonsai_fast_profiles_use_dense_path(monkeypatch, profile):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "qwen3"
            self.sub_configs = {}
            self.dtype = None

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def eval(self):
            return self

    config_calls = []
    tokenizer_calls = []
    model_calls = []
    auto_dtype_calls = []
    native_spec = loader.internal_gguf.GGUFQuantizedCheckpointSpec(
        model_type="qwen3",
        bits_alias="q1_0_g128",
        tensor_qtype=loader.internal_gguf.GGMLQuantizationType.Q1_0_g128,
        lm_head_quantized=True,
    )

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    @loader.ModelLoader
    class DummyQModel:
        loader = None
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.config = model.config
            self.quantized = kwargs.get("quantized")

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(path, config=None, **kwargs):
            model_calls.append({"path": path, "kwargs": kwargs})
            return FakeModel(config)

    DummyQModel.loader = FakeInnerLoader

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        loader,
        "auto_dtype",
        lambda *args, **kwargs: auto_dtype_calls.append(kwargs) or torch.bfloat16,
    )
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **kwargs: config_calls.append(kwargs) or FakeConfig(),
    )
    monkeypatch.setattr(
        loader.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **kwargs: tokenizer_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loader,
        "_resolve_native_quantized_gguf_checkpoint",
        lambda *_args, **_kwargs: ("/tmp/fake-model/bonsai.gguf", native_spec),
    )
    monkeypatch.setattr(
        DummyQModel,
        "from_quantized",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected low-memory redirect"))),
    )

    result = DummyQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
        profile=profile,
    )

    assert result.quantized is False
    assert config_calls[0]["gguf_file"] == "bonsai.gguf"
    assert tokenizer_calls[0]["gguf_file"] == "bonsai.gguf"
    assert auto_dtype_calls[0]["quant_inference"] is False
    assert model_calls[0]["path"] == "/tmp/fake-model"
    assert model_calls[0]["kwargs"]["gguf_file"] == "bonsai.gguf"
    assert model_calls[0]["kwargs"]["dtype"] is torch.bfloat16
    assert result.config.dtype is torch.bfloat16


def test_model_loader_from_pretrained_native_bonsai_fast_auto_enables_fa2_on_cuda(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "qwen3"
            self.sub_configs = {}
            self.dtype = None
            self.architectures = ["FakeModelForCausalLM"]

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.generation_config = GenerationConfig()

        def eval(self):
            return self

    model_calls = []
    native_spec = loader.internal_gguf.GGUFQuantizedCheckpointSpec(
        model_type="qwen3",
        bits_alias="q1_0_g128",
        tensor_qtype=loader.internal_gguf.GGMLQuantizationType.Q1_0_g128,
        lm_head_quantized=True,
    )

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    @loader.ModelLoader
    class DummyQModel:
        loader = None
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.config = model.config
            self.quantized = kwargs.get("quantized")

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(path, config=None, **kwargs):
            if "attn_implementation" in kwargs:
                config._attn_implementation = kwargs["attn_implementation"]
            model_calls.append({"path": path, "kwargs": kwargs})
            return FakeModel(config)

    DummyQModel.loader = FakeInnerLoader

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cuda"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.bfloat16)
    monkeypatch.setattr(loader, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeConfig(),
    )
    monkeypatch.setattr(
        loader.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loader,
        "_resolve_native_quantized_gguf_checkpoint",
        lambda *_args, **_kwargs: ("/tmp/fake-model/bonsai.gguf", native_spec),
    )
    monkeypatch.setattr(
        loader.transformers,
        "FakeModelForCausalLM",
        type("FakeModelForCausalLM", (), {"_supports_flash_attn_2": True}),
        raising=False,
    )

    DummyQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
        profile=PROFILE.FAST,
    )

    assert model_calls[0]["kwargs"]["attn_implementation"] == "flash_attention_2"


def test_model_loader_from_pretrained_native_bonsai_fast_keeps_explicit_attn_impl(monkeypatch):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "qwen3"
            self.sub_configs = {}
            self.dtype = None
            self.architectures = ["FakeModelForCausalLM"]

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.generation_config = GenerationConfig()

        def eval(self):
            return self

    model_calls = []
    native_spec = loader.internal_gguf.GGUFQuantizedCheckpointSpec(
        model_type="qwen3",
        bits_alias="q1_0_g128",
        tensor_qtype=loader.internal_gguf.GGMLQuantizationType.Q1_0_g128,
        lm_head_quantized=True,
    )

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    @loader.ModelLoader
    class DummyQModel:
        loader = None
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **kwargs):
            self.model = model
            self.config = model.config
            self.quantized = kwargs.get("quantized")

    class FakeInnerLoader:
        @staticmethod
        def from_pretrained(path, config=None, **kwargs):
            if "attn_implementation" in kwargs:
                config._attn_implementation = kwargs["attn_implementation"]
            model_calls.append({"path": path, "kwargs": kwargs})
            return FakeModel(config)

    DummyQModel.loader = FakeInnerLoader

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cuda"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.bfloat16)
    monkeypatch.setattr(loader, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeConfig(),
    )
    monkeypatch.setattr(
        loader.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loader,
        "_resolve_native_quantized_gguf_checkpoint",
        lambda *_args, **_kwargs: ("/tmp/fake-model/bonsai.gguf", native_spec),
    )
    monkeypatch.setattr(
        loader.transformers,
        "FakeModelForCausalLM",
        type("FakeModelForCausalLM", (), {"_supports_flash_attn_2": True}),
        raising=False,
    )

    result = DummyQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
        profile=PROFILE.FAST,
        attn_implementation="sdpa",
    )

    assert model_calls[0]["kwargs"]["attn_implementation"] == "sdpa"
    assert getattr(result.model.config, "_gptqmodel_bonsai_dense_fast_cb", False) is False


@pytest.mark.parametrize(
    ("extra_kwargs", "expected_backend"),
    [
        ({"profile": PROFILE.LOW_MEMORY}, BACKEND.AUTO),
        ({"backend": BACKEND.GGUF_TORCH, "profile": PROFILE.LOW_MEMORY}, BACKEND.GGUF_TORCH),
    ],
)
def test_model_loader_from_pretrained_native_bonsai_low_memory_redirects_with_backend(monkeypatch, extra_kwargs, expected_backend):
    class FakeConfig:
        def __init__(self):
            self._experts_implementation = None
            self.model_type = "qwen3"
            self.sub_configs = {}
            self.dtype = None

    config_calls = []
    tokenizer_calls = []
    captured = {}
    sentinel = object()
    native_spec = loader.internal_gguf.GGUFQuantizedCheckpointSpec(
        model_type="qwen3",
        bits_alias="q1_0_g128",
        tensor_qtype=loader.internal_gguf.GGMLQuantizationType.Q1_0_g128,
        lm_head_quantized=True,
    )

    def fake_normalize(model_id_or_path, kwargs, *, api_name):
        assert model_id_or_path in {"/tmp/fake-model/bonsai.gguf", "/tmp/fake-model"}
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "bonsai.gguf"
        return "/tmp/fake-model"

    @loader.ModelLoader
    class DummyQModel:
        loader = object()
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

    monkeypatch.setattr(loader, "check_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", fake_normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_args, **_kwargs: "/tmp/fake-model")
    monkeypatch.setattr(loader, "auto_select_device", lambda *_args, **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_args, **_kwargs: torch.bfloat16)
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **kwargs: config_calls.append(kwargs) or FakeConfig(),
    )
    monkeypatch.setattr(
        loader.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **kwargs: tokenizer_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(loader, "print_module_tree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loader,
        "_resolve_native_quantized_gguf_checkpoint",
        lambda *_args, **_kwargs: ("/tmp/fake-model/bonsai.gguf", native_spec),
    )
    monkeypatch.setattr(
        DummyQModel,
        "from_quantized",
        classmethod(lambda cls, model_id_or_path, **kwargs: captured.update(path=model_id_or_path, kwargs=kwargs) or sentinel),
    )

    result = DummyQModel.from_pretrained(
        "/tmp/fake-model/bonsai.gguf",
        quantize_config=None,
        **extra_kwargs,
    )

    assert result is sentinel
    assert config_calls[0]["gguf_file"] == "bonsai.gguf"
    assert tokenizer_calls[0]["gguf_file"] == "bonsai.gguf"
    assert captured["path"] == "/tmp/fake-model"
    assert captured["kwargs"][INTERNAL_HF_GGUF_FILE_KWARG] == "bonsai.gguf"
    assert captured["kwargs"]["backend"] == expected_backend
    assert captured["kwargs"]["dtype"] is torch.bfloat16


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
        loader_requires_dtype = False
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
