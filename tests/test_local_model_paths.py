from types import SimpleNamespace

from gptqmodel.models import GPTQModel, auto, loader


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
