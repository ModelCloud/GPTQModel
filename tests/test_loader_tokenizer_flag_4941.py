# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A definition that predates ``require_load_tokenizer`` must still load (#4941).

#4894 added ``BaseQModel.require_load_tokenizer`` and made ``ModelLoader`` read
it as ``cls.require_load_tokenizer``. Only ``BaseQModel`` subclasses inherit the
default, so every plain-class test double (and any third-party definition that
does not subclass it) raised

    AttributeError: type object 'DummyQModel' has no attribute 'require_load_tokenizer'

and the whole load aborted (#4923, #4941). An absent attribute has to mean
"text model", which is what every definition did before #4894; only an explicit
``require_load_tokenizer = False`` may skip the text path.

These tests drive the real ``ModelLoader.from_pretrained`` with the ``.bin``
source harness from ``test_lazy_turtle_conversion_cache.py``.
"""

from __future__ import annotations

import pytest

from gptqmodel.models import loader
from gptqmodel.models.loader import _requires_tokenizer
from gptqmodel.quantization import QuantizeConfig
from tests.test_lazy_turtle_conversion_cache import _FakeCheckpointModel, _patch_bin_source


def _install_loader_stubs(monkeypatch, *, tokenizer_calls, source_dir, base_config):
    """Neutralise everything ``from_pretrained`` does before the tokenizer step."""

    def _fake_build_shell_model(_loader, config, **_kwargs):
        model = _FakeCheckpointModel()
        model.config = config
        return model

    def _fake_load_tokenizer(*_args, **_kwargs):
        tokenizer_calls.append(_args)
        return object()

    def _fake_convert_model(model, cleanup_original=False):
        assert cleanup_original is False
        return model

    monkeypatch.setattr(loader, "check_versions", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_a, **_k: str(source_dir))
    monkeypatch.setattr(loader, "auto_select_device", lambda *_a, **_k: "cpu")
    monkeypatch.setattr(loader, "auto_dtype", lambda *_a, **_k: "float16")
    monkeypatch.setattr(loader, "print_module_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "load_model_config", lambda *_a, **_k: base_config)
    monkeypatch.setattr(loader, "load_hf_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr("gptqmodel.utils.hf.build_shell_model", _fake_build_shell_model)
    monkeypatch.setattr(loader.defuser, "convert_model", _fake_convert_model)


class _FakeConfig:
    def __init__(self):
        self._experts_implementation = None
        self.model_type = "fake_loader_flag"
        self.sub_configs = {}
        self.dtype = None

    def to_dict(self):
        return {"max_position_embeddings": 128}


def _dummy_qmodel(*, declare_flag: bool | None):
    """A loader double in the style the existing tests use (no ``BaseQModel``)."""

    class _FakeInnerLoader:
        @staticmethod
        def from_pretrained(_path, config=None, **_kwargs):
            raise AssertionError("the inner loader must not run for this harness")

    class _DummyQModel:
        loader = _FakeInnerLoader
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
            self.config = model.config

    if declare_flag is not None:
        _DummyQModel.require_load_tokenizer = declare_flag
    return loader.ModelLoader(_DummyQModel)


def _run_load(monkeypatch, tmp_path, *, declare_flag):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)
    tokenizer_calls = []
    _install_loader_stubs(
        monkeypatch,
        tokenizer_calls=tokenizer_calls,
        source_dir=source_dir,
        base_config=_FakeConfig(),
    )
    definition = _dummy_qmodel(declare_flag=declare_flag)
    definition.from_pretrained(
        str(source_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        trust_remote_code=False,
    )
    return tokenizer_calls


def test_plain_class_double_without_the_attribute_still_loads(monkeypatch, tmp_path):
    """The pre-#4894 contract: an absent attribute means "this is a text model"."""

    tokenizer_calls = _run_load(monkeypatch, tmp_path, declare_flag=None)

    assert len(tokenizer_calls) == 1, (
        "the loader skipped the tokenizer for a definition that never declared "
        "require_load_tokenizer; an absent attribute must default to the text path"
    )


def test_definition_that_declares_false_skips_the_tokenizer(monkeypatch, tmp_path):
    """An explicit opt-out still skips the whole text path (Decision Transformer)."""

    assert _run_load(monkeypatch, tmp_path, declare_flag=False) == []


def test_definition_that_declares_true_keeps_the_text_path(monkeypatch, tmp_path):
    assert len(_run_load(monkeypatch, tmp_path, declare_flag=True)) == 1


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (True, True),
        (False, False),
        (None, False),
    ],
)
def test_requires_tokenizer_reads_a_declared_value(declared, expected):
    class _Definition:
        require_load_tokenizer = declared

    assert _requires_tokenizer(_Definition) is expected


def test_requires_tokenizer_defaults_to_true_when_absent():
    class _Definition:
        pass

    assert _requires_tokenizer(_Definition) is True
