import json
import sys
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.hf import (
    INTERNAL_HF_GGUF_FILE_KWARG,
    get_hf_config_dtype,
    get_hf_gguf_load_kwargs,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
    resolve_trust_remote_code,
)
from gptqmodel.utils.kernel import KernelConfig, configure_kernel


def test_get_hf_config_dtype_avoids_modern_deprecated_alias():
    class ModernConfig:
        dtype = None

        @property
        def torch_dtype(self):
            raise AssertionError("deprecated torch_dtype alias should not be read")

    assert get_hf_config_dtype(ModernConfig()) is None


def test_get_hf_config_dtype_preserves_stored_legacy_value():
    config = SimpleNamespace(dtype=None, torch_dtype="float16")

    assert get_hf_config_dtype(config) is torch.float16


def test_get_hf_config_dtype_accepts_the_torch_prefixed_string():
    config = SimpleNamespace(dtype="torch.bfloat16")

    assert get_hf_config_dtype(config) is torch.bfloat16


def test_get_hf_config_dtype_rejects_a_non_dtype_torch_attribute():
    config = SimpleNamespace(dtype="torch.pi")

    with pytest.raises(ValueError, match="Invalid dtype string"):
        get_hf_config_dtype(config)


def test_normalize_torch_dtype_kwarg_moves_alias_to_dtype():
    kwargs = {"torch_dtype": torch.float16}

    resolved = normalize_torch_dtype_kwarg(kwargs, api_name="test")

    assert resolved is torch.float16
    assert kwargs == {"dtype": torch.float16}


def test_normalize_torch_dtype_kwarg_resolves_explicit_dtype_parameter():
    kwargs = {"torch_dtype": torch.bfloat16}

    resolved = normalize_torch_dtype_kwarg(kwargs, api_name="test", explicit_dtype="auto")

    assert resolved is torch.bfloat16
    assert kwargs == {}


@pytest.mark.parametrize("value", ["bfloat16", "torch.bfloat16"])
def test_normalize_torch_dtype_kwarg_resolves_dtype_strings(value):
    kwargs = {"dtype": value}

    resolved = normalize_torch_dtype_kwarg(kwargs, api_name="test")

    assert resolved is torch.bfloat16
    assert kwargs == {"dtype": torch.bfloat16}


def test_normalize_torch_dtype_kwarg_rejects_invalid_dtype_string():
    with pytest.raises(ValueError, match="Invalid dtype"):
        normalize_torch_dtype_kwarg(
            {"dtype": "not-a-torch-dtype"},
            api_name="test",
        )


def test_normalize_torch_dtype_kwarg_rejects_conflicting_values():
    kwargs = {"dtype": torch.float16, "torch_dtype": torch.bfloat16}

    with pytest.raises(ValueError, match="both `dtype` and deprecated `torch_dtype`"):
        normalize_torch_dtype_kwarg(kwargs, api_name="test")


def test_normalize_model_id_or_path_for_hf_gguf_rejects_public_kwarg():
    kwargs = {"gguf_file": "bonsai.gguf"}

    with pytest.raises(TypeError, match="does not accept `gguf_file`"):
        normalize_model_id_or_path_for_hf_gguf("/tmp/fake-model", kwargs, api_name="test")


def test_normalize_model_id_or_path_for_hf_gguf_normalizes_local_file(monkeypatch, tmp_path):
    gguf_path = tmp_path / "bonsai.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(hf_utils, "_patch_transformers_prism_gguf_compat", lambda **_kwargs: None)

    kwargs = {}
    model_root = normalize_model_id_or_path_for_hf_gguf(str(gguf_path), kwargs, api_name="test")

    assert model_root == str(tmp_path)
    assert kwargs[INTERNAL_HF_GGUF_FILE_KWARG] == "bonsai.gguf"
    assert get_hf_gguf_load_kwargs(kwargs) == {"gguf_file": "bonsai.gguf"}


def test_patch_transformers_prism_gguf_compat_registers_internal_runtime(monkeypatch):
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.utils import import_utils as hf_import_utils

    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(gguf_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_import_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_utils, "_transformers_has_native_prism_gguf_support", lambda: False)

    hf_utils._patch_transformers_prism_gguf_compat(api_name="test")

    assert sys.modules["gguf"] is internal_gguf
    assert gguf_utils.is_gguf_available() is True
    assert hf_import_utils.is_gguf_available() is True
    assert gguf_utils.PRISM_Q1_0_G128_NAME == hf_utils.PRISM_Q1_0_G128_NAME
    assert gguf_utils._dequantize_prism_q1_0_g128 is hf_utils._dequantize_prism_q1_0_g128


def test_patch_transformers_prism_gguf_compat_wraps_load_checkpoint_for_torch_loader(monkeypatch):
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.utils import import_utils as hf_import_utils

    calls = {"direct": 0, "torch_dtype": None}

    def _original_load_gguf_checkpoint(*args, **kwargs):
        return {"variant": "original"}

    def _direct_loader(**kwargs):
        calls["direct"] += 1
        calls["torch_dtype"] = kwargs["torch_dtype"]
        return {"variant": "direct", "path": kwargs["gguf_checkpoint_path"]}

    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(gguf_utils, "load_gguf_checkpoint", _original_load_gguf_checkpoint)
    monkeypatch.delattr(gguf_utils, "_GPTQMODEL_GGUF_TORCH_LOADER_PATCHED", raising=False)
    monkeypatch.delattr(gguf_utils, "_gptqmodel_original_load_gguf_checkpoint", raising=False)
    monkeypatch.setattr(gguf_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_import_utils, "is_gguf_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(hf_utils, "_transformers_has_native_prism_gguf_support", lambda: False)
    monkeypatch.setattr(hf_utils, "_load_gguf_checkpoint_torch_direct", _direct_loader)

    hf_utils._patch_transformers_prism_gguf_compat(api_name="test")
    configure_kernel(KernelConfig(gguf_torch_dequant_loader=True))
    result = gguf_utils.load_gguf_checkpoint(
        "bonsai.gguf",
        return_tensors=True,
        model_to_load=object(),
        torch_dtype=torch.bfloat16,
    )

    assert calls["direct"] == 1
    assert calls["torch_dtype"] is torch.bfloat16
    assert result == {"variant": "direct", "path": "bonsai.gguf"}


def test_local_brumby_runs_without_legacy_default_rope_init(monkeypatch):
    import transformers.modeling_rope_utils as rope_utils

    from gptqmodel.models.ports.brumby.configuration_brumby import BrumbyConfig
    from gptqmodel.models.ports.brumby.modeling_brumby import BrumbyForCausalLM

    monkeypatch.delitem(rope_utils.ROPE_INIT_FUNCTIONS, "default", raising=False)
    config = BrumbyConfig(
        vocab_size=24, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, chunk_size=4,
    )
    assert "default" not in rope_utils.ROPE_INIT_FUNCTIONS

    model = BrumbyForCausalLM(config).eval()
    tokens = torch.tensor([[2, 4, 6, 8, 10]])
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        prefix = model(tokens[:, :3], use_cache=True)
        actual = model(tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits
    assert expected.std() > 0.001
    torch.testing.assert_close(actual, expected[:, 3:], rtol=1e-4, atol=1e-5)


def test_resolve_trust_remote_code_overrides_when_native_support_exists(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    # A MODEL_MAP type (e.g. phi3) is owned by its LocalModelLoader and returns
    # before the Transformers probe; use a native type GPT-QModel does not define.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "biogpt"}))
    hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
    monkeypatch.setattr(
        hf_utils,
        "_detect_native_transformers_causallm_support",
        lambda model_id_or_path: (True, "biogpt", "BioGptForCausalLM"),
    )

    # The warning is emitted through the library logger, whose sink is a logbar
    # handler bound at import time; ``caplog`` observes the record, ``capsys``
    # does not see that handler's stream.
    with caplog.at_level("WARNING"):
        resolved = resolve_trust_remote_code(str(tmp_path), trust_remote_code=True)

    assert resolved is False
    assert "overriding trust_remote_code=True to False" in caplog.text


def test_resolve_trust_remote_code_keeps_true_without_native_support(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "compat_remote_fixture"}))
    hf_utils._TRUST_REMOTE_CODE_OVERRIDE_WARNED.clear()
    monkeypatch.setattr(
        hf_utils,
        "_detect_native_transformers_causallm_support",
        lambda model_id_or_path: (False, None, None),
    )

    with caplog.at_level("WARNING"):
        resolved = resolve_trust_remote_code(str(tmp_path), trust_remote_code=True)

    assert resolved is True
    assert caplog.text == ""


