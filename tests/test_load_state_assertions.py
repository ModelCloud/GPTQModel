# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The post-load safety net must fail loudly on a corrupted model.

Issue #4129: overlapping ``GPTQModel.load()`` calls interleaved transformers'
process-wide default-dtype and meta-init state, so a model could come back with
mixed fp16/fp32 parameters, a tied weight left on the meta device, or the
process default dtype stuck at the load dtype. The loaders assert these
invariants after every build; a silent mixed model is worse than a failed load.
"""

import pytest
import torch
from torch import nn

from gptqmodel.utils.hf import assert_loaded_model_state


class _Tiny(nn.Module):
    """Model stand-in with one floating parameter per requested dtype."""

    def __init__(self, dtypes, meta_indices=()):
        super().__init__()
        for index, dtype in enumerate(dtypes):
            device = "meta" if index in meta_indices else "cpu"
            self.register_parameter(
                f"p{index}",
                nn.Parameter(torch.empty(2, dtype=dtype, device=device)),
            )


def test_accepts_a_consistent_load():
    model = _Tiny([torch.float16, torch.float16])

    assert_loaded_model_state(
        model,
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
        expected_dtype=torch.float16,
        require_uniform_dtype=True,
    )


def test_rejects_mixed_dtypes():
    model = _Tiny([torch.float16, torch.float32])

    with pytest.raises(RuntimeError, match="mixed dtypes"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            require_uniform_dtype=True,
        )


def test_rejects_a_parameter_outside_the_requested_dtype():
    model = _Tiny([torch.float16, torch.float32])

    with pytest.raises(RuntimeError, match="not all torch.float16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_rejects_a_meta_tied_weight():
    """The measured failure: ``lm_head.weight`` stayed on the meta device."""

    model = _Tiny([torch.float16, torch.float16], meta_indices=(1,))

    with pytest.raises(RuntimeError, match="stayed on the meta device"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_rejects_a_changed_process_default_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = _Tiny([torch.float16])
        with pytest.raises(RuntimeError, match="get_default_dtype"):
            assert_loaded_model_state(
                model,
                api_name="test.load",
                default_dtype_before=torch.float32,
                expected_dtype=torch.float16,
            )
    finally:
        torch.set_default_dtype(previous)


def test_meta_shell_only_checks_the_process_default_dtype():
    """``offload_to_disk`` builds a meta shell that is filled in later."""

    model = _Tiny([torch.float16, torch.float16], meta_indices=(0, 1))

    assert_loaded_model_state(
        model,
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
        require_materialized=False,
        require_uniform_dtype=False,
    )


def test_non_torch_model_only_checks_the_process_default_dtype():
    """The MLX runtime object has no ``named_parameters``."""

    assert_loaded_model_state(
        {"weight": torch.zeros(2)},
        api_name="test.load",
        default_dtype_before=torch.get_default_dtype(),
    )


class _DeclaredKeepInFp32(nn.Module):
    """Two linears; ``keep`` stays fp32 and ``other`` is cast to fp16."""

    def __init__(self, attr="_keep_in_fp32_modules_strict", declared=("keep",)):
        super().__init__()
        self.keep = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)
        if attr is not None:
            setattr(self, attr, list(declared))


def _declared_mixed_model(attr="_keep_in_fp32_modules_strict", declared=("keep",)):
    model = _DeclaredKeepInFp32(attr=attr, declared=declared)
    model.other.to(torch.float16)
    return model


def test_declared_keep_in_fp32_modules_are_exempt_from_both_dtype_checks():
    """Issue #4170: a declared fp32 pin inside a half model is not corruption."""

    model = _declared_mixed_model()

    assert_loaded_model_state(
        model,
        api_name="test.load",
        expected_dtype=torch.float16,
        require_uniform_dtype=True,
    )
    assert_loaded_model_state(
        model,
        api_name="test.load",
        require_uniform_dtype=True,
    )


def test_keep_in_fp32_declaration_on_a_submodule_is_honored():
    """Transformers can register the declaration on a child, not the root."""

    model = _declared_mixed_model(attr=None)
    model.keep._keep_in_fp32_modules_strict = ["keep"]

    assert_loaded_model_state(
        model,
        api_name="test.load",
        expected_dtype=torch.float16,
        require_uniform_dtype=True,
    )


def test_declared_module_left_in_a_half_dtype_is_still_reported():
    """The exemption only covers the float32 pin transformers applies."""

    model = _declared_mixed_model()
    model.keep.weight.data = model.keep.weight.data.to(torch.float16)
    model.keep.bias.data = model.keep.bias.data.to(torch.float16)

    with pytest.raises(RuntimeError, match="not all torch.float32"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float32,
            require_uniform_dtype=True,
        )


def test_keep_in_fp32_patterns_follow_transformers_glob_matching():
    """transformers 5.22 compiles each entry as a regex with ``*`` -> ``.*``."""

    matching = _declared_mixed_model(declared=("keep.w*",))
    matching.keep.bias.data = matching.keep.bias.data.to(torch.float16)
    assert_loaded_model_state(
        matching,
        api_name="test.load",
        expected_dtype=torch.float16,
        require_uniform_dtype=True,
    )

    non_matching = _declared_mixed_model(declared=("keep.bias",))
    non_matching.keep.bias.data = non_matching.keep.bias.data.to(torch.float16)
    with pytest.raises(RuntimeError, match="not all torch.float16"):
        assert_loaded_model_state(
            non_matching,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_undeclared_mixed_dtypes_still_raise():
    """Negative control: #4140's protection is not weakened model-wide."""

    model = _Tiny([torch.float16, torch.float32])

    with pytest.raises(RuntimeError, match="mixed dtypes"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            require_uniform_dtype=True,
        )
    with pytest.raises(RuntimeError, match="not all torch.float16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_clearing_keep_in_fp32_declaration_restores_the_guard():
    """Negative control: removing the declaration exposes the same fp32 pin."""

    model = _declared_mixed_model()
    model._keep_in_fp32_modules = None
    model._keep_in_fp32_modules_strict = None

    with pytest.raises(RuntimeError, match="not all torch.float16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def test_declared_fp32_parameter_on_meta_still_raises():
    """Dtype exemptions do not hide an unmaterialized parameter."""

    model = _declared_mixed_model()
    model.keep.weight = nn.Parameter(torch.empty(2, 2, device="meta"))

    with pytest.raises(RuntimeError, match="stayed on the meta device"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.float16,
            require_uniform_dtype=True,
        )


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = (
        "[UNK]", "[PAD]", "<s>", "</s>", "A", "short", "calibration",
        "sentence", "about", "fox", "Another", "moon", ".", "runs",
    )
    inner = Tokenizer(
        models.WordLevel(vocab={word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )


def _write_tiny_ernie4_5_moe(root):
    from transformers import Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM

    config = Ernie4_5_MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_num_experts=4,
        moe_k=2,
        moe_layer_start_index=1,
        max_position_embeddings=64,
    )
    config.dtype = torch.bfloat16
    Ernie4_5_MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(root)
    _tiny_tokenizer().save_pretrained(root)


def _write_tiny_gpt_oss(root):
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        num_hidden_layers=2,
        num_local_experts=2,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_experts_per_tok=1,
        layer_types=["sliding_attention", "sliding_attention"],
    )
    GptOssForCausalLM(config).to(torch.float32).save_pretrained(root)
    _tiny_tokenizer().save_pretrained(root)


def _floating_histogram(model):
    from collections import Counter

    return Counter(
        parameter.dtype
        for _, parameter in model.named_parameters()
        if parameter.is_floating_point()
    )


def _fp32_parameter_names(model):
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype == torch.float32
    }


def _clear_keep_in_fp32_declarations(model):
    """Shadow every declaration with ``None`` (including inherited class ones)."""

    for module in model.modules():
        for attr in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict"):
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)


def test_ernie4_5_moe_loads_with_declared_keep_in_fp32_strict_modules(tmp_path):
    """Issue #4170: Ernie4.5-MoE pins ``gate.weight`` and ``moe_statics``."""

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    source = tmp_path / "ernie"
    _write_tiny_ernie4_5_moe(source)
    expected_fp32 = {
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.moe_statics.e_score_correction_bias",
    }

    native = GPTQModel.load(str(source), device="cpu")
    native_histogram = _floating_histogram(native.model)
    print("Ernie4.5-MoE native dtype histogram:", dict(native_histogram))
    assert native_histogram[torch.bfloat16] > 0
    assert _fp32_parameter_names(native.model) == expected_fp32

    quantize_path = GPTQModel.load(
        str(source),
        QuantizeConfig(offload_to_disk=False),
        device="cpu",
        dtype=torch.bfloat16,
    )
    quantize_histogram = _floating_histogram(quantize_path.model)
    print("Ernie4.5-MoE quantize-path dtype histogram:", dict(quantize_histogram))
    assert quantize_histogram[torch.bfloat16] > 0
    assert _fp32_parameter_names(quantize_path.model) == expected_fp32


def test_gpt_oss_loads_in_fp16_with_declared_keep_in_fp32_norms(tmp_path):
    """Issue #4170: GPT-OSS norms stay fp32 when the load dtype is fp16."""

    from gptqmodel import GPTQModel

    source = tmp_path / "gpt_oss"
    _write_tiny_gpt_oss(source)

    loaded = GPTQModel.load(str(source), device="cpu", dtype=torch.float16)
    histogram = _floating_histogram(loaded.model)
    print("GPT-OSS fp16 dtype histogram:", dict(histogram))
    assert histogram[torch.float16] > 0
    assert _fp32_parameter_names(loaded.model) == {
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
    }


def test_ernie4_5_moe_negative_controls_still_raise(tmp_path):
    """#4140 still catches fp32 leakage the model never declared (issue #4170)."""

    from gptqmodel import GPTQModel

    source = tmp_path / "ernie"
    _write_tiny_ernie4_5_moe(source)
    loaded = GPTQModel.load(str(source), device="cpu", dtype=torch.bfloat16)
    model = loaded.model

    # 1. Cast a non-exempt decoder projection to fp32: undeclared mixed dtypes.
    exempt = "model.layers.1.self_attn.q_proj.weight"
    parameter = dict(model.named_parameters())[exempt]
    original = parameter.data
    parameter.data = original.float()
    with pytest.raises(RuntimeError, match="not all torch.bfloat16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.bfloat16,
            require_uniform_dtype=True,
        )
    parameter.data = original

    # 2. Clear the declaration: the same fp32 pins become corruption again.
    _clear_keep_in_fp32_declarations(model)
    with pytest.raises(RuntimeError, match="mixed dtypes"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            require_uniform_dtype=True,
        )


class _PackedStorage(nn.Module):
    """Module whose scale parameter is kernel storage, not an activation operand."""

    CHECKPOINT_EXACT_DTYPE_PARAMETERS = ("scales",)

    def __init__(self):
        super().__init__()
        self.scales = nn.Parameter(
            torch.empty(2, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.dense = nn.Parameter(torch.empty(2, dtype=torch.bfloat16))


class _StorageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedStorage()


def test_accepts_declared_storage_parameters_at_the_checkpoint_dtype():
    # A declared parameter keeps the checkpoint dtype (float8) while the rest of
    # the model is bf16; that is the kernel contract, not corruption.
    assert_loaded_model_state(
        _StorageModel(),
        api_name="test.load",
        expected_dtype=torch.bfloat16,
        require_uniform_dtype=True,
    )


def test_still_rejects_an_undeclared_storage_parameter():
    class _Undeclared(_PackedStorage):
        CHECKPOINT_EXACT_DTYPE_PARAMETERS = ()

    model = nn.Module()
    model.experts = _Undeclared()

    with pytest.raises(RuntimeError, match="not all torch.bfloat16"):
        assert_loaded_model_state(
            model,
            api_name="test.load",
            expected_dtype=torch.bfloat16,
            require_uniform_dtype=True,
        )


def test_declared_storage_parameters_must_still_be_materialized():
    class _MetaStorage(_PackedStorage):
        def __init__(self):
            super().__init__()
            self.scales = nn.Parameter(
                torch.empty(2, dtype=torch.float8_e4m3fn, device="meta"),
                requires_grad=False,
            )

    model = nn.Module()
    model.experts = _MetaStorage()

    with pytest.raises(RuntimeError, match="stayed on the meta device"):
        assert_loaded_model_state(model, api_name="test.load")
