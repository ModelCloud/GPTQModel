# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the Ricursive Ripple (``ricursive``) quantization path.

``Ripple-65B-Preview`` ships custom code (``RicursiveForCausalLM``), so the
checkpoint must be loaded with ``trust_remote_code=True``. These tests keep the
cheap contract checks on CPU: registry wiring, the module tree shape, and -- when
the reference custom code is importable -- a tiny end-to-end GPTQ
quantize/save/reload smoke run.

The end-to-end test is skipped when the reference code cannot be fetched. Point
``GPTQMODEL_RIPPLE_REF`` at a directory holding ``configuration_ricursive.py``
and ``modeling_ricursive.py`` to run it offline.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models import auto
from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


_RIPPLE_REPO = "ricursiveAI/Ripple-65B-Preview"

# Modules the definition must quantize in every decoder layer.
_QUANTIZED_PER_LAYER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.g_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Modules the definition must leave dense (norms + the value-residual scalar).
_DENSE_PER_LAYER = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


def test_ricursive_registry_selects_definition(monkeypatch):
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="ricursive"),
    )

    assert auto.MODEL_MAP["ricursive"] is RicursiveQModel
    assert auto.check_and_get_model_definition("/tmp/ripple", trust_remote_code=True) is RicursiveQModel


def test_ricursive_definition_contract():
    assert RicursiveQModel.require_trust_remote_code is True
    assert RicursiveQModel.pre_lm_head_norm_module == "model.norm"
    assert RicursiveQModel.extract_layers_node() == ["model.layers"]


def test_ricursive_module_tree_covers_perinorm_and_gated_attention():
    layer = RicursiveQModel.module_tree[3]

    assert set(layer) == {
        "input_layernorm",
        "self_attn:!",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "mlp",
        "post_feedforward_layernorm",
    }

    attn = layer["self_attn:!"]
    # Per-head Q/K RMSNorm must stay dense.
    assert "q_norm:!:norm" in attn
    assert "k_norm:!:norm" in attn
    # Gated attention: g_proj shares the hidden states with q/k.
    assert "g_proj:0:in=x" in attn
    for name in ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "o_proj:1:o"):
        assert name in attn
    # `use_valres` makes v_proj read a lerp of the first layer's hidden states, so it must
    # NOT carry the `in=x` tag that groups q/k/g onto one Hessian.
    assert "v_proj:0:v" in attn
    assert not any(token.startswith("v_proj:") and "in=" in token for token in attn)

    assert layer["mlp"] == ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down")


def test_ricursive_simple_layer_modules_quantize_eight_projects_per_layer():
    model_config = SimpleNamespace(model_type="ricursive", num_hidden_layers=80)
    quantize_config = SimpleNamespace(dynamic=None)

    simple = RicursiveQModel.simple_layer_modules(model_config, quantize_config)
    names = [name for block in simple for name in block]

    assert names == list(_QUANTIZED_PER_LAYER)

    full_names = {name for block in RicursiveQModel.full_layer_modules(model_config) for name in block}
    for name in _DENSE_PER_LAYER:
        assert f"{name}:!" in full_names, name


def test_ricursive_shared_input_plan_excludes_value_residual_projection():
    """`v_proj` must not share a Hessian group with q/k/g when `use_valres` is on.

    Ripple's attention reads `v_proj(lerp(value_residual, hidden_states, blend))` while
    q/k/g read `hidden_states` directly. Grouping `v_proj` with them (as a plain Llama
    tree would) would collect the wrong `X^T X` for it.
    """

    from gptqmodel.models.shared_input import build_shared_input_plan

    model_config = SimpleNamespace(model_type="ricursive", num_hidden_layers=80)
    quantize_config = SimpleNamespace(dynamic=None)

    # The definition stays singleton-only (`shared_input_verified_model_types` is empty):
    # opting in would require a real-forward case in
    # `tests/module_tree/test_shared_input_cpu_forward.py`, and the tags below are asserted
    # against a live forward here instead. Pass `explicit_tags=True` to check the tags.
    assert not RicursiveQModel.shared_input_verified(model_config)
    assert RicursiveQModel.shared_input_plan(model_config, quantize_config).shared_groups == ()
    plan = build_shared_input_plan(
        RicursiveQModel.module_tree,
        RicursiveQModel.simple_layer_modules(model_config, quantize_config),
        explicit_tags=True,
    )

    groups = {frozenset(group.modules) for group in plan.groups}
    assert frozenset({"self_attn.q_proj", "self_attn.k_proj", "self_attn.g_proj"}) in groups
    assert frozenset({"mlp.gate_proj", "mlp.up_proj"}) in groups
    for group in groups:
        assert not ("self_attn.v_proj" in group and len(group) > 1), group
        assert not ("self_attn.o_proj" in group and len(group) > 1), group
        assert not ("mlp.down_proj" in group and len(group) > 1), group


def test_plan_matches_real_forward_inputs(tmp_path: Path):
    """Opt-in check: the declared groups really do share one tensor in a live forward.

    Also proves the `v_proj` exclusion is required rather than merely conservative --
    without it the probe would report `v_proj` as a mismatch.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    from gptqmodel.models.shared_input import probe_shared_inputs

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    from gptqmodel.models.shared_input import build_shared_input_plan

    model_config = SimpleNamespace(model_type="ricursive", num_hidden_layers=5)
    quantize_config = SimpleNamespace(dynamic=None)
    template_plan = build_shared_input_plan(
        RicursiveQModel.module_tree,
        RicursiveQModel.simple_layer_modules(model_config, quantize_config),
        explicit_tags=True,
    )
    assert template_plan.shared_groups

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    input_ids = torch.randint(0, 256, (2, 16))

    def forward():
        with torch.no_grad():
            model(input_ids=input_ids)

    # Layer 0 is sliding-attention, layer 4 is full-attention (NoPE).
    for layer_index in (0, 4):
        layer = model.model.layers[layer_index]
        plan = template_plan.filter_modules(set(dict(layer.named_modules())))
        report = probe_shared_inputs(layer, plan, forward)

        assert report.ok, report.describe()
        assert report.fully_verified and not report.has_errors
        assert report.mismatches == (), report.describe()
        assert report.undeclared == (), report.describe()
        assert report.missing_modules == ()

        # Dedup is opt-in only: every multi-module group must be explicit.
        for group in plan.shared_groups:
            assert group.explicit, group

        assert plan.shares_input("self_attn.q_proj", "self_attn.k_proj")
        assert plan.shares_input("self_attn.q_proj", "self_attn.g_proj")
        assert not plan.shares_input("self_attn.q_proj", "self_attn.v_proj")
        assert not plan.shares_input("self_attn.q_proj", "self_attn.o_proj")


def _reference_dir() -> Path | None:
    """Locate the Ripple custom-code directory, or ``None`` when unavailable."""

    env = os.environ.get("GPTQMODEL_RIPPLE_REF")
    if env:
        candidate = Path(env)
        if (candidate / "configuration_ricursive.py").is_file():
            return candidate
        return None

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            _RIPPLE_REPO,
            allow_patterns=["config.json", "*.py", "tokenizer*", "special_tokens_map.json"],
        )
    except Exception:
        return None
    candidate = Path(path)
    return candidate if (candidate / "configuration_ricursive.py").is_file() else None


def _build_tokenizer(model_dir: Path) -> None:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"t{i}": i for i in range(3, 256)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))

    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            indent=2,
        )
    )
    (model_dir / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<unk>"}, indent=2)
    )


def _build_tiny_fixture(ref_dir: Path, model_dir: Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("configuration_ricursive.py", "modeling_ricursive.py"):
        shutil.copy(ref_dir / name, model_dir / name)

    # Five layers exercise both the sliding-attention and the full-attention (NoPE) path.
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["RicursiveForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_ricursive.RicursiveConfig",
                    "AutoModel": "modeling_ricursive.RicursiveModel",
                    "AutoModelForCausalLM": "modeling_ricursive.RicursiveForCausalLM",
                },
                "model_type": "ricursive",
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 96,
                "num_hidden_layers": 5,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 128,
                "sliding_window": 16,
                "rms_norm_eps": 1e-06,
                "rope_theta": 10000.0,
                "use_gated_attention": True,
                "use_valres": True,
                "ffn_clamp_threshold": 20.0,
                "tie_word_embeddings": False,
            },
            indent=2,
        )
    )
    _build_tokenizer(model_dir)

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    AutoModelForCausalLM.from_config(config, trust_remote_code=True).save_pretrained(str(model_dir))


def _apply_compat_patch_for_test(model) -> None:
    """Exercise the legacy checkpoint path even when the reference is fixed."""
    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    # The reference repository is a moving target. Its current custom code
    # declares native TF5 fixes, so production correctly skips these shims.
    # Clear only the fixture class's marker to keep the compatibility tests
    # exercising GPT-QModel's patch on current and older reference snapshots.
    cls = type(model)
    had_marker = "_ricursive_tf5_native_fix" in vars(cls)
    original_marker = vars(cls).get("_ricursive_tf5_native_fix")
    try:
        cls._ricursive_tf5_native_fix = None
        apply_ricursive_generate_patch(model)
    finally:
        if had_marker:
            cls._ricursive_tf5_native_fix = original_marker
        else:
            del cls._ricursive_tf5_native_fix


def test_tiny_ricursive_gptq_quantization_smoke(tmp_path: Path):
    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    _build_tiny_fixture(ref_dir, model_dir)

    from transformers import AutoModelForCausalLM

    native = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        backend=BACKEND.TORCH,
        trust_remote_code=True,
    )
    calibration = [{"input_ids": torch.randint(0, 256, (64,))} for _ in range(8)]
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(str(quantized_dir))

    quantized = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
        trust_remote_code=True,
    )
    modules = dict(quantized.model.named_modules())

    for layer_index in (0, 4):
        for name in _QUANTIZED_PER_LAYER:
            full_name = f"model.layers.{layer_index}.{name}"
            assert isinstance(modules[full_name], TorchLinear), full_name
        for name in _DENSE_PER_LAYER:
            full_name = f"model.layers.{layer_index}.{name}"
            assert not isinstance(modules[full_name], TorchLinear), full_name

    # The value-residual blend scalar is a Parameter (not a Module), so it never appears
    # in `named_modules()` and can never be picked up as a quantizable projection.
    parameters = dict(quantized.model.named_parameters())
    assert isinstance(parameters["model.layers.0.self_attn.value_res_lambda"], torch.nn.Parameter)
    assert "model.layers.0.self_attn.value_res_lambda" not in modules

    input_ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        native_logits = native(input_ids=input_ids).logits.float()
        quantized_logits = quantized.model(input_ids=input_ids).logits.float()

    diff = (native_logits - quantized_logits).abs().mean().item()
    scale = native_logits.abs().mean().item()
    assert diff < 0.5 * scale, (diff, scale)


def test_generate_patch_keeps_input_shape_on_empty_cache():
    """The upstream bug: an empty cache plus `cache_position=None` added a bogus axis."""

    from transformers.cache_utils import DynamicCache

    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    class _Dummy:
        device = torch.device("cpu")

    input_ids = torch.arange(6).unsqueeze(0)
    assert input_ids.shape == (1, 6)

    out = _prepare_inputs_for_generation(
        _Dummy(),
        input_ids,
        past_key_values=DynamicCache(),
        cache_position=None,
        use_cache=True,
    )

    # Prefill must stay 2-D; the buggy code produced (1, 1, 6).
    assert out["input_ids"].shape == (1, 6)
    assert out["position_ids"].shape == (1, 6)
    assert out["cache_position"].tolist() == [0, 1, 2, 3, 4, 5]


def test_generate_patch_keeps_single_token_decode_shape():
    from transformers.cache_utils import DynamicCache

    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    class _Dummy:
        device = torch.device("cpu")

    # Decode step: a single new token against a warm cache is already the right shape.
    out = _prepare_inputs_for_generation(
        _Dummy(),
        torch.tensor([[7]]),
        past_key_values=DynamicCache(),
        cache_position=None,
        use_cache=True,
    )
    assert out["input_ids"].shape == (1, 1)


@pytest.mark.parametrize(
    "past_length,input_ids,mask_length,explicit_cache,expected_ids,expected_positions",
    [
        (10, [17, 18], None, False, [17, 18], [10, 11]),
        (1, [17, 18], 3, True, [17, 18], [1, 2]),
        (4, [1, 2, 3, 4, 17, 18], 6, False, [17, 18], [4, 5]),
    ],
)
def test_generate_patch_distinguishes_new_blocks_from_full_prompts(
    past_length, input_ids, mask_length, explicit_cache, expected_ids, expected_positions
):
    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    class _Dummy:
        device = torch.device("cpu")

    class _Cache:
        def get_seq_length(self):
            return past_length

    out = _prepare_inputs_for_generation(
        _Dummy(),
        torch.tensor([input_ids]),
        past_key_values=_Cache(),
        attention_mask=torch.ones((1, mask_length), dtype=torch.long) if mask_length else None,
        cache_position=torch.arange(past_length, past_length + len(input_ids)) if explicit_cache else None,
        use_cache=True,
    )
    assert out["input_ids"].tolist() == [expected_ids]
    assert out["cache_position"].tolist() == expected_positions
    assert out["position_ids"].tolist() == [expected_positions]


@pytest.mark.parametrize("capacity", [10, 12])
def test_generate_patch_full_prompt_with_capacity_padded_mask(capacity):
    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    class _Dummy:
        device = torch.device("cpu")

    class _Cache:
        def get_seq_length(self):
            return 4

    out = _prepare_inputs_for_generation(
        _Dummy(),
        torch.tensor([[1, 2, 3, 4, 17, 18]]),
        past_key_values=_Cache(),
        attention_mask=torch.tensor([[1] * 6 + [0] * (capacity - 6)]),
        position_ids=torch.arange(6).unsqueeze(0),
        cache_position=None,
        use_cache=True,
    )
    assert out["input_ids"].tolist() == [[17, 18]]
    assert out["cache_position"].tolist() == [4, 5]
    assert out["position_ids"].tolist() == [[4, 5]]


def test_ricursive_generate_works_and_matches_uncached(tmp_path: Path):
    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    from gptqmodel.utils.ricursive_patch import _PATCH_FLAG, apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    # The class may already carry the patch if an earlier test loaded through GPTQModel.
    _apply_compat_patch_for_test(model)
    assert getattr(type(model), _PATCH_FLAG, False) is True
    # Idempotent.
    assert not apply_ricursive_generate_patch(model)

    input_ids = torch.randint(0, 256, (1, 6))
    cached = model.generate(input_ids, max_new_tokens=8, do_sample=False, use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=8, do_sample=False, use_cache=False)

    assert cached.shape == (1, 14)
    assert torch.equal(cached, uncached)


def test_ricursive_definition_installs_generate_patch():
    from gptqmodel.utils.ricursive_patch import _PATCH_FLAG

    class _FakeRemote:
        device = torch.device("cpu")

        def prepare_inputs_for_generation(self, *args, **kwargs):  # pragma: no cover - replaced
            raise AssertionError("original must not be used")

    qmodel = object.__new__(RicursiveQModel)
    model = _FakeRemote()
    assert qmodel.after_model_load(model) is model
    assert getattr(type(model), _PATCH_FLAG, False) is True


def test_generate_patch_stands_down_for_native_fix():
    """A checkpoint that already ships the transformers-5 fixes must not be patched.

    ``modeling_ricursive.py`` sets ``_ricursive_tf5_native_fix`` once the generate,
    causal-mask and sliding-window fixes are inline. Re-applying the runtime
    monkeypatch on top would be redundant and would replace the shipped methods.
    """

    from gptqmodel.utils.ricursive_patch import _PATCH_FLAG, apply_ricursive_generate_patch

    class _NativeFixed:
        device = torch.device("cpu")
        _ricursive_tf5_native_fix = "2026-11-09.1"

        def prepare_inputs_for_generation(self, *args, **kwargs):  # pragma: no cover - untouched
            return {"input_ids": None}

    model = _NativeFixed()
    assert not apply_ricursive_generate_patch(model)
    assert not getattr(_NativeFixed, _PATCH_FLAG, False)


def test_cache_compat_shim_provides_max_cache_shape():
    """transformers 5 dropped ``get_max_cache_shape``; the shim reports "unknown"."""

    from transformers.cache_utils import DynamicCache

    from gptqmodel.utils.ricursive_patch import _ensure_cache_compat

    cache = DynamicCache()
    assert not hasattr(cache, "get_max_cache_shape")
    assert _ensure_cache_compat(cache) is cache
    # An empty cache has no layers, so 5.x `get_max_length()` raises; treat as unknown.
    assert cache.get_max_cache_shape() is None

    class _Native:
        def get_max_cache_shape(self):
            return 42

    native = _Native()
    assert _ensure_cache_compat(native) is native
    assert native.get_max_cache_shape() == 42


def test_ricursive_generate_with_padding_matches_unpadded(tmp_path: Path):
    """A padded batch must decode identically to the same row run alone.

    Padding is the only path that builds an explicit causal mask. The published
    mask builder seeds every entry with ``finfo.min`` and skips the triangular fill
    when ``sequence_length == 1``, so each decode step came out fully masked and a
    batched row collapsed to a constant token.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    from gptqmodel.utils.ricursive_patch import _MASK_PATCH_FLAG, apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    _apply_compat_patch_for_test(model)
    assert getattr(type(model.model), _MASK_PATCH_FLAG, False) is True
    assert not apply_ricursive_generate_patch(model)

    model.generation_config.pad_token_id = 0
    a = torch.tensor([[5, 6, 7, 8, 9, 10]])
    single = model.generate(a, max_new_tokens=6, do_sample=False)

    # Left padding: the short row's real tokens start at a higher index.
    left_ids = torch.cat([a, torch.tensor([[0, 0, 11, 12, 13, 14]])], 0)
    left_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    left = model.generate(left_ids, attention_mask=left_mask, max_new_tokens=6, do_sample=False)
    assert left.shape == (2, 12)
    assert torch.equal(left[0], single[0])

    # Right padding: the short row's tokens are trailing-padded instead.
    right_ids = torch.cat([a, torch.tensor([[11, 12, 13, 14, 0, 0]])], 0)
    right_mask = torch.ones(2, 6, dtype=torch.long)
    right_mask[1, 4:] = 0
    right = model.generate(right_ids, attention_mask=right_mask, max_new_tokens=6, do_sample=False)
    assert torch.equal(right[0], single[0])


def test_sliding_window_mask_fast_path_matches_upstream(tmp_path: Path):
    """The decode fast path must be a no-op relative to the upstream mask.

    When ``kv_len <= sliding_window`` the window admits every key, so upstream either
    returns an all-True mask or leaves the padding mask untouched. The fast path skips
    building that tensor entirely, so the two must agree.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    from gptqmodel.utils.ricursive_patch import _WINDOW_PATCH_FLAG

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    native_fix = bool(getattr(type(model), "_ricursive_tf5_native_fix", False))
    _apply_compat_patch_for_test(model)

    attn = model.model.layers[0].self_attn
    cls = type(attn)
    assert getattr(cls, _WINDOW_PATCH_FLAG, False) is True

    patched = cls._apply_sliding_window_mask
    original = patched._gptqmodel_original
    window = attn.config.sliding_window

    # Decode inside the window: older upstream builds an all-True mask; the
    # current native fix and our fast path both return None.
    cache_position = torch.tensor([7])
    upstream = original(attn, None, 1, 8, torch.float32, torch.device("cpu"), cache_position)
    if native_fix:
        assert upstream is None
    else:
        assert upstream.dtype == torch.bool
        assert bool(upstream.all()), "precondition: window must admit every key here"
    assert patched(attn, None, 1, 8, torch.float32, torch.device("cpu"), cache_position) is None

    # Padded decode: upstream leaves the float mask untouched, the fast path returns it.
    padding = torch.zeros(2, 1, 1, 8, dtype=torch.float32)
    upstream_pad = original(attn, padding, 1, 8, torch.float32, torch.device("cpu"), cache_position)
    assert torch.equal(upstream_pad, padding)
    assert patched(attn, padding, 1, 8, torch.float32, torch.device("cpu"), cache_position) is padding

    # Past the window the real mask is needed again: both paths must agree exactly.
    cache_position = torch.tensor([window + 5])
    kv_len = window + 6
    upstream_win = original(attn, None, 1, kv_len, torch.float32, torch.device("cpu"), cache_position)
    patched_win = patched(attn, None, 1, kv_len, torch.float32, torch.device("cpu"), cache_position)
    assert patched_win is not None, "window must still be applied past sliding_window"
    assert torch.equal(upstream_win, patched_win)


def test_sliding_window_fast_path_preserves_generation(tmp_path: Path):
    """Generation must be unchanged with the fast path enabled."""

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    _apply_compat_patch_for_test(model)
    model.generation_config.pad_token_id = 0

    cls = type(model.model.layers[0].self_attn)
    patched = cls._apply_sliding_window_mask
    original = patched._gptqmodel_original

    a = torch.tensor([[5, 6, 7, 8, 9, 10]])
    fast = model.generate(a, max_new_tokens=8, do_sample=False)

    cls._apply_sliding_window_mask = original
    try:
        slow = model.generate(a, max_new_tokens=8, do_sample=False)
    finally:
        cls._apply_sliding_window_mask = patched

    assert torch.equal(fast, slow)


def test_sliding_window_mask_trims_evicted_keys(tmp_path: Path):
    """Sliding layers evict keys, so the shared mask can be wider than the layer's KV.

    transformers 5 builds a hybrid cache: sliding layers keep only ``sliding_window``
    keys while full layers keep growing. One mask is built per forward, so a sliding
    layer must trim it to the keys it actually holds -- otherwise SDPA rejects the
    shape once decoding passes the window.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    _apply_compat_patch_for_test(model)

    attn = model.model.layers[0].self_attn
    cls = type(attn)
    patched = cls._apply_sliding_window_mask
    original = patched._gptqmodel_original
    window = attn.config.sliding_window
    cpu = torch.device("cpu")

    # One key more than the window: the layer holds `window` keys while the mask
    # covers the full span. Every surviving key is inside the window, so the result
    # is the trimmed mask unchanged.
    mask = torch.zeros(2, 1, 1, window + 1, dtype=torch.float32)
    out = patched(attn, mask, 1, window, torch.float32, cpu, torch.tensor([window]))
    assert out is not None
    assert out.shape[-1] == window, "mask must be trimmed to the live key count"
    assert torch.equal(out, mask[..., -window:])

    # Past the window the mask really binds: no trim, and the upstream result stands.
    kv_len = window + 6
    wide = torch.zeros(2, 1, 1, kv_len, dtype=torch.float32)
    cache_position = torch.tensor([kv_len - 1])
    out_wide = patched(attn, wide, 1, kv_len, torch.float32, cpu, cache_position)
    assert out_wide is not None and out_wide.shape[-1] == kv_len
    assert torch.equal(out_wide, original(attn, wide, 1, kv_len, torch.float32, cpu, cache_position))


def test_sliding_window_mask_aligns_to_written_positions(tmp_path: Path):
    """A capacity-sized mask must align to ``[total_kv - kv_len, total_kv)``.

    ``[-kv_len:]`` is only correct when the mask length equals the number of
    written keys. With a preallocated cache the shared mask covers the whole
    capacity, so the tail slice selects slots that were never written and the
    layer attends to the wrong positions.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    apply_ricursive_generate_patch(model)

    attn = model.model.layers[0].self_attn
    cls = type(attn)
    patched = cls._apply_sliding_window_mask
    window = attn.config.sliding_window
    capacity = window * 4
    total_kv = capacity - 2  # written tokens, fewer than the allocated capacity
    cpu = torch.device("cpu")

    # Distinct value per position so a wrong slice cannot compare equal by luck.
    mask = (
        torch.arange(capacity, dtype=torch.float32)
        .view(1, 1, 1, capacity)
        .expand(2, 1, 1, capacity)
        .clone()
    )
    out = patched(
        attn,
        mask,
        1,
        window,
        torch.float32,
        cpu,
        torch.tensor([total_kv - 1]),
        total_kv=total_kv,
    )

    assert out is not None and out.shape[-1] == window
    expected = mask[..., total_kv - window : total_kv]
    assert torch.equal(out, expected), "mask must align to the written key span"
    assert not torch.equal(out, mask[..., -window:]), "the unwritten tail must not be used"


def test_static_cache_decode_matches_full_forward(tmp_path: Path):
    """A preallocated cache must append, not overwrite, when decoding.

    ``past_key_value.update()`` needs ``cache_position``; a transformers 4.x
    ``StaticCache`` otherwise falls back to ``arange(q_len)`` and the first
    decode step rewrites the prompt slots.
    """

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import StaticCache

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    apply_ricursive_generate_patch(model)

    total = 10
    ids = torch.randint(0, model.config.vocab_size, (1, total))
    with torch.no_grad():
        full = model(input_ids=ids).logits

    cache = StaticCache(config=model.config, max_cache_len=16)
    with torch.no_grad():
        model(
            input_ids=ids[:, :6],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(6),
        )
        errors = []
        for pos in range(6, total):
            out = model(
                input_ids=ids[:, pos : pos + 1],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([pos]),
            )
            errors.append((out.logits[:, -1] - full[:, pos]).abs().max().item())

    assert max(errors) < 2e-4, f"decode diverged from a full forward: {max(errors):.3e}"


def test_prepare_inputs_moves_position_ids_to_model_device():
    """A CPU prompt must not leave `position_ids` on CPU.

    transformers derives `position_ids` from the caller's tensors before the hook
    runs, so a CPU `input_ids` yields CPU positions. RoPE then multiplies them
    against CUDA frequencies and fails with a "mat2 is on cpu" bmm error.
    """

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the device move")

    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    class _Dummy:
        device = torch.device("cuda:0")

    input_ids = torch.arange(6).unsqueeze(0)  # CPU on purpose
    position_ids = torch.arange(6).unsqueeze(0)  # CPU, as transformers would supply

    out = _prepare_inputs_for_generation(
        _Dummy(),
        input_ids,
        past_key_values=None,
        attention_mask=None,
        position_ids=position_ids,
        cache_position=None,
        use_cache=True,
    )

    assert out["input_ids"].device.type == "cuda"
    assert out["position_ids"].device.type == "cuda", "position_ids must follow the inputs"
    assert out["cache_position"].device.type == "cuda"


def _load_tiny_with_originals(tmp_path: Path):
    """Load the tiny fixture and return the *reference* forwards.

    The custom-code class is shared across a pytest session, so an earlier test may
    already have installed the decode patches. ``_gptqmodel_original`` keeps the
    upstream implementation reachable, which is what the A/B assertions need.
    """
    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()

    attn = model.model.layers[0].self_attn
    norm = model.model.layers[0].input_layernorm
    attn_cls, norm_cls = type(attn), type(norm)
    orig_attn = getattr(attn_cls.forward, "_gptqmodel_original", attn_cls.forward)
    orig_norm = getattr(norm_cls.forward, "_gptqmodel_original", norm_cls.forward)
    return model, attn, norm, orig_attn, orig_norm


def test_decode_patches_install_and_stay_idempotent(tmp_path: Path):
    from gptqmodel.utils.ricursive_patch import (
        _GQA_PATCH_FLAG,
        _NORM_PATCH_FLAG,
        apply_ricursive_generate_patch,
    )

    model, _attn, _norm, _orig_attn, _orig_norm = _load_tiny_with_originals(tmp_path)
    attn_cls, norm_cls = type(model.model.layers[0].self_attn), type(model.model.layers[0].input_layernorm)

    apply_ricursive_generate_patch(model)
    assert getattr(attn_cls, _GQA_PATCH_FLAG, False) is True
    assert getattr(norm_cls, _NORM_PATCH_FLAG, False) is True
    # The reference implementations stay reachable for A/B tests.
    assert callable(attn_cls.forward._gptqmodel_original)
    assert callable(norm_cls.forward._gptqmodel_original)
    assert attn_cls.forward._gptqmodel_original is not attn_cls.forward
    assert norm_cls.forward._gptqmodel_original is not norm_cls.forward

    patched_attn, patched_norm = attn_cls.forward, norm_cls.forward
    apply_ricursive_generate_patch(model)
    assert attn_cls.forward is patched_attn
    assert norm_cls.forward is patched_norm


def test_chunked_mlp_is_bit_identical_and_idempotent(monkeypatch):
    """Splitting the MLP over rows must only bound the peak, never change the result.

    Ripple's MLP expands 5120 -> 40960, so a long-prompt prefill materialises three
    ``tokens x 40960`` tensors at once. Chunking the row dimension is what keeps a
    33 GiB 4-bit model inside a 48 GiB card at eval batch sizes, so it has to be
    exactly equivalent to the unchunked forward.
    """
    import types

    from gptqmodel.utils.ricursive_patch import _MLP_CHUNK_PATCH_FLAG, _patch_chunked_mlp

    class _MLP(torch.nn.Module):
        def __init__(self, hidden: int = 128, intermediate: int = 512, clamp: float = 10.0):
            super().__init__()
            self.clamp_threshold = clamp
            self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.clamp_threshold:
                gate = gate.clamp(min=None, max=self.clamp_threshold)
                up = up.clamp(min=-self.clamp_threshold, max=self.clamp_threshold)
            return self.down_proj(torch.nn.functional.silu(gate) * up)

    monkeypatch.setenv("RIPPLE_MLP_CHUNK", "64")
    torch.manual_seed(0)
    mlp = _MLP().to(torch.bfloat16)
    holder = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=mlp)])
    )

    x = torch.randn(3, 200, 128, dtype=torch.bfloat16)
    reference = mlp(x).clone()

    assert _patch_chunked_mlp(holder) is True
    assert getattr(type(mlp), _MLP_CHUNK_PATCH_FLAG, False) is True

    chunked = mlp(x)
    assert chunked.dtype == x.dtype
    assert torch.equal(chunked, reference), "chunked MLP diverged from the row-wise reference"

    # 2-D inputs (already flattened by some callers) take the same path.
    assert torch.equal(mlp(x.reshape(-1, 128)), reference.reshape(-1, 128))
    # A prefill shorter than the chunk budget must not be split at all.
    assert torch.equal(mlp(x[:, :10]), reference[:, :10])
    # Installing twice must not wrap the wrapper.
    patched = type(mlp).forward
    assert _patch_chunked_mlp(holder) is False
    assert type(mlp).forward is patched


def test_gqa_attention_is_bit_identical_to_repeat_kv(tmp_path: Path):
    """``enable_gqa`` must reproduce the expanded-KV path exactly."""
    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, attn, _norm, orig_attn, _orig_norm = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    assert attn.num_kv_groups > 1, "fixture must exercise GQA"
    for q_len in (1, 4):
        torch.manual_seed(1234 + q_len)
        hidden = torch.randn(2, q_len, model.config.hidden_size) * 0.05
        pos = torch.arange(q_len).unsqueeze(0)
        with torch.no_grad():
            ref = orig_attn(attn, hidden, None, pos, None, False, False, pos)[0]
            got = attn(hidden, None, pos, None, False, False, pos)[0]
        assert got.dtype == hidden.dtype
        assert torch.equal(ref, got), f"q_len={q_len} diverged from the reference path"


def test_gqa_patch_never_materialises_kv(tmp_path: Path, monkeypatch):
    """The patched forward must not call ``repeat_kv`` at all."""
    import sys

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, attn, _norm, _orig_attn, _orig_norm = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    module = sys.modules[type(attn).__module__]
    calls = []

    def _spy(hidden_states, n_rep):
        calls.append(n_rep)
        raise AssertionError("repeat_kv must not be used by the patched attention")

    monkeypatch.setattr(module, "repeat_kv", _spy)
    hidden = torch.randn(2, 3, model.config.hidden_size) * 0.05
    pos = torch.arange(3).unsqueeze(0)
    with torch.no_grad():
        out = attn(hidden, None, pos, None, False, False, pos)[0]

    assert calls == []
    assert out.shape == (2, 3, model.config.hidden_size)


def test_fused_rms_norm_matches_reference(tmp_path: Path):
    """The fused norm must agree with the reference within float tolerance."""
    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, _attn, norm, _orig_attn, orig_norm = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    torch.manual_seed(7)
    hidden = torch.randn(2, 5, model.config.hidden_size) * 0.5
    with torch.no_grad():
        ref = orig_norm(norm, hidden)
        got = norm(hidden)

    assert got.dtype == hidden.dtype
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5), (ref - got).abs().max()


def test_multi_token_block_on_warm_cache_matches_full_forward(tmp_path: Path):
    """A multi-token verify block on a warm cache must match the full forward.

    SDPA's implicit causal mask is aligned top-left, so it is only correct when the
    query block is the whole sequence. Speculative decoders verify a block that sits
    at the *end* of a warm cache; using the implicit mask there hides every earlier
    key and corrupts the block. The patch must build the end-anchored mask instead.
    """
    from transformers import DynamicCache

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, *_ = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    torch.manual_seed(11)
    ids = torch.randint(0, model.config.vocab_size, (1, 24))
    split = 12
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        cache = DynamicCache(config=model.config)
        cache.activate_past_recording()
        model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True)
        block = model(input_ids=ids[:, split:], past_key_values=cache, use_cache=True).logits

    assert block.shape[1] == ids.shape[1] - split
    torch.testing.assert_close(block[0], full[0, split:], atol=1e-5, rtol=1e-4)


def test_logits_to_keep_projects_only_the_kept_positions(tmp_path: Path):
    """``logits_to_keep`` must return the trailing row(s) and match the full logits."""
    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, *_ = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    torch.manual_seed(13)
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    with torch.no_grad():
        full = model(input_ids=ids).logits
        kept = model(input_ids=ids, logits_to_keep=1).logits

    assert kept.shape == (2, 1, model.config.vocab_size)
    torch.testing.assert_close(kept[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)


def test_static_cache_uses_actual_positions_not_capacity(tmp_path: Path):
    """A preallocated cache must not be treated as occupied context.

    A six-token prompt in a 16-slot ``StaticCache`` used to get query positions
    10-15 because the block mask used ``k.shape[2]`` (capacity). Changing the last
    token then changed earlier tokens' outputs in the full-attention layers, which
    is impossible under causal attention. The mask must use the real cache length.
    """
    from transformers import StaticCache

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, *_ = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)
    cfg = model.config

    torch.manual_seed(17)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    changed = ids.clone()
    changed[0, -1] = (int(changed[0, -1]) + 7) % cfg.vocab_size

    with torch.no_grad():
        a = model(
            input_ids=ids,
            past_key_values=StaticCache(config=cfg, max_cache_len=16),
            use_cache=True,
        ).logits
        b = model(
            input_ids=changed,
            past_key_values=StaticCache(config=cfg, max_cache_len=16),
            use_cache=True,
        ).logits

    # Only the last token changed, so every earlier position must be identical.
    assert torch.equal(a[:, :5], b[:, :5]), (a[:, :5] - b[:, :5]).abs().max()

    # A multi-token verify block on the preallocated cache must match a full forward.
    full_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    cache = StaticCache(config=cfg, max_cache_len=16)
    with torch.no_grad():
        full = model(input_ids=full_ids, use_cache=False).logits
        model(input_ids=full_ids[:, :6], past_key_values=cache, use_cache=True)
        block = model(
            input_ids=full_ids[:, 6:],
            past_key_values=cache,
            cache_position=torch.arange(6, 10),
            use_cache=True,
        ).logits

    torch.testing.assert_close(block[0], full[0, 6:], atol=1e-5, rtol=1e-4)


def test_labels_with_logits_to_keep_keeps_finite_loss(tmp_path: Path):
    """A labeled call must keep the full sequence so the shifted loss is defined.

    Trimming both logits and labels to one position leaves the causal shift empty
    and returns NaN. The patch must skip the LM-head input slice when labels are
    present, and the loss must match the full (untrimmed) forward.
    """
    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    model, *_ = _load_tiny_with_originals(tmp_path)
    apply_ricursive_generate_patch(model)

    torch.manual_seed(19)
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(input_ids=ids, labels=ids)
        kept = model(input_ids=ids, labels=ids, logits_to_keep=1)
        # `labels` may also arrive as positional argument 5.
        kept_pos = model(ids, None, None, None, None, ids, logits_to_keep=1)

    assert torch.isfinite(kept.loss)
    torch.testing.assert_close(kept.loss, full.loss, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(kept_pos.loss)
    torch.testing.assert_close(kept_pos.loss, full.loss, atol=1e-6, rtol=1e-6)
    assert kept.logits.shape == (1, 1, model.config.vocab_size)


def test_patch_classes_are_disjoint_and_documented():
    """The upstream/local patch split is a contract, not a comment.

    ``ricursiveAI/Ripple-65B-Preview`` may only receive accuracy and memory
    fixes. Fast paths are bound to Ripple's 80-query/8-KV head split and to
    this Torch build's ``enable_gqa`` / fused ``rms_norm``, so they must never
    be proposed upstream. Guard the split in CI so a new patch cannot land in
    the wrong class by accident, and so the markers that document it cannot be
    dropped.
    """
    from gptqmodel.utils import ricursive_patch as rp

    upstream = set(rp.UPSTREAM_ELIGIBLE_PATCHES)
    local = set(rp.LOCAL_ONLY_FAST_PATHS)

    assert upstream, "the upstream-eligible set must not be empty"
    assert local, "the local-only fast-path set must not be empty"
    assert not (upstream & local), "a patch cannot be both upstream-eligible and local-only"

    for name in local:
        fn = getattr(rp, name, None)
        assert callable(fn), f"{name} is listed in LOCAL_ONLY_FAST_PATHS but is missing"
        doc = fn.__doc__ or ""
        assert "LOCAL-ONLY FAST PATH" in doc, f"{name} must carry the local-only marker"
        assert "never upstream" in doc.lower(), f"{name} must state it is never upstream"

    for name in upstream:
        fn = getattr(rp, name, None)
        if not name.startswith("_patch_"):
            # Implementation helpers (``_update_causal_mask`` and friends) are
            # nested inside their installer and carry no separate marker.
            continue
        assert callable(fn), f"{name} is listed in UPSTREAM_ELIGIBLE_PATCHES but is missing"
        doc = fn.__doc__ or ""
        assert "upstream-eligible" in doc, f"{name} must carry the upstream marker"


@pytest.fixture(scope="module")
def cached_generation_model(tmp_path_factory):
    from types import MethodType

    from transformers import AutoModelForCausalLM

    from gptqmodel.utils.ricursive_patch import _prepare_inputs_for_generation

    ref_dir = _reference_dir()
    if ref_dir is None:
        pytest.skip("Ripple custom code unavailable (set GPTQMODEL_RIPPLE_REF to run offline)")
    model_dir = tmp_path_factory.mktemp("ripple_cached_generation")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        _build_tiny_fixture(ref_dir, model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).eval()
    # Exercise GPT-QModel's compatibility method even when the reference source
    # declares native support and the normal patch installer stands down.
    model.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, model)
    return model


def _cache_test_positions(mask):
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)


@pytest.mark.parametrize("cache_type", ["dynamic", "static"])
@pytest.mark.parametrize(
    "past_length,new_length,full_prompt,capacity,visible_tail,explicit_cache",
    [
        pytest.param(4, 2, True, 10, True, False, id="review-visible-capacity"),
        pytest.param(10, 2, False, 16, False, False, id="review-short-padded-block"),
        pytest.param(4, 2, True, 10, False, False, id="zero-exact-capacity"),
        pytest.param(4, 2, True, 12, False, False, id="zero-long-capacity"),
        pytest.param(4, 2, True, 6, True, False, id="full-prompt"),
        pytest.param(4, 4, False, 8, True, False, id="equal-length-block"),
        pytest.param(4, 1, False, 10, False, False, id="single-token-padded"),
        pytest.param(4, 6, False, 10, True, True, id="long-block-explicit-cache"),
        pytest.param(4, 6, False, 16, False, True, id="long-block-padded"),
        pytest.param(4, 2, True, 10, True, True, id="full-prompt-explicit-cache"),
        pytest.param(0, 6, True, 10, False, False, id="prefill-padded-capacity"),
    ],
)
@pytest.mark.parametrize("left_padding", [0, 2])
@torch.inference_mode()
def test_generate_patch_cached_logits_match_full_forward(
    cached_generation_model, cache_type, past_length, new_length, full_prompt, capacity,
    visible_tail, explicit_cache, left_padding,
):
    from transformers.cache_utils import DynamicCache, StaticCache

    total = past_length + new_length
    tokens = torch.arange(3, 3 + total).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, capacity, dtype=torch.long)
    mask[0, :left_padding] = 0
    if not visible_tail:
        mask[:, total:] = 0
    cache = (
        StaticCache(config=cached_generation_model.config, max_cache_len=capacity)
        if cache_type == "static" else DynamicCache()
    )
    if past_length:
        cached_generation_model(
            input_ids=tokens[:, :past_length],
            attention_mask=mask[:, :past_length],
            position_ids=_cache_test_positions(mask[:, :past_length]),
            past_key_values=cache,
            use_cache=True,
        )
    prepared = cached_generation_model.prepare_inputs_for_generation(
        tokens if full_prompt else tokens[:, past_length:],
        past_key_values=cache,
        attention_mask=mask,
        cache_position=torch.arange(past_length, total) if explicit_cache else None,
        use_cache=True,
    )
    # Execute even on the buggy source to capture the cache and numerical regression.
    prepared["logits_to_keep"] = 0
    actual = cached_generation_model(**prepared).logits
    expected = cached_generation_model(
        input_ids=tokens,
        attention_mask=mask[:, :total],
        position_ids=_cache_test_positions(mask[:, :total]),
        use_cache=False,
    ).logits[:, past_length:]
    error = (actual[:, -new_length:] - expected).abs().max().item()
    detail = (
        f"cache_length={int(cache.get_seq_length())}, "
        f"positions={prepared['position_ids'].tolist()}, max_logit_error={error:.8g}"
    )
    assert int(cache.get_seq_length()) == total, detail
    assert torch.equal(prepared["input_ids"], tokens[:, past_length:]), detail
    assert torch.equal(prepared["cache_position"], torch.arange(past_length, total)), detail
    assert torch.equal(prepared["position_ids"], _cache_test_positions(mask)[:, past_length:total]), detail
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5, msg=detail)


@pytest.mark.parametrize("provided_positions", ["full", "capacity", "new"])
@torch.inference_mode()
def test_generate_patch_explicit_position_ids(cached_generation_model, provided_positions):
    tokens = torch.arange(3, 9).unsqueeze(0)
    cache = cached_generation_model(input_ids=tokens[:, :4], use_cache=True).past_key_values
    # Custom caller positions must be preserved, including when already sliced.
    position_ids = {
        "full": torch.arange(20, 26).unsqueeze(0),
        "capacity": torch.arange(20, 30).unsqueeze(0),
        "new": torch.tensor([[24, 25]]),
    }[provided_positions]
    prepared = cached_generation_model.prepare_inputs_for_generation(
        tokens,
        past_key_values=cache,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        position_ids=position_ids,
    )
    assert prepared["input_ids"].tolist() == [[7, 8]]
    assert prepared["position_ids"].tolist() == [[24, 25]]


@torch.inference_mode()
def test_generate_patch_left_padded_batch(cached_generation_model):
    tokens = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0, "eos_token_id": None}
    batched = cached_generation_model.generate(tokens, attention_mask=mask, **options)
    for row, length in enumerate([2, 4]):
        unpadded = tokens[row:row + 1, -length:]
        single = cached_generation_model.generate(unpadded, attention_mask=torch.ones_like(unpadded), **options)
        assert torch.equal(batched[row, -3:], single[0, -3:])
