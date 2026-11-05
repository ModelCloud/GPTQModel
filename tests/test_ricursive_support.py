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
        "self_attn",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "mlp",
        "post_feedforward_layernorm",
    }

    attn = layer["self_attn"]
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
    apply_ricursive_generate_patch(model)
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
        _ricursive_tf5_native_fix = "2026-11-05"

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
    apply_ricursive_generate_patch(model)
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

    from gptqmodel.utils.ricursive_patch import (
        _WINDOW_PATCH_FLAG,
        apply_ricursive_generate_patch,
    )

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    apply_ricursive_generate_patch(model)

    attn = model.model.layers[0].self_attn
    cls = type(attn)
    assert getattr(cls, _WINDOW_PATCH_FLAG, False) is True

    patched = cls._apply_sliding_window_mask
    original = patched._gptqmodel_original
    window = attn.config.sliding_window

    # Decode step well inside the window, unpadded: upstream builds an all-True mask,
    # the fast path returns None so SDPA keeps its no-mask path.
    cache_position = torch.tensor([7])
    upstream = original(attn, None, 1, 8, torch.float32, torch.device("cpu"), cache_position)
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

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    apply_ricursive_generate_patch(model)
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

    from gptqmodel.utils.ricursive_patch import apply_ricursive_generate_patch

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(ref_dir, model_dir)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True).eval()
    apply_ricursive_generate_patch(model)

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
