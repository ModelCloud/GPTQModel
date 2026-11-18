# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Offline CPU contracts for package-owned Ripple loading and quantization."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from ricursive_test_utils import make_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models import auto
from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


# Modules the definition must quantize in every decoder layer.
_QUANTIZED_PER_LAYER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.g_proj",
    "self_attn.v_proj",
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


def test_ricursive_registry_selects_definition(tmp_path):
    make_model().config.save_pretrained(tmp_path)
    assert auto.MODEL_MAP["ricursive"] is RicursiveQModel
    for trust in (False, True):
        assert auto.check_and_get_model_definition(tmp_path, trust_remote_code=trust) is RicursiveQModel


def test_ricursive_definition_contract():
    assert not RicursiveQModel.require_trust_remote_code
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
    # Gated attention: q/k/g read the input norm, but g_proj's sigmoid gate is
    # applied to the attention output, so it may not be `o_proj`'s fold source.
    assert "g_proj:0:in=x:nosrc" in attn
    for name in ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:1:v", "o_proj:2:o"):
        assert name in attn
    # `use_valres` makes v_proj read a lerp of the value residual and the norm
    # output, so it must be its own stage and must not carry the `in=x` tag
    # that groups q/k/g onto one Hessian.
    assert not any(token.startswith("v_proj:") and "in=" in token for token in attn)
    # The lerp means no single v_proj scale can compensate an input-norm fold.
    assert "input_layernorm:!:norm:nosrc" in layer["input_layernorm"]

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
    """Always-run check: the declared groups really do share one tensor in a live forward.

    Also proves the `v_proj` exclusion is required rather than merely conservative --
    without it the probe would report `v_proj` as a mismatch.
    """



    from gptqmodel.models.shared_input import probe_shared_inputs

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    _build_tiny_fixture(model_dir)

    from gptqmodel.models.shared_input import build_shared_input_plan

    model_config = SimpleNamespace(model_type="ricursive", num_hidden_layers=5)
    quantize_config = SimpleNamespace(dynamic=None)
    template_plan = build_shared_input_plan(
        RicursiveQModel.module_tree,
        RicursiveQModel.simple_layer_modules(model_config, quantize_config),
        explicit_tags=True,
    )
    assert template_plan.shared_groups

    model = _load_tiny_model(str(model_dir)).eval()
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


def _build_tiny_fixture(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(
        vocab_size=256, hidden_size=64, intermediate_size=96, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=128, sliding_window=16, rms_norm_eps=1e-6,
        rope_theta=10000.0, use_gated_attention=True, use_valres=True,
        ffn_clamp_threshold=20.0, tie_word_embeddings=False,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
    )
    model.save_pretrained(model_dir)
    _build_tokenizer(model_dir)


def _load_tiny_model(model_dir, **kwargs):
    # The isolated class executes the package files; no checkpoint Python.
    return type(make_model()).from_pretrained(model_dir, **kwargs)


def test_tiny_ricursive_gptq_quantization_smoke(tmp_path: Path):

    torch.manual_seed(0)
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    _build_tiny_fixture(model_dir)


    native = _load_tiny_model(str(model_dir)).eval()

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        backend=BACKEND.TORCH,
    )
    calibration = [{"input_ids": torch.randint(0, 256, (64,))} for _ in range(8)]
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    model.save(str(quantized_dir))

    quantized = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
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


def test_config_owned_chunked_mlp_is_bit_identical():
    mlp = make_model(hidden_size=128, intermediate_size=512,
                     ffn_clamp_threshold=10.0, mlp_chunk_size=64).model.layers[0].mlp.to(torch.bfloat16)
    unchunked = make_model(hidden_size=128, intermediate_size=512,
                           ffn_clamp_threshold=10.0, mlp_chunk_size=0).model.layers[0].mlp.to(torch.bfloat16)
    unchunked.load_state_dict(mlp.state_dict())
    x = torch.randn(3, 200, 128, dtype=torch.bfloat16)
    reference = unchunked(x)
    rows = []
    hook = mlp.gate_proj.register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
    try:
        assert torch.equal(mlp(x), reference)
    finally:
        hook.remove()
    assert len(rows) > 1 and sum(rows) == 600 and max(rows) <= 64
    assert torch.equal(mlp(x.reshape(-1, 128)), reference.reshape(-1, 128))
    assert torch.equal(mlp(x[:, :10]), reference[:, :10])


