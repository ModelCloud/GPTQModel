# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Adapters attach only to the modules their PEFT config targets (issue #4210).

Before this fix one global adapter was passed to every quantized module, so a
standard partial PEFT LoRA (``q_proj``/``v_proj``) asserted on ``k_proj`` and
other uncovered modules while loading. Adapter weights for modules that stay
dense (an unquantized ``lm_head``) were never applied at all.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from gptqmodel import CalibrationConfig
from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.adapter.targeting import module_matches_targets
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.model import (
    apply_dense_module_adapters,
    create_quant_module,
    gptqmodel_post_init,
)


CPU = torch.device("cpu")
IN_FEATURES = 4
OUT_FEATURES = 3


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


def _write_adapter(
    directory,
    *,
    target_modules,
    module_keys,
    rank=2,
    exclude_modules=None,
    in_features=IN_FEATURES,
    out_features=OUT_FEATURES,
):
    """Write a PEFT adapter whose weights cover ``module_keys``."""

    directory.mkdir(parents=True, exist_ok=True)
    config = {"r": rank, "lora_alpha": rank, "target_modules": target_modules}
    if exclude_modules is not None:
        config["exclude_modules"] = exclude_modules
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = {}
    for module_key in module_keys:
        weights[f"base_model.model.{module_key}.lora_A.weight"] = torch.randn(
            rank, in_features
        )
        weights[f"base_model.model.{module_key}.lora_B.weight"] = torch.randn(
            out_features, rank
        )
    save_file(weights, str(directory / "adapter_model.safetensors"))


def _dense_root(*names):
    root = nn.Module()
    for name in names:
        setattr(root, name, nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    return root


def test_target_list_uses_exact_and_dotted_suffix_matching():
    targets = ["q_proj", "v_proj"]

    assert module_matches_targets("q_proj", targets)
    assert module_matches_targets("model.layers.0.self_attn.q_proj", targets)
    assert module_matches_targets("model.layers.0.self_attn.v_proj", targets)
    # A bare suffix must start at a dotted boundary.
    assert not module_matches_targets("model.layers.0.self_attn.my_q_proj", targets)
    assert not module_matches_targets("model.layers.0.self_attn.k_proj", targets)


def test_target_string_and_re_prefix_are_regular_expressions():
    assert module_matches_targets(
        "model.layers.0.self_attn.q_proj", r".*\.q_proj"
    )
    assert module_matches_targets(
        "model.layers.0.self_attn.q_proj", ["re:.*q_proj$"]
    )
    assert not module_matches_targets(
        "model.layers.0.self_attn.k_proj", ["re:.*q_proj$"]
    )
    # ``re:`` is anchored like PEFT's fullmatch, not a free substring search.
    assert not module_matches_targets(
        "model.layers.0.self_attn.q_proj", ["re:q_proj"]
    )


def test_exclude_modules_wins_over_a_target_match():
    assert not module_matches_targets(
        "model.layers.0.self_attn.q_proj",
        ["q_proj"],
        r".*\.layers\.0\..*",
    )
    assert module_matches_targets(
        "model.layers.1.self_attn.q_proj",
        ["q_proj"],
        r".*\.layers\.0\..*",
    )


def test_create_quant_module_attaches_adapter_only_to_targeted_modules(tmp_path):
    """A partial PEFT LoRA must not be attached to k_proj (the crash source)."""

    _write_adapter(
        tmp_path,
        target_modules=["q_proj", "v_proj"],
        module_keys=["q_proj", "v_proj"],
    )
    adapter = Lora(rank=2, path=str(tmp_path))
    captured = {}

    class DummyQuantLinear(nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            self.bias = None
            captured[kwargs["name"]] = kwargs.get("adapter")

    root = _dense_root("q_proj", "k_proj", "v_proj")
    for name in ("q_proj", "k_proj", "v_proj"):
        create_quant_module(
            name=name,
            linear_cls=DummyQuantLinear,
            bits=4,
            desc_act=False,
            dynamic=None,
            group_size=32,
            module=root,
            submodule=getattr(root, name),
            sym=True,
            device="cpu",
            lm_head_name="lm_head",
            pack_dtype=torch.int32,
            adapter=adapter,
        )

    assert captured["q_proj"] is not None
    assert captured["v_proj"] is not None
    assert captured["k_proj"] is None


def test_targeted_module_without_tensors_gets_no_adapter(tmp_path):
    """A target list may cover a module the export has no tensors for.

    ``layers_to_transform`` and partial exports produce exactly this shape; it
    must skip the module instead of asserting in ``Lora.post_init``.
    """

    _write_adapter(
        tmp_path,
        target_modules=["q_proj", "k_proj", "v_proj"],
        module_keys=["q_proj", "v_proj"],
    )
    adapter = Lora(rank=2, path=str(tmp_path))

    assert adapter.targets_module("k_proj")
    assert not adapter.has_weights("k_proj")
    assert adapter.has_weights("q_proj")
    assert adapter.has_weights("v_proj")


def test_partial_lora_post_init_skips_uncovered_modules(tmp_path):
    """The load path must not assert on modules the adapter does not cover."""

    _write_adapter(
        tmp_path,
        target_modules=["q_proj", "v_proj"],
        module_keys=[
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ],
    )
    adapter = Lora(rank=2, path=str(tmp_path))

    # This is exactly what gptqmodel_post_init does for each quant module: only
    # the modules the adapter targets reach Lora.post_init. k_proj is covered by
    # neither the target list nor the weights file.
    assert adapter.targets_module("model.layers.0.self_attn.q_proj")
    assert not adapter.targets_module("model.layers.0.self_attn.k_proj")
    adapter.post_init("model.layers.0.self_attn.q_proj", CPU)
    assert adapter.lora_A.shape == (IN_FEATURES, 2)
    assert adapter.lora_B.shape == (2, OUT_FEATURES)


def test_dense_lm_head_adapter_is_applied_and_changes_output(tmp_path):
    _write_adapter(
        tmp_path,
        target_modules=["lm_head"],
        module_keys=["lm_head"],
    )
    torch.manual_seed(7)
    root = _dense_root("lm_head")
    x = torch.randn(2, IN_FEATURES)
    base = root.lm_head(x).detach().clone()

    adapter = Lora(rank=2, path=str(tmp_path))
    applied = apply_dense_module_adapters(root, adapter)

    assert applied == 1
    assert root.lm_head._gptqmodel_dense_adapter is not None
    out = root.lm_head(x)
    assert not torch.allclose(out, base)

    # Independent oracle from the serialized source tensors, not the adapter
    # object under test. lora_alpha == r, so the PEFT scale is 1.
    source = load_file(str(tmp_path / "adapter_model.safetensors"))
    lora_a = source["base_model.model.lm_head.lora_A.weight"]
    lora_b = source["base_model.model.lm_head.lora_B.weight"]
    expected = base + (x.double() @ lora_a.double().T) @ lora_b.double().T
    torch.testing.assert_close(out.double(), expected, rtol=1e-6, atol=1e-6)


def test_dense_untargeted_modules_get_no_adapter(tmp_path):
    _write_adapter(
        tmp_path,
        target_modules=["lm_head"],
        module_keys=["lm_head"],
    )
    root = _dense_root("lm_head", "proj")
    adapter = Lora(rank=2, path=str(tmp_path))

    assert apply_dense_module_adapters(root, adapter) == 1
    assert root.lm_head._gptqmodel_dense_adapter is not None
    assert getattr(root.proj, "_gptqmodel_dense_adapter", None) is None
    assert not root.proj._forward_hooks


def test_gptqmodel_post_init_applies_dense_adapter(tmp_path):
    _write_adapter(
        tmp_path,
        target_modules=["lm_head"],
        module_keys=["lm_head"],
    )
    root = _dense_root("lm_head")
    x = torch.randn(1, IN_FEATURES)
    base = root.lm_head(x).detach().clone()
    quantize_config = QuantizeConfig(bits=4, group_size=32, adapter=Lora(rank=2, path=str(tmp_path)))

    gptqmodel_post_init(root, use_act_order=False, quantize_config=quantize_config)

    assert root.lm_head._gptqmodel_dense_adapter is not None
    assert not torch.allclose(root.lm_head(x), base)


def test_eora_output_path_without_config_keeps_attach_all_behavior(tmp_path):
    """A not-yet-saved EoRA output directory must not fail target resolution."""

    adapter = Lora(rank=2, path=str(tmp_path / "eora-output-not-created-yet"))

    assert adapter.targets_module("model.layers.0.self_attn.q_proj")
    assert adapter.targets_module("model.lm_head")


def test_adapter_without_config_keeps_attach_all_behavior():
    """Merged/EoRA adapters have no target list and must keep working."""

    adapter = Lora(rank=2)
    adapter.lora_A = torch.randn(IN_FEATURES, 2)
    adapter.lora_B = torch.randn(2, OUT_FEATURES)

    assert adapter.targets_module("model.layers.0.self_attn.q_proj")
    assert adapter.targets_module("model.lm_head")


def _write_tiny_llama_tokenizer(model_dir):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    texts = [
        "tiny llama calibration sample one with enough tokens to survive filtering",
        "tiny llama calibration sample two with enough tokens to survive filtering",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(texts, trainer=WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    ))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)
    return fast, texts


def test_partial_lora_and_dense_lm_head_load_end_to_end(tmp_path):
    """Quantize a tiny Llama, then load it with the two adapters from #4210."""

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel import BACKEND, GPTQModel

    torch.set_num_threads(4)
    hidden = 64
    vocab = 128
    native = tmp_path / "native"
    quantized = tmp_path / "quantized"
    native.mkdir()

    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=hidden,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(native)
    tokenizer, texts = _write_tiny_llama_tokenizer(native)
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in texts]

    model = GPTQModel.load(
        str(native),
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )
    model.save(quantized)

    # (a) PEFT's default Llama target is q_proj/v_proj. Loading used to assert
    # on k_proj/o_proj because the adapter was attached to every quant module.
    partial = tmp_path / "partial"
    _write_adapter(
        partial,
        target_modules=["q_proj", "v_proj"],
        module_keys=[
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ],
        in_features=hidden,
        out_features=hidden,
    )
    loaded = GPTQModel.load(
        str(quantized),
        adapter=Lora(rank=2, path=str(partial)),
        backend=BACKEND.TORCH,
        device="cpu",
    )
    modules = dict(loaded.model.named_modules())
    assert modules["model.layers.0.self_attn.q_proj"].adapter is not None
    assert modules["model.layers.0.self_attn.v_proj"].adapter is not None
    assert modules["model.layers.0.self_attn.k_proj"].adapter is None
    assert modules["model.layers.0.self_attn.o_proj"].adapter is None

    # (b) lm_head stays dense. Its adapter used to be dropped silently; it must
    # now be attached and change the logits.
    dense_adapter = tmp_path / "dense"
    _write_adapter(
        dense_adapter,
        target_modules=["lm_head"],
        module_keys=["lm_head"],
        in_features=hidden,
        out_features=vocab,
    )
    AdapterCache.reset()
    base_model = GPTQModel.load(str(quantized), backend=BACKEND.TORCH, device="cpu")
    input_ids = torch.randint(0, vocab, (1, 8))
    with torch.inference_mode():
        base_logits = base_model.model(input_ids).logits.clone()

    AdapterCache.reset()
    dense_model = GPTQModel.load(
        str(quantized),
        adapter=Lora(rank=2, path=str(dense_adapter)),
        backend=BACKEND.TORCH,
        device="cpu",
    )
    assert isinstance(dense_model.model.lm_head, nn.Linear)
    assert dense_model.model.lm_head._gptqmodel_dense_adapter is not None
    with torch.inference_mode():
        dense_logits = dense_model.model(input_ids).logits
    assert not torch.allclose(dense_logits, base_logits)


def test_legacy_lowercase_targets_attach_when_the_tensors_match(tmp_path):
    """Legacy EoRA exports used lowercase targets and tensor keys (#5418)."""

    _write_adapter(
        tmp_path,
        target_modules=["wqkv"],
        module_keys=["transformer.blocks.0.antn.wqkv"],
    )
    adapter = Lora(rank=2, path=str(tmp_path))

    assert adapter.targets_module("transformer.blocks.0.antn.Wqkv")
    assert adapter.has_weights("transformer.blocks.0.antn.Wqkv")


def test_legacy_lowercase_targets_need_the_matching_tensor_pair(tmp_path):
    """The compatibility match is tensor-confirmed; a bare target is not enough."""

    _write_adapter(
        tmp_path,
        target_modules=["wqkv"],
        module_keys=["transformer.blocks.0.antn.other"],
    )
    adapter = Lora(rank=2, path=str(tmp_path))

    assert not adapter.targets_module("transformer.blocks.0.antn.Wqkv")


def test_canonical_targets_stay_case_sensitive(tmp_path):
    _write_adapter(
        tmp_path,
        target_modules=["Wqkv"],
        module_keys=["transformer.blocks.0.antn.Wqkv"],
    )
    adapter = Lora(rank=2, path=str(tmp_path))

    assert adapter.targets_module("transformer.blocks.0.antn.Wqkv")
    assert not adapter.targets_module("transformer.blocks.0.antn.wqkv")


def test_legacy_lowercase_adapter_attaches_in_create_quant_module(tmp_path):
    _write_adapter(tmp_path, target_modules=["wqkv"], module_keys=["wqkv"])
    adapter = Lora(rank=2, path=str(tmp_path))
    captured = {}

    class DummyQuantLinear(nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            self.bias = None
            captured[kwargs["name"]] = kwargs.get("adapter")

    root = _dense_root("Wqkv")
    create_quant_module(
        name="Wqkv",
        linear_cls=DummyQuantLinear,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=32,
        module=root,
        submodule=root.Wqkv,
        sym=True,
        device="cpu",
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        adapter=adapter,
    )

    assert captured["Wqkv"] is not None
