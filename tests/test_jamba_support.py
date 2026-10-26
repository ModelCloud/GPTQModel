# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the Jamba hybrid (Mamba + attention) quantization path.

The real `AI21-Jamba2-3B` checkpoint is GPU-evaluated by
`tests/models/test_jamba.py`. These tests keep the cheap contract checks on
CPU: the registry wiring, the module tree shape, the attention-layer replay
mask rebuild, and a tiny end-to-end GPTQ quantize/save/reload smoke run.
"""

from pathlib import Path
from types import SimpleNamespace

import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import JambaConfig, JambaForCausalLM, PreTrainedTokenizerFast
from transformers.masking_utils import create_causal_mask

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models import auto
from gptqmodel.models.definitions.jamba import JambaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


class _JambaReplayProbe(JambaQModel):
    """Minimal instance stand-in: the replay hook only needs `model.config`."""

    def __init__(self, model):
        object.__setattr__(self, "model", model)


def _use_torch_reference_kernels(monkeypatch):
    """Keep this CPU test independent of the CUDA-only mamba-ssm / causal-conv1d packages.

    When either package is installed, `JambaMambaMixer` binds its CUDA kernels at import
    time and would raise on CPU tensors; force the torch reference implementations.
    """

    import transformers.models.jamba.modeling_jamba as modeling_jamba

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba_selective_scan",
        "mamba_selective_state_update",
    ):
        fallback = getattr(getattr(modeling_jamba, name), "__wrapped__", None)
        if fallback is not None:
            monkeypatch.setattr(modeling_jamba, name, fallback)


_CALIBRATION_TEXTS = [
    "tiny jamba calibration sample one with enough tokens to survive minimum length filtering",
    "tiny jamba calibration sample two with repeated words to exercise the hybrid quant path",
    "another synthetic calibration example that is intentionally verbose so token filtering keeps it",
] * 2


def _tiny_jamba_config() -> JambaConfig:
    return JambaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_layer_offset=1,
        attn_layer_period=2,
        expert_layer_period=2,
        expert_layer_offset=1,
        num_experts=1,
        num_experts_per_tok=1,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return fast_tokenizer


def _build_tiny_jamba_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
    model = JambaForCausalLM(_tiny_jamba_config())
    model.save_pretrained(model_dir)
    return _build_local_tokenizer(model_dir)


def _build_calibration_dataset(tokenizer: PreTrainedTokenizerFast, pad_tail: int = 0):
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        attention_mask = encoded["attention_mask"].clone()
        if pad_tail > 0:
            attention_mask[0, -pad_tail:] = 0
        dataset.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": attention_mask,
            }
        )
    return dataset


def test_jamba_registry_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="jamba")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.MODEL_MAP["jamba"] is JambaQModel
    assert auto.check_and_get_model_definition("/tmp/jamba", trust_remote_code=False) is JambaQModel


def test_jamba_definition_contract_and_module_tree():
    assert JambaQModel.layer_modules_strict is False
    assert JambaQModel.pre_lm_head_norm_module == "model.final_layernorm"
    assert JambaQModel.extract_layers_node() == ["model.layers"]

    quantize_config = SimpleNamespace(dynamic=None)
    simple = JambaQModel.simple_layer_modules(_tiny_jamba_config(), quantize_config)
    full = JambaQModel.full_layer_modules(_tiny_jamba_config())

    assert simple == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mamba.in_proj"],
        ["mamba.out_proj"],
        ["feed_forward.gate_proj", "feed_forward.up_proj"],
        ["feed_forward.down_proj"],
    ]

    full_names = {name for block in full for name in block}
    assert "input_layernorm:!" in full_names
    assert "pre_ff_layernorm:!" in full_names
    # The mixer consumes `dt_proj.weight` and `x_proj` output directly, so the
    # definition intentionally leaves them dense.
    assert not any("dt_proj" in name or "x_proj" in name for name in full_names)


def test_jamba_replay_rebuilds_causal_mask_for_attention_layers():
    config = _tiny_jamba_config()
    model = JambaForCausalLM(config)
    qmodel = _JambaReplayProbe(model)

    hidden_states = torch.randn(1, 6, config.hidden_size)
    padding_mask = torch.ones(1, 6, dtype=torch.bool)
    padding_mask[0, -2:] = False
    position_ids = torch.arange(6).unsqueeze(0)

    attention_layer = model.model.layers[1]
    mamba_layer = model.model.layers[0]

    attention_inputs = JambaQModel.prepare_layer_replay_kwargs(
        qmodel,
        layer=attention_layer,
        layer_input=[hidden_states],
        additional_inputs={"attention_mask": padding_mask, "position_ids": position_ids},
        target_device=torch.device("cpu"),
    )
    expected = create_causal_mask(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=padding_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    assert attention_inputs["attention_mask"].dtype == torch.bool
    assert attention_inputs["attention_mask"].shape == expected.shape
    assert torch.equal(attention_inputs["attention_mask"], expected)

    mamba_inputs = JambaQModel.prepare_layer_replay_kwargs(
        qmodel,
        layer=mamba_layer,
        layer_input=[hidden_states],
        additional_inputs={"attention_mask": padding_mask, "position_ids": position_ids},
        target_device=torch.device("cpu"),
    )
    assert torch.equal(mamba_inputs["attention_mask"], padding_mask)


def test_jamba_replay_mask_is_causal_not_bidirectional():
    """Guard the bug the replay override fixes: a raw 2D padding mask is not a causal mask."""

    config = _tiny_jamba_config()
    model = JambaForCausalLM(config).eval()
    attention_layer = model.model.layers[1]
    qmodel = _JambaReplayProbe(model)

    hidden_states = torch.randn(1, 6, config.hidden_size)
    padding_mask = torch.ones(1, 6, dtype=torch.bool)

    with torch.no_grad():
        rebuilt = JambaQModel.prepare_layer_replay_kwargs(
            qmodel,
            layer=attention_layer,
            layer_input=[hidden_states],
            additional_inputs={"attention_mask": padding_mask, "position_ids": None},
            target_device=torch.device("cpu"),
        )["attention_mask"]
        causal_output = attention_layer(
            hidden_states,
            attention_mask=rebuilt,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )
        bidirectional_output = attention_layer(
            hidden_states,
            attention_mask=padding_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )

    assert not torch.allclose(causal_output, bidirectional_output)


def test_tiny_jamba_gptq_quantization_smoke(tmp_path: Path, monkeypatch):
    _use_torch_reference_kernels(monkeypatch)

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    tokenizer = _build_tiny_jamba_fixture(model_dir)
    # Right-padded rows keep the attention replay mask rebuild on the hot path.
    calibration = _build_calibration_dataset(tokenizer, pad_tail=3)

    native = JambaForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    native.eval()

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(quantized_dir)

    quantized = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    modules = dict(quantized.model.named_modules())

    for name in (
        "model.layers.0.mamba.in_proj",
        "model.layers.0.mamba.out_proj",
        "model.layers.0.feed_forward.gate_proj",
        "model.layers.0.feed_forward.down_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.o_proj",
    ):
        assert isinstance(modules[name], TorchLinear), name

    # Mamba mixer modules that consume dense weights stay dense.
    assert isinstance(modules["model.layers.0.mamba.dt_proj"], torch.nn.Linear)
    assert isinstance(modules["model.layers.0.mamba.x_proj"], torch.nn.Linear)
    assert isinstance(modules["model.layers.0.mamba.conv1d"], torch.nn.Conv1d)

    input_ids = tokenizer("tiny jamba calibration sample one", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        native_logits = native(input_ids=input_ids).logits
        quantized_logits = quantized.model(input_ids=input_ids).logits

    diff = (native_logits - quantized_logits).abs().mean().item()
    scale = native_logits.abs().mean().item()
    assert diff < 0.5 * scale, (diff, scale)

    # Cached decode must agree with the non-cached prefill path for the hybrid layers.
    cached = quantized.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, use_cache=True)
    uncached = quantized.generate(input_ids=input_ids, max_new_tokens=6, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)
