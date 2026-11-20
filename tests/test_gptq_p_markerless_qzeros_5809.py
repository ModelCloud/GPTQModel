# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #5809: markerless ``gptq_p`` loads must emit a logical zero marker.

A ``gptq_p`` checkpoint stores logical zero points, like ``gptq_v2``. The
markerless load hook only recognized ``gptq``/``gptq_v2``, so a freshly loaded
planar module kept the constructor's v1 placeholder. The next ``state_dict()``
labelled the logical words as v1, and copying it into another ``gptq_p``
module shifted every zero point by one.
"""

import os

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.checkpoint_state import checkpoint_serialization
from gptqmodel.utils.packing import hf_pack_layer


BITS = (2, 3, 4, 5, 6, 7, 8)
GROUP = 32
IN_FEATURES = 32
OUT_FEATURES = 32


def _module(bits: int) -> TorchLinear:
    return TorchLinear(
        bits=bits,
        group_size=GROUP,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        format=FORMAT.GPTQ_P,
        compile=False,
    ).eval()


@torch.no_grad()
def _packed(bits: int, fill: float = 1.0) -> TorchLinear:
    zero_point = 1 << (bits - 1)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    linear.weight.fill_(fill)
    module = hf_pack_layer(
        _module(bits),
        linear,
        torch.ones(IN_FEATURES, 1),
        torch.full((OUT_FEATURES, 1), zero_point),
        torch.zeros(IN_FEATURES, dtype=torch.int32),
        checkpoint_format=FORMAT.GPTQ_P,
        post_init=True,
    )
    # ``hf_pack_layer`` runs under ``inference_mode``, so its buffers are
    # inference tensors. Materialize ordinary buffers before the initialized
    # target case copies into them outside that mode.
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        value = getattr(module, name, None)
        if isinstance(value, torch.Tensor):
            setattr(module, name, value.clone())
    module._clear_runtime_weight_cache()
    return module


def _inputs() -> torch.Tensor:
    return torch.eye(IN_FEATURES, dtype=torch.float16)[:1]


def _assert_logical(module: TorchLinear, zero_point: int) -> None:
    zeros = module.unpack_zero_points()
    torch.testing.assert_close(zeros, torch.full_like(zeros, zero_point))


def _write_markerless(path, bits: int) -> TorchLinear:
    source = _packed(bits)
    expected = torch.ones(1, OUT_FEATURES, dtype=torch.float16)
    assert torch.equal(source(_inputs()), expected)
    assert int(source.state_dict()["qzeros_format"]) == 2

    with checkpoint_serialization():
        disk = source.state_dict()
    assert "qzeros_format" not in disk
    save_file(
        {name: value.contiguous() for name, value in disk.items()},
        str(path),
        metadata={"format": "gptq_p", "bits": str(bits)},
    )
    return source


def test_hf_bridge_marks_gptq_p_as_logical():
    """The HF/optimum bridge must record ``gptq_p`` as a logical layout."""

    from gptqmodel.utils.model import hf_convert_gptq_v1_to_v2_format

    source = _packed(4)
    state = {name: value.clone() for name, value in source.state_dict().items()}
    words = source.qzeros.detach().clone()

    fresh = _module(4)
    fresh.load_state_dict(state)
    fresh.post_init()

    model = torch.nn.Module()
    model.add_module("q_proj", fresh)
    converted, changed = hf_convert_gptq_v1_to_v2_format(
        model,
        bits=4,
        qlinear_kernel=TorchLinear,
        checkpoint_format=FORMAT.GPTQ_P,
        meta=None,
    )
    module = converted.q_proj
    assert changed is False
    assert module._checkpoint_format_on_disk == FORMAT.GPTQ_P
    assert module.qzero_format() == 2
    assert module._buffers["qzeros_format"].item() == 2
    assert torch.equal(module.qzeros, words)
    assert int(module.state_dict()["qzeros_format"]) == 2


@torch.no_grad()
@pytest.mark.parametrize("bits", BITS)
def test_markerless_gptq_p_load_emits_logical_marker_and_copies(tmp_path, bits):
    """Every width must survive markerless load -> state-dict copy -> forward."""

    zero_point = 1 << (bits - 1)
    path = tmp_path / f"gptq_p_{bits}.safetensors"
    source = _write_markerless(path, bits)
    expected = torch.ones(1, OUT_FEATURES, dtype=torch.float16)

    loaded = _module(bits)
    loaded.load_state_dict(load_file(str(path)))
    loaded.post_init()

    assert loaded.qzero_format() == 2, f"{bits}-bit markerless load kept the v1 placeholder"
    assert torch.equal(loaded.qzeros, source.qzeros), f"{bits}-bit load changed qzero words"
    _assert_logical(loaded, zero_point)
    assert torch.equal(loaded(_inputs()), expected)

    state = loaded.state_dict()
    assert int(state["qzeros_format"]) == 2, f"{bits}-bit load emitted a v1 marker"

    for assign in (False, True):
        fresh = _module(bits)
        fresh.load_state_dict(state, assign=assign)
        fresh.post_init()
        assert int(fresh.state_dict()["qzeros_format"]) == 2
        assert torch.equal(fresh.qzeros, source.qzeros), (
            f"{bits}-bit fresh copy (assign={assign}) changed qzero words"
        )
        _assert_logical(fresh, zero_point)
        assert torch.equal(fresh(_inputs()), expected)

        initialized = _packed(bits, fill=-1.0)
        assert torch.equal(initialized(_inputs()), -expected)
        initialized.load_state_dict(state, assign=assign)
        initialized.post_init()
        assert torch.equal(initialized.qzeros, source.qzeros), (
            f"{bits}-bit initialized copy (assign={assign}) changed qzero words"
        )
        _assert_logical(initialized, zero_point)
        assert torch.equal(initialized(_inputs()), expected), (
            f"{bits}-bit initialized copy (assign={assign}) did not replace the old words"
        )


def _save_tokenizer(directory) -> None:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog", "a tiny llama sample"],
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


def _write_tiny_model_checkpoint(directory) -> dict:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=IN_FEATURES,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).half()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=GROUP,
        desc_act=False,
        sym=True,
        format=FORMAT.GPTQ_P,
        offload_to_disk=False,
    )
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=4,
            group_size=GROUP,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_P,
        )
        scales = torch.full((linear.out_features, linear.in_features // GROUP), 0.01)
        quant.pack_original(linear, scales, torch.full_like(scales, 8))
        quant._checkpoint_format_on_disk = FORMAT.GPTQ_P
        model.set_submodule(name, quant)

    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".qzeros_format")
    }
    os.makedirs(directory, exist_ok=True)
    save_file(state, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(directory)
    qcfg.save_pretrained(directory)
    _save_tokenizer(directory)
    return state


def test_model_loader_marks_gptq_p_runtime_as_logical(tmp_path):
    """``GPTQModel.load`` must not leave a gptq_p module on the v1 placeholder."""

    checkpoint = str(tmp_path / "checkpoint")
    state = _write_tiny_model_checkpoint(checkpoint)

    model = GPTQModel.load(
        checkpoint,
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.float16,
        local_files_only=True,
    )
    name = "model.layers.0.self_attn.q_proj"
    module = model.model.get_submodule(name)
    assert isinstance(module, TorchLinear)
    assert module._checkpoint_format_on_disk == FORMAT.GPTQ_P
    assert module.qzero_format() == 2
    assert module._buffers["qzeros_format"].item() == 2
    assert int(module.state_dict()["qzeros_format"]) == 2
    assert torch.equal(module.qzeros, state[f"{name}.qzeros"])
    _assert_logical(module, 8)
