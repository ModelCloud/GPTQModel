# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4380: a disk-offloaded GPTQ layer must decode as v2 in-session.

``module.state_dict()`` writes checkpoint-layout (v1) ``qzeros`` plus a
``qzeros_format=1`` marker, and ``accelerate.disk_offload`` copies those
tensors straight back into the live module.  It never runs
``_load_from_state_dict``, so the v1 -> v2 conversion was skipped and every
zero point decoded one too low (NLL 14.06 vs 1.65 on the report's model).

The bundle must stay in checkpoint layout -- it is the durable record a save or
a resumed run reads -- so the restore path converts instead.
"""

import os

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.offload import offload_to_disk, undo_offload_to_disk


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 32
GROUP = 32
BITS = 4
QPROJ = "model.layers.0.self_attn.q_proj"


class _Container(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.q_proj = module


def _torchlinear(in_features, out_features, seed, sym=False):
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    scales = torch.rand(out_features, in_features // GROUP) * 0.05 + 1e-3
    zeros = torch.randint(
        0, 1 << BITS, (out_features, in_features // GROUP), dtype=torch.int32,
    )
    module = TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=FORMAT.GPTQ,
    )
    module.pack_original(linear, scales, zeros)
    return module


def _offload(module, tmp_path):
    model = _Container(module)
    offload_to_disk(
        module=module,
        model=model,
        disk_path=str(tmp_path),
        force=True,
    )
    return model


def _bundle(tmp_path):
    return load_file(str(tmp_path / "q_proj" / "module.safetensors"))


def test_disk_offload_forward_matches_pre_offload(tmp_path):
    """The accelerate hook must reload qzeros in the live v2 layout."""

    module = _torchlinear(HIDDEN, HIDDEN, seed=0)
    assert module.qzero_format() == 2
    inputs = torch.randn(3, HIDDEN, dtype=torch.float16)
    reference = module(inputs).clone()
    live_qzeros = module.qzeros.clone()
    checkpoint_qzeros = module.state_dict()["qzeros"].clone()
    assert not torch.equal(live_qzeros, checkpoint_qzeros)

    _offload(module, tmp_path)

    # Guard: the durable bundle still holds checkpoint-layout words and the
    # checkpoint marker, so saving/resuming from it is unchanged.
    bundle = _bundle(tmp_path)
    assert torch.equal(bundle["qzeros"], checkpoint_qzeros)
    assert int(bundle["qzeros_format"]) == 1

    assert torch.equal(module(inputs), reference), (
        "the restored layer decoded v1 qzeros as v2; in-session output drifted"
    )


def test_undo_offload_restores_live_qzeros(tmp_path):
    """``undo_offload_to_disk`` must decode to the pre-offload weights."""

    module = _torchlinear(HIDDEN, HIDDEN, seed=1)
    live_qzeros = module.qzeros.clone()
    reference_weight = module.dequantize_weight().clone()
    inputs = torch.randn(2, HIDDEN, dtype=torch.float16)
    reference = module(inputs).clone()

    _offload(module, tmp_path)
    undo_offload_to_disk(module, device=torch.device("cpu"))

    assert module.qzero_format() == 2
    assert torch.equal(module.qzeros, live_qzeros), (
        "restored qzeros are checkpoint-layout v1 words, not the live v2 words"
    )
    assert int(module._buffers["qzeros_format"].item()) == 2
    assert torch.equal(module.dequantize_weight(), reference_weight)
    assert torch.equal(module(inputs), reference)

    # The restored marker must describe the restored words: a refill built from
    # the live buffers must not convert the v2 words a second time.
    persistent = set(module.state_dict())
    refill = {
        name: value.detach().clone()
        for name, value in module.named_buffers()
        if name in persistent
    }
    module.load_state_dict(refill)
    assert torch.equal(module.qzeros, live_qzeros)


def _tiny_llama():
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    return config, LlamaForCausalLM(config).half()


def _save_tokenizer(directory):
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


def test_offload_bundle_round_trips_through_gptqmodel_load(tmp_path):
    """The bundle stays a valid checkpoint payload for ``GPTQModel.load``."""

    config, model = _tiny_llama()
    live_qzeros = {}
    for index, (name, linear) in enumerate(list(model.named_modules())):
        if not isinstance(linear, nn.Linear) or name == "lm_head":
            continue
        quant = _torchlinear(linear.in_features, linear.out_features, seed=index, sym=True)
        model.set_submodule(name, quant)
        live_qzeros[name] = quant.qzeros.clone()

    module = model.get_submodule(QPROJ)
    checkpoint_state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".qzeros_format")
    }

    offload_to_disk(
        module=module,
        model=model,
        disk_path=str(tmp_path / "offload"),
        force=True,
    )
    bundle_path = tmp_path / "offload" / QPROJ / "module.safetensors"
    bundle = load_file(str(bundle_path))
    assert torch.equal(bundle["qzeros"], checkpoint_state[f"{QPROJ}.qzeros"])

    checkpoint = tmp_path / "checkpoint"
    os.makedirs(checkpoint)
    save_file(checkpoint_state, str(checkpoint / "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(str(checkpoint))
    QuantizeConfig(
        bits=BITS,
        group_size=GROUP,
        desc_act=False,
        sym=True,
        format=FORMAT.GPTQ,
        offload_to_disk=False,
    ).save_pretrained(str(checkpoint))
    _save_tokenizer(str(checkpoint))

    reloaded = GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.float16,
        local_files_only=True,
    )
    reloaded_module = reloaded.model.get_submodule(QPROJ)
    assert reloaded_module.qzero_format() == 2
    assert torch.equal(reloaded_module.qzeros, live_qzeros[QPROJ]), (
        "the checkpoint built from the offload bundle did not decode to the live zeros"
    )


def _state_dict_snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_dict_equal(actual, expected):
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), f"state-dict tensor drifted: {name}"


def test_forced_offload_converts_qzeros_once(tmp_path, monkeypatch):
    """A forced offload must not build the unused size-preflight state dict (#5850).

    ``force=True`` bypasses the 4 KiB size check, so the preflight state dict is
    discarded. Each state dict runs the v2 -> v1 qzero conversion; building two
    doubles the conversion work and leaves both converted copies live at once.
    """

    from gptqmodel.utils import model_dequant

    module = _torchlinear(HIDDEN, HIDDEN, seed=0)
    assert module.qzero_format() == 2
    checkpoint_before = _state_dict_snapshot(module)
    inputs = torch.randn(3, HIDDEN, dtype=torch.float16)
    reference = module(inputs).clone()

    calls = []
    original = model_dequant._revert_gptq_v1_qzeros_correction

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_dequant, "_revert_gptq_v1_qzeros_correction", counting)

    model = _Container(module)
    offload_to_disk(module=module, model=model, disk_path=str(tmp_path), force=True)

    assert len(calls) == 1, (
        "forced disk offload built a second state dict for the unused size "
        f"preflight ({len(calls)} v2 -> v1 conversions)"
    )

    bundle = _bundle(tmp_path)
    _assert_state_dict_equal(bundle, checkpoint_before)
    assert int(bundle["qzeros_format"]) == 1
    assert torch.equal(module(inputs), reference), "disk-offloaded forward drifted"

    undo_offload_to_disk(module, device=torch.device("cpu"))
    assert module.qzero_format() == 2
    _assert_state_dict_equal(_state_dict_snapshot(module), checkpoint_before)
    assert torch.equal(module(inputs), reference), "restored forward drifted"


def test_non_forced_offload_still_enforces_size_threshold(tmp_path):
    """Without ``force`` the 4 KiB size preflight must still skip small modules."""

    small = _torchlinear(64, 64, seed=1)
    small_inputs = torch.randn(2, 64, dtype=torch.float16)
    small_reference = small(small_inputs).clone()
    small_model = _Container(small)

    offload_to_disk(module=small, model=small_model, disk_path=str(tmp_path / "small"))

    assert not (tmp_path / "small" / "q_proj").exists(), (
        "a below-threshold module was offloaded without force"
    )
    assert not small.qweight.is_meta
    assert torch.equal(small(small_inputs), small_reference)

    large = _torchlinear(256, 256, seed=2)
    checkpoint_before = _state_dict_snapshot(large)
    inputs = torch.randn(2, 256, dtype=torch.float16)
    reference = large(inputs).clone()
    large_model = _Container(large)

    offload_to_disk(module=large, model=large_model, disk_path=str(tmp_path / "large"))

    bundle = load_file(str(tmp_path / "large" / "q_proj" / "module.safetensors"))
    _assert_state_dict_equal(bundle, checkpoint_before)
    assert int(bundle["qzeros_format"]) == 1
    assert large.qweight.is_meta
    assert torch.equal(large(inputs), reference), "disk-offloaded forward drifted"
