# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The ``qzeros`` layout invariant across load, save, and raw packing.

Issues #4166, #4167 and #4171 all describe the same missing contract:

* the live ``qzeros`` buffer stays in the kernel's runtime layout, recorded by
  ``_qzeros_format``;
* ``_checkpoint_format_on_disk`` records only the layout a checkpoint write
  emits;
* the v1/v2 conversion runs at the two serialization boundaries, is idempotent,
  and follows the marker carried by the same incoming state dict;
* a checkpoint never persists the marker.

The load->save cases below compare the raw ``qzeros`` bytes, which is what the
silent-corruption bugs changed. The sharded case is the one that catches the
per-shard double conversion: accelerate refills every key a shard omits from
the module's own buffers.
"""

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 32
GROUP = 32
BITS = 4

QPROJ = "model.layers.0.self_attn.q_proj"


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


def _tiny_llama(seed=0):
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
    torch.manual_seed(seed)
    return config, LlamaForCausalLM(config).half()


def _checkpoint_state(fmt):
    config, model = _tiny_llama()
    qcfg = QuantizeConfig(
        bits=BITS, group_size=GROUP, desc_act=False, sym=True, format=fmt,
        offload_to_disk=False,
    )
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=BITS,
            group_size=GROUP,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
        )
        scales = torch.full((linear.out_features, linear.in_features // GROUP), 0.01)
        zeros = torch.full_like(scales, 8)
        quant.pack_original(linear, scales, zeros)
        model.set_submodule(name, quant)
    if fmt == FORMAT.GPTQ:
        convert_gptq_v2_to_v1_format(model, qcfg, TorchLinear)
    for module in model.modules():
        if isinstance(module, TorchLinear):
            module._checkpoint_format_on_disk = fmt
    # A real checkpoint never carries the layout marker (#4155).
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".qzeros_format")
    }
    return config, qcfg, state


def _write_checkpoint(path, fmt, sharded=False):
    os.makedirs(path, exist_ok=True)
    config, qcfg, state = _checkpoint_state(fmt)
    if sharded:
        # Split the quantized modules across two files so each shard omits
        # most of the model. With ``device_map=None`` accelerate refills the
        # omitted keys from the module's own buffers once per shard.
        quantized = [
            name for name in state
            if any(part in name for part in (".q_proj.", ".k_proj.", ".v_proj."))
        ]
        quantized_set = set(quantized)
        rest = [name for name in state if name not in quantized_set]
        shards = {
            "model-00001-of-00002.safetensors": quantized,
            "model-00002-of-00002.safetensors": rest,
        }
        weight_map = {}
        for filename, names in shards.items():
            save_file(
                {name: state[name].contiguous() for name in names},
                os.path.join(path, filename),
                metadata={"format": "pt"},
            )
            weight_map.update(dict.fromkeys(names, filename))
        with open(os.path.join(path, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
            json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, handle)
    else:
        save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(path)
    qcfg.save_pretrained(path)
    _save_tokenizer(path)
    return state


def _read_qzeros(directory):
    single = os.path.join(directory, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    tensors = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(directory, filename)))
    return tensors


def _load(path):
    return GPTQModel.load(
        path,
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.float16,
        local_files_only=True,
    )


@pytest.mark.parametrize("sharded", [False, True], ids=["single", "sharded"])
@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2], ids=["gptq_v1", "gptq_v2"])
def test_load_then_save_preserves_qzeros_bytes(tmp_path, fmt, sharded):
    """Issue #4166: ``load().save()`` must not shift a single zero point."""

    source = tmp_path / "source"
    target = tmp_path / "target"
    source_state = _write_checkpoint(str(source), fmt, sharded=sharded)

    model = _load(str(source))
    module = model.model.get_submodule(QPROJ)
    assert getattr(type(module), "REQUIRES_FORMAT_V2", False), (
        "the CPU Torch kernel must require the v2 runtime layout for this case to "
        "exercise the conversion"
    )
    model.save(str(target))

    saved = _read_qzeros(str(target))
    for name, expected in source_state.items():
        if not name.endswith(".qzeros"):
            continue
        assert torch.equal(saved[name], expected), (
            f"{fmt.value}/{sharded}: `{name}` changed across load -> save"
        )


def test_direct_pack_labels_logical_zero_points(tmp_path):
    """Issue #4167: a raw ``pack()`` caller must emit logical (v2) words."""

    linear = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    scales = torch.rand(INTERMEDIATE, HIDDEN // GROUP) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << BITS, (INTERMEDIATE, HIDDEN // GROUP), dtype=torch.int32)
    g_idx = torch.arange(HIDDEN, dtype=torch.int32) // GROUP

    module = TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        in_features=HIDDEN,
        out_features=INTERMEDIATE,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
    )
    module.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    assert module.qzero_format() == 2, (
        "the packer wrote logical zero points but left the v1 label"
    )
    emitted = module.state_dict()
    assert torch.equal(emitted["qzeros"], module.qzeros), (
        "state_dict() relabelled or converted freshly packed logical words"
    )

    fresh = TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        in_features=HIDDEN,
        out_features=INTERMEDIATE,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
    )
    fresh.load_state_dict(emitted)
    assert torch.equal(fresh.qzeros, module.qzeros)


def test_load_state_dict_is_idempotent_with_refilled_shard(tmp_path):
    """Issue #4166: a dict refilled from live buffers must not convert twice."""

    source = tmp_path / "source"
    _write_checkpoint(str(source), FORMAT.GPTQ, sharded=False)
    module = _load(str(source)).model.get_submodule(QPROJ)
    assert module.qzero_format() == 2
    reference = module.qzeros.detach().clone()

    # Reproduce accelerate's ``set_model_state_dict`` dict: every live buffer,
    # including the persistent layout marker, under fresh storage.
    merged = {
        name: value.detach().clone()
        for name, value in module.named_buffers()
    }
    merged.update(
        {name: value.detach().clone() for name, value in module.named_parameters()}
    )
    persistent = set(module.state_dict())
    merged = {name: value for name, value in merged.items() if name in persistent}
    assert "qzeros_format" in merged
    assert merged["qzeros_format"].item() == 2

    module.load_state_dict(merged)
    assert torch.equal(module.qzeros, reference), "first refilled load shifted zeros"
    module.load_state_dict(merged)
    assert torch.equal(module.qzeros, reference), "second refilled load shifted zeros"


def _nll(model, input_ids):
    with torch.inference_mode():
        logits = model.model(input_ids=input_ids).logits.float()
    return torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    ).item()


def _quantized_nll(tmp_path, fmt):
    source = tmp_path / f"source_{fmt.value}"
    if not source.exists():
        os.makedirs(source)
        config, dense = _tiny_llama()
        dense.save_pretrained(str(source))
        _save_tokenizer(str(source))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(source))
    calibration = []
    for text in (
        "the quick brown fox jumps over the lazy dog",
        "a tiny llama calibration sample with enough tokens",
        "another synthetic calibration example for the test",
    ):
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )

    qcfg = QuantizeConfig(
        bits=BITS,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        format=fmt,
        offload_to_disk=False,
    )
    model = GPTQModel.load(
        str(source),
        quantize_config=qcfg,
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    module = model.model.get_submodule(QPROJ)
    assert module.qzero_format() == 2
    live = module.qzeros.detach().clone()

    target = tmp_path / f"target_{fmt.value}"
    model.save(str(target))
    saved = load_file(str(target / "model.safetensors"))[f"{QPROJ}.qzeros"]

    reloaded = _load(str(target))
    reloaded_module = reloaded.model.get_submodule(QPROJ)
    assert torch.equal(reloaded_module.qzeros.cpu(), live), (
        f"{fmt.value}: reloaded zero points differ from the in-memory model"
    )

    ids = tokenizer("the quick brown fox", return_tensors="pt")["input_ids"]
    return {
        "live": live,
        "saved": saved,
        "in_memory": _nll(model, ids),
        "nll": _nll(reloaded, ids),
    }


@pytest.mark.slow
def test_gptq_v2_quantize_save_reload_matches_v1(tmp_path):
    """Issue #4167: ``FORMAT.GPTQ_V2`` must not add one to every zero point."""

    v1 = _quantized_nll(tmp_path, FORMAT.GPTQ)
    v2 = _quantized_nll(tmp_path, FORMAT.GPTQ_V2)

    # v2 stores the logical zero points, so its checkpoint is byte-identical to
    # the live buffer. v1 stores ``zero_point - 1`` and reloads to the same
    # live words through the conversion.
    assert torch.equal(v2["saved"], v2["live"])
    assert not torch.equal(v1["saved"], v1["live"])

    # The corrupt v2 checkpoint measured NLL ~16.7 against ~1.5 for v1 on the
    # issue's model. On this single-layer fixture the drift is smaller but still
    # measurable (4.2229 vs 4.2178), so pin the reloaded NLL to the in-memory
    # model that produced the checkpoint.
    for label, result in (("gptq", v1), ("gptq_v2", v2)):
        assert abs(result["nll"] - result["in_memory"]) < 1e-3, (
            f"{label}: reloaded NLL {result['nll']:.6f} differs from the "
            f"in-memory model {result['in_memory']:.6f}"
        )
    assert abs(v2["nll"] - v1["nll"]) < 1e-3, (
        f"GPTQ_V2 NLL {v2['nll']:.6f} diverged from GPTQ {v1['nll']:.6f}"
    )
