# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AUTO_NF12 must never serialize an excluded layer's calibration source storage.

A `dynamic` `-:` rule that excludes every module of a non-final layer still
runs one replay forward so the next layer receives activations. That replay
must leave the layer dense: an excluded layer never takes the `quant_source`
role that restores a packed AUTO_NF12 source, so a cast there would write
`nf12_packed` / `nf12_escapes` / `nf12_exp_base` instead of `weight` while
`quantize_config.json` still describes the module as plain unquantized, and a
reload would leave the dense weights uninitialised (NaN logits).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.adapter.adapter import Lora
from gptqmodel.quantization.config import AWQConfig, NF12Config
from gptqmodel.utils import auto_dtype_cast


pytestmark = pytest.mark.slow

_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny llama calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 3

_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _build_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)
    return fast


@pytest.fixture(scope="module")
def native_llama(tmp_path_factory) -> Path:
    """Tiny four-layer BF16 Llama; a non-final layer is layer 0, 1 or 2."""

    model_dir = tmp_path_factory.mktemp("auto-nf12-native")
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.bfloat16).eval().save_pretrained(model_dir)
    _build_tokenizer(model_dir)
    return model_dir


def _require_nf12_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AUTO_NF12 eligibility requires a CUDA/HIP device")
    if not auto_dtype_cast.nf12_supported_device(torch.device("cuda:0")):
        pytest.skip("AUTO_NF12 is not supported on this accelerator")


def _quantize_config(pattern: str, method: str, cast: str) -> QuantizeConfig:
    settings = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "device": "cuda:0",
        "dynamic": {pattern: {}},
        "auto_dtype_cast": cast,
    }
    if method == "awq":
        return AWQConfig(**settings)
    return QuantizeConfig(**settings)


def _quantize_and_save(native: Path, output: Path, pattern: str, method: str, cast: str) -> Path:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(native)
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    model = GPTQModel.load(
        str(native),
        quantize_config=_quantize_config(pattern, method, cast),
        backend=BACKEND.TORCH,
    )
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(str(output))
    return output


def _saved_tensor_names(checkpoint: Path) -> list[str]:
    names: list[str] = []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        names.extend(load_file(str(shard)).keys())
    return names


def _reload_logits(checkpoint: Path, native: Path) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(native)
    ids = tokenizer("the quick brown fox", return_tensors="pt")["input_ids"].to("cuda:0")
    reloaded = GPTQModel.load(str(checkpoint), backend=BACKEND.TORCH, device="cuda:0")
    reference = GPTQModel.load(str(native), backend=BACKEND.TORCH, device="cuda:0")
    with torch.inference_mode():
        return reloaded.model(input_ids=ids).logits.float(), reference.model(input_ids=ids).logits.float()


def _assert_no_nf12_payload(names: list[str]) -> None:
    leaked = [name for name in names if ".nf12_" in name]
    assert not leaked, f"checkpoint serialized AUTO_NF12 source storage: {leaked[:8]}"


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("layers", [(0,), (1,), (2,), (1, 2)])
def test_fully_excluded_non_final_layer_saves_dense_weights(tmp_path, native_llama, layers, method):
    """Layers 0, 1, 2 and 1+2 are all non-final, so each one is replayed once."""

    _require_nf12_gpu()
    pattern = "-:.*layers\\.(" + "|".join(str(layer) for layer in layers) + ")\\..*"
    checkpoint = _quantize_and_save(native_llama, tmp_path / "quantized", pattern, method, "auto_nf12")

    names = _saved_tensor_names(checkpoint)
    _assert_no_nf12_payload(names)
    for layer in layers:
        for projection in _PROJECTIONS:
            name = f"model.layers.{layer}.{projection}.weight"
            assert name in names, f"excluded layer saved without a dense weight: {name}"
    # The exclusion must not disable quantization for the remaining layers.
    assert any(name.endswith(".qweight") for name in names)

    with open(checkpoint / "quantize_config.json", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    assert pattern in saved_config["dynamic"]

    logits, reference = _reload_logits(checkpoint, native_llama)
    assert torch.isfinite(logits).all(), "reloaded logits are not finite"
    relative_error = ((logits - reference).norm() / reference.norm()).item()
    assert relative_error < 0.5, relative_error


@pytest.mark.gpu
@pytest.mark.parametrize(
    "pattern, cast",
    [
        (r"-:.*layers\.3\..*", "auto_nf12"),  # last layer: no replay forward
        (r"-:.*mlp\.down_proj", "auto_nf12"),  # single module, not a whole layer
        (r"-:.*layers\.1\.self_attn\..*", "auto_nf12"),  # one sub-block
        (r"-:.*layers\.1\..*", "off"),  # AUTO_NF12 disabled
    ],
    ids=["last-layer", "single-module", "sub-block", "cast-off"],
)
def test_other_exclusions_never_serialize_nf12_buffers(tmp_path, native_llama, pattern, cast):
    _require_nf12_gpu()
    checkpoint = _quantize_and_save(native_llama, tmp_path / "quantized", pattern, "gptq", cast)

    names = _saved_tensor_names(checkpoint)
    _assert_no_nf12_payload(names)
    logits, reference = _reload_logits(checkpoint, native_llama)
    assert torch.isfinite(logits).all(), "reloaded logits are not finite"
    relative_error = ((logits - reference).norm() / reference.norm()).item()
    # The invariant here is the checkpoint layout above: these exclusions never
    # install NF12 storage. Their logit drift is not the subject of the issue and
    # varies with the accelerator (an excluded attention sub-block measured
    # 0.57 on the R9700), so only reject a catastrophic difference.
    assert relative_error < 1.0, relative_error


@pytest.mark.gpu
def test_gptq_dynamic_exclusion_declares_optimum_inclusion_list(tmp_path, native_llama):
    """A GPTQ `-:` checkpoint must tell optimum exactly which modules are packed.

    Optimum builds an uninitialized quantized layer for every module it decides
    to quantize and matches ``modules_in_block_to_quantize`` entries with
    ``name.endswith``. Without the explicit list a plain HF load silently builds
    Marlin layers for the excluded modules and returns garbage logits.
    """

    _require_nf12_gpu()
    excluded = "model.layers.1.self_attn.q_proj"
    checkpoint = _quantize_and_save(
        native_llama,
        tmp_path / "quantized",
        r"-:.*layers\.1\.self_attn\.q_proj$",
        "gptq",
        "auto_nf12",
    )

    with open(checkpoint / "config.json", encoding="utf-8") as handle:
        saved = json.load(handle)["quantization_config"]
    inclusion = saved["modules_in_block_to_quantize"]
    # Optimum flattens the field with `sum(..., [])`, which requires groups.
    assert isinstance(inclusion, list) and inclusion
    assert all(isinstance(group, list) for group in inclusion)
    flat = [name for group in inclusion for name in group]

    quantized = {
        name[: -len(".qweight")]
        for name in _saved_tensor_names(checkpoint)
        if name.endswith(".qweight")
    }
    assert quantized, "fixture quantized no modules"
    assert quantized <= set(flat), sorted(quantized - set(flat))[:8]
    assert excluded not in flat

    # Emulate optimum's per-layer decision: `name.endswith(entry)`.
    def _optimum_quantizes(full_name: str) -> bool:
        return any(full_name.endswith(entry) for entry in flat)

    assert not _optimum_quantizes(excluded)
    for name in quantized:
        assert _optimum_quantizes(name), name

    # Our own loader must still honor the same payload on reload.
    logits, reference = _reload_logits(checkpoint, native_llama)
    assert torch.isfinite(logits).all(), "reloaded logits are not finite"
    relative_error = ((logits - reference).norm() / reference.norm()).item()
    assert relative_error < 0.5, relative_error


@pytest.mark.gpu
@pytest.mark.parametrize(
    "pattern, cast",
    [
        (r"-:^$", "auto_nf12"),
        (r"-:.*layers\.1\.self_attn\.q_proj$", "off"),
    ],
    ids=["auto-nf12-source", "dynamic-exclusion"],
)
def test_post_quant_eora_generation_survives_auto_nf12_and_dynamic(
    tmp_path, native_llama, pattern, cast
):
    """EoRA generation must not forward or dequantize an AUTO_NF12 wrapper.

    The generator needs a dense native weight for `w - wq` and rewrites it with
    `wq + BA`; a packed wrapper has neither a `weight` attribute nor a slot for
    that rewrite. A checkpoint whose `-:` rules exclude a module must also not
    ask the quantized-weight map for that dense module.
    """

    _require_nf12_gpu()
    checkpoint = _quantize_and_save(
        native_llama, tmp_path / "quantized", pattern, "gptq", cast
    )
    adapter_dir = tmp_path / "adapter"
    GPTQModel.adapter.generate(
        adapter=Lora(path=str(adapter_dir), rank=4),
        model_id_or_path=str(native_llama),
        quantized_model_id_or_path=str(checkpoint),
        calibration_dataset=_CALIBRATION_TEXTS,
        calibration_dataset_concat_size=0,
        batch_size=1,
    )
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()


def _packed_tree(module_path: str) -> nn.Module:
    """Build a module tree with AUTO_NF12 payload buffers at `module_path`."""

    root = nn.Module()
    parent = root
    parts = module_path.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        setattr(parent, part, child)
        parent = child
    leaf = nn.Linear(4, 4)
    leaf.register_buffer("nf12_packed", torch.zeros((4, 6), dtype=torch.uint8))
    leaf.register_buffer("nf12_escapes", torch.zeros((0, 8), dtype=torch.int16))
    leaf.register_buffer("nf12_exp_base", torch.zeros((1,), dtype=torch.uint8))
    setattr(parent, parts[-1], leaf)
    return root


def _owner(module_path: str, quantize_config) -> SimpleNamespace:
    return SimpleNamespace(model=_packed_tree(module_path), quantize_config=quantize_config)


def _validate(owner, **kwargs) -> None:
    # Imported at use so a checkout without the writer guard fails these tests
    # individually instead of breaking collection of the whole module.
    from gptqmodel.models.writer import _validate_nf12_payload_ownership

    _validate_nf12_payload_ownership(owner, **kwargs)


def test_save_guard_rejects_nf12_buffers_under_an_excluded_module():
    owner = _owner(
        "model.layers.1.mlp.up_proj",
        QuantizeConfig(bits=4, group_size=32, dynamic={r"-:.*layers\.1\..*": {}}),
    )

    with pytest.raises(ValueError, match="AUTO_NF12 source storage"):
        _validate(owner)


def test_save_guard_rejects_nf12_buffers_for_a_non_nf12_config():
    owner = _owner("model.layers.1.mlp.up_proj", QuantizeConfig(bits=4, group_size=32))

    with pytest.raises(ValueError, match="model.layers.1.mlp.up_proj"):
        _validate(owner)


def test_save_guard_allows_modules_the_config_declares_as_nf12():
    owner = _owner("model.layers.1.mlp.up_proj", NF12Config())

    _validate(owner)


def test_save_guard_ignores_modules_outside_the_requested_prefixes():
    owner = _owner(
        "model.layers.1.mlp.up_proj",
        QuantizeConfig(bits=4, group_size=32, dynamic={r"-:.*layers\.1\..*": {}}),
    )

    _validate(owner, prefixes=["model.layers.2.mlp.up_proj"])
