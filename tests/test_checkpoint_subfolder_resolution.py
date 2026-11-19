# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Nested quantized checkpoints resolve every artifact under ``subfolder``.

`from_quantized(..., subfolder="nested")` used to forward ``subfolder`` only to
``AutoConfig.from_pretrained``. The quantize config, the checkpoint shards, and
a relative adapter manifest were all resolved at the repository root, so a
nested checkpoint either failed to load or mixed metadata from two directories.
"""

import json
from pathlib import Path

import pytest

import gptqmodel.utils.model as model_utils
from gptqmodel.models import loader as loader_module
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.hf import resolve_checkpoint_subfolder


def _write_nested_quantize_config(root: Path, bits: int = 4) -> Path:
    nested = root / "nested"
    nested.mkdir()
    QuantizeConfig(bits=bits).save_pretrained(nested)
    return nested


@pytest.mark.parametrize("filename", ["quantize_config.json", "quant_config.json", "config.json"])
def test_quantize_config_from_pretrained_honors_subfolder(tmp_path, filename):
    nested = tmp_path / "nested"
    nested.mkdir()
    payload = {"bits": 8, "group_size": 64, "sym": True}
    body = {"quantization_config": payload} if filename == "config.json" else payload
    (nested / filename).write_text(json.dumps(body), encoding="utf-8")

    # The repository root has no config, so success proves the subfolder was used.
    assert not (tmp_path / filename).exists()
    config = QuantizeConfig.from_pretrained(str(tmp_path), subfolder="nested")

    assert config.bits == 8
    assert config.group_size == 64


def test_quantize_config_from_pretrained_rejects_escaping_subfolder(tmp_path):
    (tmp_path / "checkpoint").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    QuantizeConfig(bits=4).save_pretrained(outside)

    for subfolder in ("../outside", "nested/../../outside", "/etc", "C:\\checkpoint", "nested\\..\\.."):
        with pytest.raises(ValueError, match="subfolder"):
            QuantizeConfig.from_pretrained(str(tmp_path / "checkpoint"), subfolder=subfolder)


def test_quantize_config_from_pretrained_rejects_symlinked_subfolder(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    QuantizeConfig(bits=4).save_pretrained(outside)
    (checkpoint / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inside the checkpoint"):
        QuantizeConfig.from_pretrained(str(checkpoint), subfolder="nested")


def test_quantize_config_from_pretrained_without_subfolder_is_unchanged(tmp_path):
    nested = _write_nested_quantize_config(tmp_path, bits=3)

    assert QuantizeConfig.from_pretrained(str(nested)).bits == 3
    assert QuantizeConfig.from_pretrained(str(nested), subfolder="").bits == 3
    assert QuantizeConfig.from_pretrained(str(nested), subfolder=".").bits == 3


def test_get_checkpoints_resolves_local_subfolder(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "model.safetensors").write_bytes(b"")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}), encoding="utf-8"
    )

    is_sharded, archive, basename = model_utils.get_checkpoints(
        str(tmp_path), [".safetensors"], ["model"], subfolder="nested"
    )
    assert is_sharded is True
    assert archive == str(nested / "model.safetensors.index.json")
    assert basename == str(nested / "model")

    (nested / "model.safetensors.index.json").unlink()
    is_sharded, archive, basename = model_utils.get_checkpoints(
        str(tmp_path), [".safetensors"], ["model"], subfolder="nested"
    )
    assert is_sharded is False
    assert archive == str(nested / "model.safetensors")
    assert basename == "model"


def test_get_checkpoints_does_not_fall_back_to_the_repository_root(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="nested"):
        model_utils.get_checkpoints(str(tmp_path), [".safetensors"], ["model"], subfolder="nested")


def test_get_checkpoints_hub_branch_forwards_subfolder(tmp_path, monkeypatch):
    calls = []
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"")

    def fake_cached_file(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs.get("subfolder")))
        if filename == "model.safetensors":
            return str(weights)
        return None

    monkeypatch.setattr(model_utils, "cached_file", fake_cached_file)

    is_sharded, archive, _ = model_utils.get_checkpoints(
        "org/repo", [".safetensors"], ["model"], subfolder="nested", revision="abc"
    )

    assert is_sharded is False
    assert archive == str(weights)
    assert calls == [
        ("org/repo", "model.safetensors.index.json", "nested"),
        ("org/repo", "model.safetensors", "nested"),
    ]


def test_resolve_checkpoint_subfolder_narrows_without_normalizing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()

    assert resolve_checkpoint_subfolder(str(tmp_path), None) == str(tmp_path)
    assert resolve_checkpoint_subfolder(str(tmp_path), "") == str(tmp_path)
    assert resolve_checkpoint_subfolder(str(tmp_path), "a/b") == str(tmp_path / "a" / "b")
    assert resolve_checkpoint_subfolder(str(tmp_path), "a//b/") == str(tmp_path / "a" / "b")


def test_loader_resolves_adapter_manifest_under_subfolder(tmp_path, monkeypatch):
    """`from_quantized` must hand the subfolder root to the manifest resolver."""
    from gptqmodel.models.base import BaseQModel

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "adapter").mkdir()
    (nested / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                    "adapter": {"name": "lora", "rank": 2, "path": "adapter"},
                },
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    class _Stop(Exception):
        """Abort once the loader has resolved the manifest root."""

    def fake_resolve(path, checkpoint_path):
        seen.update(path=path, checkpoint=checkpoint_path)
        raise _Stop

    monkeypatch.setattr(loader_module, "resolve_checkpoint_adapter_path", fake_resolve)

    with pytest.raises(_Stop):
        BaseQModel.from_quantized(str(tmp_path), subfolder="nested", device="cpu")

    assert seen == {"path": "adapter", "checkpoint": str(nested)}


def _write_nested_llama_checkpoint(root: Path) -> Path:
    """Write a nested checkpoint whose config is only readable via ``subfolder``."""
    nested = root / "nested"
    nested.mkdir()
    (nested / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "vocab_size": 8,
                "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
            }
        ),
        encoding="utf-8",
    )
    return nested


def test_config_load_kwargs_forward_subfolder_and_revision():
    """Model-type detection must read the same config file the loader reads."""
    from gptqmodel.models import auto

    assert auto._get_config_load_kwargs({"subfolder": "nested", "revision": "abc123"}) == {
        "subfolder": "nested",
        "revision": "abc123",
    }
    # An unset option keeps the AutoConfig default instead of forcing None.
    assert auto._get_config_load_kwargs({}) == {}
    assert auto._get_config_load_kwargs({"revision": None}) == {}
    # An explicit empty subfolder means "the repository root" and is preserved.
    assert auto._get_config_load_kwargs({"subfolder": ""}) == {"subfolder": ""}
    # The GGUF selector keeps working alongside the checkpoint selectors.
    assert auto._get_config_load_kwargs(
        {loader_module.INTERNAL_HF_GGUF_FILE_KWARG: "m.gguf", "subfolder": "nested"}
    ) == {"subfolder": "nested", "gguf_file": "m.gguf"}


def test_tokenizer_load_kwargs_forward_subfolder_and_revision():
    from gptqmodel.models import loader as loader_module

    assert loader_module._get_tokenizer_load_kwargs({"subfolder": "nested", "revision": "abc123"}) == {
        "subfolder": "nested",
        "revision": "abc123",
    }
    assert loader_module._get_tokenizer_load_kwargs({}) == {}


@pytest.mark.parametrize("entry_point", ["from_quantized", "load"])
def test_public_dispatch_resolves_a_nested_checkpoint_config(tmp_path, entry_point):
    """The reported failure: public dispatch dropped ``subfolder`` before the loader.

    With only ``nested/config.json`` present, dispatch raised
    ``ValueError: Unrecognized model in <root>`` because ``_get_config_load_kwargs``
    returned ``{}`` for ``{"subfolder": "nested"}``. The assertion is on that
    specific failure mode: any later error (missing weights or tokenizer files)
    is outside the dispatch contract being pinned here.
    """
    from gptqmodel import GPTQModel

    _write_nested_llama_checkpoint(tmp_path)
    assert not (tmp_path / "config.json").exists()

    try:
        getattr(GPTQModel, entry_point)(str(tmp_path), subfolder="nested", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        assert "Unrecognized model" not in str(exc), exc


def test_loader_forwards_subfolder_and_revision_to_the_tokenizer(tmp_path, monkeypatch):
    """The tokenizer must come from the same directory and revision as the weights."""
    from gptqmodel.models.base import BaseQModel

    _write_nested_llama_checkpoint(tmp_path)
    seen = {}

    class _Stop(Exception):
        """Abort once the loader has reached the tokenizer stage."""

    def fake_tokenizer(path, **kwargs):
        seen.update(path=path, kwargs=kwargs)
        raise _Stop

    monkeypatch.setattr(loader_module, "load_hf_tokenizer", fake_tokenizer)

    with pytest.raises(_Stop):
        BaseQModel.from_quantized(str(tmp_path), subfolder="nested", revision="abc123", device="cpu")

    assert seen["path"] == str(tmp_path)
    assert seen["kwargs"]["subfolder"] == "nested"
    assert seen["kwargs"]["revision"] == "abc123"


def test_public_dispatch_rejects_an_escaping_subfolder(tmp_path):
    """An escaping subfolder is refused before any config read."""
    from gptqmodel import GPTQModel

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}}),
        encoding="utf-8",
    )

    for subfolder in ("../outside", "/etc", "C:\\checkpoint", "nested\\..\\.."):
        for entry_point in ("from_quantized", "load"):
            with pytest.raises(ValueError, match="subfolder"):
                getattr(GPTQModel, entry_point)(str(checkpoint), subfolder=subfolder, device="cpu")


def test_normalize_checkpoint_subfolder_canonicalizes_without_touching_the_root():
    from gptqmodel.utils.hf import normalize_checkpoint_subfolder

    assert normalize_checkpoint_subfolder(None) == ""
    assert normalize_checkpoint_subfolder("") == ""
    assert normalize_checkpoint_subfolder(".") == ""
    assert normalize_checkpoint_subfolder("a/b") == "a/b"
    assert normalize_checkpoint_subfolder("a//b/") == "a/b"
    with pytest.raises(ValueError, match="subfolder"):
        normalize_checkpoint_subfolder("../x")


def test_quantize_config_from_pretrained_reports_missing_config(tmp_path):
    with pytest.raises(ValueError) as exc:
        QuantizeConfig.from_pretrained(str(tmp_path))

    message = str(exc.value)
    assert str(tmp_path) in message
    for filename in ("quantize_config.json", "quant_config.json", "config.json"):
        assert filename in message


def test_quantize_config_from_pretrained_keeps_parse_errors(tmp_path):
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        QuantizeConfig.from_pretrained(str(tmp_path))


# --------------------------------------------------------------------------- #
# Multimodal processors follow the same subfolder as the config and weights.
#
# `GPTQModel.load(parent, subfolder="nested")` forwards `subfolder` to the
# Transformers model loader but used to construct the wrapper with the
# repository root. `BaseQModel.__init__` then called
# `AutoProcessor.from_pretrained(root)`, so a checkpoint stored entirely under
# `parent/nested` either raised "Unrecognized processing class" or silently
# used an unrelated root processor. The fixtures below put a different valid
# processor at the root so a silent fallback cannot pass.
# --------------------------------------------------------------------------- #

_GEMMA3_SPECIAL_TOKENS = {
    "boi_token": "<start_of_image>",
    "eoi_token": "<end_of_image>",
    "image_token": "<image_soft_token>",
}


def _gemma3_wordlevel_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3, "hello": 4, "world": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_special_tokens=_GEMMA3_SPECIAL_TOKENS,
    )


def _write_gemma3_processor(directory: Path, *, image_seq_length: int) -> None:
    """Write a valid Gemma3 processor whose marker is `image_seq_length`."""
    from transformers import Gemma3ImageProcessorPil, Gemma3Processor

    directory.mkdir(parents=True, exist_ok=True)
    processor = Gemma3Processor(
        # The PIL backend keeps the fixture usable in CPU images without
        # torchvision; AutoProcessor selects it automatically on load.
        image_processor=Gemma3ImageProcessorPil(size={"height": 28, "width": 28}),
        tokenizer=_gemma3_wordlevel_tokenizer(),
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        image_seq_length=image_seq_length,
    )
    processor.save_pretrained(directory)


def _write_gemma3_checkpoint(directory: Path) -> None:
    """Write a tiny local Gemma3 VL checkpoint with its processor files."""
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config(
        text_config={
            "vocab_size": 512,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "sliding_window": 32,
            "layer_types": ["full_attention"],
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
        },
        mm_tokens_per_image=4,
    )
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    Gemma3ForConditionalGeneration(config).save_pretrained(directory)
    _write_gemma3_processor(directory, image_seq_length=4)


def _quantize_gemma3(source: Path, destination: Path) -> None:
    """Quantize the tiny checkpoint on CPU and save it under `destination`."""
    import torch

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig

    wrapped = GPTQModel.load(
        str(source),
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            device="cpu",
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.quantize(
        CalibrationConfig(
            dataset=[
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 8), dtype=torch.long),
                }
            ],
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.save(destination)
    # Saving copies the checkpoint's own processor files; rewrite them so the
    # nested processor is the marker the assertion reads.
    _write_gemma3_processor(destination, image_seq_length=4)


def test_public_load_reads_the_processor_from_the_requested_subfolder(tmp_path):
    """The dense loader must hand the wrapper the selected checkpoint directory."""
    from transformers import AutoProcessor

    from gptqmodel import GPTQModel

    nested = tmp_path / "nested"
    _write_gemma3_checkpoint(nested)
    # A different valid processor at the root: a silent root fallback returns
    # `image_seq_length=8` instead of raising.
    _write_gemma3_processor(tmp_path, image_seq_length=8)

    # Control: Transformers itself resolves the nested processor.
    assert AutoProcessor.from_pretrained(tmp_path, subfolder="nested").image_seq_length == 4

    direct = GPTQModel.load(str(nested), device="cpu")
    assert direct.processor.image_seq_length == 4

    selected = GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu")
    assert selected.processor.image_seq_length == 4


def test_quantized_load_reads_the_processor_from_the_requested_subfolder(tmp_path):
    """`from_quantized` must hand the wrapper the same directory it reads tensors from."""
    from gptqmodel import GPTQModel

    parent = tmp_path / "parent"
    nested = parent / "nested"
    source = tmp_path / "source"
    _write_gemma3_checkpoint(source)
    _quantize_gemma3(source, nested)
    _write_gemma3_processor(parent, image_seq_length=8)

    loaded = GPTQModel.load(str(parent), subfolder="nested", device="cpu")

    assert loaded.processor.image_seq_length == 4
