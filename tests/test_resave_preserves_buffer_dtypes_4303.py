# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Re-saving a loaded checkpoint must not cast packed/scale buffers (#4303).

The export shell built by ``get_model_with_quantize`` allocates each quantized
module's buffers in its storage dtype. Loading the checkpoint with the model
dtype instead made accelerate cast every floating tensor through
``set_module_tensor_to_device``, which narrowed packed FP8 weights and their
FP32 scales to bf16 (the re-saved checkpoint then refused to load) and GPTQ
FP16 scales to bf16 (a lossy re-save).
"""

import os

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import QuantizeConfig


CALIBRATION = [
    {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }
]


def _save_tokenizer(directory):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        ["tiny calibration text", "another dtype round trip"],
        trainer=trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            vocab_size=64,
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(directory)


def _tiny_source(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        dtype=torch.bfloat16,
    )
    LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory)
    _save_tokenizer(directory)
    return directory


def _quantize(source, save_dir, qcfg, backend):
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=backend)
    model.quantize(
        CalibrationConfig(
            dataset=CALIBRATION,
            batch_size=1,
            min_length=1,
        ),
        backend=backend,
    )
    model.save(str(save_dir))
    del model


def _resave(checkpoint, save_dir, backend):
    model = GPTQModel.load(str(checkpoint), backend=backend, device="cpu")
    model.save(str(save_dir))
    del model


def _shard_tensors(directory):
    """Map every stored tensor name to (dtype, raw bytes)."""
    from safetensors import safe_open

    tensors = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                tensors[name] = (tensor.dtype, tensor.contiguous().view(torch.uint8).clone())
    return tensors


def _checkpoint_bytes(directory):
    return sum(
        os.path.getsize(os.path.join(directory, filename))
        for filename in os.listdir(directory)
        if filename.endswith(".safetensors")
    )


def _capture_export_shell_load(monkeypatch):
    """Record the export-shell load kwargs while still running the real load."""
    from gptqmodel.models import writer

    recorded = []
    real_load = writer.load_checkpoint_in_model_then_tie_weights

    def spy(model, *args, **kwargs):
        recorded.append(dict(kwargs))
        return real_load(model, *args, **kwargs)

    monkeypatch.setattr(writer, "load_checkpoint_in_model_then_tie_weights", spy)
    return recorded


def test_export_shell_load_does_not_cast_dtypes(tmp_path, monkeypatch):
    source = _tiny_source(tmp_path / "source")
    quantized = tmp_path / "quantized"
    resaved = tmp_path / "resaved"
    _quantize(source, quantized, QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"), BACKEND.TORCH)

    recorded = _capture_export_shell_load(monkeypatch)
    _resave(quantized, resaved, BACKEND.TORCH)

    assert len(recorded) == 1
    assert recorded[0]["dtype"] is None, "the export shell must not be cast to the model dtype"

    before = _shard_tensors(quantized)
    after = _shard_tensors(resaved)
    assert set(before) == set(after)
    scales = [name for name in before if name.endswith(".scales")]
    assert scales, "expected GPTQ scales in the checkpoint"
    for name in scales:
        assert after[name][0] is torch.float16, (name, after[name][0])
        assert torch.equal(after[name][1], before[name][1]), name


def test_gptq_resave_is_byte_identical(tmp_path):
    source = _tiny_source(tmp_path / "source")
    quantized = tmp_path / "quantized"
    resaved = tmp_path / "resaved"
    _quantize(source, quantized, QuantizeConfig(bits=4, group_size=32, desc_act=False, device="cpu"), BACKEND.TORCH)
    _resave(quantized, resaved, BACKEND.TORCH)

    before = _shard_tensors(quantized)
    after = _shard_tensors(resaved)
    assert set(before) == set(after)
    for name, (dtype, payload) in before.items():
        assert after[name][0] is dtype, (name, dtype, after[name][0])
        assert torch.equal(after[name][1], payload), name
    assert _checkpoint_bytes(resaved) == _checkpoint_bytes(quantized)


def test_fp8_resave_keeps_packed_weight_and_scales(tmp_path):
    source = _tiny_source(tmp_path / "source")
    quantized = tmp_path / "quantized"
    resaved = tmp_path / "resaved"
    _quantize(
        source,
        quantized,
        QuantizeConfig(method="fp8", device="cpu", offload_to_disk=False),
        BACKEND.FP8_TORCH,
    )
    _resave(quantized, resaved, BACKEND.FP8_TORCH)

    before = _shard_tensors(quantized)
    after = _shard_tensors(resaved)
    assert set(before) == set(after)
    scales = [name for name in before if name.endswith(".weight_scale_inv")]
    packed = [name[: -len("weight_scale_inv")] + "weight" for name in scales]
    assert packed and scales
    for name in packed:
        assert before[name][0] is torch.float8_e4m3fn, (name, before[name][0])
        assert after[name][0] is torch.float8_e4m3fn, (name, after[name][0])
        assert torch.equal(after[name][1], before[name][1]), name
    for name in scales:
        assert after[name][0] is torch.float32, (name, after[name][0])
        assert torch.equal(after[name][1], before[name][1]), name
    assert _checkpoint_bytes(resaved) == _checkpoint_bytes(quantized)

    # The pre-fix re-save wrote bf16 packed weights and refused to reload.
    GPTQModel.load(str(resaved), backend=BACKEND.FP8_TORCH, device="cpu")
