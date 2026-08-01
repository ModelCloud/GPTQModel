# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end and unit tests for per-layer incremental quantization."""

import os
import shutil
import tempfile
from typing import Dict

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT


pytestmark = [pytest.mark.model, pytest.mark.slow]


def _make_dummy_tokenizer(vocab_size: int = 132) -> PreTrainedTokenizerFast:
    """Create a minimal fast tokenizer with a fixed vocabulary."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=128,
    )
    fast.add_tokens([f"<t{i}>" for i in range(vocab_size)])
    return fast


def _build_tiny_llama(vocab_size: int) -> LlamaForCausalLM:
    """Return a 2-layer Llama-shaped model with tiny hidden dims."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
        use_cache=False,
    )
    return LlamaForCausalLM(cfg)


def _calibration(vocab_size: int, length: int = 16, count: int = 1):
    """Return a small list of calibration dicts."""
    return [
        {"input_ids": torch.randint(0, vocab_size, (length,)).tolist(), "attention_mask": [1] * length}
        for _ in range(count)
    ]


def _count_quantized_modules_by_layer(model: LlamaForCausalLM) -> Dict[str, int]:
    """Count BaseQuantLinear modules under each transformer layer prefix."""
    counts: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear):
            if name.startswith("model.layers."):
                layer_idx = name.split(".")[2]
                counts[layer_idx] = counts.get(layer_idx, 0) + 1
    return counts


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true" and not torch.cuda.is_available(),
    reason="CPU-only GitHub Actions runners are too slow for this end-to-end test.",
)
def test_per_layer_quant_and_requant():
    """Quantize layer 0, save, reload, requant layer 1 with a different config, save full, reload, forward."""
    torch.manual_seed(0)
    torch.set_num_threads(8)

    tokenizer = _make_dummy_tokenizer()
    vocab_size = len(tokenizer)
    model = _build_tiny_llama(vocab_size)

    tmp_dir = tempfile.mkdtemp()
    partial_dir = tempfile.mkdtemp()
    full_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        qcfg0 = QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            format=FORMAT.GPTQ,
            offload_to_disk=False,
        )

        qm0 = GPTQModel.load(tmp_dir, qcfg0, device="cpu", backend="auto")
        qm0.quantize(_calibration(vocab_size), layer_scope=0)

        counts0 = _count_quantized_modules_by_layer(qm0.model)
        assert counts0 == {"0": 7}, f"expected layer 0 quantized, got {counts0}"

        qm0.save(partial_dir)

        # Reload the partial checkpoint and continue with a *different* config.
        qcfg1 = QuantizeConfig(
            bits=3,
            group_size=32,
            desc_act=False,
            sym=True,
            format=FORMAT.GPTQ,
            offload_to_disk=False,
        )

        qm1 = GPTQModel.load(partial_dir, device="cpu", backend="auto")
        qm1.requant(_calibration(vocab_size, length=24), quantize_config=qcfg1, layer_scope=1)

        counts1 = _count_quantized_modules_by_layer(qm1.model)
        assert counts1 == {"0": 7, "1": 7}, f"expected both layers quantized, got {counts1}"

        # The saved dynamic map must remember layer 0 used 4 bits while base is 3 bits.
        dynamic = qm1.quantize_config.dynamic or {}
        positive_patterns = {k: v for k, v in dynamic.items() if k.startswith("+:")}
        negative_patterns = {k: v for k, v in dynamic.items() if k.startswith("-:")}
        assert not negative_patterns, f"saved config should not contain scope negatives: {negative_patterns}"
        assert positive_patterns, "saved config should contain positive override for differently-quantized layer"
        assert any(v.get("bits") == 4 for v in positive_patterns.values()), positive_patterns

        qm1.save(full_dir)

        # Full checkpoint reload and forward pass.
        qm2 = GPTQModel.load(full_dir, device="cpu", backend="auto")
        ids = torch.randint(0, vocab_size, (1, 8))
        with torch.inference_mode():
            out = qm2.model(ids)
        assert out.logits.shape == (1, 8, vocab_size)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(partial_dir, ignore_errors=True)
        shutil.rmtree(full_dir, ignore_errors=True)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true" and not torch.cuda.is_available(),
    reason="CPU-only GitHub Actions runners are too slow for this end-to-end test.",
)
def test_per_layer_quant_slice_scope():
    """Quantize a slice of layers and verify only that slice is quantized."""
    torch.manual_seed(0)
    torch.set_num_threads(8)

    tokenizer = _make_dummy_tokenizer()
    vocab_size = len(tokenizer)
    model = _build_tiny_llama(vocab_size)

    tmp_dir = tempfile.mkdtemp()
    save_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        qcfg = QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            format=FORMAT.GPTQ,
            offload_to_disk=False,
        )

        qm = GPTQModel.load(tmp_dir, qcfg, device="cpu", backend="auto")
        qm.quantize(_calibration(vocab_size), layer_scope=slice(0, 1))

        counts = _count_quantized_modules_by_layer(qm.model)
        assert counts == {"0": 7}, f"expected layer 0 quantized, got {counts}"

        qm.save(save_dir)

        qm2 = GPTQModel.load(save_dir, device="cpu", backend="auto")
        ids = torch.randint(0, vocab_size, (1, 8))
        with torch.inference_mode():
            out = qm2.model(ids)
        assert out.logits.shape == (1, 8, vocab_size)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(save_dir, ignore_errors=True)


def test_resolve_layer_scope_indices_unit():
    """Unit-test the layer-scope index normalization helper."""
    from gptqmodel.models.base import BaseQModel

    class FakeModel(BaseQModel):
        @property
        def model_type(self):
            return "test"

        def extract_layers_node(self):
            return ["model.layers"]

    fake = FakeModel.__new__(FakeModel)
    layer_names = ["model.layers.0", "model.layers.1", "model.layers.2"]

    assert fake._resolve_layer_scope_indices(None, layer_names, 3) == {0, 1, 2}
    assert fake._resolve_layer_scope_indices(1, layer_names, 3) == {1}
    assert fake._resolve_layer_scope_indices(slice(0, 2), layer_names, 3) == {0, 1}
    assert fake._resolve_layer_scope_indices([0, 2], layer_names, 3) == {0, 2}
    assert fake._resolve_layer_scope_indices(["model\\.layers\\.1"], layer_names, 3) == {1}

    with pytest.raises(IndexError):
        fake._resolve_layer_scope_indices(5, layer_names, 3)

    with pytest.raises(ValueError):
        fake._resolve_layer_scope_indices("non_matching_regex", layer_names, 3)
