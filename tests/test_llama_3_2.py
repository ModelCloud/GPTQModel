# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end CPU GPTQ quantization test on a tiny Llama 3.2 1B-like model.

This uses a single-layer Llama architecture with the 3.2 1B MLP ratio
(hidden_size=2048, intermediate_size=8192) so the `mlp.down_proj` hot path
exercises the compiled CPU block code for a realistic large shape, while
keeping the full-model test small enough for CPU-only CI.
"""

import os
import shutil
import tempfile

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT

pytestmark = [pytest.mark.model, pytest.mark.slow]


def _make_dummy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal fast tokenizer that can be saved alongside the model."""
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
    # Add a small vocabulary so random input ids are valid.
    fast.add_tokens([f"<t{i}>" for i in range(256)])
    return fast


def _build_llama_1b_like(vocab_size: int) -> LlamaForCausalLM:
    """Return a single-layer Llama 3.2 1B-shaped model with random weights."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
        use_cache=False,
    )
    return LlamaForCausalLM(cfg)


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true" and not torch.cuda.is_available(),
    reason="CPU-only GitHub Actions runners are too slow for this end-to-end test.",
)
def test_llama_3_2_1b_cpu_gptq():
    """Quantize, save, reload, and run inference on a tiny Llama-3.2-1B-like model."""
    torch.manual_seed(0)
    torch.set_num_threads(8)

    tokenizer = _make_dummy_tokenizer()
    vocab_size = len(tokenizer)
    model = _build_llama_1b_like(vocab_size)

    tmp_dir = tempfile.mkdtemp()
    save_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        qcfg = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            format=FORMAT.GPTQ,
            offload_to_disk=False,
        )

        # 16-token calibration sequence on CPU.
        input_ids = torch.randint(0, vocab_size, (16,)).tolist()
        calibration = [{"input_ids": input_ids, "attention_mask": [1] * 16}]

        qm = GPTQModel.load(
            tmp_dir,
            qcfg,
            device="cpu",
            backend="auto",
            trust_remote_code=False,
        )
        qm.quantize(calibration, batch_size=1)

        # The model should now contain quantized linear layers.
        assert any(isinstance(m, BaseQuantLinear) for m in qm.model.modules())

        qm.save(save_dir)

        # Reload quantized and run a forward pass.
        qm2 = GPTQModel.load(save_dir, device="cpu", backend="auto")
        ids = torch.randint(0, vocab_size, (1, 8))
        with torch.inference_mode():
            out = qm2.model(ids)
        assert out.logits.shape == (1, 8, vocab_size)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(save_dir, ignore_errors=True)
