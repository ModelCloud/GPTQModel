# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ADMM-Q quantize -> save -> reload on a tiny local Llama.

The saved artifact must stay an ordinary GPTQ-packed checkpoint while the
published ``method`` and the ADMM solver settings recorded in ``meta`` survive
the round trip, so a reload restores ``ADMMQConfig`` with the same values.
"""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import METHOD, ADMMQConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _tokenizer():
    words = (
        "[UNK]", "[PAD]", "<s>", "</s>", "A", "short", "calibration", "sentence",
        "about", "fox", "Another", "moon", ".", "runs",
    )
    inner = Tokenizer(
        models.WordLevel(vocab={word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )


def _packed_modules(model):
    return [
        (name, module)
        for name, module in model.model.named_modules()
        if hasattr(module, "qweight")
    ]


def _packed_state(module):
    return {
        key: tensor.detach().cpu()
        for key, tensor in module.state_dict().items()
        if key in ("qweight", "qzeros", "scales", "g_idx")
    }


@pytest.mark.parametrize("desc_act", [False, True])
def test_tiny_llama_admmq_roundtrip(tmp_path, desc_act):
    torch.manual_seed(41)
    source = tmp_path / "source"
    saved = tmp_path / "quantized"
    source.mkdir()
    tokenizer = _tokenizer()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer.save_pretrained(source)

    qcfg = ADMMQConfig(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=desc_act,
        max_iter=24,
        update_iter=3,
        switch_iter=6,
        early_stop_tol=1e-8,
        pairswap_rounds=1,
        pairswap_pairs=8,
        pairswap_batch=2,
        offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), qcfg, device=DEVICE, dtype=DTYPE)
    calibration = [
        "A short calibration sentence about a fox. A short calibration sentence about a fox.",
        "Another sentence about the moon. Another sentence about the moon.",
    ]
    # The portable GPTQ kernel is available on every device family and keeps
    # the packed layout identical on both sides of the round trip.
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1),
        backend=BACKEND.GPTQ_TORCH,
    )

    # ADMM-Q keeps its own method; the tensors are ordinary GPTQ packing.
    assert model.quantize_config.method is METHOD.ADMMQ
    selected = _packed_modules(model)
    assert len(selected) == 14
    expected = {name: _packed_state(module) for name, module in selected}

    model.save(str(saved))
    loaded = GPTQModel.load(
        str(saved), backend=BACKEND.GPTQ_TORCH, device=DEVICE, dtype=DTYPE
    )

    assert isinstance(loaded.quantize_config, ADMMQConfig)
    assert loaded.quantize_config.method is METHOD.ADMMQ
    assert loaded.quantize_config.max_iter == 24
    assert loaded.quantize_config.switch_iter == 6
    assert loaded.quantize_config.pairswap_pairs == 8
    assert loaded.quantize_config.desc_act is desc_act

    restored = _packed_modules(loaded)
    assert [name for name, _ in restored] == [name for name, _ in selected]
    for name, module in restored:
        state = _packed_state(module)
        assert state.keys() == expected[name].keys()
        for key, tensor in expected[name].items():
            torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)

    ids = tokenizer("A fox runs", return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode():
        logits = loaded.model(input_ids=ids).logits
    assert logits.shape == (1, ids.shape[1], 32)
    assert torch.isfinite(logits).all()
