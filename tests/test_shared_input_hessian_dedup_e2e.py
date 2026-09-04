# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end CPU GPTQ quantization of a tiny Llama with shared-input Hessian dedup on vs off."""

import tempfile
from typing import Dict, List, Tuple

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.quantization import GPTQ
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig


@pytest.fixture(scope="module")
def tiny_llama_dir() -> str:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    tmp = tempfile.mkdtemp()
    LlamaForCausalLM(config).save_pretrained(tmp)

    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    tok.train_from_iterator(
        ["the quick brown fox", "lorem ipsum dolor", "hello world"],
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=128),
    )
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>").save_pretrained(tmp)
    return tmp


def _calibration() -> List[Dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(1)
    return [
        {
            "input_ids": torch.randint(4, 128, (n,), generator=g),
            "attention_mask": torch.ones(n, dtype=torch.long),
        }
        for n in (12, 9, 15)
    ]


def _quantize(model_dir: str, dedup: bool, monkeypatch) -> Tuple[List[Tuple[Dict[str, str], List[str]]], Dict[str, torch.Tensor], Dict[str, int]]:
    with monkeypatch.context() as mp:
        return _quantize_patched(model_dir, dedup, mp)


def _quantize_patched(model_dir: str, dedup: bool, monkeypatch) -> Tuple[List[Tuple[Dict[str, str], List[str]]], Dict[str, torch.Tensor], Dict[str, int]]:
    elections: List[Tuple[Dict[str, str], List[str]]] = []
    weights: Dict[str, torch.Tensor] = {}
    captured: Dict[str, int] = {}

    orig_end = GPTQProcessor.end_shared_input_capture

    def spy_end(self, subset_names):
        elections.append((dict(self._shared_input_leaders), list(subset_names)))
        for name in subset_names:
            task = self.tasks.get(name)
            if task is not None:
                captured[task._named_module.full_name] = task.fwd_counter
        return orig_end(self, subset_names)

    orig_quantize = GPTQ.quantize

    def spy_quantize(self, *args, **kwargs):
        result = orig_quantize(self, *args, **kwargs)
        weights[self._named_module.full_name] = result[0].detach().to("cpu").clone()
        return result

    monkeypatch.setattr(GPTQProcessor, "end_shared_input_capture", spy_end)
    monkeypatch.setattr(GPTQ, "quantize", spy_quantize)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        hessian=HessianConfig(dedup_shared_inputs=dedup),
    )
    model = GPTQModel.load(model_dir, quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(_calibration(), batch_size=1, backend=BACKEND.GPTQ_TORCH, calibration_data_min_length=1)
    return elections, weights, captured


def test_tiny_llama_dedup_matches_independent_hessians(tiny_llama_dir, monkeypatch):
    elections_on, weights_on, captured_on = _quantize(tiny_llama_dir, True, monkeypatch)
    elections_off, weights_off, captured_off = _quantize(tiny_llama_dir, False, monkeypatch)

    qkv = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    gate_up = ["mlp.gate_proj", "mlp.up_proj"]
    expected_on = [
        ({"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj"}, qkv),
        ({}, ["self_attn.o_proj"]),
        ({"mlp.up_proj": "mlp.gate_proj"}, gate_up),
        ({}, ["mlp.down_proj"]),
    ] * 2
    assert elections_on == expected_on
    assert elections_off == [({}, names) for _, names in expected_on]

    # Followers never ran their own capture hook; leaders and singletons captured every batch.
    for full_name, count in captured_on.items():
        if full_name.endswith(("k_proj", "v_proj", "up_proj")):
            assert count == 0, full_name
        else:
            assert count == captured_off[full_name] > 0, full_name

    # k_proj/v_proj have a narrower output than q_proj (GQA) yet share q_proj's Hessian.
    assert weights_on["model.layers.0.self_attn.k_proj"].shape[0] < weights_on["model.layers.0.self_attn.q_proj"].shape[0]

    assert weights_on.keys() == weights_off.keys()
    assert len(weights_on) == 14
    for full_name in weights_on:
        assert torch.equal(weights_on[full_name], weights_off[full_name]), full_name
