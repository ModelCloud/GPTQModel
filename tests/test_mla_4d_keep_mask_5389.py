# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MLA `kv_b_proj` must obey the calibration keep mask like every sibling (#5389).

transformers' MLA attention expands the compressed KV latent as a single-head
``[B, 1, S, kv_lora_rank]`` tensor before calling ``kv_b_proj``. Every capture
hook matched a keep mask by its leading dims, so ``shape[:2]`` was ``(B, 1)``
and the masked path never ran: at ``batch_size > 1`` the padded positions of
the whole layer landed in that one projection's sample (46% of it on
DeepSeek-V2-Lite with unsorted calibration), while ``q_proj``/``o_proj``/...
saw only real rows. ``exclude_attention_sinks`` was ignored there too.

#5343 fixed only the ``.t()`` crash in GPTAQ/FOEM/QQQ. The masking predicates
are shared by GPTQ, the GPTAQ/FOEM native side, AWQ, QQQ, MACaQ, EXL3 and the
NVFP4 activation range, so the fix belongs in one helper.
"""

from __future__ import annotations

import threading

import pytest
import torch
from tokenizers import Tokenizer, models
from torch import nn
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.looper import gptq_processor as gptq_processor_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.quantization import gptq as gptq_module
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig
from gptqmodel.utils.attn_mask import align_single_head_capture, apply_keep_mask_bt


BATCH = 2
SEQ = 3
LATENT = 5
# Four kept rows out of six: a right pad on row 0 and a left pad on row 1.
MASK = torch.tensor([[True, True, False], [False, True, True]])


def _latent(hidden: int = LATENT) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5389)
    return torch.randn(BATCH, 1, SEQ, hidden, generator=generator)


def _mask_tls(mask=MASK):
    tls = threading.local()
    tls.value = mask
    tls.padding_value = mask
    return tls


def test_align_helper_squeezes_only_a_single_head_capture():
    latent = _latent()
    squeezed = align_single_head_capture(latent, MASK)
    assert tuple(squeezed.shape) == (BATCH, SEQ, LATENT)

    # Anything else is returned untouched.
    for other in (
        torch.randn(BATCH, SEQ, LATENT),
        torch.randn(BATCH * SEQ, LATENT),
        torch.randn(BATCH, 2, SEQ, LATENT),
        torch.randn(BATCH, 1, SEQ + 1, LATENT),
    ):
        assert align_single_head_capture(other, MASK) is other
    assert align_single_head_capture(latent, None) is latent


def test_apply_keep_mask_handles_a_single_head_capture():
    kept = apply_keep_mask_bt(_latent(), MASK)
    assert tuple(kept.shape) == (int(MASK.sum()), LATENT)


def test_gptq_hook_masks_a_single_head_capture():
    class _Task:
        module = nn.Linear(LATENT, LATENT)

        def __init__(self):
            self.batches = []

        def add_batch(self, inp, out, batch_index=None):
            self.batches.append(inp.clone())

    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor._mask_tls = _mask_tls()
    processor._batch_tls = threading.local()
    processor._batch_tls.index = 0
    processor.preserve_batch_keep_mask = True
    processor.qcfg = None
    processor._activation_headroom_probe = False
    task = _Task()
    processor.tasks = {"self_attn.kv_b_proj": task}

    latent = _latent()
    processor.pre_process_fwd_hook("self_attn.kv_b_proj")(
        nn.Linear(LATENT, LATENT), (latent,), torch.randn(BATCH, 1, SEQ, LATENT)
    )

    # One entry per sample with at least one real token, padding dropped.
    assert [tuple(batch.shape) for batch in task.batches] == [
        (1, int(MASK[0].sum()), LATENT),
        (1, int(MASK[1].sum()), LATENT),
    ]
    torch.testing.assert_close(task.batches[0][0], latent[0, 0][MASK[0]])
    torch.testing.assert_close(task.batches[1][0], latent[1, 0][MASK[1]])


def test_native_split_masks_a_single_head_capture():
    processor = NativeProcessor.__new__(NativeProcessor)
    processor._native_share_lock = threading.Lock()
    processor._native_share_caches = {}

    latent = _latent()
    entries = processor._split_batch_samples(latent, MASK, "cpu")

    assert [tuple(entry.shape) for entry in entries] == [
        (int(MASK[0].sum()), LATENT),
        (int(MASK[1].sum()), LATENT),
    ]
    torch.testing.assert_close(entries[0], latent[0, 0][MASK[0]])
    torch.testing.assert_close(entries[1], latent[1, 0][MASK[1]])


def test_awq_capture_masks_a_single_head_capture():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor._mask_tls = _mask_tls()

    kept = processor._apply_active_keep_mask_to_feature(_latent())

    assert tuple(kept.shape) == (int(MASK.sum()), LATENT)


def test_calibration_rows_mask_a_single_head_capture():
    processor = LoopProcessor.__new__(LoopProcessor)
    processor._mask_tls = threading.local()
    no_padding = torch.ones(BATCH, SEQ, dtype=torch.bool)
    sink_mask = no_padding.clone()
    sink_mask[:, 0] = False
    # `exclude_attention_sinks` on: the calibration mask is the padding mask
    # minus the sink positions.
    processor._mask_tls.value = sink_mask
    processor._mask_tls.padding_value = no_padding

    kept = processor.calibration_rows(_latent())

    assert tuple(kept.shape) == (BATCH * (SEQ - 1), LATENT)


def test_activation_amax_ignores_padded_rows_of_a_single_head_capture(monkeypatch):
    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.lock = threading.Lock()
    processor._mask_tls = _mask_tls()
    processor._activation_amax = {}
    processor._activation_headroom_probe = False
    monkeypatch.setattr(GPTQProcessor, "_target_recipe", lambda self, name, module=None: "nvfp4")
    monkeypatch.setattr(GPTQProcessor, "_activation_key", staticmethod(lambda name, module=None: "k"))
    monkeypatch.setattr(gptq_processor_module, "nvfp4_uses_headroom", lambda recipe: False)

    latent = torch.ones(BATCH, 1, SEQ, LATENT)
    latent[0, 0, 2] = 1e6  # right pad of sample 0
    latent[1, 0, 0] = 1e6  # left pad of sample 1

    processor._record_activation_amax("self_attn.kv_b_proj", latent, None)

    # Padding is never a real activation, so it cannot widen the range.
    assert float(processor._activation_amax["k"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The issue's equivalence check: batch_size=1 and batch_size>1 must agree
# ---------------------------------------------------------------------------

VOCAB = 128
LENGTHS = (9, 14, 6, 11, 17, 8, 13, 10)
ATTENTION = (
    "self_attn.q_a_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.q_b_proj",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)


def _tiny_mla_model(directory):
    config = Glm4MoeLiteConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        max_position_embeddings=64,
        first_k_dense_replace=2,
        n_group=1,
        topk_group=1,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    Glm4MoeLiteForCausalLM(config).eval().save_pretrained(directory)
    tokenizer = Tokenizer(
        models.WordLevel(vocab={str(index): index for index in range(VOCAB)}, unk_token="0")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="0",
    ).save_pretrained(directory)


def _attention_rows(directory, batch_size, monkeypatch):
    rows = {}
    original = gptq_module.GPTQ.add_batch

    def spy(self, inp, out, batch_index=None):
        rows[str(self.name)] = rows.get(str(self.name), 0) + int(inp.numel() // inp.shape[-1])
        return original(self, inp, out, batch_index=batch_index)

    monkeypatch.setattr(gptq_module.GPTQ, "add_batch", spy)
    calibration = [
        {"input_ids": torch.randint(4, VOCAB, (length,)).tolist()} for length in LENGTHS
    ]
    model = GPTQModel.load(
        str(directory),
        GPTQConfig(bits=4, group_size=32, device="cpu"),
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=batch_size, min_length=1))
    return {name: rows[name] for name in rows if name in ATTENTION}


@pytest.mark.slow
def test_kv_b_proj_sample_is_the_same_at_batch_size_one_and_many(tmp_path, monkeypatch):
    """`kv_b_proj` must not collect the padded rows the other projections drop."""

    single = tmp_path / "single"
    batched = tmp_path / "batched"
    single.mkdir()
    batched.mkdir()
    _tiny_mla_model(single)
    _tiny_mla_model(batched)

    at_one = _attention_rows(single, 1, monkeypatch)
    at_many = _attention_rows(batched, 4, monkeypatch)

    assert set(at_one) == set(ATTENTION), at_one
    assert at_many == at_one, (
        "batched calibration must accumulate the same rows per attention projection; "
        f"batch_size=4 {at_many} vs batch_size=1 {at_one}"
    )
