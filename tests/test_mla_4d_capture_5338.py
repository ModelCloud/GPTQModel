# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MLA attention captures a 4-D latent for ``kv_b_proj`` (#5338).

transformers 5.22 expands the compressed KV latent as a single-head 4-D tensor
(``[B, 1, S, kv_lora_rank]``) before calling ``kv_b_proj``. The GPTAQ, FOEM and
QQQ accumulators only flattened 2-D and 3-D captures, so ``inp.t()`` raised
``t() expects a tensor with <= 2 dimensions`` on every MLA model. GPTQ already
flattens unconditionally.

Each case feeds one 4-D capture and asserts the accumulator saw every
``B * S`` row: the Hessian is ``[hidden, hidden]`` and ``nsamples`` counts the
flattened tokens, not the outer dimensions.
"""

from __future__ import annotations

import torch
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.qqq import QQQ


BATCH = 2
SEQ = 5
HIDDEN = 8
TOKENS = BATCH * SEQ


def _latent() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5338)
    return torch.randn(BATCH, 1, SEQ, HIDDEN, generator=generator)


def _named(linear: nn.Linear) -> NamedModule:
    named = NamedModule(
        linear,
        name="kv_b_proj",
        full_name="model.layers.0.self_attn.kv_b_proj",
        layer_index=0,
    )
    # A routed expert skips the native-delta pairing, which keeps this test on
    # the flatten path without a NativeProcessor capture. GPTAQ pops the
    # native-input list at construction, so the empty capture is still set.
    named.state["is_routed_expert"] = True
    named.state["native_inp"] = []
    return named


def test_gptaq_flattens_a_4d_mla_capture():
    linear = nn.Linear(HIDDEN, 4, bias=False, dtype=torch.float32).eval()
    task = GPTAQ(
        _named(linear),
        qcfg=QuantizeConfig(
            bits=4,
            group_size=4,
            sym=True,
            desc_act=False,
            gptaq=GPTAQConfig(alpha=0.25, device="cpu"),
        ),
    )
    task.quantizer.configure(perchannel=True)
    task.H = None
    task.dXXT = None
    task.nsamples = 0

    token_count, xtx, device = task.process_batch(_latent(), batch_index=0)

    assert token_count == TOKENS
    assert task.nsamples == TOKENS
    assert xtx is None
    assert device == torch.device("cpu")
    assert tuple(task.H.shape) == (HIDDEN, HIDDEN)


def test_foem_flattens_a_4d_mla_capture():
    linear = nn.Linear(HIDDEN, 4, bias=False, dtype=torch.float32).eval()
    task = FOEM(
        _named(linear),
        qcfg=QuantizeConfig(
            bits=4,
            group_size=4,
            sym=True,
            desc_act=False,
            foem=FOEMConfig(alpha=0, beta=0.2, device="cpu"),
        ),
    )
    task.quantizer.configure(perchannel=True)
    task.H = None
    task.nsamples = 0

    token_count, xtx, device = task.process_batch(_latent(), batch_index=0)

    assert token_count == TOKENS
    assert task.nsamples == TOKENS
    assert xtx is None
    assert device == torch.device("cpu")
    assert tuple(task.H.shape) == (HIDDEN, HIDDEN)


def test_qqq_flattens_a_4d_mla_capture():
    linear = nn.Linear(HIDDEN, 4, bias=False, dtype=torch.float32).eval()
    task = QQQ(linear)
    task.quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=4)

    task.add_batch(_latent(), None)
    hessian = task.materialize_hessian()

    assert task.nsamples == TOKENS
    assert tuple(hessian.shape) == (HIDDEN, HIDDEN)
