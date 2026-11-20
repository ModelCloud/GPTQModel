# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The full fp32 Hessian must not stay resident through the block pass.

After the inverse, the block pass only reads ``Hinv``/``P``. GPTQ kept a local
``H_permuted`` alive next to ``self.H`` in the ``desc_act`` and
``act_group_aware`` orderings, and GPTAQ/FOEM kept the local ``H`` captured by
their inverse closures, so a second ``columns ** 2 * 4`` buffer stayed
resident for the whole module (issue #5717).
"""

import copy
import weakref

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import quantizer as quantizer_module
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


IN_FEATURES = 512
OUT_FEATURES = 64
TOKENS = 1024


def _named_linear(seed: int = 0) -> NamedModule:
    torch.manual_seed(seed)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).eval()
    module = NamedModule(
        copy.deepcopy(linear), name="proj", full_name="l.0.proj", layer_index=0
    )
    module.state["native_inp"] = []
    return module


def _probe(kind: str, base: dict):
    """Run one quantize pass and record whether the Hessian is alive in it."""

    torch.manual_seed(0)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).eval()
    x = torch.randn(TOKENS, IN_FEATURES)
    hessian = 2.0 / TOKENS * (x.T @ x)

    state = {}
    original_inverse = GPTQ.hessian_inverse
    original_quantize = quantizer_module.Quantizer.quantize

    def spy_inverse(self, H):
        state["ref"] = weakref.ref(H)
        state["alive"] = []
        return original_inverse(self, H)

    def spy_quantize(self, w):
        if "ref" in state:
            state["alive"].append(state["ref"]() is not None)
        return original_quantize(self, w)

    GPTQ.hessian_inverse = spy_inverse
    quantizer_module.Quantizer.quantize = spy_quantize
    try:
        module = NamedModule(
            copy.deepcopy(linear), name="proj", full_name="l.0.proj", layer_index=0
        )
        module.state["native_inp"] = []
        if kind == "gptq":
            quantizer = GPTQ(module, qcfg=QuantizeConfig(**base))
            quantizer.quantizer.configure(perchannel=True)
            quantizer.add_batch(x.unsqueeze(0), None)
        elif kind == "gptaq":
            quantizer = GPTAQ(
                module,
                qcfg=QuantizeConfig(
                    **base, gptaq=GPTAQConfig(alpha=0.25, device="cpu")
                ),
            )
            quantizer.quantizer.configure(perchannel=True)
            quantizer.H = hessian.clone()
            quantizer.dXXT = torch.zeros_like(hessian)
            quantizer.nsamples = TOKENS
            quantizer._gptaq_batch_request_count = 1
        else:
            quantizer = FOEM(
                module,
                qcfg=QuantizeConfig(
                    **base, foem=FOEMConfig(alpha=0.0, beta=0.2, device="cpu")
                ),
            )
            quantizer.quantizer.configure(perchannel=True)
            quantizer.H = hessian.clone()
            quantizer.nsamples = TOKENS
        quantizer.quantize(blocksize=128)
    finally:
        GPTQ.hessian_inverse = original_inverse
        quantizer_module.Quantizer.quantize = original_quantize
    return state["alive"]


CASES = [
    pytest.param(
        "gptq",
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "act_group_aware": False,
        },
        id="gptq-natural",
    ),
    pytest.param(
        "gptq",
        {"bits": 4, "group_size": 128, "sym": True, "desc_act": False},
        id="gptq-act-group-aware",
    ),
    pytest.param(
        "gptq",
        {"bits": 4, "group_size": 128, "sym": True, "desc_act": True},
        id="gptq-desc-act",
    ),
    pytest.param(
        "gptaq",
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "act_group_aware": False,
        },
        id="gptaq",
    ),
    pytest.param(
        "foem",
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "act_group_aware": False,
        },
        id="foem",
    ),
]


@pytest.mark.parametrize("kind,base", CASES)
def test_hessian_is_released_before_the_block_pass(kind, base):
    alive = _probe(kind, base)

    assert alive, "the block pass never quantized a column"
    assert not any(alive), (
        f"{kind}: the fp32 Hessian stayed alive during "
        f"{sum(alive)}/{len(alive)} per-column quantize calls"
    )
