# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#5720: static-group quantizer lookup must not read the permutation per column.

With ``static_groups`` plus activation ordering (``desc_act`` or
``act_group_aware``) the block pass turned the column permutation into a Python
int once per column. Each ``perm[column].item()`` is an
``aten::_local_scalar_dense`` call, which on CUDA drains the stream and blocks
the host - one sync per column, before every fused-kernel launch. GPTAQ and FOEM
paid the same read through ``Tensor.__index__`` when indexing ``groups`` with a
0-dim device tensor.

The map is now built once per module, so the counts below are constant in the
column count. The old code returned ``columns + 2`` host reads for GPTQ and
``columns`` ``__index__`` calls for GPTAQ/FOEM.
"""

import copy

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import gptq as gptq_module
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


IN, OUT, TOK, GROUP = 256, 4, 64, 128


def test_the_static_group_map_matches_the_per_column_lookup():
    from gptqmodel.quantization.gptq import _static_group_of_column

    perm = torch.tensor([0, 1, 129, 127, 255, 256, 5])
    assert _static_group_of_column(perm, 128) == [0, 0, 1, 0, 1, 2, 0]
    # No permutation means the natural group of each column.
    assert _static_group_of_column(None, 128) is None
    # Ungrouped modules never index ``groups``.
    assert _static_group_of_column(perm, -1) is None


def _layer():
    torch.manual_seed(0)
    return nn.Linear(IN, OUT, bias=False).eval(), torch.randn(TOK, IN)


def _gptq(desc_act=False, act_group_aware=False):
    linear, X = _layer()
    qcfg = QuantizeConfig(
        bits=4, group_size=GROUP, sym=True, static_groups=True,
        desc_act=desc_act, act_group_aware=act_group_aware,
    )
    task = GPTQ(copy.deepcopy(linear), qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(X.unsqueeze(0), None)
    return task


def _count_host_reads(task, monkeypatch, attribute):
    calls = {"n": 0}
    original = getattr(torch.Tensor, attribute)

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(torch.Tensor, attribute, counting)
    try:
        task.quantize(blocksize=128)
    finally:
        monkeypatch.setattr(torch.Tensor, attribute, original)
    return calls["n"]


@pytest.mark.parametrize("ordering", ["desc_act", "act_group_aware"])
def test_gptq_reads_the_static_group_permutation_once_per_module(ordering, monkeypatch):
    task = _gptq(**{ordering: True})
    reads = _count_host_reads(task, monkeypatch, "item")
    # The two remaining reads are the module's loss reductions, not the
    # permutation; the pre-fix code added one per column.
    assert reads <= 8, f"{ordering}: {reads} host reads for {IN} columns"
    assert reads < IN


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize("ordering", ["desc_act", "act_group_aware"])
def test_gptaq_and_foem_never_index_a_device_tensor_per_column(kind, ordering, monkeypatch):
    linear, X = _layer()
    if kind == "gptaq":
        qcfg = QuantizeConfig(bits=4, group_size=GROUP, sym=True, static_groups=True,
                              gptaq=GPTAQConfig(), **{ordering: True})
        task = GPTAQ(_named(linear, X), qcfg=qcfg)
    else:
        qcfg = QuantizeConfig(bits=4, group_size=GROUP, sym=True, static_groups=True,
                              foem=FOEMConfig(), **{ordering: True})
        task = FOEM(_named(linear, X), qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(X.unsqueeze(0), None)

    assert _count_host_reads(task, monkeypatch, "__index__") == 0


def _named(linear, X):
    named = NamedModule(copy.deepcopy(linear), name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state["native_inp"] = [X]
    return named


def test_the_map_decides_which_static_group_serves_each_column(monkeypatch):
    """A wrong map must change the output: the block pass really reads it."""

    task = _gptq(desc_act=True)
    reference = task.quantize(blocksize=128)[0].clone()

    task = _gptq(desc_act=True)
    monkeypatch.setattr(
        gptq_module, "_static_group_of_column", lambda perm, group_size: [0] * IN
    )
    shifted = task.quantize(blocksize=128)[0].clone()

    assert not torch.equal(reference, shifted)
