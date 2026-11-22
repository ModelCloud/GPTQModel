# SPDX-License-Identifier: Apache-2.0

import threading
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest
import torch
from torch import nn

from gptqmodel.looper import (
    awq_processor,
    gptq_processor,
    paro_processor,
    qqq_processor,
    weight_only_processor,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import (
    AWQConfig,
    GPTQConfig,
    ParoConfig,
    QQQConfig,
    RTNConfig,
)
from gptqmodel.utils.backend import BACKEND


class _UnrelatedSubtree(nn.Module):
    def named_children(self) -> Iterator[tuple[str, nn.Module]]:
        raise AssertionError("Finalizing a known module traversed an unrelated subtree")


@pytest.mark.parametrize(
    "implementation,processor_cls,config_cls,kernel,embedding",
    [
        (gptq_processor, gptq_processor.GPTQProcessor, GPTQConfig, TorchLinear, False),
        (gptq_processor, gptq_processor.GPTQProcessor, GPTQConfig, TorchLinear, True),
        (
            weight_only_processor,
            weight_only_processor.WeightOnlyProcessor,
            RTNConfig,
            TorchLinear,
            False,
        ),
        (
            weight_only_processor,
            weight_only_processor.WeightOnlyProcessor,
            RTNConfig,
            TorchLinear,
            True,
        ),
        (qqq_processor, qqq_processor.QQQProcessor, QQQConfig, QQQTorchLinear, False),
        (awq_processor, awq_processor.AWQProcessor, AWQConfig, AwqTorchLinear, False),
        (
            paro_processor,
            paro_processor.ParoProcessor,
            ParoConfig,
            ParoLinear,
            False,
        ),
    ],
    ids=["gptq", "gptq-embedding", "rtn", "rtn-embedding", "qqq", "awq", "paro"],
)
def test_finalize_looks_up_only_the_module_being_packed(
    monkeypatch: pytest.MonkeyPatch,
    implementation: ModuleType,
    processor_cls: type,
    config_cls: type,
    kernel: type[nn.Module],
    embedding: bool,
) -> None:
    original = (
        nn.Embedding(128, 128, dtype=torch.float16)
        if embedding
        else nn.Linear(128, 128, bias=False, dtype=torch.float16)
    )
    tree = nn.Module()
    tree.layers = nn.ModuleList([nn.ModuleDict({"proj": original})])
    tree.unrelated = _UnrelatedSubtree()
    name = "layers.0.proj"
    module = NamedModule(original, name="proj", full_name=name, layer_index=0)
    module.state.update(
        q_scales=torch.ones(128, 1),
        q_zeros=torch.zeros(128, 1),
        q_g_idx=torch.zeros(128, dtype=torch.int32),
        q_scales_extra=torch.ones(128),
        pack_weight=original.weight.detach().clone(),
        pairs=torch.zeros(1, dtype=torch.int16),
        theta=torch.zeros(1),
        channel_scales=torch.ones(1),
    )
    processor = object.__new__(processor_cls)
    processor.lock = threading.Lock()
    processor._results = {}
    processor._results_lock = threading.Lock()
    processor.calculate_w_wq_diff = False
    processor.qcfg = config_cls(
        bits=4, group_size=128, sym=True, device="cpu", offload_to_disk=False
    )
    processor.format = processor.qcfg.format
    model = SimpleNamespace(model=tree, qlinear_kernel=kernel, lm_head="lm_head")
    processor.gptq_model = model
    if processor_cls is qqq_processor.QQQProcessor:
        processor._quant_linear_kernel = lambda dtype=None: (kernel, BACKEND.QQQ_TORCH)

    def check_pack_inputs(**kwargs: object) -> None:
        packed = kwargs["qModules"][name]
        expected_type = TorchQuantEmbeddings if embedding else kernel
        assert isinstance(packed, expected_type)
        assert packed is not original
        assert kwargs["layers"] == {name: original}
        # Atomic publication (#5089): until packing returns, the model tree
        # still holds the original float projection, so an interrupted pack
        # cannot leave a kernel without its packed buffers installed.
        assert tree.get_submodule(name) is original
        return "module.pack"

    monkeypatch.setattr(implementation, "pack_module", check_pack_inputs)
    processor.submodule_finalize(module, model)

    replacement = tree.get_submodule(name)
    expected_type = TorchQuantEmbeddings if embedding else kernel
    assert isinstance(replacement, expected_type)
    assert replacement is not original
