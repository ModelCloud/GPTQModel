# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.qqq_processor import QQQProcessor


@pytest.mark.parametrize("method", ["qqq", "paro"])
def test_eora_handoff_preserves_independent_reconstruction(method):
    cls = QQQProcessor if method == "qqq" else ParoQuantProcessor
    processor = object.__new__(cls)
    processor.lock = threading.Lock()
    processor.calculate_w_wq_diff = True
    module = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(4, 4)),
        name="proj",
        layer_index=0,
        state={},
    )
    reconstructed = torch.full((4, 4), 0.75)
    if method == "qqq":
        processor.tasks = {
            "proj": SimpleNamespace(
                quantize=lambda: (
                    reconstructed,
                    torch.ones(4),
                    torch.zeros(4),
                    torch.arange(4),
                    0.0,
                    0.0,
                    0.0,
                    torch.ones(4),
                    1,
                ),
                free=lambda: None,
            )
        }
        processor.qcfg = SimpleNamespace(dynamic=None)
        (
            processor.durations,
            processor.avg_losses,
            processor.module_names,
            processor.log,
        ) = [], [], [], []
        for name in (
            "draw_progress",
            "module_feature_summary",
            "module_dtype_size_summary",
            "log_new_row",
        ):
            setattr(processor, name, lambda *args: None)
        processor.formatted_fwd_time = lambda: "0"
        processor.process(module)
    else:
        result = SimpleNamespace(
            pseudo_weight=reconstructed,
            pack_weight=reconstructed,
            q_scales=torch.ones(4),
            q_zeros=torch.zeros(4),
            pairs=torch.zeros(4, dtype=torch.int16),
            theta=torch.zeros(4),
            channel_scales=torch.ones(4),
        )
        processor._apply_optimization_result(
            module, result, module.weight.detach().clone()
        )
    assert torch.equal(module.state["w_wq_diff"], torch.full((4, 4), 0.25))
    module.weight.data.fill_(9)
    assert torch.equal(module.state["wq"], torch.full((4, 4), 0.75))
    assert module.state["wq"].device.type == "cpu"


def test_qqq_packs_uncorrected_weight_after_eora(monkeypatch):
    import gptqmodel.looper.qqq_processor as implementation

    processor = object.__new__(QQQProcessor)
    processor.lock = threading.Lock()
    processor.calculate_w_wq_diff = True
    base = torch.full((4, 4), 0.75)
    module = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(4, 4)),
        name="proj",
        state={
            "wq": base,
            "q_zeros": None,
            "q_scales": None,
            "q_g_idx": None,
            "q_scales_extra": None,
        },
    )
    monkeypatch.setattr(implementation, "find_modules", lambda model: {})

    def stop_before_packing():
        raise RuntimeError("packing boundary")

    processor._quant_linear_kernel = stop_before_packing
    with pytest.raises(RuntimeError, match="packing boundary"):
        processor.submodule_finalize(module, SimpleNamespace(model=None))
    assert torch.equal(module.weight, base)
    assert "wq" not in module.state
