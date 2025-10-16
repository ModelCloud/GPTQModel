import copy

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@torch.no_grad()
def test_out_of_order_batches_finalize_matches_reference():
    torch.manual_seed(0)

    module = torch.nn.Linear(4, 4)
    reference_module = copy.deepcopy(module)

    cfg = QuantizeConfig()
    gptq = GPTQ(module, cfg)
    reference = GPTQ(reference_module, copy.deepcopy(cfg))

    x0 = torch.randn(1, 1, 4)
    x1 = torch.randn(1, 1, 4)

    y0 = module(x0)
    y1 = module(x1)

    # Add batches out of order to ensure accumulation is order agnostic.
    gptq.add_batch(x1, y1, batch_index=1)
    gptq.add_batch(x0, y0, batch_index=0)

    gptq.finalize_hessian()

    reference.add_batch(x0, y0, batch_index=0)
    reference.add_batch(x1, y1, batch_index=1)
    reference.finalize_hessian()

    assert gptq.H is not None
    torch.testing.assert_close(gptq.H, reference.H)
    assert gptq.nsamples == reference.nsamples
    assert not gptq._device_hessian_partials


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finalize_hessian_preserves_device(monkeypatch):
    module = torch.nn.Linear(4, 4).cuda()
    cfg = QuantizeConfig()
    gptq = GPTQ(module, cfg)

    module_device = module.weight.device

    def fake_process_batch(self, inp):
        xtx = torch.eye(self.columns, dtype=torch.float32, device=module_device)
        return 1, xtx.clone(), module_device

    monkeypatch.setattr(GPTQ, "process_batch", fake_process_batch, raising=False)

    inp = torch.zeros(1, device=module_device)

    gptq.add_batch(inp, inp, batch_index=1)
    gptq.add_batch(inp, inp, batch_index=0)

    # No Hessian materialized until finalize is invoked.
    assert gptq.H is None
    assert module_device in gptq._device_hessian_partials

    gptq.finalize_hessian()

    assert gptq.H is not None
    assert gptq.H.device == module_device
    assert not gptq._device_hessian_partials

    torch.cuda.synchronize()
