import copy

import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@torch.no_grad()
def test_out_of_order_batches_flush_in_sequence():
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

    gptq.add_batch(x1, y1, batch_index=1)
    assert gptq.nsamples == 0

    gptq.add_batch(x0, y0, batch_index=0)

    reference.add_batch(x0, y0, batch_index=0)
    reference.add_batch(x1, y1, batch_index=1)

    torch.testing.assert_close(gptq.H, reference.H)
    assert gptq.nsamples == reference.nsamples
    assert not gptq._pending_updates
