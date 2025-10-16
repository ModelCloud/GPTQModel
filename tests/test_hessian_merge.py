import copy

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="requires at least 4 CUDA devices",
)
@torch.no_grad()
def test_hessian_merge_multi_gpu_matches_serial():
    torch.manual_seed(0)

    in_features = 16
    out_features = 8
    batch_count = 100
    per_device = batch_count // 4
    devices = [torch.device(f"cuda:{idx}") for idx in range(4)]

    base = torch.nn.Linear(in_features, out_features, bias=False).eval()
    cfg_serial = QuantizeConfig()
    cfg_multi = copy.deepcopy(cfg_serial)

    serial_module = copy.deepcopy(base)
    multi_module = copy.deepcopy(base).to(devices[0])

    gptq_serial = GPTQ(serial_module, cfg_serial)
    gptq_multi = GPTQ(multi_module, cfg_multi)

    samples = [torch.randn(1, 1, in_features) for _ in range(batch_count)]

    for idx, sample in enumerate(samples):
        gptq_serial.add_batch(sample, torch.empty(0), batch_index=idx)

    gptq_serial.finalize_hessian()
    serial_hessian = gptq_serial.H.detach().cpu()
    assert gptq_serial.nsamples == batch_count

    for device_idx, device in enumerate(devices):
        start = device_idx * per_device
        end = start + per_device
        for idx in range(start, end):
            sample_gpu = samples[idx].to(device)
            gptq_multi.add_batch(sample_gpu, torch.empty(0, device=device), batch_index=idx)
            del sample_gpu
        torch.cuda.synchronize(device=device)

    gptq_multi.finalize_hessian()
    merged_hessian = gptq_multi.H.detach().cpu()
    assert gptq_multi.nsamples == batch_count

    torch.testing.assert_close(merged_hessian, serial_hessian, atol=1e-6, rtol=1e-6)
