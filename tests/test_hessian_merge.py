import copy

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
)
@torch.no_grad()
def test_hessian_merge_multi_gpu_matches_serial():
    torch.manual_seed(0)

    in_features = 16
    out_features = 8
    batch_count = 64
    device_count = 2
    per_device = batch_count // device_count
    devices = [torch.device(f"cuda:{idx}") for idx in range(device_count)]

    base = torch.nn.Linear(in_features, out_features, bias=False).eval()
    cfg_serial = QuantizeConfig()
    cfg_multi = copy.deepcopy(cfg_serial)

    serial_module = copy.deepcopy(base).to(devices[0])
    multi_module = copy.deepcopy(base).to(devices[0])

    gptq_serial = GPTQ(serial_module, cfg_serial)
    gptq_multi = GPTQ(multi_module, cfg_multi)

    samples = [torch.randn(1, 1, in_features) for _ in range(batch_count)]

    for idx, sample in enumerate(samples):
        sample_gpu = sample.to(devices[0])
        gptq_serial.add_batch(sample_gpu, torch.empty(0, device=devices[0]), batch_index=idx)
        del sample_gpu
    torch.cuda.synchronize(device=devices[0])

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

    partials_snapshot = {
        dev: tensor.clone()
        for dev, tensor in gptq_multi._device_hessian_partials.items()
    }
    sample_counts_snapshot = dict(gptq_multi._device_sample_counts)

    gptq_multi.finalize_hessian()
    merged_hessian = gptq_multi.H.detach().cpu()
    assert gptq_multi.nsamples == batch_count

    total_samples = sum(sample_counts_snapshot.values())
    assert total_samples == batch_count

    manual_device = gptq_multi.H.device
    manual_accum = torch.zeros(
        (gptq_multi.columns, gptq_multi.columns),
        dtype=torch.float64,
        device=manual_device,
    )
    for dev, tensor in partials_snapshot.items():
        manual_accum.add_(tensor.to(device=manual_device, dtype=torch.float64))
    manual_accum.mul_(2.0 / float(total_samples))
    manual_result = manual_accum.to(dtype=torch.float32).cpu()

    # The materialized Hessian should match the explicit fp64 reduction exactly.
    assert torch.equal(merged_hessian, manual_result)

    # And the merged Hessian should agree with the serial reference to float32 resolution.
    torch.testing.assert_close(merged_hessian, serial_hessian, atol=5e-7, rtol=5e-7)
