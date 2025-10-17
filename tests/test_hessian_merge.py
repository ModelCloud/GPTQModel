import copy

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask


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

    max_abs_diff = (merged_hessian - serial_hessian).abs().max().item()
    print(
        "[hessian-no-mask] "
        f"serial_nsamples={gptq_serial.nsamples} "
        f"multi_nsamples={gptq_multi.nsamples} "
        f"max_abs_diff={max_abs_diff:.6e}"
    )

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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason="requires CUDA devices >= 8 to exercise GPUs 6 and 7",
)
@torch.no_grad()
def test_hessian_merge_multi_gpu_with_attention_mask():
    torch.manual_seed(123)

    in_features = 32
    out_features = 16
    batch_size = 3
    seq_len = 21
    batch_count = 10

    device_serial = torch.device("cuda:6")
    devices = [torch.device("cuda:6"), torch.device("cuda:7")]

    base = torch.nn.Linear(in_features, out_features, bias=False).eval()
    cfg_serial = QuantizeConfig(mock_quantization=True, desc_act=False)
    cfg_multi = copy.deepcopy(cfg_serial)

    serial_module = copy.deepcopy(base).to(device_serial)
    multi_module = copy.deepcopy(base).to(device_serial)

    gptq_serial = GPTQ(serial_module, cfg_serial)
    gptq_multi = GPTQ(multi_module, cfg_multi)

    samples = []
    for _ in range(batch_count):
        hidden = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)
        mask = torch.ones(batch_size, seq_len, dtype=torch.int32)
        for row in range(batch_size):
            # ensure at least one valid token per row, trim a random tail portion
            cutoff = torch.randint(1, seq_len + 1, ()).item()
            if cutoff < seq_len:
                mask[row, cutoff:] = 0
        samples.append((hidden, mask))

    total_kept_tokens = 0
    for idx, (hidden, mask) in enumerate(samples):
        hidden_gpu = hidden.to(device_serial)
        mask_gpu = mask.to(device_serial)
        keep = normalize_seq_mask(mask_gpu)
        trimmed = apply_keep_mask_bt(hidden_gpu, keep)
        total_kept_tokens += trimmed.shape[0]
        gptq_serial.add_batch(trimmed, torch.empty(0, device=device_serial), batch_index=idx)
    torch.cuda.synchronize(device=device_serial)

    gptq_serial.finalize_hessian()
    serial_hessian = gptq_serial.H.detach().cpu()
    assert gptq_serial.nsamples == total_kept_tokens

    per_device = batch_count // len(devices)
    remainder = batch_count % len(devices)
    start = 0

    device_token_counts = {}
    for device_idx, device in enumerate(devices):
        extra = 1 if device_idx < remainder else 0
        end = start + per_device + extra
        for idx in range(start, end):
            hidden, mask = samples[idx]
            hidden_gpu = hidden.to(device)
            mask_gpu = mask.to(device)
            keep = normalize_seq_mask(mask_gpu)
            trimmed = apply_keep_mask_bt(hidden_gpu, keep)
            device_token_counts[device] = device_token_counts.get(device, 0) + trimmed.shape[0]
            gptq_multi.add_batch(trimmed, torch.empty(0, device=device), batch_index=idx)
        torch.cuda.synchronize(device=device)
        start = end

    assert sum(device_token_counts.values()) == total_kept_tokens

    partial_counts_snapshot = dict(gptq_multi._device_sample_counts)
    assert partial_counts_snapshot == device_token_counts

    gptq_multi.finalize_hessian()
    merged_hessian = gptq_multi.H.detach().cpu()

    max_abs_diff = (merged_hessian - serial_hessian).abs().max().item()
    print(
        "[hessian-mask] "
        f"serial_tokens={total_kept_tokens} "
        f"multi_tokens={gptq_multi.nsamples} "
        f"per_device={{{', '.join(f'{str(dev)}:{count}' for dev, count in device_token_counts.items())}}} "
        f"max_abs_diff={max_abs_diff:.6e}"
    )

    assert gptq_multi.nsamples == total_kept_tokens
    torch.testing.assert_close(merged_hessian, serial_hessian, atol=5e-7, rtol=5e-7)
