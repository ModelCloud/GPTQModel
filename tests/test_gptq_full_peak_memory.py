"""Checks for dropping the dense Hessian after GPTQ inversion."""

import pytest
import torch

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.kernel import KernelConfig, configure_kernel


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("ordering", ["none", "desc_act", "act_group_aware"])
def test_nan_retry_uses_original_hessian_diagonal(device_type, ordering, monkeypatch):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    configure_kernel(KernelConfig(gptq_fused_block_update=False))
    device = torch.device(device_type)
    torch.manual_seed(821)
    weight = torch.randn(4, 16, device=device)
    inputs = torch.randn(1, 48, 16, device=device)

    def make_task(mock_quantization):
        layer = torch.nn.Linear(16, 4, bias=False, device=device)
        layer.weight.data.copy_(weight)
        config = QuantizeConfig(
            bits=4,
            group_size=4,
            desc_act=ordering == "desc_act",
            act_group_aware=ordering == "act_group_aware",
            mock_quantization=mock_quantization,
        )
        task = GPTQ(layer, qcfg=config)
        task.quantizer.configure(perchannel=True)
        return task

    task = make_task(mock_quantization=False)
    task.add_batch(inputs, None)
    captured_hessian = None
    natural_diag = None
    original_neutralize = gptq_mod._neutralize_dead_columns

    def recording_neutralize(hessian, weight, *extra):
        # This runs before desc_act / act_group_aware permutes H, so it is the
        # natural-order diagonal the retry must restore. Keep the first call:
        # the retry also neutralizes, but by then H is already permuted.
        nonlocal natural_diag
        if natural_diag is None:
            natural_diag = hessian.diagonal().clone()
        return original_neutralize(hessian, weight, *extra)

    monkeypatch.setattr(gptq_mod, "_neutralize_dead_columns", recording_neutralize)

    def nan_inverse(hessian):
        nonlocal captured_hessian
        captured_hessian = hessian.clone()
        factor = torch.eye(16, device=device)
        factor[0, 0] = float("nan")
        return factor, task.qcfg.damp_percent

    task.hessian_inverse = nan_inverse
    retried = task.quantize(blocksize=8)
    assert task.qcfg.mock_quantization
    assert captured_hessian is not None
    assert natural_diag is not None

    reference = make_task(mock_quantization=True)
    reference.H = torch.diag(natural_diag)
    reference.nsamples = task.nsamples
    expected = reference.quantize(blocksize=8)

    for index in (0, 1, 2, 3):
        torch.testing.assert_close(retried[index], expected[index], rtol=0, atol=0)
    assert retried[5] == expected[5]


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_hessian_is_released_before_block_updates(device_type, monkeypatch):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    configure_kernel(KernelConfig(gptq_fused_block_update=False))
    device = torch.device(device_type)
    task = GPTQ(
        torch.nn.Linear(16, 4, bias=False, device=device),
        qcfg=QuantizeConfig(bits=4, group_size=4, act_group_aware=True),
    )
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.randn(1, 48, 16, device=device), None)
    original_quantize = task.quantizer.quantize
    observed = []

    def check_hessian_released(*args, **kwargs):
        observed.append(task.H is None)
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(task.quantizer, "quantize", check_hessian_released)
    task.quantize(blocksize=8)
    assert observed and all(observed)
