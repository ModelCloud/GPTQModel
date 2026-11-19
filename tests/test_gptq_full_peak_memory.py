"""Checks for peak memory during GPTQ and QQQ quantization."""

import pytest
import torch

from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import QQQConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ
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


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["gptq", "qqq"])
def test_block_error_propagation_respects_product_budget(device_type, method, monkeypatch):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    blocksize = 4
    base = torch.tensor([0.75, -2.75, 5.25, -8.0, 1.625, 4.625, 7.0, -1.125, 2.625, -6.25])
    weight = torch.stack([base.roll(row) * 2.0**row for row in range(5)])
    rows, columns = weight.shape
    hessian = torch.eye(columns)
    for left, right, coupling in [(0, 4, 0.75), (2, 8, -0.625), (5, 9, 0.7)]:
        hessian[left, right] = hessian[right, left] = coupling

    damp = 0.01
    config_type = QuantizeConfig if method == "gptq" else QQQConfig
    config = config_type(
        bits=4, group_size=-1, sym=False, desc_act=False, act_group_aware=False,
        damp_percent=damp, fallback=None, offload_to_disk=False,
    )
    layer = torch.nn.Linear(columns, rows, device=device_type, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    if method == "gptq":
        task = GPTQ(layer, config)
        task.quantizer.configure(perchannel=True)
    else:
        task = QQQ(layer, config)
        task.quantizer.configure(bits=4, perchannel=True, sym=False, groupsize=-1)
        task._materialized_samples = 1
    task.H = hessian.to(device_type, copy=True)
    task.nsamples = 1

    # The first update needs several row tiles, including a ragged last tile.
    # The narrower second update fits in a single product.
    budget = 2 * (columns - blocksize) * weight.element_size()
    monkeypatch.setattr(gptq_mod, "_BLOCK_ERROR_PRODUCT_BYTES", budget, raising=False)
    products = []
    matmul = torch.Tensor.matmul

    def track_matmul(left, right):
        product = matmul(left, right)
        if left.ndim == 2 and left.shape[1] == blocksize:
            products.append((product.shape, product.numel() * product.element_size()))
        return product

    monkeypatch.setattr(torch.Tensor, "matmul", track_matmul)
    actual, scales, zeros, groups, _, loss, *_ = task.quantize(blocksize=blocksize)

    # An independent FP64 oracle immediately propagates each column's error
    # to every subsequent column, without any block partitioning.
    damped = hessian.double()
    damped.diagonal().add_(damp * damped.diagonal().mean())
    factor = torch.linalg.cholesky(torch.linalg.inv(damped), upper=True)
    work = weight.double()
    scale = (work.amax(1, keepdim=True) - work.amin(1, keepdim=True)) / 15
    zero = (-work.amin(1, keepdim=True) / scale).round()
    without_feedback = scale * ((work / scale).round().add(zero).clamp(0, 15) - zero)
    expected = torch.empty_like(work)
    expected_loss = 0.0
    for column in range(columns):
        codes = (work[:, column] / scale[:, 0]).round().add(zero[:, 0]).clamp(0, 15)
        expected[:, column] = scale[:, 0] * (codes - zero[:, 0])
        error = (work[:, column] - expected[:, column]) / factor[column, column]
        work[:, column + 1:] -= error[:, None] * factor[column, column + 1:]
        expected_loss += error.square().sum().item() / 2

    assert (expected[:, blocksize:] != without_feedback[:, blocksize:]).any()
    torch.testing.assert_close(actual.cpu(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), scale.float(), rtol=0, atol=0)
    torch.testing.assert_close(zeros.cpu(), zero.float(), rtol=0, atol=0)
    torch.testing.assert_close(groups.cpu(), torch.zeros(columns, dtype=torch.int32))
    assert loss == pytest.approx(expected_loss, rel=1e-6, abs=1e-6)
    assert products and max(size for _, size in products) <= budget
    assert [shape for shape, _ in products if shape[1] < columns - blocksize] == [
        (rows, columns - 2 * blocksize)
    ]
