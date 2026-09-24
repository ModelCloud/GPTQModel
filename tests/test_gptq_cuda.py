import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "ordering,symmetric,columns,group_size,static_groups,actual_hessian",
    [
        ("group_aware", True, 256, 128, False, False),
        ("act_order", True, 256, 128, False, False),
        ("none", False, 256, 128, False, False),
        ("group_aware", True, 192, 128, False, False),
        ("group_aware", True, 256, 128, False, True),
        ("none", True, 256, 16, False, False),
        ("group_aware", False, 256, 32, False, True),
        ("none", True, 256, 64, False, False),
        ("none", True, 256, 256, False, True),
        ("none", True, 256, 512, False, False),
        ("none", True, 256, 1024, False, False),
        ("none", True, 256, -1, False, False),
        ("act_order", True, 256, 32, True, False),
        ("none", False, 256, 64, True, False),
        ("none", True, 256, 48, False, False),
        ("none", True, 256, 1, False, False),
        ("none", False, 256, 7, False, False),
        ("none", True, 256, 129, False, False),
        ("none", True, 192, 48, False, False),
    ],
)
def test_fused_gptq_matches_eager_exactly(
    monkeypatch, ordering, symmetric, columns, group_size, static_groups, actual_hessian,
):
    from gptqmodel.quantization import gptq_cuda

    if not gptq_cuda.block_update_available():
        pytest.skip(gptq_cuda._EXTENSION.last_error_message())

    device = torch.device("cuda:0")
    torch.manual_seed(117)
    weight = torch.randn(13, columns, device=device)
    inputs = torch.randn(1, 192, columns, device=device)
    factor = torch.triu(torch.randn(columns, columns, device=device) * 0.001)
    factor.diagonal().fill_(1.0)

    def quantize(disable_fusion):
        if disable_fusion:
            monkeypatch.setenv("GPTQMODEL_DISABLE_GPTQ_CUDA", "1")
        else:
            monkeypatch.delenv("GPTQMODEL_DISABLE_GPTQ_CUDA", raising=False)

        layer = torch.nn.Linear(columns, 13, bias=False, device=device)
        layer.weight.data.copy_(weight)
        qcfg = QuantizeConfig(
            bits=4,
            group_size=group_size,
            sym=symmetric,
            desc_act=ordering == "act_order",
            act_group_aware=ordering == "group_aware",
            static_groups=static_groups,
        )
        task = GPTQ(layer, qcfg=qcfg)
        task.quantizer.configure(perchannel=True)
        task.add_batch(inputs, None)
        if not actual_hessian:
            task.hessian_inverse = lambda _h: (factor, qcfg.damp_percent)
        return task.quantize(blocksize=128)

    original = gptq_cuda.gptq_block_update
    launches = []

    def counted(*args, **kwargs):
        launches.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gptq_cuda, "gptq_block_update", counted)
    eager = quantize(disable_fusion=True)
    fused = quantize(disable_fusion=False)
    torch.cuda.synchronize(device)

    assert len(launches) == columns // 128
    for index in (0, 1, 2, 3):
        torch.testing.assert_close(fused[index], eager[index], rtol=0, atol=0)
    assert fused[5] == eager[5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_falls_back_when_cuda_extension_is_unavailable(monkeypatch):
    from gptqmodel.quantization import gptq_cuda

    monkeypatch.delenv("GPTQMODEL_DISABLE_GPTQ_CUDA", raising=False)
    monkeypatch.setattr(gptq_cuda, "block_update_available", lambda: False)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("CUDA block update should not run without the extension")

    monkeypatch.setattr(gptq_cuda, "gptq_block_update", should_not_run)
    torch.manual_seed(119)
    layer = torch.nn.Linear(128, 8, bias=False, device="cuda")
    cfg = QuantizeConfig(bits=4, group_size=128, act_group_aware=True)
    task = GPTQ(layer, qcfg=cfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.randn(1, 192, 128, device="cuda"), None)
    result = task.quantize(blocksize=128)
    assert result[0].shape == layer.weight.shape


def test_cpu_quantization_keeps_python_block_loop(monkeypatch):
    from gptqmodel.quantization import gptq_cuda

    def should_not_load():
        raise AssertionError("CPU quantization must not load the CUDA extension")

    monkeypatch.setattr(gptq_cuda, "block_update_available", should_not_load)
    torch.manual_seed(121)
    layer = torch.nn.Linear(128, 8, bias=False)
    cfg = QuantizeConfig(bits=4, group_size=128, act_group_aware=True)
    task = GPTQ(layer, qcfg=cfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.randn(1, 192, 128), None)
    result = task.quantize(blocksize=128)
    assert result[0].shape == layer.weight.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_block_rounding_ties_and_saturation_match_eager():
    from gptqmodel.quantization import gptq_cuda

    if not gptq_cuda.block_update_available():
        pytest.skip(gptq_cuda._EXTENSION.last_error_message())

    values = torch.tensor(
        [-100.0, -2.125, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 2.125, 100.0],
        device="cuda", dtype=torch.float32,
    )
    work = values.repeat(13)[:128].unsqueeze(0).repeat(3, 1)
    hinv = torch.eye(128, device="cuda")
    scale = torch.full((1, 3), 0.25, device="cuda")
    zero = torch.tensor([[8.0, 7.0, 3.0]], device="cuda")
    column_group = torch.zeros(128, device="cuda", dtype=torch.int32)

    eager_work = work.clone()
    eager_q = torch.empty_like(work)
    eager_error = torch.empty_like(work)
    eager_loss = torch.empty_like(work)
    for column in range(128):
        w = eager_work[:, column]
        q = scale.flatten() * (
            torch.clamp(torch.round(w / scale.flatten()) + zero.flatten(), 0, 15)
            - zero.flatten()
        )
        eager_q[:, column] = q
        delta = eager_error[:, column]
        torch.sub(w, q, out=delta)
        eager_loss[:, column] = delta.square()
        eager_work[:, column:] -= delta.unsqueeze(1) * hinv[column, column:]

    cuda_work = work.clone()
    cuda_q = torch.empty_like(work)
    cuda_error = torch.empty_like(work)
    cuda_loss = torch.empty_like(work)
    gptq_cuda.gptq_block_update(
        cuda_work, hinv, scale, zero, column_group,
        cuda_q, cuda_error, cuda_loss,
    )
    for actual, expected in zip(
        (cuda_work, cuda_q, cuda_error, cuda_loss),
        (eager_work, eager_q, eager_error, eager_loss),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_block_preserves_nan_quantization():
    from gptqmodel.quantization import gptq_cuda

    if not gptq_cuda.block_update_available():
        pytest.skip(gptq_cuda._EXTENSION.last_error_message())

    work = torch.ones(1, 128, device="cuda")
    work[0, 0] = float("nan")
    hinv = torch.eye(128, device="cuda")
    scale = torch.ones(1, 1, device="cuda")
    zero = torch.full((1, 1), 8.0, device="cuda")
    column_group = torch.zeros(128, device="cuda", dtype=torch.int32)
    q = torch.empty_like(work)
    errors = torch.empty_like(work)
    losses = torch.empty_like(work)
    gptq_cuda.gptq_block_update(work, hinv, scale, zero, column_group, q, errors, losses)
    assert torch.isnan(q[0, 0])
    assert torch.isnan(errors[0, 0])
    assert torch.isnan(losses[0, 0])
