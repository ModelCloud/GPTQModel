# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_module


def _build_module() -> qqq_module.QQQLinear:
    return qqq_module.QQQLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )


def test_qqq_forward_uses_jit_kernel(monkeypatch):
    module = _build_module()
    calls = {}

    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: True)

    def fake_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par):
        calls["gemm"] = {
            "A_shape": tuple(A.shape),
            "A_dtype": A.dtype,
            "B_shape": tuple(B.shape),
            "D_shape": tuple(D.shape),
            "s1_shape": tuple(s1.shape),
            "s2_shape": tuple(s2.shape),
            "s3_shape": tuple(s3.shape),
            "workspace_shape": tuple(workspace.shape),
            "thread_k": thread_k,
            "thread_n": thread_n,
            "sms": sms,
            "max_par": max_par,
        }
        D.copy_(torch.full_like(D, 3.0))

    monkeypatch.setattr(qqq_module, "qqq_gemm", fake_gemm)

    x = torch.randn((2, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["gemm"] == {
        "A_shape": (2, module.in_features),
        "A_dtype": torch.int8,
        "B_shape": tuple(module.B.shape),
        "D_shape": (2, module.out_features),
        "s1_shape": (2, 1),
        "s2_shape": tuple(module.s_channel.shape),
        "s3_shape": tuple(module.s_group.shape),
        "workspace_shape": tuple(module.workspace.shape),
        "thread_k": -1,
        "thread_n": -1,
        "sms": -1,
        "max_par": module.max_par,
    }
    assert out.shape == (2, module.out_features)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_qqq_forward_raises_runtime_error_when_jit_ops_missing(monkeypatch):
    module = _build_module()

    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: False)
    monkeypatch.setattr(qqq_module, "qqq_runtime_error", lambda: "missing qqq jit ops")

    with pytest.raises(ModuleNotFoundError, match="missing qqq jit ops"):
        module(torch.randn((1, module.in_features), dtype=torch.float16))


def _build_grouped_parity_modules(
    cases: list[tuple[int, float, int]],
) -> tuple[qqq_module.QQQLinear, qqq_module.QQQTorchLinear]:
    in_features = 256
    out_features = 128
    group_size = 128
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.zero_()
    scales = torch.ones((out_features, in_features // group_size), dtype=torch.float16)

    for output_index, (code, scale, _) in enumerate(cases):
        scales[output_index, 0] = scale
        linear.weight.data[output_index, 0] = (code - 8) * scale

    s_channel = torch.ones(out_features, dtype=torch.float32)
    modules = []
    for module_cls in (qqq_module.QQQLinear, qqq_module.QQQTorchLinear):
        module = module_cls(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        module.pack(linear, scales, s_channel)
        module.post_init()
        modules.append(module.eval())

    return modules[0].cuda(), modules[1]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qqq_grouped_cuda_matches_torch_rounding_and_saturation():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("QQQ CUDA requires compute capability >= 8.0")
    if not qqq_module.qqq_runtime_available():
        pytest.skip(qqq_module.qqq_runtime_error())

    cases = [
        (15, 22.0, 127),
        (0, 20.0, -128),
        (15, 18.0, 126),
        (9, 0.5, 0),
        (11, 0.5, 2),
        (13, 0.5, 2),
        (7, 0.5, 0),
        (5, 0.5, -2),
        (3, 0.5, -2),
    ]
    cuda_module, torch_module = _build_grouped_parity_modules(cases)
    x = torch.zeros((1, cuda_module.in_features), dtype=torch.float16)
    x[0, 0] = 1.0

    expected = torch.tensor([case[2] for case in cases], dtype=torch.float16)
    cuda_output = cuda_module(x.cuda()).cpu()[0, : len(cases)]
    torch_output = torch_module(x)[0, : len(cases)]

    torch.testing.assert_close(torch_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(cuda_output, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qqq_grouped_cuda_matches_torch_for_regular_values(tokens, dtype):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("QQQ CUDA requires compute capability >= 8.0")
    if not qqq_module.qqq_runtime_available():
        pytest.skip(qqq_module.qqq_runtime_error())

    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.normal_(0, 0.08)
    scales = torch.rand((out_features, 2), dtype=torch.float16) * 0.02 + 0.01
    s_channel = torch.rand(out_features, dtype=torch.float32) * 0.5 + 0.75

    modules = []
    for module_cls in (qqq_module.QQQLinear, qqq_module.QQQTorchLinear):
        module = module_cls(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        module.pack(linear, scales, s_channel)
        module.post_init()
        modules.append(module.eval())

    x = torch.randn((tokens, in_features), dtype=dtype)
    cuda_output = modules[0].cuda()(x.cuda()).cpu()
    torch_output = modules[1](x)

    assert cuda_output.dtype == dtype
    torch.testing.assert_close(cuda_output, torch_output, rtol=0.02, atol=0.02)
