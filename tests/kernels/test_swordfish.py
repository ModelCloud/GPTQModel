# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import unittest

import pytest
import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.swordfish import SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.swordfish import (
    prewarm_swordfish_extension,
    _validate_swordfish_device_support,
    swordfish_runtime_available,
    swordfish_runtime_error,
)


def _supported_swordfish_device(device_index: int) -> bool:
    try:
        with torch.cuda.device(device_index):
            return _validate_swordfish_device_support()
    except (RuntimeError, AssertionError):
        return False


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    if not _supported_swordfish_device(torch.cuda.current_device()):
        major, minor = torch.cuda.get_device_capability()
        return f"Swordfish does not support the current CUDA device (sm{major}{minor})"
    return None


_SKIP_REASON = _skip_reason()


def _quantize_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool,
    g_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make deterministic GPTQ codes in the same [out, group] convention as pack()."""

    out_features, in_features = weight.shape
    effective_group_size = in_features if group_size == -1 else group_size
    if g_idx is None:
        g_idx = torch.arange(in_features, dtype=torch.int32) // effective_group_size
    num_groups = int(g_idx.max().item()) + 1
    levels = (1 << bits) - 1
    half_range = 1 << (bits - 1)

    q = torch.empty_like(weight)
    scales = torch.empty(
        (out_features, num_groups), dtype=weight.dtype, device=weight.device
    )
    zeros = torch.empty_like(scales)
    for group in range(num_groups):
        block = weight[:, g_idx == group]
        if sym:
            scale = block.abs().amax(dim=1, keepdim=True) / (half_range - 1)
            zero = torch.full_like(scale, half_range)
        else:
            block_min = block.amin(dim=1, keepdim=True)
            block_max = block.amax(dim=1, keepdim=True)
            scale = (block_max - block_min) / levels
            zero = torch.round(
                -block_min / scale.clamp_min(torch.finfo(weight.dtype).eps)
            )
            zero.clamp_(0, levels)
        scale = scale.clamp_min(torch.finfo(weight.dtype).eps)
        codes = torch.round(block / scale + (half_range if sym else zero))
        codes.clamp_(0, levels)
        q[:, g_idx == group] = codes
        scales[:, group : group + 1] = scale
        zeros[:, group : group + 1] = zero
    return q, scales, zeros, g_idx


def _pack_torch_reference(
    *,
    bits: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
    in_features: int,
    out_features: int,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    device: torch.device,
) -> TorchLinear:
    linear = nn.Linear(
        in_features, out_features, bias=False, dtype=weight.dtype, device="cpu"
    )
    with torch.no_grad():
        linear.weight.copy_(weight)

    torch_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
        pack_dtype=torch.int32,
    )
    torch_linear.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    torch_linear = torch_linear.to(device=device)
    torch_linear.post_init()
    return torch_linear


def _copy_to_swordfish(
    torch_linear: TorchLinear,
    device: torch.device,
    dtype: torch.dtype,
) -> SwordfishLinear:
    sf = SwordfishLinear(
        bits=torch_linear.bits,
        group_size=torch_linear.requested_group_size,
        desc_act=torch_linear.desc_act,
        sym=torch_linear.sym,
        in_features=torch_linear.in_features,
        out_features=torch_linear.out_features,
        bias=False,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    sf.qweight = nn.Parameter(
        torch_linear.qweight.data.detach().clone().to(device=device).contiguous(),
        requires_grad=False,
    )
    sf.scales = nn.Parameter(
        torch_linear.scales.data.detach()
        .clone()
        .to(device=device, dtype=dtype)
        .contiguous(),
        requires_grad=False,
    )
    if torch_linear.g_idx is not None and torch_linear.g_idx.numel() > 0:
        sf.g_idx = nn.Parameter(
            torch_linear.g_idx.data.detach().clone().to(device=device).contiguous(),
            requires_grad=False,
        )
    if torch_linear.qzeros is not None and torch_linear.qzeros.numel() > 0:
        sf.qzeros = nn.Parameter(
            torch_linear.qzeros.data.detach().clone().to(device=device).contiguous(),
            requires_grad=False,
        )
    sf.post_init()
    return sf


def _make_pair(
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[TorchLinear, SwordfishLinear]:
    torch.manual_seed(17 + in_features + out_features + group_size)
    weight = torch.randn((out_features, in_features), dtype=dtype, device="cpu") * 0.5
    g_idx = None
    if desc_act:
        permutation = torch.randperm(in_features)
        weight = weight[:, permutation]
        effective_group_size = in_features if group_size == -1 else group_size
        g_idx = (torch.arange(in_features, dtype=torch.int32) // effective_group_size)[
            permutation
        ]
    _, scales, zeros, g_idx = _quantize_weight(
        weight, bits=4, group_size=group_size, sym=sym, g_idx=g_idx
    )
    torch_linear = _pack_torch_reference(
        bits=4,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        weight=weight,
        scales=scales,
        zeros=zeros,
        g_idx=g_idx,
        device=device,
    )
    return torch_linear, _copy_to_swordfish(torch_linear, device, dtype)


def _shape_for_rows(rows: int, in_features: int, rank: int) -> tuple[int, ...]:
    if rank == 2:
        return rows, in_features
    if rank == 3:
        return 1, rows, in_features
    if rank == 4:
        return 1, 1, rows, in_features
    raise ValueError(f"unsupported test rank {rank}")


def _assert_pair(
    torch_linear: TorchLinear,
    sf: SwordfishLinear,
    x: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        ref = torch.matmul(x, torch_linear.dequantize_weight().to(dtype=x.dtype))
        actual = sf(x)
    assert actual.shape == ref.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, ref, rtol=0, atol=0.15)
    assert (actual - ref).abs().float().mean().item() < 0.05
    return actual


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestSwordfishKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            prewarm_swordfish_extension()
        except Exception as exc:
            raise unittest.SkipTest(
                f"Swordfish runtime unavailable: {swordfish_runtime_error() or exc}"
            ) from exc
        if not swordfish_runtime_available():
            raise unittest.SkipTest(
                f"Swordfish runtime unavailable: {swordfish_runtime_error()}"
            )

    @parameterized.expand(
        [
            (
                "fp16_channelwise_decode_edges",
                64,
                64,
                -1,
                True,
                False,
                torch.float16,
                2,
                (1, 16, 17, 32, 33),
            ),
            (
                "bf16_asym_rank3_kn_tails",
                192,
                320,
                32,
                False,
                False,
                torch.bfloat16,
                3,
                (47, 48, 49, 64, 65),
            ),
            (
                "fp16_desc_act_rank4_n_tail",
                256,
                320,
                64,
                True,
                True,
                torch.float16,
                4,
                (95, 96, 127, 128),
            ),
            (
                "bf16_asym_desc_act_n_tail",
                256,
                192,
                128,
                False,
                True,
                torch.bfloat16,
                2,
                (1, 17, 65, 129),
            ),
            (
                "bf16_prefill_edges",
                256,
                256,
                128,
                True,
                False,
                torch.bfloat16,
                2,
                (55, 56, 57, 95, 96, 97),
            ),
        ]
    )
    def test_swordfish_matches_deterministic_torch_reference(
        self,
        _case_name,
        in_features,
        out_features,
        group_size,
        sym,
        desc_act,
        dtype,
        rank,
        rows,
    ):
        device = torch.device("cuda", torch.cuda.current_device())
        torch_linear, sf = _make_pair(
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            dtype=dtype,
            device=device,
        )
        sf.eval()
        for row_count in rows:
            x = torch.randn(
                _shape_for_rows(row_count, in_features, rank),
                dtype=dtype,
                device=device,
            )
            _assert_pair(torch_linear, sf, x)

    def test_swordfish_accepts_only_int32_packing(self):
        kwargs = dict(
            bits=4,
            group_size=64,
            desc_act=False,
            sym=True,
            in_features=64,
            out_features=64,
            dtype=torch.float16,
        )
        ok, error = SwordfishLinear.validate(pack_dtype=torch.int32, **kwargs)
        self.assertTrue(ok, error)
        self.assertIsNone(error)
        for pack_dtype in (torch.int8, torch.int16, torch.int64):
            ok, error = SwordfishLinear.validate(pack_dtype=pack_dtype, **kwargs)
            self.assertFalse(ok)
            self.assertIsInstance(error, NotImplementedError)

    def test_swordfish_rejects_trainable_nonempty_configuration(self):
        ok, error = SwordfishLinear.validate(
            bits=4,
            group_size=64,
            desc_act=False,
            sym=True,
            in_features=64,
            out_features=64,
            pack_dtype=torch.int32,
            dtype=torch.float16,
            trainable=True,
        )
        self.assertFalse(ok)
        self.assertIsInstance(error, NotImplementedError)

        device = torch.device("cuda", torch.cuda.current_device())
        _, sf = _make_pair(
            in_features=64,
            out_features=64,
            group_size=64,
            sym=True,
            desc_act=False,
            dtype=torch.float16,
            device=device,
        )
        sf.eval()
        with self.assertRaises(NotImplementedError):
            sf.train()

    def test_swordfish_noncontiguous_activation(self):
        device = torch.device("cuda", torch.cuda.current_device())
        torch_linear, sf = _make_pair(
            in_features=192,
            out_features=320,
            group_size=32,
            sym=True,
            desc_act=False,
            dtype=torch.float16,
            device=device,
        )
        x = torch.randn((17, 384), dtype=torch.float16, device=device)[:, ::2]
        self.assertFalse(x.is_contiguous())
        _assert_pair(torch_linear, sf, x)

    def test_swordfish_empty_backward_is_supported(self):
        device = torch.device("cuda", torch.cuda.current_device())
        torch_linear, sf = _make_pair(
            in_features=64,
            out_features=64,
            group_size=-1,
            sym=True,
            desc_act=False,
            dtype=torch.float16,
            device=device,
        )
        del torch_linear
        sf.eval()
        x = torch.empty((0, 64), dtype=torch.float16, device=device, requires_grad=True)
        out = sf(x)
        self.assertEqual(tuple(out.shape), (0, 64))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(tuple(x.grad.shape), tuple(x.shape))
        self.assertEqual(int(torch.count_nonzero(x.grad)), 0)

    @pytest.mark.xfail(
        strict=True,
        reason="Swordfish torch.ops do not yet register fake/meta kernels for fullgraph capture",
    )
    def test_swordfish_compile_fullgraph_eager_contract(self):
        device = torch.device("cuda", torch.cuda.current_device())
        torch_linear, sf = _make_pair(
            in_features=64,
            out_features=64,
            group_size=-1,
            sym=True,
            desc_act=False,
            dtype=torch.float16,
            device=device,
        )
        del torch_linear
        sf.eval()
        x = torch.randn((1, 64), dtype=torch.float16, device=device)
        compiled = torch.compile(sf, fullgraph=True, backend="eager")
        with torch.inference_mode():
            eager = sf(x)
            actual = compiled(x)
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)

    def test_swordfish_cuda_graph_capture_and_replay(self):
        device = torch.device("cuda", torch.cuda.current_device())
        _, sf = _make_pair(
            in_features=64,
            out_features=64,
            group_size=-1,
            sym=True,
            desc_act=False,
            dtype=torch.float16,
            device=device,
        )
        sf.eval()
        static_x = torch.randn((1, 64), dtype=torch.float16, device=device)
        replay_inputs = (torch.randn_like(static_x), torch.randn_like(static_x))
        with torch.inference_mode():
            sf(static_x)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = sf(static_x)
            for replay_x in replay_inputs:
                static_x.copy_(replay_x)
                graph.replay()
                actual = captured.clone()
                expected = sf(replay_x)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_swordfish_uses_tensor_device_when_current_device_differs(self):
        supported = [
            index
            for index in range(torch.cuda.device_count())
            if _supported_swordfish_device(index)
        ]
        if len(supported) < 2:
            self.skipTest("requires at least two supported Blackwell GPUs")

        current_device = supported[0]
        tensor_device = torch.device("cuda", supported[1])
        previous_device = torch.cuda.current_device()
        with torch.cuda.device(current_device):
            self.assertEqual(torch.cuda.current_device(), current_device)
            torch_linear, sf = _make_pair(
                in_features=64,
                out_features=64,
                group_size=-1,
                sym=True,
                desc_act=False,
                dtype=torch.float16,
                device=tensor_device,
            )
            x = torch.randn((1, 64), dtype=torch.float16, device=tensor_device)
            actual = _assert_pair(torch_linear, sf, x)
            self.assertEqual(actual.device, tensor_device)
            self.assertEqual(torch.cuda.current_device(), current_device)
        self.assertEqual(torch.cuda.current_device(), previous_device)


if __name__ == "__main__":
    unittest.main()
