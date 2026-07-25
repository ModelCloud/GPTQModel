# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import random

import pytest
import torch
import unittest

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.gptaq import GPTAQ
from models.model_test import ModelTest


class TestQwen2_5_GPTAQ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    EVAL_TASKS = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.2739, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3055, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    GPTAQ = GPTAQConfig()

    def test_qwen2_5(self):
        self.quantize_and_evaluate()


class TestGPTAQHessian(unittest.TestCase):
    """Verify GPTAQ Hessian accumulation matches a reference implementation."""

    def _process_input(self, x):
        """Mirror GPTAQ's input preprocessing for 3D (B, S, C) activations."""
        x = x.to(dtype=torch.float32)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.t()
        return x

    def _reference(self, inputs, natives):
        """Compute 2/N * (X^T X) and 2/N * ((X_native - X)^T X)."""
        target = inputs[0].device
        total = sum(x.shape[0] for x in inputs)

        X = torch.cat([self._process_input(x.to(target)) for x in inputs], dim=1)
        X_native = torch.cat([self._process_input(n.to(target)) for n in natives], dim=1)

        scale = 2.0 / float(total) if total > 0 else 0.0
        H_ref = scale * X.matmul(X.t())
        dXXT_ref = scale * (X_native - X).matmul(X.t())
        return H_ref, dXXT_ref

    def _make_gptaq(self, layer, native_list):
        named = NamedModule(layer, name="l", full_name="m.l", layer_index=0)
        # Clone so the test reference tensors are not mutated by GPTAQ's in-place diff.
        named.state["native_inp"] = [n.clone() for n in native_list]
        return GPTAQ(named, QuantizeConfig(gptaq=GPTAQConfig()))

    def test_single_batch_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        x = torch.randn(2, 8, 16, dtype=torch.float32)
        native = torch.randn(2, 8, 16, dtype=torch.float32)
        gptaq = self._make_gptaq(layer, [native])
        gptaq.add_batch(x, None)
        gptaq.materialize_global_hessian()
        H_ref, dXXT_ref = self._reference([x], [native])
        self.assertTrue(torch.allclose(gptaq.H, H_ref, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(gptaq.dXXT, dXXT_ref, rtol=1e-4, atol=1e-5))

    def test_multiple_batches_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        xs = [torch.randn(2, 8, 16, dtype=torch.float32) for _ in range(3)]
        natives = [torch.randn(2, 8, 16, dtype=torch.float32) for _ in range(3)]
        gptaq = self._make_gptaq(layer, natives)
        for x in xs:
            gptaq.add_batch(x, None)
        H_ref, dXXT_ref = self._reference(xs, natives)
        gptaq.materialize_global_hessian()
        self.assertTrue(torch.allclose(gptaq.H, H_ref, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(gptaq.dXXT, dXXT_ref, rtol=1e-4, atol=1e-5))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_multi_device_matches_reference(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32, device="cpu")
        x_cpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cpu")
        native_cpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cpu")
        x_gpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cuda:0")
        native_gpu = torch.randn(2, 8, 16, dtype=torch.float32, device="cuda:0")
        gptaq = self._make_gptaq(layer, [native_cpu, native_gpu])
        gptaq.add_batch(x_cpu, None)
        gptaq.add_batch(x_gpu, None)
        H_ref, dXXT_ref = self._reference([x_cpu, x_gpu], [native_cpu, native_gpu])
        gptaq.materialize_global_hessian()
        self.assertTrue(torch.allclose(gptaq.H, H_ref, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(gptaq.dXXT, dXXT_ref, rtol=1e-4, atol=1e-5))

    def test_empty_batch_is_noop(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(16, 8, dtype=torch.float32)
        gptaq = self._make_gptaq(layer, [torch.zeros(0, 8, 16, dtype=torch.float32)])
        gptaq.add_batch(torch.zeros(0, 8, 16, dtype=torch.float32), None)
        gptaq.materialize_global_hessian()
        self.assertEqual(gptaq.H.shape, (16, 16))
        self.assertTrue(torch.allclose(gptaq.H, torch.zeros_like(gptaq.H)))
        self.assertTrue(torch.allclose(gptaq.dXXT, torch.zeros_like(gptaq.dXXT)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_second_batch_avoids_hessian_temporary(self):
        """A second batch should not allocate another columns x columns temporary."""
        columns = 4096
        layer = torch.nn.Linear(columns, columns // 2, dtype=torch.float16, device="cuda:0")
        x = torch.randn(1, 64, columns, dtype=torch.float16, device="cuda:0")
        native = x.clone()
        gptaq = self._make_gptaq(layer, [native, native.clone()])

        gptaq.add_batch(x, None)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        gptaq.add_batch(x, None)
        peak = torch.cuda.max_memory_allocated()

        # H and dXXT partials are already allocated.  A second in-place batch
        # should only allocate the small activation cast/workspace, not another
        # two columns x columns fp32 temporaries.
        self.assertLess(peak - base, (columns * columns * 4) // 2)


def _gptaq_reference(inputs, natives):
    target = inputs[0].device
    total = sum(x.shape[0] for x in inputs)
    X = torch.cat(
        [x.reshape(-1, x.shape[-1]).to(target).float().t() for x in inputs],
        dim=1,
    )
    X_native = torch.cat(
        [n.reshape(-1, n.shape[-1]).to(target).float().t() for n in natives],
        dim=1,
    )
    if total == 0:
        zeros = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32, device=target)
        return zeros, zeros
    scale = 2.0 / float(total)
    H_ref = scale * X.matmul(X.t())
    dXXT_ref = scale * (X_native - X).matmul(X.t())
    return H_ref, dXXT_ref


def _make_gptaq_for_random(layer, native_list):
    named = NamedModule(layer, name="l", full_name="m.l", layer_index=0)
    named.state["native_inp"] = [n.clone() for n in native_list]
    return GPTAQ(named, QuantizeConfig(gptaq=GPTAQConfig()))


@pytest.mark.parametrize("seed", range(10000))
def test_gptaq_hessian_randomized(seed: int):
    """Run many random shapes/datasets through the GPTAQ Hessian path."""
    rng = random.Random(seed)
    columns = rng.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_count = rng.choice([1, 2, 3, 5])
    samples = rng.choice([1, 2, 3])
    seq_len = rng.choice([1, 4, 8, 16, 32, 64])

    max_tokens = 1_000_000
    total = columns * samples * seq_len * batch_count
    if total > max_tokens:
        seq_len = max(1, max_tokens // (columns * samples * batch_count))

    torch.manual_seed(seed)
    layer = torch.nn.Linear(columns, columns // 2, dtype=torch.float32)
    batches = [
        torch.randn(samples, seq_len, columns, dtype=torch.float32)
        for _ in range(batch_count)
    ]
    natives = [x.clone() for x in batches]
    gptaq = _make_gptaq_for_random(layer, natives)

    for x in batches:
        gptaq.add_batch(x, None)
    gptaq.materialize_global_hessian()

    H_ref, dXXT_ref = _gptaq_reference(batches, natives)
    assert torch.allclose(gptaq.H, H_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(gptaq.dXXT, dXXT_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seed", range(1000))
def test_gptaq_hessian_randomized_multi_device(seed: int):
    """Randomized multi-device accumulation (cpu + cuda:0)."""
    rng = random.Random(seed)
    columns = rng.choice([8, 16, 32, 64, 128, 256, 512, 1024])
    seq_len = rng.choice([1, 4, 8, 16, 32])

    torch.manual_seed(seed)
    layer = torch.nn.Linear(columns, columns // 2, dtype=torch.float32, device="cpu")

    x_cpu = torch.randn(1, seq_len, columns, dtype=torch.float32, device="cpu")
    native_cpu = x_cpu.clone()
    x_gpu = torch.randn(1, seq_len, columns, dtype=torch.float32, device="cpu").to("cuda:0")
    native_gpu = torch.randn(1, seq_len, columns, dtype=torch.float32, device="cpu").to("cuda:0")

    gptaq = _make_gptaq_for_random(layer, [native_cpu, native_gpu])
    gptaq.add_batch(x_cpu, None)
    gptaq.add_batch(x_gpu, None)
    gptaq.materialize_global_hessian()

    H_ref, dXXT_ref = _gptaq_reference([x_cpu, x_gpu], [native_cpu, native_gpu])
    assert torch.allclose(gptaq.H, H_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(gptaq.dXXT, dXXT_ref, rtol=1e-4, atol=1e-5)
