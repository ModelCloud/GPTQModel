# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

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
        return GPTAQ(named, QuantizeConfig())

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

    def test_multi_device_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
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
