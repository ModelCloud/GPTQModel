# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real CUDA graph ownership tests; synthetic factors test runtime algebra only."""

import pytest
import torch

from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    _encode,
    _metadata,
    prepare_rank8,
)
from gptqmodel.quantization.qvq_window_graphs import P32WindowGraphs


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from test_qvq_grouped_runtime import _child
        from test_qvq_window_recovery import _kernel_rank8

        self.q = _child("q_proj", device="cuda").eval()
        self.k = _child("k_proj", device="cuda").eval()
        for child in (self.q, self.k):
            _kernel_rank8(child)
        metadata = _metadata(self.k)
        metadata["selected"] = False
        self.k.rank8_metadata = _encode(metadata, self.k.trellis.device)
        self.eval()

    def forward(self, x):
        return {"q": self.q(x), "k": self.k(x)}


@pytest.fixture
def model():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    return Pair()


def eager(model, x, mode):
    for child in (model.q, model.k):
        prepare_rank8(child, P32WindowConfig(recovery_mode="auto", quality_mode=mode))
    with torch.no_grad():
        return model(x)


def test_modes_requests_streams_and_output_lifetime(model):
    owner = P32WindowGraphs(model)
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.01
    before = [getattr(child, "_p32_window_config", None) for child in (model.q, model.k)]
    owner.capture("decode16", {"x": x})
    assert [getattr(child, "_p32_window_config", None) for child in (model.q, model.k)] == before
    inputs = [x, x * 0.5, -x]
    references = {(i, mode): eager(model, value, mode) for i, value in enumerate(inputs)
                  for mode in ("fast", "balanced", "quality")}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    retained = []
    for i, mode in ((0, "fast"), (1, "quality"), (2, "balanced"), (0, "quality"), (0, "fast")):
        with torch.cuda.stream(stream if i % 2 else torch.cuda.current_stream()):
            result = owner.replay("decode16", mode, x=inputs[i])
            retained.append((result, references[i, mode]))
    torch.cuda.synchronize()
    for result, expected in retained:
        for name in result:
            torch.testing.assert_close(result[name], expected[name], atol=0, rtol=0)
    assert torch.equal(references[0, "fast"]["k"], references[0, "balanced"]["k"])
    assert not torch.equal(references[0, "balanced"]["k"], references[0, "quality"]["k"])
    assert torch.equal(references[0, "balanced"]["q"], references[0, "quality"]["q"])
    owner.invalidate()
    with pytest.raises(KeyError):
        owner.replay("decode16", "fast", x=x)


def test_mutation_signature_and_atomic_failed_capture(model):
    owner = P32WindowGraphs(model)
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16) * 0.01
    owner.capture("one", {"x": x})
    previous = getattr(model.q, "_p32_window_config", None)
    bad = {"quality": {"q": P32WindowConfig(), "k": P32WindowConfig(algorithm="hopper_m16")}}
    # Corrupt an accepted artifact: preparation must fail after earlier modes
    # and modules were prepared, and restore their eager policies atomically.
    model.k.rank8_B.add_(0.01)
    with pytest.raises(ValueError, match="factors hash"):
        owner.capture("failed", {"x": x}, configs=bad)
    assert getattr(model.q, "_p32_window_config", None) == previous
    with pytest.raises(KeyError):
        owner.replay("failed", "fast", x=x)
    # Off does not inspect correction payload and its graph remains valid.
    owner.replay("one", "fast", x=x)
    with pytest.raises(RuntimeError, match="model state changed"):
        owner.replay("one", "quality", x=x)
    with pytest.raises(ValueError, match="signature"):
        owner.replay("one", "fast", x=x.repeat(2, 1))
    with torch.no_grad():
        model.q.SV.add_(0.01)
    with pytest.raises(RuntimeError, match="model state changed"):
        owner.replay("one", "fast", x=x)
    owner.invalidate()


def test_request_guard_and_config_validation(model):
    owner = P32WindowGraphs(model)
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="already has"):
        P32WindowGraphs(model)
    with owner._exclusive(), pytest.raises(RuntimeError, match="already active"):
        owner.capture("x", {"x": x})
    with pytest.raises(ValueError, match="names"):
        owner.capture("x", {"x": x}, configs={"fast": {"missing": P32WindowConfig()}})
    model.train()
    with pytest.raises(ValueError, match="eval"):
        owner.capture("x", {"x": x})
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.capture("x", {"x": x})
    P32WindowGraphs(model).close()


def test_full_tiny_llama_graphs(model):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32, hidden_size=256, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    llama = LlamaForCausalLM(config).half().cuda().eval()
    llama.model.layers[0].self_attn.q_proj = model.q
    llama.model.layers[0].self_attn.k_proj = model.k
    owner = P32WindowGraphs(llama)
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    llama.config._attn_implementation = "eager"
    mask = torch.full((1, 1, 4, 4), torch.finfo(torch.float16).min, device="cuda").half().triu(1)
    owner.capture(
        "prefill4", {"input_ids": ids, "attention_mask": mask},
        static_kwargs={"use_cache": False, "return_dict": False},
    )
    for mode in ("fast", "quality", "balanced", "fast"):
        for child in (model.q, model.k):
            prepare_rank8(child, P32WindowConfig(recovery_mode="auto", quality_mode=mode))
        with torch.no_grad():
            expected = llama(ids, use_cache=False, return_dict=False)[0]
        actual = owner.replay("prefill4", mode, input_ids=ids, attention_mask=mask)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    owner.invalidate()


def test_grouped_payload_survives_policy_restore_and_allocator_reuse(model):
    from test_qvq_grouped_runtime import _child
    from test_qvq_window_recovery import _kernel_rank8

    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups

    class GateUp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _child("gate_proj", device="cuda").eval()
            self.up_proj = _child("up_proj", device="cuda", su=self.gate_proj.SU).eval()
            _kernel_rank8(self.gate_proj)
            self.eval()

        def forward(self, x):
            return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)

    grouped = GateUp()
    assert sum(install_qvq_hopper_groups(grouped).values()) == 1
    owner = P32WindowGraphs(grouped)
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.01
    choices = {mode: {name: P32WindowConfig(recovery_kernel="fused_epilogue")
                     for name in owner.layers} for mode in ("fast", "balanced", "quality")}
    owner.capture("group", {"x": x}, configs=choices)
    runtime = grouped.gate_proj._gptqmodel_qvq_grouped_runtime
    assert runtime._payload is None
    scratch = [torch.empty(4096, device="cuda").fill_(123) for _ in range(40)]
    for mode in ("fast", "quality", "balanced", "fast"):
        for child in (grouped.gate_proj, grouped.up_proj):
            prepare_rank8(child, P32WindowConfig(
                recovery_mode="auto", quality_mode=mode, recovery_kernel="fused_epilogue",
            ))
        with torch.no_grad():
            expected = grouped(x)
        actual = owner.replay("group", mode, x=x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert scratch[0][0] == 123
    owner.invalidate()


def test_cpu_operator_remains_available_without_graph_capture():
    from test_qvq_window_recovery import fixture

    layer, _, train, _ = fixture()
    owner = P32WindowGraphs(layer)
    with pytest.raises(ValueError, match="CUDA"):
        owner.capture("cpu", {"x": train})
    with torch.no_grad():
        assert torch.isfinite(layer(train)).all()


def test_external_geometry_is_captured_without_overriding_quality(model):
    from dataclasses import replace

    owner = P32WindowGraphs(model)
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.01
    choices = {
        "fast": {name: P32WindowConfig(algorithm="hopper_m16", recovery_mode="on") for name in owner.layers},
        "balanced": {name: P32WindowConfig(
            algorithm="hopper_direct_decode_mma", block_m=32, block_n=64,
            recovery_mode="off", recovery_kernel="fused_epilogue",
        ) for name in owner.layers},
        "quality": {name: P32WindowConfig(
            algorithm="hopper_direct_decode_mma", block_m=64, block_n=128,
            recovery_mode="off", recovery_kernel="fused_epilogue",
        ) for name in owner.layers},
    }
    owner.capture("tiles", {"x": x}, configs=choices)
    for mode in ("quality", "fast", "balanced"):
        for name, child in owner.layers.items():
            prepare_rank8(child, replace(choices[mode][name], recovery_mode="auto", quality_mode=mode))
        with torch.no_grad():
            expected = model(x)
        actual = owner.replay("tiles", mode, x=x)
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    owner.close()


@pytest.mark.filterwarnings("ignore:.*CUDA Graph is empty.*")
def test_cold_qvq_payload_cache_fails_closed_during_capture(model):
    """Preparation and dtype-cache allocation must happen before graph capture."""
    child = model.q
    child._dtype_cache_clear()
    with child._qvq_cuda_bank_cache_lock:
        child._qvq_cuda_bank_cache = None
        child._qvq_cuda_window_cache = None
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.01
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="must be prepared"), torch.cuda.graph(graph):
        child(x)
