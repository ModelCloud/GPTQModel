# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A compiled static-cache ``generate()`` must not run on graph-breaking kernels (#4669).

Transformers auto-compiles the forward pass of a static-cache generation
(``CompileConfig``, ``mode="reduce-overhead"``). Every GPT-QModel kernel but
the Torch-AWQ reference path calls a custom op without a fake implementation or
branches on host values (``data_ptr()``/``id()``/``._version``) inside
``forward``, so dynamo breaks the graph on every quantized linear and CUDA-graph
capture never covers a layer. The decode step then pays for the fragments and
guard checks: the static cache measured 2-4x slower than eager on Llama-1B.

``GPTQModel.generate()`` now turns that auto-compile off unless the selected
kernels declare ``SUPPORTS_COMPILE``. These tests pin the policy: the opt-out
reaches the model, every explicit caller choice wins, and a dense or
compile-clean model keeps compiling.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import CompileConfig, GenerationConfig

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


pytestmark = [pytest.mark.cpu]


class _GraphBreakingLinear(BaseQuantLinear):
    """Stands in for a custom-op kernel: ``SUPPORTS_COMPILE`` stays default False."""

    def __init__(self):
        nn.Module.__init__(self)


class _CompileCleanLinear(BaseQuantLinear):
    SUPPORTS_COMPILE = True

    def __init__(self):
        nn.Module.__init__(self)


class _RecordingModel(nn.Module):
    """The inner HF model: records the generation kwargs it receives."""

    def __init__(self, projection: nn.Module):
        super().__init__()
        self.projection = projection
        self.device = torch.device("cpu")
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, 1, dtype=torch.long)


class _Harness:
    """``BaseQModel``'s policy helpers and ``generate()`` on a fake model."""

    _kernels_support_compile = BaseQModel._kernels_support_compile
    _generation_compile_kwargs = BaseQModel._generation_compile_kwargs
    generate = BaseQModel.generate

    def __init__(self, projection: nn.Module):
        self.model = _RecordingModel(projection)
        self.tokenizer = None


def test_graph_breaking_kernels_are_not_compile_clean():
    harness = _Harness(_GraphBreakingLinear())

    assert harness._kernels_support_compile() is False
    assert harness._generation_compile_kwargs({}) == {"disable_compile": True}


def test_dense_models_keep_the_callers_compile_policy():
    harness = _Harness(nn.Linear(4, 4))

    assert harness._kernels_support_compile() is True
    assert harness._generation_compile_kwargs({}) == {}


def test_compile_clean_kernel_keeps_auto_compile():
    harness = _Harness(_CompileCleanLinear())

    assert harness._kernels_support_compile() is True
    assert harness._generation_compile_kwargs({}) == {}


def test_an_explicit_disable_compile_false_is_respected():
    """A caller who asks to compile keeps compiling, even on a breaking kernel."""

    harness = _Harness(_GraphBreakingLinear())

    assert harness._generation_compile_kwargs({"disable_compile": False}) == {"disable_compile": False}
    assert harness._generation_compile_kwargs({"disable_compile": True}) == {"disable_compile": True}


def test_an_explicit_compile_config_is_respected():
    harness = _Harness(_GraphBreakingLinear())
    compile_config = CompileConfig()
    config = GenerationConfig(compile_config=compile_config)

    passed = harness._generation_compile_kwargs({"compile_config": compile_config})
    assert passed["compile_config"] is compile_config
    assert "disable_compile" not in passed

    passed = harness._generation_compile_kwargs({"generation_config": config})
    assert passed["generation_config"] is config
    assert config.disable_compile is None


def test_an_explicit_generation_config_choice_is_respected():
    harness = _Harness(_GraphBreakingLinear())

    opted_in = GenerationConfig(disable_compile=False)
    passed = harness._generation_compile_kwargs({"generation_config": opted_in})

    assert passed["generation_config"] is opted_in
    assert opted_in.disable_compile is False


def test_the_opt_out_never_mutates_the_callers_generation_config():
    harness = _Harness(_GraphBreakingLinear())
    config = GenerationConfig()

    added = harness._generation_compile_kwargs({"generation_config": config})

    assert added["generation_config"] is not config
    assert added["generation_config"].disable_compile is True
    # The caller's object keeps its own default.
    assert config.disable_compile is None


def test_generate_passes_the_opt_out_to_the_model():
    harness = _Harness(_GraphBreakingLinear())
    inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    harness.generate(inputs, cache_implementation="static")

    assert harness.model.calls[-1]["disable_compile"] is True
    assert harness.model.calls[-1]["cache_implementation"] == "static"


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        return _FakeBatch(input_ids=torch.zeros(len(prompts), 3, dtype=torch.long))


def test_generate_keeps_the_callers_kwargs_on_the_tensor_branch():
    """The policy is additive: it must never drop the caller's options."""

    harness = _Harness(_GraphBreakingLinear())

    harness.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=7, cache_implementation="static")

    call = harness.model.calls[-1]
    assert call["max_new_tokens"] == 7
    assert call["cache_implementation"] == "static"
    assert call["disable_compile"] is True


def test_generate_keeps_the_callers_kwargs_on_the_string_branch():
    harness = _Harness(_GraphBreakingLinear())
    harness.tokenizer = _FakeTokenizer()

    harness.generate(["a prompt", "another prompt"], max_new_tokens=5, cache_implementation="static")

    call = harness.model.calls[-1]
    assert call["max_new_tokens"] == 5
    assert call["cache_implementation"] == "static"
    assert call["disable_compile"] is True
    assert call["input_ids"].shape == (2, 3)


def test_generate_leaves_a_dense_model_alone():
    harness = _Harness(nn.Linear(4, 4))
    inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    harness.generate(inputs, cache_implementation="static")

    assert "disable_compile" not in harness.model.calls[-1]


def test_generate_keeps_an_explicit_opt_in():
    harness = _Harness(_GraphBreakingLinear())
    inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    harness.generate(inputs, cache_implementation="static", disable_compile=False)

    assert harness.model.calls[-1]["disable_compile"] is False


def test_optimize_wrapper_is_unwrapped():
    """``optimize()`` wraps the model; the kernels live on ``_orig_mod``."""

    wrapper = type("OptimizedLike", (), {})()
    wrapper._orig_mod = _RecordingModel(_GraphBreakingLinear())
    harness = _Harness(nn.Linear(4, 4))
    harness.model = wrapper

    assert harness._kernels_support_compile() is False


def test_kernel_declarations():
    """Only a trace-clean, measured win opts in; subclasses do not inherit it."""

    assert MarlinLinear.SUPPORTS_COMPILE is False
    assert TorchLinear.SUPPORTS_COMPILE is False
    # `tensor_cache_key()` calls `tensor.is_inference()`, a hard dynamo break, so
    # `fullgraph=True` fails and the compiled static cache is slower (#4739).
    assert AwqTorchLinear.SUPPORTS_COMPILE is False
    assert ParoLinear.SUPPORTS_COMPILE is False
    # FP8 traces clean under `fullgraph=True` and its static-cache decode is 2.3x
    # faster than eager here (#4739).
    assert TorchFP8Linear.SUPPORTS_COMPILE is True


def test_mlx_fp8_does_not_inherit_the_torch_fp8_opt_in():
    """The MPS holder runs native Metal kernels that were never traced (#4739)."""

    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear

    assert issubclass(FP8MlxQuantLinear, TorchFP8Linear)
    assert FP8MlxQuantLinear.SUPPORTS_COMPILE is False
