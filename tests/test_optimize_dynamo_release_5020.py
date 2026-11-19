# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#5020: ``optimize()`` + ``generate()`` must not pin the model until a GC.

#4986 removed the ``model -> forward -> model`` cycle from ``optimize()`` by
compiling the unbound class function and reaching the instance through a weak
reference (#4937). The remaining holder is torch.compile's own trace state:
every frame dynamo traces builds ``VariableTracker`` objects, and a graph break
leaves an ``UnspecializedNNModuleVariable`` -> ``SideEffectsProxyDict`` ->
``DunderDictVariable`` cycle that keeps the traced module reachable. Those
cycles are unreachable garbage, but reference counting cannot reclaim them, so
the quantized model - and all of its VRAM - survived the last user reference
until a full cyclic ``gc.collect()``.

``_CompiledModelForward`` now collects them as soon as a compiled call traces a
new frame, while the caller still owns the model. The model is then reachable
only from its owner, so dropping it frees its weights immediately with cyclic
GC disabled.
"""

from __future__ import annotations

import gc
import json
import os
import types
import weakref
from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GPTQConfig
from gptqmodel.quantization.config import CalibrationConfig
from tests.accelerator_gates import requires_nvidia_cuda
from tests.test_optimize_no_reference_cycle_4937 import _Harness, _TinyLM


_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
_LAYERS = 4
_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose for the tokenizer",
    "gravity attracts objects with mass and measurements help test predictions",
    "the baker mixed flour with water and yeast before leaving the dough to rise",
    "a function transforms inputs into results and tests compare expected values",
    "water evaporates from lakes, clouds form when vapor cools, and rain returns it",
    "a young child watched a small bird spread its wings and fly back to its nest",
]

def _cuda_ready() -> bool:
    """Whether this host can hold the regression's quantized model on a GPU."""

    if not torch.cuda.is_available():
        return False
    free, _total = torch.cuda.mem_get_info()
    return free >= 2 * 2**30


def _require_nvidia_for(backend) -> None:
    """Skip a CUDA-only backend on a HIP build, like the #5056 gates."""

    if backend == BACKEND.GPTQ_TRITON and not requires_nvidia_cuda():
        pytest.skip(f"{backend.name} is CUDA-only; ROCm (HIP) is not supported")


@pytest.fixture(scope="module")
def quantized_checkpoint(tmp_path_factory):
    """A 4-layer GPTQ 4-bit checkpoint, quantized once on the CPU."""

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = snapshot_download(
        _MODEL_ID,
        local_files_only=True,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
    )
    truncated = tmp_path_factory.mktemp("optimize-5020-src") / "smollm-4l"
    truncated.mkdir()
    for name in os.listdir(source):
        target = truncated / name
        if name == "config.json":
            payload = json.loads(Path(source, name).read_text())
            payload["num_hidden_layers"] = _LAYERS
            target.write_text(json.dumps(payload))
        else:
            os.symlink(os.path.join(source, name), target)

    tokenizer = AutoTokenizer.from_pretrained(str(truncated))
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in _CALIBRATION_TEXTS]

    model = GPTQModel.load(
        str(truncated),
        quantize_config=GPTQConfig(
            bits=4,
            group_size=128,
            device="cpu",
            offload_to_disk=False,
        ),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    destination = tmp_path_factory.mktemp("optimize-5020-ckpt") / "gptq-4bit-g128"
    model.save(str(destination))
    del model
    return str(destination)


def test_optimize_reports_the_compiled_forward():
    """``optimize()`` sets the ``compiled`` flag its docstring promises.

    The flag was only ever written as ``False`` (initialisation and the
    no-compile branches), so callers could not tell a compiled model from an
    eager one even after ``optimize()`` installed the proxy (#5020).
    """

    torch.manual_seed(0)
    harness = _Harness(_TinyLM())
    harness.optimize(backend="inductor")

    if not _compiled_proxy_installed(harness.model):
        pytest.skip("torch.compile is unavailable on this build")

    # `_Harness` bypasses `BaseQModel.__init__`, which is the only place the
    # flag used to be written, so read it defensively: the failure this pins is
    # "never set on the success path", not "attribute missing".
    assert getattr(harness, "compiled", False) is True


def _input_ids() -> torch.Tensor:
    return torch.tensor([[128, 791, 6864, 315, 9822, 374, 13]], device="cuda:0")


def _compiled_proxy_installed(model) -> bool:
    forward = model.__dict__.get("forward")
    return not isinstance(forward, types.MethodType) and hasattr(
        forward, "_torchdynamo_orig_callable"
    )


def _run_two_generates(checkpoint):
    """Two generates with different prompt lengths; return the model weakref."""

    model = GPTQModel.load(
        checkpoint,
        device="cuda:0",
        backend=BACKEND.GPTQ_TRITON,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.optimize()
    hf_model = model.model
    if not _compiled_proxy_installed(hf_model):
        pytest.skip("torch.compile is unavailable on this build")
    model_ref = weakref.ref(hf_model)
    # The second prompt is shorter, so dynamo traces another prefill frame and
    # a second set of decode frames.
    model.generate(_input_ids(), max_new_tokens=4, do_sample=False)
    model.generate(_input_ids()[:, :4], max_new_tokens=2, do_sample=False)
    return model_ref


def _allocated_gib() -> float:
    return torch.cuda.memory_allocated() / 2**30


def _run_optimized_generate(checkpoint, backend, baseline):
    """Load, compile, generate; return the model's weakref and resident GiB."""

    model = GPTQModel.load(
        checkpoint,
        device="cuda:0",
        backend=backend,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.optimize()
    hf_model = model.model
    if not _compiled_proxy_installed(hf_model):
        pytest.skip("torch.compile is unavailable on this build")
    model.generate(_input_ids(), max_new_tokens=4, do_sample=False)
    resident = _allocated_gib() - baseline
    return weakref.ref(hf_model), resident


@pytest.mark.parametrize("backend", [BACKEND.GPTQ_TRITON, BACKEND.GPTQ_TORCH])
def test_optimize_generate_does_not_pin_the_model(quantized_checkpoint, backend):
    """Dropping the model must free it with cyclic GC disabled (#5020)."""

    _require_nvidia_for(backend)
    if not _cuda_ready():
        pytest.skip("needs a CUDA device with 2 GiB free for the regression")

    gc.collect()
    torch.cuda.empty_cache()
    baseline = _allocated_gib()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        model_ref, resident = _run_optimized_generate(quantized_checkpoint, backend, baseline)
        assert resident > 0.0, "the quantized model was never on the GPU"

        assert model_ref() is None, (
            "the compiled model survived its last reference with cyclic GC "
            "disabled: optimize() + generate() left dynamo trace cycles holding "
            "the model (#5020)"
        )
        held = _allocated_gib() - baseline
        # Scale the bound to the model so the node does not depend on the
        # checkpoint's width; a leak holds essentially all of `resident`.
        assert held < max(0.02, 0.25 * resident), (
            f"{held:.3f} GiB stayed allocated after the model was dropped "
            f"(resident {resident:.3f} GiB); only a user-issued gc.collect() "
            "released it (#5020)"
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def test_repeated_generate_does_not_accumulate(quantized_checkpoint):
    """A second generate re-traces decode frames; those must be released too."""

    _require_nvidia_for(BACKEND.GPTQ_TRITON)
    if not _cuda_ready():
        pytest.skip("needs a CUDA device with 2 GiB free for the regression")

    gc.collect()
    torch.cuda.empty_cache()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        # A helper keeps every strong reference in a frame that has returned by
        # the time the liveness assertion runs.
        model_ref = _run_two_generates(quantized_checkpoint)
        assert model_ref() is None, (
            "the model stayed alive after a second compiled generate() "
            "re-traced frames (#5020)"
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def test_triton_node_uses_the_shared_nvidia_gate(monkeypatch):
    """``[gptq_triton]`` must skip on HIP through the #5056 predicate."""

    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(pytest.skip.Exception):
        _require_nvidia_for(BACKEND.GPTQ_TRITON)
    # The portable Torch backend is not gated.
    _require_nvidia_for(BACKEND.GPTQ_TORCH)
