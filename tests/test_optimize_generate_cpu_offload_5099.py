# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#5099: ``optimize()`` + ``generate()`` must work on a CPU-offloaded model.

``BaseQModel.generate`` ran every non-XLA model under ``torch.inference_mode``.
An Accelerate CPU-offload hook moves its layer with ``Module._apply`` on every
forward. Parameters are rebound with ``param.data = ...``, which keeps their
version counter, but buffers are replaced outright
(``self._buffers[key] = fn(buf)``). Inside inference mode the second move of an
inference tensor produces a tensor without a version counter (#4572), so a
layer that is offloaded and materialized again loses the counter on its packed
``qweight``/``qzeros``/``scales`` buffers. Dynamo reads version counters, so
the compiled forward installed by ``optimize()`` crashed on the decode step of
the first ``generate()``::

    RuntimeError: Inference tensors do not track version counter.

The eager path carried the same defect even without ``optimize()``: after one
``generate()`` on an offloaded placement every re-materialized buffer had no
version counter, which is what broke the next compiled call.

``BaseQModel.generate`` now selects ``torch.no_grad()`` for a model that carries
an offload hook, so the moved tensors keep their version counter. The first
test pins that tensor-lifecycle invariant without a compile; the second drives
the documented ``optimize()`` then ``generate()`` flow.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import RTNConfig
from gptqmodel.quantization.config import CalibrationConfig
from gptqmodel.utils.model import has_cpu_offload_hooks
from tests.tiny_remote_models import build_wordlevel_tokenizer


_VOCAB = 256
_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2

# Both decoder layers are offloaded. The second layer's hook moves the first
# one back to CPU before the forward, and the decode step moves it out again:
# that second materialization is what stripped the buffers' version counters.
_OFFLOAD_MAP = {
    "model.embed_tokens": 0,
    "model.layers.0": "cpu",
    "model.layers.1": "cpu",
    "model.norm": 0,
    "lm_head": 0,
}

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CPU offload needs one CUDA device plus the CPU",
)


def _config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _input_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(5099)
    return torch.randint(100, _VOCAB - 8, (1, 6), generator=generator).to("cuda:0")


def _tensors_without_version_counter(model) -> list[str]:
    """Names of parameters/buffers that cannot report a version counter."""

    missing = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            tensor._version
        except RuntimeError:
            missing.append(name)
    return missing


def _offloaded_model(checkpoint):
    model = GPTQModel.load(
        checkpoint,
        device_map=dict(_OFFLOAD_MAP),
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    assert has_cpu_offload_hooks(model.model), "the offload map did not install a moving hook"
    return model


@pytest.fixture(scope="module")
def offloaded_gptq_checkpoint(tmp_path_factory):
    """A tiny 4-bit RTN checkpoint plus a tokenizer, built locally on the CPU."""

    if not torch.cuda.is_available():
        pytest.skip("a CPU-offload device map needs an accelerator for the resident layers")

    root = tmp_path_factory.mktemp("optimize-5099")
    base = root / "base"
    base.mkdir()
    torch.manual_seed(5099)
    LlamaForCausalLM(_config()).save_pretrained(base)
    tokenizer = build_wordlevel_tokenizer(vocab_size=_VOCAB)
    tokenizer.save_pretrained(base)

    texts = [
        "the quick brown fox jumps over the lazy dog and keeps running through the field",
        "a tiny calibration sample with enough tokens to survive length filtering",
        "another synthetic calibration example deliberately verbose for the tokenizer",
        "gravity attracts objects with mass and measurements help test predictions",
    ]
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in texts]

    model = GPTQModel.load(
        str(base),
        quantize_config=RTNConfig(bits=4, group_size=128, device="cpu", offload_to_disk=False),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    destination = root / "rtn-4bit-g128"
    model.save(str(destination))
    del model
    return str(destination)


@requires_cuda
def test_eager_generate_keeps_version_counters_on_offloaded_layers(offloaded_gptq_checkpoint):
    """One eager ``generate()`` must not strip the offloaded buffers' counters.

    The trigger is the real dispatch path: ``simple_dispatch_model`` installs
    the ``CpuOffload`` hooks, the first offloaded layer is materialized and then
    put back to CPU, and ``generate()`` is the public entry point that used to
    run those moves under inference mode.
    """

    model = _offloaded_model(offloaded_gptq_checkpoint)
    try:
        packed = model.model.model.layers[0].self_attn.q_proj.qweight
        assert packed.device.type == "cpu"
        assert packed._version >= 0  # readable before generate()

        ids = _input_ids()
        out = model.generate(ids, max_new_tokens=2, do_sample=False)

        assert out.shape[-1] > ids.shape[-1], "generate() produced no new token"
        missing = _tensors_without_version_counter(model.model)
        assert not missing, (
            f"{len(missing)} offloaded parameters/buffers have no version counter "
            f"after one generate() (e.g. {missing[:3]}): generate() ran the "
            "offload hook under inference_mode (#5099)"
        )
        # The map still offloads; the fix must not pin the layer on the accelerator.
        assert model.model.model.layers[0].self_attn.q_proj.qweight.device.type == "cpu"
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
@requires_cuda
def test_optimize_then_generate_on_an_offloaded_placement(offloaded_gptq_checkpoint):
    """The documented ``optimize()`` then ``generate()`` flow must produce tokens.

    This is the end-to-end reproduction: a real quantized checkpoint with
    ``cpu`` device-map entries, a compiled forward, then ``generate()``. It used
    to raise ``RuntimeError: Inference tensors do not track version counter``
    while tracing the decode step of the first layer (#5099).
    """

    model = _offloaded_model(offloaded_gptq_checkpoint)
    try:
        model.optimize()
        if not getattr(model, "compiled", False):
            pytest.skip("torch.compile is unavailable on this build")

        ids = _input_ids()
        out = model.generate(ids, max_new_tokens=3, do_sample=False)

        assert out.shape[-1] > ids.shape[-1], "optimize()+generate() produced no new token"
        assert torch.isfinite(out).all()
        assert not _tensors_without_version_counter(model.model), (
            "the compiled generate() left offloaded tensors without a version "
            "counter (#5099)"
        )
    finally:
        del model
        torch.cuda.empty_cache()
