# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU-offload ``device_map`` loads must survive their first forward (#4866).

``simple_dispatch_model`` is the loader's own placement path for an explicit
``device_map`` (``loader.py`` calls it instead of ``accelerate.dispatch_model``).
Two of its steps only work when every mapped module has parameters on one
device:

* ``retie_parameters`` rebinds a tied group to a single tensor. When the map put
  ``lm_head`` on CPU and the embedding on the accelerator, the CPU copy won the
  tie while the embedding kept an ``AlignDevicesHook`` that moved ``input_ids``
  to the accelerator: the first forward raised ``RuntimeError`` from
  ``index_select``.
* ``accelerate.cpu_offload_with_hook`` reads ``next(module.parameters())`` in
  ``CpuOffload.pre_forward``, so a parameterless structural module mapped to
  ``cpu`` (``model.rotary_emb``) raised ``StopIteration``.

Both loaded without error and crashed on the first forward, so these tests drive
the real dispatch function and then run a forward.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.utils.model import simple_dispatch_model


_LAYERS = 2
_VOCAB = 256
_HIDDEN = 64
_INTERMEDIATE = 128

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="explicit CPU offload needs one CUDA device plus the CPU",
)


def _config(*, tied: bool) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
    )


def _base_state_dict(config: LlamaConfig):
    torch.manual_seed(0)
    return LlamaForCausalLM(config).to(torch.float16).state_dict()


def _dispatched(config: LlamaConfig, state_dict, device_map, *, distinct_head: bool = False):
    model = LlamaForCausalLM(config).to(torch.float16).eval()
    model.load_state_dict(state_dict)
    if distinct_head:
        # An untied checkpoint stores a real head; it must never be retied.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone() + 0.5)
    return simple_dispatch_model(model, dict(device_map))


def _split(first: str, second: str) -> dict:
    """The issue's map: the first layer block on one device, the tail on another."""

    device_map = {
        "model.embed_tokens": first,
        "model.rotary_emb": first,
        "model.norm": second,
        "lm_head": second,
    }
    device_map.update(
        {f"model.layers.{index}": (first if index < _LAYERS // 2 else second) for index in range(_LAYERS)}
    )
    return device_map


def _input_ids(device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4866)
    return torch.randint(100, _VOCAB - 8, (2, 8), generator=generator).to(device)


def _forward(model, ids=None):
    device = model.model.embed_tokens.weight.device
    if ids is None:
        ids = _input_ids(device)
    else:
        ids = ids.to(device)
    with torch.inference_mode():
        logits = model(ids).logits
    return logits.float().cpu()


_OFFLOAD_MAPS = {
    # `lm_head` (the tied partner of the embedding) offloaded to CPU.
    "head_cpu": {**_split("cuda:0", "cuda:0"), "lm_head": "cpu"},
    # The whole tail, including `lm_head`, on CPU.
    "1_then_cpu": _split("cuda:0", "cpu"),
    # A parameterless structural module mapped to CPU.
    "rotary_cpu": {**_split("cuda:0", "cuda:0"), "model.rotary_emb": "cpu"},
    # The head block on CPU, so the embedding's tied partner starts there.
    "cpu_then1": _split("cpu", "cuda:0"),
}


@requires_cuda
@pytest.mark.parametrize("map_name", sorted(_OFFLOAD_MAPS))
def test_tied_head_cpu_offload_map_forwards(map_name):
    """A tied model placed by an explicit CPU-offload map must run a forward."""

    config = _config(tied=True)
    model = _dispatched(config, _base_state_dict(config), _OFFLOAD_MAPS[map_name])

    logits = _forward(model)

    assert logits.shape == (2, 8, _VOCAB)
    assert torch.isfinite(logits).all(), f"{map_name}: forward produced non-finite logits"


@requires_cuda
@pytest.mark.parametrize("map_name", ["head_cpu", "rotary_cpu", "cpu_then1"])
def test_tied_group_follows_its_accelerator_member(map_name):
    """A tied group is never split, so the head and the embedding stay one tensor."""

    config = _config(tied=True)
    model = _dispatched(config, _base_state_dict(config), _OFFLOAD_MAPS[map_name])
    assert model.lm_head.weight is model.model.embed_tokens.weight

    _forward(model)

    assert model.lm_head.weight is model.model.embed_tokens.weight, (
        f"{map_name}: the first forward broke the tie between lm_head and embed_tokens"
    )
    assert model.lm_head.weight.device == model.model.embed_tokens.weight.device
    assert model.hf_device_map["lm_head"] == model.hf_device_map["model.embed_tokens"]


@requires_cuda
@pytest.mark.parametrize("map_name", ["head_cpu", "rotary_cpu", "cpu_then1"])
def test_offloaded_map_matches_the_all_accelerator_result(map_name):
    """Keeping the tie must not change the numbers the model produces."""

    config = _config(tied=True)
    state_dict = _base_state_dict(config)
    all_accelerator = dict.fromkeys(_OFFLOAD_MAPS[map_name], "cuda:0")
    ids = _input_ids(torch.device("cpu"))
    expected = _forward(_dispatched(config, state_dict, all_accelerator), ids)

    actual = _forward(_dispatched(config, state_dict, _OFFLOAD_MAPS[map_name]), ids)

    assert torch.equal(actual, expected), (
        f"{map_name}: CPU offload changed the logits, max|delta|={(actual - expected).abs().max().item():.3e}"
    )


@requires_cuda
def test_untied_head_keeps_its_own_placement_and_tensor():
    """An untied head is not part of a tied group, so the map is left alone (#4799)."""

    config = _config(tied=False)
    model = _dispatched(
        config,
        _base_state_dict(config),
        {**_split("cuda:0", "cuda:0"), "lm_head": "cpu"},
        distinct_head=True,
    )

    assert model.hf_device_map["lm_head"] == "cpu"
    assert model.lm_head.weight.device.type == "cpu"
    assert model.lm_head.weight is not model.model.embed_tokens.weight

    logits = _forward(model)

    assert torch.isfinite(logits).all()
    assert model.lm_head.weight is not model.model.embed_tokens.weight


@requires_cuda
def test_parameterless_module_gets_no_offload_hook():
    """A parameterless CPU entry cannot carry `CpuOffload`; it stays hookless on an accelerator."""

    config = _config(tied=True)
    model = _dispatched(config, _base_state_dict(config), _OFFLOAD_MAPS["rotary_cpu"])

    rotary = model.model.rotary_emb
    assert not any(True for _ in rotary.parameters())
    hook = getattr(rotary, "_hf_hook", None)
    # The loader installs its own package-owned subclass, which keeps every
    # placement tensor move from emptying the device cache. What matters here
    # is that the parameterless module is placed by an alignment hook instead
    # of ``cpu_offload_with_hook`` (which raised on its empty parameter list).
    from gptqmodel.utils.offload_hooks import GPTQModelAlignDevicesHook

    assert isinstance(hook, GPTQModelAlignDevicesHook), (
        f"expected a GPTQModelAlignDevicesHook on the parameterless module, got {type(hook).__name__}"
    )
    assert torch.isfinite(_forward(model)).all()
