# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-state invariant for every quantized kernel.

For a module that holds a checkpoint-layout state, collecting ``state_dict()``
and loading it back into the *same* module must be an identity: the emitted
tensors and the layer output are unchanged. Every serialization surface relies
on this:

* ``torch.save(model.state_dict())`` restores,
* ``Trainer``/``accelerate`` resumption restores,
* copying a state dict between two models restores,
* PEFT's ``set_peft_model_state_dict`` (which calls ``state_dict()``) cannot
  corrupt the layer it is attaching to.

The inventory is data-driven: :func:`_gptq_cases` and :func:`_awq_cases` list
every kernel that implements ``_save_to_state_dict``/``_load_from_state_dict``.
A new kernel added there is immediately held to the invariant.

Each GPTQ case runs against **both** on-disk formats, ``gptq`` (v1 words) and
``gptq_v2`` (v2 words), because the load hook converts between them.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.packing import hf_repack_layer


BITS = 4
GROUP_SIZE = 32
IN_FEATURES = 128
OUT_FEATURES = 64


def _inputs(seed: int = 0, group_size: int = GROUP_SIZE):
    torch.manual_seed(seed)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    groups = IN_FEATURES // group_size
    g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // group_size
    scales = torch.rand(OUT_FEATURES, groups, dtype=torch.float32) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << BITS, (OUT_FEATURES, groups), dtype=torch.int32)
    return linear, scales, zeros, g_idx


def _materialize(module: BaseQuantLinear) -> BaseQuantLinear:
    """Turn packing-time inference tensors into ordinary ones.

    The packers run under ``torch.inference_mode()``, so the buffers they
    register cannot be written by ``load_state_dict``. A module loaded from a
    checkpoint holds ordinary tensors, which is the state this invariant is
    about.
    """

    with torch.inference_mode(False):
        for name in list(module._parameters):
            value = module._parameters[name]
            if torch.is_tensor(value) and torch.is_inference(value):
                module._parameters[name] = nn.Parameter(value.detach().clone(), requires_grad=False)
        for name in list(module._buffers):
            value = module._buffers[name]
            if torch.is_tensor(value) and torch.is_inference(value):
                module._buffers[name] = value.detach().clone()
    return module


def _gptq_source(cls, backend, *, group_size: int = GROUP_SIZE, sym: bool = False):
    """A freshly packed GPTQ kernel holding checkpoint-layout words."""

    module = cls(
        bits=BITS,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=torch.int32,
        backend=backend,
        bias=False,
        register_buffers=True,
    )
    linear, scales, zeros, g_idx = _inputs(group_size=group_size)
    module.pack_original(linear, scales, zeros, g_idx)
    # The int4pack transforms read the shift tables directly; the loader and
    # the fusion path initialize them before a transformed forward
    # (`gptqmodel/models/fusion.py`). A directly packed module must do the same.
    module._init_wf_unsqueeze_buffers()
    return _materialize(module)


def _as_loaded_from(module: BaseQuantLinear, on_disk: FORMAT) -> BaseQuantLinear:
    """Model a module just loaded from a checkpoint written in ``on_disk``.

    Packing writes logical (v2) zero points, and the loader converts a v1
    checkpoint to that same runtime layout. Setting the marker reproduces the
    post-load state without a round trip through disk.
    """

    module.qzero_format(format=2)
    module._checkpoint_format_on_disk = on_disk
    # Inference kernels only run their int4pack/AVX512 transform in eval mode,
    # and the transform is what makes a serialized runtime layout observable.
    return module.eval()


def _gptq_cases():
    """Kernel inventory: (id, build, device)."""

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    cases = [
        ("torch", lambda: _gptq_source(TorchLinear, BACKEND.GPTQ_TORCH), "cpu"),
    ]

    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
    from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear

    cases += [
        (
            "torch_aten",
            lambda: _gptq_source(TorchAtenLinear, BACKEND.GPTQ_TORCH_ATEN),
            "cpu",
        ),
        (
            "torch_fused",
            lambda: _gptq_source(TorchFusedLinear, BACKEND.GPTQ_TORCH_FUSED),
            "cpu",
        ),
        (
            "torch_cpu_avx512",
            lambda: _gptq_source(TorchCpuAvx512Linear, BACKEND.GPTQ_TORCH_CPU_AVX512),
            "cpu",
        ),
    ]
    return cases


def _awq_cases():
    """AWQ kernel inventory: (id, build, device)."""

    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    def _awq_source():
        module = AwqTorchLinear(
            bits=BITS,
            group_size=GROUP_SIZE,
            sym=False,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            pack_dtype=torch.int32,
            backend=BACKEND.AWQ_TORCH,
            bias=False,
            register_buffers=True,
        )
        linear, scales, zeros, g_idx = _inputs()
        # ``AwqTorchLinear.pack`` takes ``out_features x groups`` scales/zeros.
        module.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
        return _materialize(module)

    def _repacked(backend):
        return lambda: hf_repack_layer(_awq_source(), backend, post_init=False)

    return [
        ("awq_torch", _awq_source, "cpu"),
        ("awq_torch_aten", _repacked(BACKEND.AWQ_TORCH_ATEN), "cpu"),
        ("awq_torch_fused", _repacked(BACKEND.AWQ_TORCH_FUSED), "cpu"),
        ("awq_torch_cpu_avx512", _repacked(BACKEND.AWQ_TORCH_CPU_AVX512), "cpu"),
    ]


def _require(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _run(module: BaseQuantLinear, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module(x).float()


def _assert_round_trip_identity(module: BaseQuantLinear, device: str, dtype: torch.dtype, label: str) -> None:
    x = torch.randn(3, IN_FEATURES, dtype=dtype, device=device)
    before = _run(module, x)

    state = module.state_dict()
    emitted = {key: value.clone() for key, value in state.items() if torch.is_tensor(value)}

    result = module.load_state_dict(state)
    assert not result.missing_keys, f"{label}: missing keys {result.missing_keys}"
    assert not result.unexpected_keys, f"{label}: unexpected keys {result.unexpected_keys}"

    reloaded = module.state_dict()
    for key, value in emitted.items():
        assert key in reloaded, f"{label}: `{key}` disappeared from state_dict()"
        assert torch.equal(reloaded[key], value), (
            f"{label}: state_dict()['{key}'] changed across "
            "load_state_dict(state_dict())"
        )

    after = _run(module, x)
    torch.testing.assert_close(after, before, atol=2e-3, rtol=2e-3)


def _assert_state_dict_is_a_checkpoint(
    build,
    module: BaseQuantLinear,
    on_disk,
    device: str,
    dtype: torch.dtype,
    label: str,
) -> None:
    """``state_dict()`` must rebuild the layer in a *fresh* module.

    A self round-trip alone cannot catch a runtime layout that overwrites the
    checkpoint tensors: copying the transformed buffers back into the same
    transformed module is trivially an identity. Saving in one process and
    loading in another starts from a module that never ran a forward, which is
    the case that exposes a corrupt ``state_dict()``.
    """

    x = torch.randn(3, IN_FEATURES, dtype=dtype, device=device)
    reference = _run(module, x)
    state = module.state_dict()

    fresh = build()
    if on_disk is not None:
        fresh = _as_loaded_from(fresh, on_disk)
    fresh = fresh.to(device).eval()
    result = fresh.load_state_dict(state)
    assert not result.missing_keys, f"{label}: fresh reload missing {result.missing_keys}"
    assert not result.unexpected_keys, f"{label}: fresh reload unexpected {result.unexpected_keys}"

    torch.testing.assert_close(
        _run(fresh, x), reference, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("on_disk", [FORMAT.GPTQ, FORMAT.GPTQ_V2], ids=["gptq_v1", "gptq_v2"])
@pytest.mark.parametrize("case", _gptq_cases(), ids=[case[0] for case in _gptq_cases()])
def test_gptq_state_dict_round_trip_is_identity(case, on_disk):
    name, build, device = case
    _require(device)

    module = _as_loaded_from(build(), on_disk).to(device)
    dtype = torch.bfloat16 if name == "torch_fused" else torch.float16
    label = f"{name}/{on_disk.value}"
    _assert_round_trip_identity(module, device, dtype, label)
    _assert_state_dict_is_a_checkpoint(build, module, on_disk, device, dtype, label)


@pytest.mark.parametrize("case", _awq_cases(), ids=[case[0] for case in _awq_cases()])
def test_awq_state_dict_round_trip_is_identity(case):
    name, build, device = case
    _require(device)

    module = build().to(device)
    _assert_round_trip_identity(module, device, torch.float16, name)
    _assert_state_dict_is_a_checkpoint(build, module, None, device, torch.float16, name)


def _no_inverse_cases():
    """Kernels whose runtime layout has no inverse back to checkpoint words.

    They must keep ``state_dict()`` readable (PEFT) and refuse only an actual
    checkpoint write.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    def _gptq_source(group_size: int):
        linear, scales, zeros, g_idx = _inputs(group_size=group_size)
        module = TorchLinear(
            bits=BITS,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            pack_dtype=torch.int32,
            backend=BACKEND.GPTQ_TORCH,
            bias=False,
        )
        module.pack_original(linear, scales, zeros, g_idx)
        module._init_wf_unsqueeze_buffers()
        module.qzero_format(format=2)
        module._checkpoint_format_on_disk = FORMAT.GPTQ_V2
        return _materialize(module)

    def _awq_source(group_size: int):
        linear, scales, zeros, g_idx = _inputs(group_size=group_size)
        module = AwqTorchLinear(
            bits=BITS,
            group_size=group_size,
            sym=False,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            pack_dtype=torch.int32,
            backend=BACKEND.AWQ_TORCH,
            bias=False,
            register_buffers=True,
        )
        module.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
        return _materialize(module)

    def _repack(source, backend):
        return lambda: hf_repack_layer(source(), backend, device="cuda:0")

    return [
        ("marlin", _repack(lambda: _gptq_source(128), BACKEND.MARLIN)),
        ("awq_marlin", _repack(lambda: _awq_source(128), BACKEND.MARLIN)),
    ]


@pytest.mark.cuda
@pytest.mark.parametrize("case", _no_inverse_cases(), ids=[case[0] for case in _no_inverse_cases()])
def test_no_inverse_kernel_state_dict_contract(case):
    from gptqmodel.utils.checkpoint_state import checkpoint_serialization

    name, build = case
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.version.hip is not None:
        pytest.skip("the Marlin kernels are CUDA-only")

    try:
        module = build()
    except (ImportError, NotImplementedError, ValueError) as error:  # pragma: no cover - runtime gate
        pytest.skip(f"{name} runtime unavailable: {error}")
    assert isinstance(module, BaseQuantLinear)

    # Read-only collection (the PEFT path) must succeed and round-trip.
    _assert_round_trip_identity(module, "cuda", torch.float16, name)

    # A checkpoint write must refuse instead of publishing the runtime layout.
    with checkpoint_serialization():
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            module.state_dict()
