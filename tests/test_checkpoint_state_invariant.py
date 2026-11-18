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
from gptqmodel.nn_modules.qlinear._fused_train_state import QZEROS_FORMAT_BUFFER
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.packing import hf_repack_layer


BITS = 4
GROUP_SIZE = 32
IN_FEATURES = 128
OUT_FEATURES = 64


def _inputs(
    seed: int = 0,
    group_size: int = GROUP_SIZE,
    in_features: int = IN_FEATURES,
    out_features: int = OUT_FEATURES,
):
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.1)
    groups = in_features // group_size
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << BITS, (out_features, groups), dtype=torch.int32)
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
    """AWQ kernel inventory: (id, build, device, activation dtype).

    The CPU AWQ kernels rewrite ``qweight`` in place on the first eval forward,
    so they are the kernels that expose whether ``AWQuantLinear`` emits the
    captured checkpoint snapshot. ``TorchCpuAvx512AwqLinear`` only accepts
    bfloat16 activations; the others take float16.
    """

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
        ("awq_torch", _awq_source, "cpu", torch.float16),
        ("awq_torch_aten", _repacked(BACKEND.AWQ_TORCH_ATEN), "cpu", torch.float16),
        ("awq_torch_fused", _repacked(BACKEND.AWQ_TORCH_FUSED), "cpu", torch.bfloat16),
        (
            "awq_torch_cpu_avx512",
            _repacked(BACKEND.AWQ_TORCH_CPU_AVX512),
            "cpu",
            torch.bfloat16,
        ),
    ]


def _require(device: str, name: str = "") -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if name.endswith("cpu_avx512"):
        from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16

        if not has_avx512_bf16():
            pytest.skip(avx512_bf16_error())


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
    _require(device, name)

    module = _as_loaded_from(build(), on_disk).to(device)
    dtype = torch.bfloat16 if name == "torch_fused" else torch.float16
    label = f"{name}/{on_disk.value}"
    _assert_round_trip_identity(module, device, dtype, label)
    _assert_state_dict_is_a_checkpoint(build, module, on_disk, device, dtype, label)


@pytest.mark.parametrize("case", _awq_cases(), ids=[case[0] for case in _awq_cases()])
def test_awq_state_dict_round_trip_is_identity(case):
    name, build, device, dtype = case
    _require(device, name)

    # ``eval()`` is required: the CPU int4pack/AVX512 transform only runs
    # outside training, and it is what overwrites the checkpoint tensors.
    module = build().to(device).eval()
    _assert_round_trip_identity(module, device, dtype, name)
    _assert_state_dict_is_a_checkpoint(build, module, None, device, dtype, name)


def _transformed_cpu_awq_cases():
    """AWQ CPU kernels that rewrite ``qweight`` in place."""

    return [
        case
        for case in _awq_cases()
        if case[0] in {"awq_torch_aten", "awq_torch_fused", "awq_torch_cpu_avx512"}
    ]


@pytest.mark.parametrize(
    "case", _transformed_cpu_awq_cases(), ids=[case[0] for case in _transformed_cpu_awq_cases()]
)
def test_awq_cpu_kernels_emit_checkpoint_layout_after_transform(case):
    """Issue #4125 item 2: the AWQ transform must not leak into ``state_dict()``.

    ``AWQuantLinear`` had no ``_save_to_state_dict`` hook, so the CPU AWQ
    kernels captured their checkpoint tensors but never emitted them: an eval
    forward replaced ``qweight`` with the int4pack operand and a subsequent
    ``save_pretrained()`` wrote that shape under the checkpoint key. Reloading
    then failed with a shape mismatch.
    """

    name, build, device, dtype = case
    _require(device, name)

    module = build().to(device).eval()
    checkpoint_shape = tuple(module.qweight.shape)
    x = torch.randn(3, IN_FEATURES, dtype=dtype, device=device)
    reference = _run(module, x)

    # The first eval forward is what installs the runtime layout.
    assert getattr(module, "_fused_train_state", None) is not None, (
        f"{name}: the int4pack transform did not run, so this case cannot "
        "observe the defect"
    )
    assert tuple(module.qweight.shape) != checkpoint_shape, (
        f"{name}: qweight kept its checkpoint shape, so the transform is a no-op"
    )

    emitted = module.state_dict()
    assert tuple(emitted["qweight"].shape) == checkpoint_shape, (
        f"{name}: state_dict() emitted the runtime qweight "
        f"{tuple(emitted['qweight'].shape)} instead of the checkpoint shape "
        f"{checkpoint_shape}"
    )
    # The float affine offsets must never reach a checkpoint.
    assert not emitted["qzeros"].is_floating_point(), (
        f"{name}: state_dict() emitted float qzeros ({emitted['qzeros'].dtype}); "
        "the v1/v2 word conversion only applies to integer words"
    )
    assert torch.isfinite(emitted["qzeros"].to(torch.float32)).all(), (
        f"{name}: state_dict() qzeros contains non-finite values"
    )

    fresh = build().to(device).eval()
    result = fresh.load_state_dict(emitted)
    assert not result.missing_keys, f"{name}: reload missing {result.missing_keys}"
    assert not result.unexpected_keys, f"{name}: reload unexpected {result.unexpected_keys}"
    torch.testing.assert_close(_run(fresh, x), reference, atol=2e-3, rtol=2e-3)


def _no_inverse_cases():
    """Kernels whose runtime layout has no inverse back to checkpoint words.

    They must keep ``state_dict()`` readable (PEFT) and refuse only an actual
    checkpoint write.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    def _gptq_source(
        group_size: int,
        *,
        in_features: int = IN_FEATURES,
        out_features: int = OUT_FEATURES,
        checkpoint_format: FORMAT = FORMAT.GPTQ_V2,
    ):
        linear, scales, zeros, g_idx = _inputs(
            group_size=group_size, in_features=in_features, out_features=out_features
        )
        module = TorchLinear(
            bits=BITS,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            backend=BACKEND.GPTQ_TORCH,
            bias=False,
            format=checkpoint_format,
        )
        module.pack_original(linear, scales, zeros, g_idx)
        module._init_wf_unsqueeze_buffers()
        module.qzero_format(format=2)
        module._checkpoint_format_on_disk = checkpoint_format
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
        # Machete and Swordfish prepack ``qweight`` and rewrite ``qzeros`` into
        # float offsets in ``post_init``; they must refuse a checkpoint write
        # exactly like Marlin (issue #4171 item 3). Both declare only the v1
        # ``gptq`` checkpoint format and require 128-aligned output widths.
        (
            "machete",
            _repack(
                lambda: _gptq_source(
                    128,
                    in_features=256,
                    out_features=128,
                    checkpoint_format=FORMAT.GPTQ,
                ),
                BACKEND.MACHETE,
            ),
        ),
        (
            "swordfish",
            _repack(
                lambda: _gptq_source(
                    128,
                    in_features=256,
                    out_features=128,
                    checkpoint_format=FORMAT.GPTQ,
                ),
                BACKEND.SWORDFISH,
            ),
        ),
        # BitBLAS replaces the checkpoint payload in ``repack_from_gptq``.
        ("bitblas", _repack(lambda: _gptq_source(128), BACKEND.GPTQ_BITBLAS)),
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

    # Read-only collection (the PEFT path) must succeed.
    state = module.state_dict()
    assert state["qweight"].shape == module.qweight.shape
    # The payload marks itself as a runtime layout so any other transport can
    # tell it apart from a checkpoint.
    assert state[QZEROS_FORMAT_BUFFER].item() == 0

    # A runtime-layout payload is rejected by a module that holds checkpoint
    # tensors: copying it there would silently produce a wrong model. The guard
    # runs before any shape check, so any checkpoint-layout packed kernel is a
    # valid receiver for this assertion.
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    checkpoint_layout = AwqTorchLinear(
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
    assert not checkpoint_layout._is_runtime_layout()
    with pytest.raises(RuntimeError, match="kernel-private runtime layout"):
        checkpoint_layout.load_state_dict(state)

    # A checkpoint write must refuse instead of publishing the runtime layout.
    with checkpoint_serialization():
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            module.state_dict()


def test_any_runtime_layout_flag_refuses_checkpoint_write():
    """Issue #4171 item 3: the refusal is a property of the runtime layout.

    Machete, Swordfish and BitBLAS rewrite packed tensors without going through
    a Marlin/ExLlamaV2-specific hook. The base packed-module write path must
    refuse them from the shared ``_is_runtime_layout()`` contract.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.checkpoint_state import checkpoint_serialization

    module = _as_loaded_from(_gptq_source(TorchLinear, BACKEND.GPTQ_TORCH), FORMAT.GPTQ)
    module._native_shuffled = True
    assert module._is_runtime_layout()

    # Read-only collection (PEFT) stays usable and labels the payload.
    state = module.state_dict()
    assert state[QZEROS_FORMAT_BUFFER].item() == 0

    with checkpoint_serialization():
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            module.state_dict()


def test_transformers_save_pretrained_enters_checkpoint_scope(tmp_path):
    """Issue #4125 item 1: the HF writer must be on the refusing side.

    PEFT calls ``state_dict()`` read-only, so the runtime-layout kernels refuse
    only inside ``checkpoint_serialization()``. That made the guard useless for
    ``PreTrainedModel.save_pretrained()``, which collects ``state_dict()``
    itself and therefore wrote the private layout with no error.
    """

    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel

    from gptqmodel.utils.checkpoint_state import is_checkpoint_serialization

    assert getattr(
        PreTrainedModel.save_pretrained, "_gptqmodel_runtime_layout_guard", False
    ), "PreTrainedModel.save_pretrained is not wrapped by the runtime-layout guard"

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=IN_FEATURES,
        intermediate_size=IN_FEATURES,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).half()

    seen = {}
    original_state_dict = model.state_dict

    def _recording_state_dict(*args, **kwargs):
        seen["serializing"] = is_checkpoint_serialization()
        return original_state_dict(*args, **kwargs)

    model.state_dict = _recording_state_dict
    model.save_pretrained(str(tmp_path / "probe"))
    assert seen.get("serializing") is True, (
        "save_pretrained did not enter checkpoint_serialization(), so a "
        "runtime-layout kernel would write its private layout without raising"
    )
    # The scope must not leak past the call.
    assert is_checkpoint_serialization() is False


@pytest.mark.cuda
def test_save_pretrained_refuses_runtime_layout_model(tmp_path):
    """Issue #4125 item 1: ``model.save_pretrained()`` must raise, not write garbage.

    Before this fix an ExLlamaV2/Marlin model saved cleanly and the reloaded
    directory generated ``' flags flags flags ...'``; Marlin raised a shape
    error only at load time. Both are now refused at save.
    """

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Marlin")
    if not marlin_runtime_available(torch.float16):
        pytest.skip(f"Marlin runtime unavailable: {marlin_runtime_error(torch.float16)}")

    device = "cuda:0"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=IN_FEATURES,
        intermediate_size=IN_FEATURES,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).half()

    linear, scales, zeros, g_idx = _inputs(group_size=128)
    source = TorchLinear(
        bits=BITS,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
    )
    source.pack_original(linear, scales, zeros, g_idx)
    source._init_wf_unsqueeze_buffers()
    source.qzero_format(format=2)
    source._checkpoint_format_on_disk = FORMAT.GPTQ_V2
    source = _materialize(source).to(device)

    marlin = hf_repack_layer(source, BACKEND.MARLIN, device=device)
    assert marlin._marlin_repacked
    model.model.layers[0].self_attn.q_proj = marlin

    with pytest.raises(RuntimeError, match="cannot be reconstructed"):
        model.save_pretrained(str(tmp_path / "out"))

    # The read-only collection PEFT performs must still work, and it must mark
    # the payload so any other transport fails loudly on reload.
    state = model.state_dict()
    assert state["model.layers.0.self_attn.q_proj.qzeros_format"].item() == 0



def _torch_transport(state):
    """A ``torch.save``/``torch.load`` round trip (pickle keeps ``_metadata``)."""

    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _safetensors_transport(state, tmp_path):
    """A safetensors round trip: tensors only, ``_metadata`` is dropped."""

    from safetensors.torch import load_file, save_file

    path = tmp_path / "payload.safetensors"
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(path),
        metadata={"format": "pt"},
    )
    return load_file(str(path))


def _dict_copy_transport(state):
    """A plain ``{k: v for k, v in sd.items()}`` copy: ``_metadata`` is dropped."""

    return {key: value.clone() for key, value in state.items()}


@pytest.mark.parametrize(
    ("transport", "keeps_metadata"),
    [
        ("torch", True),
        ("safetensors", False),
        ("dict", False),
    ],
)
def test_v1_v2_copy_survives_every_state_dict_transport(tmp_path, transport, keeps_metadata):
    """Issue #4125 item 3: the layout must travel inside the payload.

    The v1/v2 copy used to rely on ``state_dict._metadata``, which safetensors
    (Trainer/accelerate checkpoints) and plain dict copies drop. The receiving
    module then read v2 words as v1 and every zero point moved one step: the
    synthetic fixture drifted by a relative 0.25-0.65.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    target = _as_loaded_from(_gptq_source(TorchLinear, BACKEND.GPTQ_TORCH), FORMAT.GPTQ)
    source = _as_loaded_from(_gptq_source(TorchLinear, BACKEND.GPTQ_TORCH), FORMAT.GPTQ_V2)

    x = torch.randn(3, IN_FEATURES, dtype=torch.float16)
    reference = _run(target, x)
    # Identical weights, different on-disk labels: the copy must be a no-op.
    torch.testing.assert_close(_run(source, x), reference, atol=0, rtol=0)

    state = source.state_dict()
    assert state._metadata[""]["qzeros_format"] == 2

    if transport == "torch":
        carried = _torch_transport(state)
    elif transport == "safetensors":
        carried = _safetensors_transport(state, tmp_path)
    else:
        carried = _dict_copy_transport(state)

    assert hasattr(carried, "_metadata") is keeps_metadata
    # The tensor marker is what makes the non-metadata transports work.
    assert carried["qzeros_format"].item() == 2

    result = target.load_state_dict(carried)
    assert not result.missing_keys, f"{transport}: missing {result.missing_keys}"
    assert not result.unexpected_keys, f"{transport}: unexpected {result.unexpected_keys}"
    torch.testing.assert_close(
        _run(target, x), reference, atol=0, rtol=0,
        msg=lambda m: f"{transport}: v2 words copied into a v1-labelled module ({m})",
    )


def test_layout_marker_is_omitted_only_for_checkpoint_writes():
    """Issue #4152: a checkpoint write must not persist ``qzeros_format``.

    The marker is a *persistent* buffer, so ``nn.Module._save_to_state_dict``
    emits it before the packed hook runs. HF ``save_pretrained()`` collects
    tensors inside ``checkpoint_serialization()`` and used to publish one extra
    ``*.qzeros_format`` tensor per quantized layer: a key neither the GPTQ/AWQ
    format nor ``config.json`` describes. A reloaded checkpoint takes its
    layout from ``checkpoint_format``, so the marker is redundant there, while
    the read-only ``state_dict()`` transports from #4125 item 3 still need it.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.utils.checkpoint_state import checkpoint_serialization

    module = _as_loaded_from(_gptq_source(TorchLinear, BACKEND.GPTQ_TORCH), FORMAT.GPTQ)

    read_only = module.state_dict()
    assert QZEROS_FORMAT_BUFFER in read_only
    assert int(read_only[QZEROS_FORMAT_BUFFER].item()) in (0, 1, 2)

    with checkpoint_serialization():
        checkpoint = module.state_dict()

    assert QZEROS_FORMAT_BUFFER not in checkpoint
    assert set(checkpoint) == set(read_only) - {QZEROS_FORMAT_BUFFER}
    # The scope is re-entrant and must not leak into a later read-only collect.
    assert QZEROS_FORMAT_BUFFER in module.state_dict()
