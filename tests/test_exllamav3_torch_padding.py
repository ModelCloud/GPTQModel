# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent bitstream and FP32 arithmetic oracle for logical EXL3 outputs."""

import math
import struct

import pytest
import torch

from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear


def _packed(outputs=576, *, bits=6, codebook="mcg", bias=True):
    stored_outputs = math.ceil(outputs / 128) * 128
    gen = torch.Generator().manual_seed(787)
    trellis = torch.randint(-32768, 32768, (8, stored_outputs // 16, 16 * bits),
                            dtype=torch.int16, generator=gen)
    trellis.flatten()[:8] = torch.tensor(
        [0, 1, 32767, -32768, -1, 0x5555, -21846, 31], dtype=torch.int16)
    packed = {
        "trellis": trellis,
        "suh": (torch.randn(128, generator=gen) * .1).half(),
        # Keep padded channels nonzero: they must participate in the right
        # Hadamard even though the caller sees only the logical outputs.
        "svh": (torch.randn(stored_outputs, generator=gen) * .1).half(),
    }
    if codebook != "3inst":
        multiplier = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}[codebook]
        packed[codebook] = torch.tensor(multiplier, dtype=torch.uint32).view(torch.int32)
    if bias:
        packed["bias"] = (torch.randn(outputs, generator=gen) * .01).half()
    return packed


def _oracle_inner(packed):
    """Decode explicit circular bit windows, then place fragment coordinates."""
    trellis = packed["trellis"].cpu()
    bits = trellis.shape[-1] // 16
    words = trellis.long() & 65535
    words = words.reshape(*words.shape[:2], -1, 2).flip(-1).reshape_as(words)
    positions = torch.arange(15, -1, -1)
    stream = ((words[..., None] >> positions) & 1).flatten(-2)
    ends = (torch.arange(256) + 1) * bits
    windows = (ends[:, None] - 16 + torch.arange(16)) % (256 * bits)
    states = (stream[..., windows] * (1 << positions)).sum(-1)
    if "mul1" in packed:
        raw = (states * 0x83DCD12D) & 0xFFFFFFFF
        summed = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24)) + 0x6400
        values = summed.to(torch.uint16).contiguous().view(torch.float16).double()
        inv = struct.unpack("<e", struct.pack("<H", 0x1EEE))[0]
        offset = struct.unpack("<e", struct.pack("<H", 0xC931))[0]
        values = values * inv + offset
    else:
        raw = ((states * 0xCBAC1FED) if "mcg" in packed
               else states * 89226354 + 64248484) & 0xFFFFFFFF
        raw = (raw & 0x8FFF8FFF) ^ 0x3B603B60
        halves = torch.stack((raw & 65535, raw >> 16), -1).to(torch.uint16).contiguous()
        values = halves.view(torch.float16).double().sum(-1)
    # Native codewords round once to half (__hadd/__hfma); rotations stay FP32.
    values = values.half().float()
    tiles = torch.empty((*trellis.shape[:2], 16, 16), dtype=torch.float32)
    for lane in range(32):
        row, column = lane % 4 * 2, lane // 4
        coords = [(row, column), (row + 1, column), (row + 8, column), (row + 9, column),
                  (row, column + 8), (row + 1, column + 8),
                  (row + 8, column + 8), (row + 9, column + 8)]
        for element, (r, c) in enumerate(coords):
            tiles[..., r, c] = values[..., lane * 8 + element]
    return tiles.permute(0, 2, 1, 3).reshape(trellis.shape[0] * 16, trellis.shape[1] * 16)


def _oracle_weight(packed, *, device="cpu"):
    inner = _oracle_inner(packed).to(device)
    # Walsh parity defines the normalized Sylvester matrix independently of
    # the backend's recursive concatenation and cached transform helpers.
    had = torch.tensor([[(-1.) ** ((row & col).bit_count()) for col in range(128)]
                        for row in range(128)], device=device, dtype=torch.float32)
    had *= 1 / math.sqrt(128)
    k, n = inner.shape
    left = (had @ inner.reshape(-1, 128, n)).reshape(k, n)
    left = left * packed["suh"].to(device=device, dtype=torch.float32)[:, None]
    right = (left.reshape(k, -1, 128) @ had).reshape(k, n)
    return right * packed["svh"].to(device=device, dtype=torch.float32)[None, :]


def _module(packed, outputs=576):
    return ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=outputs, name="kv_a", tensors=packed)


@pytest.mark.cpu
@pytest.mark.parametrize("outputs", [128, 576, 1056])
@pytest.mark.parametrize("bits,codebook", [(2, "3inst"), (2, "mcg"), (6, "mcg"), (4, "mul1")])
def test_complete_stored_decode_then_logical_crop(outputs, bits, codebook):
    packed = _packed(outputs, bits=bits, codebook=codebook)
    module = _module(packed, outputs)
    expected_inner = _oracle_inner(packed)
    expected = _oracle_weight(packed)[:, :outputs]
    inner = module.get_inner_weight_tensor(dtype=torch.float32)
    assert inner.shape == (128, math.ceil(outputs / 128) * 128)
    assert torch.equal(inner, expected_inner)
    actual = module.get_weight_tensor(dtype=torch.float32)
    assert actual.shape == (128, outputs) and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    # Dense reconstruction must not change or crop the complete inner decode.
    assert torch.equal(module.get_inner_weight_tensor(dtype=torch.float32), inner)
    assert torch.equal(module.get_weight_tensor(dtype=torch.float32), actual)
    for dtype in (torch.float16, torch.bfloat16):
        assert torch.equal(module.get_weight_tensor(dtype=dtype), expected.to(dtype))


def _check_forward(*, device, dtype, outputs, bias, batch):
    packed_cpu = _packed(outputs, bias=bias)
    packed = {key: value.to(device) for key, value in packed_cpu.items()}
    module = _module(packed, outputs)
    gen = torch.Generator().manual_seed(41)
    x = (torch.randn((1, batch, 128), generator=gen) * .015).to(device=device, dtype=dtype)
    x[0, 0, :8] = torch.tensor([0., -0., .125, -.125, .25, -.25, .5, -.5], device=device, dtype=dtype)
    full_weight = _oracle_weight(packed_cpu, device=device)
    weight = full_weight[:, :outputs]
    actual_weight = module.get_weight_tensor(dtype=torch.float32)
    torch.testing.assert_close(actual_weight, weight, rtol=1e-6, atol=1e-6)
    compute_dtype = torch.float32 if device == "cpu" else torch.float16
    raw = x.to(compute_dtype).float() @ weight.to(compute_dtype).float()
    visible = raw.to(compute_dtype)
    if bias:
        raw = raw + packed["bias"].to(compute_dtype).float()
        visible = visible + packed["bias"].to(compute_dtype)
    actual = module(x)
    if device != "cpu":
        torch.cuda.synchronize()
    assert actual.dtype == dtype and actual.shape == (1, batch, outputs)
    # Check arithmetic separately from mandatory output rounding. Small scale
    # magnitudes bound even BF16's rounding below the inference absolute limit.
    torch.testing.assert_close(actual.float(), raw, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual, visible.to(dtype), rtol=2e-3, atol=2e-3)
    assert module.get_inner_weight_tensor(dtype=torch.float32).shape == full_weight.shape


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_wide_fallback_preserves_checkpoint_and_packed_buffer_lifecycle(dtype):
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    outputs = 576  # Exercise padded trellis storage with a logical-width bias.
    packed = _packed(outputs, bits=2, codebook="3inst", bias=True)
    module = ExllamaV3Linear.from_tensors(
        in_features=128, out_features=outputs, name="kv_a", tensors=packed
    )
    checkpoint = {key: tensor.clone() for key, tensor in module.state_dict().items()}
    keys = set(checkpoint)
    checkpoint_bytes = sum(t.numel() * t.element_size() for t in checkpoint.values())
    x = (torch.randn((2, 128), generator=torch.Generator().manual_seed(3738)) * .01).to(dtype)
    reference = x.float() @ _oracle_weight(packed)[:, :outputs] + packed["bias"].float()

    def check_packed_state(expected_bytes=checkpoint_bytes):
        state = module.state_dict()
        assert set(state) == keys
        assert sum(t.numel() * t.element_size() for t in state.values()) == expected_bytes
        assert "_torch_inner" not in module._modules
        helper = module._torch_inner
        assert helper is not None
        for name, tensor in helper.named_buffers():
            assert tensor is getattr(module, name)  # No copied packed buffers.

    first_output = module(x)
    torch.testing.assert_close(first_output.float(), reference, rtol=2e-3, atol=2e-3)
    first_helper = module._torch_inner
    check_packed_state()
    for key, saved in checkpoint.items():
        assert torch.equal(module.state_dict()[key], saved)
    torch.testing.assert_close(module(x), first_output, rtol=0, atol=0)
    assert module._torch_inner is first_helper

    with torch.no_grad():
        module.bias.add_(.125)
    changed_output = module(x)
    assert module._torch_inner is not first_helper
    torch.testing.assert_close(changed_output.float(), reference + .125, rtol=2e-3, atol=2e-3)
    check_packed_state()

    changed_helper = module._torch_inner
    module.load_state_dict(checkpoint, strict=True)
    restored_output = module(x)
    assert module._torch_inner is not changed_helper
    torch.testing.assert_close(restored_output, first_output, rtol=0, atol=0)
    check_packed_state()

    replacement = {key: tensor.clone() for key, tensor in checkpoint.items()}
    replacement["bias"].add_(.0625)
    pre_replace_helper = module._torch_inner
    module.load_state_dict(replacement, strict=True, assign=True)
    replaced_output = module(x)
    assert module._torch_inner is not pre_replace_helper
    torch.testing.assert_close(replaced_output.float(), reference + .0625, rtol=2e-3, atol=2e-3)
    check_packed_state()
    module.load_state_dict(checkpoint, strict=True, assign=True)
    torch.testing.assert_close(module(x), first_output, rtol=0, atol=0)
    check_packed_state()

    pre_move_helper = module._torch_inner
    module.to(dtype=torch.float32)
    assert module._torch_inner is None
    moved_output = module(x)
    assert module._torch_inner is not pre_move_helper
    torch.testing.assert_close(moved_output.float(), reference, rtol=2e-3, atol=2e-3)
    moved_bytes = checkpoint_bytes + module.bias.numel() * 2  # FP16 bias becomes FP32.
    check_packed_state(moved_bytes)

    pre_clear_helper = module._torch_inner
    module.clear_weight_cache()
    assert module._torch_inner is None
    torch.testing.assert_close(module(x), moved_output, rtol=0, atol=0)
    assert module._torch_inner is not pre_clear_helper
    check_packed_state(moved_bytes)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("outputs", [128, 576, 1056])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 33])
def test_cpu_forward_preserves_logical_shape_bias_and_input_dtype(dtype, outputs, bias, batch):
    _check_forward(device="cpu", dtype=dtype, outputs=outputs, bias=bias, batch=batch)


@pytest.mark.cpu
@pytest.mark.parametrize("case,message", [
    ("rank", "trellis geometry"), ("packing_width", "trellis geometry"),
    ("packing_dtype", "trellis geometry"), ("wrong_inputs", "storage differs"),
    ("unpadded_outputs", "storage differs"), ("extra_output_block", "storage differs"),
    ("missing_suh", "suh geometry"), ("logical_svh", "svh geometry"),
    ("broadcast_suh", "suh geometry"), ("broadcast_svh", "svh geometry"),
    ("stored_bias", "logical bias geometry"),
])
def test_incompatible_storage_fails_closed(case, message):
    packed = _packed()
    if case == "rank":
        packed["trellis"] = packed["trellis"].flatten(0, 1)
    elif case == "packing_width":
        packed["trellis"] = packed["trellis"][..., :-1]
    elif case == "packing_dtype":
        packed["trellis"] = packed["trellis"].int()
    elif case == "wrong_inputs":
        packed["trellis"] = packed["trellis"][:7]
    elif case == "unpadded_outputs":
        packed["trellis"] = packed["trellis"][:, :36]
    elif case == "extra_output_block":
        packed["trellis"] = torch.zeros(8, 48, 96, dtype=torch.int16)
    elif case == "missing_suh":
        del packed["suh"]
    elif case == "logical_svh":
        packed["svh"] = packed["svh"][:576]
    elif case == "broadcast_suh":
        packed["suh"] = packed["suh"][None, :]
    elif case == "broadcast_svh":
        packed["svh"] = packed["svh"][None, :]
    elif case == "stored_bias":
        packed["bias"] = torch.zeros(640, dtype=torch.float16)
    with pytest.raises(ValueError, match=message):
        _module(packed).get_weight_tensor()


@pytest.mark.cpu
@pytest.mark.parametrize("inputs,outputs", [(112, 576), (128, 0), (128, -1), (256, 576)])
def test_storage_must_match_valid_logical_dimensions(inputs, outputs):
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=inputs, out_features=outputs, name="bad", tensors=_packed())
    with pytest.raises(ValueError, match="storage differs"):
        module.get_weight_tensor()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires admitted CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("outputs", [128, 576, 1056])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 33])
def test_cuda_reference_forward_preserves_padding_and_dtype(dtype, outputs, bias, batch):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _check_forward(device="cuda", dtype=dtype, outputs=outputs, bias=bias, batch=batch)
    finally:
        torch.set_float32_matmul_precision(previous)



def _native_exl3_available() -> bool:
    """The native EXL3 linear declares ``SUPPORTS_DEVICES = [DEVICE.CUDA]``.

    ``torch.cuda.is_available()`` is also true on ROCm, where this backend
    raises ``NotImplementedError`` by design, so gate on the module's own
    declared support family instead of the runtime namespace.
    """
    if not torch.cuda.is_available():
        return False
    from gptqmodel.models._const import DEVICE, normalize_device
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    try:
        return normalize_device(torch.device("cuda", 0)) in ExllamaV3Linear.SUPPORTS_DEVICES
    except Exception:
        return DEVICE.CUDA in ExllamaV3Linear.SUPPORTS_DEVICES


NATIVE_EXL3 = pytest.mark.skipif(
    not _native_exl3_available(),
    reason="native EXL3 linear declares CUDA-only support",
)


@pytest.mark.parametrize("device", [
    pytest.param("cpu", marks=pytest.mark.cpu),
    pytest.param("cuda", marks=pytest.mark.cuda),
])
def test_native_inference_created_bias_tracks_mutation_without_extension(device):
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with torch.inference_mode():
        module = ExllamaV3Linear.from_tensors(
            in_features=128, out_features=576, name="kv_a",
            tensors={
                "trellis": torch.zeros((8, 40, 96), dtype=torch.int16, device=device),
                "bias": torch.zeros(576, dtype=torch.float16, device=device),
            },
        )
    assert module.bias.is_inference()
    packed_trellis = module.trellis
    with torch.inference_mode():
        first = module._current_signature()
    assert not module.bias.is_inference()
    assert module.trellis is packed_trellis
    assert module.bias.numel() * module.bias.element_size() == 1152
    normalized_bias = module.bias
    with torch.inference_mode():
        assert module._current_signature() == first
    assert module.bias is normalized_bias
    with torch.inference_mode():
        module.bias.add_(1)
    assert module._current_signature() != first


# --- The native ExllamaV3Linear path shares this logical/padded contract but
# --- needs admitted hardware (it builds LinearEXL3 through the compiled ext).
def _native_module(packed, outputs, device):
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    return ExllamaV3Linear(
        in_features=128, out_features=outputs, name="kv_a",
        tensors={k: v.to(device) for k, v in packed.items()},
    ).to(device)


@pytest.mark.cuda
@NATIVE_EXL3
@pytest.mark.parametrize("outputs", [128, 576, 1056])
@pytest.mark.parametrize("bias", [False, True])
def test_native_forward_crops_to_logical_outputs(outputs, bias):
    """The native kernel stores padded channels but must return logical outputs.

    ``ExllamaV3Linear`` pads the bias and the ``LinearEXL3`` geometry to the
    stored width, then slices the result. Nothing else in the suite exercises
    that path: the patch touches it but only ``ExllamaV3TorchLinear`` is tested.
    """
    device = "cuda"
    packed = _packed(outputs, bias=bias)
    module = _native_module(packed, outputs, device)
    weight = _oracle_weight(packed, device=device).float()[:, :outputs]
    gen = torch.Generator().manual_seed(17)
    x = (torch.randn((1, 3, 128), generator=gen) * .01).to(device=device, dtype=torch.float16)

    with torch.inference_mode():
        actual = module(x)
    raw = x.float().reshape(-1, 128) @ weight
    if bias:
        raw = raw + packed["bias"].to(device).float()
    raw = raw.reshape(1, 3, outputs)

    assert actual.dtype == torch.float16 and actual.shape == (1, 3, outputs)
    torch.testing.assert_close(actual.float(), raw, rtol=2e-3, atol=2e-3)


@pytest.mark.cuda
@NATIVE_EXL3
@pytest.mark.parametrize("outputs", [128, 576, 1056])
def test_native_padded_bias_stays_live(outputs):
    """An in-place ``bias`` write must reach the kernel output.

    The padded shape makes the backend hand ``LinearEXL3`` a *widened copy* of
    the bias. A copy is frozen at ``_ensure_inner`` time, so without an explicit
    refresh a later ``bias.add_`` (adapters, calibration, scale folding) is
    silently ignored for every non-128-aligned width.
    """
    device = "cuda"
    packed = _packed(outputs, bias=True)
    module = _native_module(packed, outputs, device)
    gen = torch.Generator().manual_seed(23)
    x = (torch.randn((1, 2, 128), generator=gen) * .01).to(device=device, dtype=torch.float16)

    with torch.inference_mode():
        before = module(x).float().clone()
    with torch.no_grad():
        module.bias.add_(10.0)
    with torch.inference_mode():
        after = module(x).float()

    # Every visible output moves by the bias delta. FP16 rounding of the
    # pre-add accumulator allows a couple of ulps at this magnitude (10.0),
    # but a frozen padded copy shows an exact 0.0 shift instead.
    delta = after - before
    torch.testing.assert_close(delta, torch.full_like(delta, 10.0), rtol=1e-2, atol=1e-2)


@pytest.mark.cuda
@NATIVE_EXL3
@pytest.mark.parametrize("outputs", [128, 576])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_inference_created_bias_keeps_versioned_refresh(outputs, dtype):
    """Inference-created checkpoint buffers must run and track a later bias edit."""
    packed = _packed(outputs, bias=True)
    weight = _oracle_weight(packed, device="cuda").float()[:, :outputs]
    gen = torch.Generator().manual_seed(31)
    x = (torch.randn((1, 2, 128), generator=gen) * .01).to(device="cuda", dtype=dtype)
    reference = x.float().reshape(-1, 128) @ weight + packed["bias"].to("cuda").float()

    with torch.inference_mode():
        module = _native_module(packed, outputs, "cuda")
        assert module.bias.is_inference()
        try:
            before = module(x)
        except ModuleNotFoundError as exc:
            if "ExLlamaV3 torch.ops kernels are not properly installed" not in str(exc):
                raise
            pytest.skip("native EXL3 torch.ops extension is unavailable")
        assert not module.bias.is_inference()
        initial_inner = module._inner
        repeated = module(x)
        assert module._inner is initial_inner
        module.bias.add_(0.25)
        after = module(x)
        assert module._inner is not initial_inner

    assert before.dtype == repeated.dtype == after.dtype == dtype
    assert before.shape == repeated.shape == after.shape == (1, 2, outputs)
    expected = reference.reshape(1, 2, outputs).to(dtype)
    changed_expected = (reference + 0.25).reshape(1, 2, outputs).to(dtype)
    torch.testing.assert_close(before, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(repeated, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(after, changed_expected, rtol=2e-3, atol=2e-3)
    assert module._inner.trellis is module.trellis
    stored_outputs = math.ceil(outputs / 128) * 128
    assert all(
        not isinstance(value, torch.Tensor) or tuple(value.shape) != (128, stored_outputs)
        for holder in (module, module._inner) for value in vars(holder).values()
    )


@pytest.mark.cuda
@NATIVE_EXL3
@pytest.mark.parametrize("outputs", [576, 1056])
def test_native_rejects_stored_bias_geometry(outputs):
    """A bias already widened to the padded width is a checkpoint bug."""
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    packed = _packed(outputs, bias=False)
    packed["bias"] = torch.zeros(math.ceil(outputs / 128) * 128, dtype=torch.float16)
    module = ExllamaV3Linear(
        in_features=128, out_features=outputs, name="kv_a",
        tensors={k: v.to("cuda") for k, v in packed.items()},
    ).to("cuda")
    with pytest.raises(ValueError, match="logical bias geometry"):
        module._ensure_inner()


@pytest.mark.cpu
@pytest.mark.parametrize("bits", [2, 4, 6, 8, 10, 12, 16])
def test_stored_features_accepts_every_supported_bit_width(bits):
    """Packing width is ``bits`` 16-bit words, so it reaches 256 at 16 bits.

    The first cut of this layer capped the trellis width at 128 (8 bits) and so
    rejected every valid 10/12/16-bit checkpoint with "invalid trellis geometry".
    """
    stored_outputs = 256
    trellis = torch.randint(-32768, 32767, (8, stored_outputs // 16, bits * 16),
                            dtype=torch.int16, generator=torch.Generator().manual_seed(bits))
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=stored_outputs, name="bits",
        tensors={"trellis": trellis,
                 "suh": torch.rand(128).half() * .1,
                 "svh": torch.rand(stored_outputs).half() * .1},
    )
    assert module._stored_features() == (128, stored_outputs)
    assert module._bits_per_weight() == bits
    assert module._decode_inner_weight().shape == (128, stored_outputs)


@pytest.mark.cpu
@pytest.mark.parametrize("words", [0, 8, 17, 24])
def test_stored_features_rejects_non_word_aligned_packing(words):
    """A width that is not a positive multiple of 16 cannot encode symbols."""
    trellis = torch.zeros(8, 16, words, dtype=torch.int16)
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=256, name="bad",
        tensors={"trellis": trellis,
                 "suh": torch.rand(128).half() * .1,
                 "svh": torch.rand(256).half() * .1},
    )
    with pytest.raises(ValueError, match="trellis geometry"):
        module._stored_features()
