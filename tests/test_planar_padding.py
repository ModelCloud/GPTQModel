# Unit tests for planar /32 auto-padding: pack-stage buffer padding, load-stage
# (state_dict/post_init) padding, and forward pad/slice correctness.
#
# Planar packing stores whole 32-code blocks, so logical dims that are not /32
# are padded once when the packed buffers are built. These tests verify the
# padded regions are inert (zero contribution), survive save/load, and that
# CPU/GPU forwards produce logically-shaped, dense-reference-accurate outputs.

import pytest
import torch

from test_planar_triton_kernels import (  # noqa: F401
    PLANAR_KERNEL_BITS,
    _packed_module,
    _reference_dequant,
    _to_cuda,
)


# (in_features, out_features, group_size): N-pad only, K-pad only, both.
_PAD_SHAPES = (
    (3072, 48, 32),   # Laguna attn head: N 48 -> 64
    (112, 64, 16),    # K 112 -> 128
    (112, 40, 16),    # K 112 -> 128 and N 40 -> 64
)

_CUDA = torch.cuda.is_available()


def _dense_reference(module, x: torch.Tensor) -> torch.Tensor:
    w = _reference_dequant(module)[: module.in_features, : module.out_features]
    out = x.cpu().float() @ w.float()
    if module.bias is not None:
        out = out + module.bias.cpu().float()
    return out.to(x.dtype)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
def test_pack_stage_padding_buffers(bits: int, shape):
    """Pack-time padding must produce aligned buffers whose padded region is inert."""
    in_features, out_features, group_size = shape
    module = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )

    padded_in = (in_features + 31) // 32 * 32
    padded_out = (out_features + 31) // 32 * 32
    assert module.padded_in_features == padded_in
    assert module.padded_out_features == padded_out

    # Buffer shapes are fully /32-aligned.
    assert module.qweight.shape == (padded_in // 32 * bits, padded_out)
    assert module.qzeros.shape[1] == padded_out // 32 * bits
    assert module.scales.shape[1] == padded_out
    assert module.g_idx.shape[0] == padded_in
    # Bias stays logical: forward adds it after the output slice.
    assert module.bias.shape[0] == out_features

    # Padded g_idx rows must reuse the last real group (no new group indices).
    if padded_in != in_features:
        assert (module.g_idx[in_features:] == module.g_idx[in_features - 1]).all()
    # Padded scale columns are neutral (1.0) so padded codes dequantize exactly.
    if padded_out != out_features:
        assert (module.scales[:, out_features:] == 1).all()

    # The dequantized padded region must be exactly zero: padded outputs get no
    # bias/noise and padded K rows contribute nothing to real outputs.
    dq = _reference_dequant(module)
    assert dq.shape == (padded_in, padded_out)
    if padded_out != out_features:
        assert (dq[:, out_features:] == 0).all()
    if padded_in != in_features:
        assert (dq[in_features:, :] == 0).all()


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
def test_load_stage_padding_state_dict_roundtrip(bits: int, shape):
    """Load-stage buffers (register_buffers=True) must match pack-stage padding exactly."""
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    in_features, out_features, group_size = shape
    src = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )
    state = {k: v.clone() for k, v in src.state_dict().items()}

    dst = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=(bits == 3),
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=FORMAT.GPTQ_P,
        register_buffers=True,
    )
    # Registered (load-stage) buffers must already have padded shapes so a
    # checkpoint written after pack-stage padding loads strictly.
    dst.load_state_dict(state, strict=True)
    assert dst.padded_in_features == src.padded_in_features
    assert dst.padded_out_features == src.padded_out_features

    x = torch.randn(5, in_features, dtype=torch.float16) * 0.5
    out_src = src(x)
    out_dst = dst(x)
    assert out_src.shape == out_dst.shape == (5, out_features)
    assert torch.equal(out_src, out_dst)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
@pytest.mark.parametrize("batch", [1, 7])
def test_forward_pad_slice_matches_dense(bits: int, shape, batch: int):
    """CPU forward: activation K-pad and output N-slice must not corrupt inference."""
    in_features, out_features, group_size = shape
    module = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )
    x = torch.randn(batch, in_features, dtype=torch.float16) * 0.5

    out = module(x)
    assert out.shape == (batch, out_features)
    assert out.dtype == x.dtype

    ref = _dense_reference(module, x)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2)

    # Repeated forwards must be deterministic (padding is one-time, not per-call state).
    assert torch.equal(out, module(x))


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
def test_forward_eager_num_itr_stays_one_for_tiny_k(bits: int):
    """in_features < 32 pads g_idx to 32 rows; num_itr must derive from the
    logical K so the eager path never enters the multi-iteration dequant."""
    module = _packed_module(bits, in_features=16, out_features=64, group_size=16, sym=(bits == 3))
    assert module.padded_in_features == 32

    x = torch.randn(3, 16, dtype=torch.float16) * 0.5
    out = module._forward_eager(x, (3, module.out_features))
    assert out.shape == (3, 64)
    ref = _dense_reference(module, x)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
@pytest.mark.skipif(not _CUDA, reason="CUDA unavailable")
def test_forward_pad_slice_matches_dense_gpu(bits: int, shape):
    """GPU (TritonV2) forward on padded buffers must match the CPU dense reference."""
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    in_features, out_features, group_size = shape
    module = _packed_module(
        bits,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        sym=(bits == 3),
        cls=TritonV2Linear,
    )
    ref_module = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )
    _to_cuda(module)

    x = torch.randn(9, in_features, dtype=torch.float16) * 0.5
    out = module(x.cuda())
    assert out.shape == (9, out_features)

    ref = _dense_reference(ref_module, x)
    assert torch.allclose(out.cpu().float(), ref.float(), atol=2e-2, rtol=1e-2)

    # Buffers must be unchanged by forward (padding happens at pack/load only).
    assert module.qweight.shape[1] == module.padded_out_features
    assert module.g_idx.shape[0] == module.padded_in_features


def _assert_logical_features(module, in_features: int, out_features: int):
    assert module.in_features == in_features
    assert module.out_features == out_features
    if module.bias is not None:
        assert module.bias.shape[0] == out_features


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
def test_padding_never_modifies_logical_features_pack_stage(bits: int, shape):
    """Pack-stage padding must never rewrite the module's public in/out features."""
    in_features, out_features, group_size = shape
    module = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )
    _assert_logical_features(module, in_features, out_features)
    # Padded dims are exposed only through the dedicated attributes.
    assert module.padded_in_features >= in_features
    assert module.padded_out_features >= out_features

    # dequantize_weight() must return the logical shape, not the padded one.
    assert module.dequantize_weight().shape == (in_features, out_features)

    # Forward must accept logical-K activations and emit logical-N outputs,
    # and must not mutate the feature attributes.
    x = torch.randn(3, in_features, dtype=torch.float16) * 0.5
    out = module(x)
    assert out.shape == (3, out_features)
    _assert_logical_features(module, in_features, out_features)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
def test_padding_never_modifies_logical_features_load_stage(bits: int, shape):
    """Load (register_buffers) + state_dict + post_init must keep logical features intact."""
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    in_features, out_features, group_size = shape
    src = _packed_module(
        bits, in_features=in_features, out_features=out_features, group_size=group_size, sym=(bits == 3)
    )
    dst = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=(bits == 3),
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=FORMAT.GPTQ_P,
        register_buffers=True,
    )
    _assert_logical_features(dst, in_features, out_features)
    dst.load_state_dict({k: v.clone() for k, v in src.state_dict().items()}, strict=True)
    _assert_logical_features(dst, in_features, out_features)
    dst.post_init()
    _assert_logical_features(dst, in_features, out_features)

    # Saved state must round-trip without feature drift even when re-saved
    # after post_init (padding must not be applied twice).
    state = dst.state_dict()
    for key in ("qweight", "qzeros", "scales", "g_idx"):
        assert torch.equal(state[key].cpu(), src.state_dict()[key].cpu()), key

    x = torch.randn(2, in_features, dtype=torch.float16) * 0.5
    assert dst(x).shape == (2, out_features)
    _assert_logical_features(dst, in_features, out_features)


def test_relayout_never_modifies_logical_features():
    """convert_to_planar (continuous gptq_v2 3-bit -> planar) must keep logical features."""
    from test_format_bit_map import _make_continuous_module

    in_features, out_features = 256, 128
    module = _make_continuous_module(in_features=in_features, out_features=out_features)
    _assert_logical_features(module, in_features, out_features)
    assert module.convert_to_planar()
    _assert_logical_features(module, in_features, out_features)
    assert module.dequantize_weight().shape == (in_features, out_features)

    x = torch.randn(4, in_features, dtype=torch.float16) * 0.5
    assert module(x).shape == (4, out_features)
    _assert_logical_features(module, in_features, out_features)


@pytest.mark.parametrize("bits", PLANAR_KERNEL_BITS)
@pytest.mark.parametrize("shape", _PAD_SHAPES)
@pytest.mark.skipif(not _CUDA, reason="CUDA unavailable")
def test_padding_never_modifies_logical_features_gpu(bits: int, shape):
    """GPU post_init (incl. Pangolin gating) must not rewrite logical in/out features."""
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    in_features, out_features, group_size = shape
    module = _packed_module(
        bits,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        sym=(bits == 3),
        cls=TritonV2Linear,
    )
    _to_cuda(module)
    module.post_init()
    _assert_logical_features(module, in_features, out_features)

    x = torch.randn(1, in_features, dtype=torch.float16, device="cuda") * 0.5
    assert module(x).shape == (1, out_features)
    _assert_logical_features(module, in_features, out_features)
