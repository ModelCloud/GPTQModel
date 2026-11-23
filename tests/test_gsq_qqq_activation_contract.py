# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ GSQ calibration must describe the deployed INT8 activation grid.

``qqq_calibration_moments`` used to narrow the activation to FP16, take the row
maximum there, and divide by 127 in FP16 before widening the scale. The deployed
portable kernel widens first and divides in FP32, so the calibration statistics
described a different operand: an FP16-rounded scale for ordinary rows, and a
raised "runtime token scale underflows" for rows whose FP16 quotient is
subnormal even though the kernel deploys a finite FP32 scale.

Every expected value below comes from the shipped inference path
(``QQQLinear.dynamic_quant``) and from FP64 accumulation, never from the
calibration helper or from ``qqq_activation_quantize``: the point of the
regression is that the two paths agree.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import qqq as qqq_mod
from gptqmodel.quantization.gsq_qqq import qqq_calibration_moments
from gptqmodel.quantization.qqq_activation import qqq_activation_quantize


CPU = pytest.mark.cpu

#: The least positive FP32 subnormal, ``2**-149``. A positive row maximum below
#: ``127 * 2**-150`` divides to zero in FP32 even though the row itself is
#: representable, so the deployed scale has to be floored here.
_FP32_MIN_SUBNORMAL = 2.0**-149


def _deployed_quant(monkeypatch, x):
    """The real portable-kernel grid, with the Triton path disabled.

    ``forward`` also accepts bfloat16 through ``QQQTorchLinear``; this exercises
    the same method the kernel calls rather than re-deriving the arithmetic, so
    a helper that drifts from the kernel fails here.
    """

    monkeypatch.setattr(qqq_mod, "_qqq_gpu_quantizer", lambda: None)
    module = qqq_mod.QQQLinear.__new__(qqq_mod.QQQLinear)
    torch.nn.Module.__init__(module)
    return module.dynamic_quant(x)


def _oracle_moments(codes, scale, teacher):
    """FP64 moments of the deployed INT8 operand."""

    deployed = codes.double() * scale.double()
    reference = teacher.double()
    return deployed.T @ deployed, (reference - deployed).T @ deployed


def _assert_calibration_matches_deployment(monkeypatch, x, *, teacher=None):
    teacher = x if teacher is None else teacher
    deployed_codes, deployed_scale = _deployed_quant(monkeypatch, x)
    hessian, cross, count = qqq_calibration_moments(x, teacher_inputs=teacher)
    h_ref, cross_ref = _oracle_moments(deployed_codes, deployed_scale, teacher)

    assert count == x.shape[0]
    assert hessian.dtype == torch.float32 and cross.dtype == torch.float32
    assert torch.isfinite(hessian).all() and torch.isfinite(cross).all()
    if h_ref.norm() > 0:
        assert (hessian.double() - h_ref).norm() / h_ref.norm() <= 1e-6
    else:
        assert (hessian.double() - h_ref).norm() == 0
    assert (cross.double() - cross_ref).abs().max() <= 1e-6 * max(1.0, cross_ref.abs().max())
    # The calibration's own codes/scale are the ones it accumulates moments
    # from; they must be the deployed ones, not merely close.
    calib_codes, calib_scale = qqq_activation_quantize(x)
    assert torch.equal(calib_codes, deployed_codes.contiguous())
    torch.testing.assert_close(calib_scale, deployed_scale, rtol=0, atol=0)
    return deployed_codes, deployed_scale


def _assert_helper_matches_deployment(monkeypatch, x):
    """Codes and scale only, for rows whose FP32 moments would underflow.

    ``_assert_calibration_matches_deployment`` also compares the accumulated
    moments with an FP64 oracle. Squaring a subnormal operand underflows to
    zero in FP32, which is a pre-existing limit of FP32 moment accumulation and
    not the grid this fixture pins, so compare the grid operands directly.
    """

    deployed_codes, deployed_scale = _deployed_quant(monkeypatch, x)
    calib_codes, calib_scale = qqq_activation_quantize(x)
    assert torch.equal(calib_codes, deployed_codes.contiguous())
    torch.testing.assert_close(calib_scale, deployed_scale, rtol=0, atol=0)
    return deployed_codes, deployed_scale


@CPU
@pytest.mark.parametrize(
    "values,expected_codes,expected_scale",
    [
        # Ordinary FP16 values: the FP16 quotient rounded the scale away from
        # the FP32 value, so the old Hessian missed by 1.220694e-4 relative.
        ((1.0, 0.25), (127, 32), 1.0 / 127.0),
        # Small-but-normal values: old relative error 2.381857e-3.
        ((0.001, 0.00025), (127, 32), None),
        # FP16 subnormal input, still a normal FP32 scale on the grid.
        ((6.1e-5, 3.05e-5), (127, 64), None),
        # The FP16 quotient 2**-20/127 is subnormal and rounds to zero, yet the
        # deployed kernel uses the representable FP32 scale.
        ((2.0**-20, 2.0**-22), (127, 32), 7.50924655790186e-9),
        # FP16 subnormal range, where the narrower grid loses precision first.
        ((2.0**-24, 0.0), (127, 0), None),
    ],
)
def test_fp16_activation_is_quantized_on_the_deployed_fp32_grid(monkeypatch, values, expected_codes, expected_scale):
    x = torch.tensor([values], dtype=torch.float16)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x)
    assert tuple(codes[0].tolist()) == expected_codes
    if expected_scale is not None:
        torch.testing.assert_close(scale, torch.tensor([[expected_scale]], dtype=torch.float32), rtol=1e-6, atol=0)
    # Every scale is the stored FP32 widening of the row's FP16 maximum, so it
    # never carries the FP16 division's rounding.
    stored_max = x[0].float().abs().amax()
    assert scale.item() == pytest.approx((stored_max / 127.0).item(), rel=0, abs=0)


@CPU
def test_tiny_fp16_row_no_longer_raises_where_the_kernel_deploys_a_scale(monkeypatch):
    """The old calibration raised where the kernel produces usable output."""

    x = torch.tensor([[2.0**-20, 2.0**-22]], dtype=torch.float16)
    codes, scale = _deployed_quant(monkeypatch, x)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert scale.item() == pytest.approx(7.50924655790186e-9, rel=1e-6)
    assert tuple(codes.flatten().tolist()) == (127, 32)

    hessian, cross, count = qqq_calibration_moments(x)
    h_ref, cross_ref = _oracle_moments(codes, scale, x)
    assert count == 1
    assert torch.isfinite(hessian).all() and torch.isfinite(cross).all()
    torch.testing.assert_close(hessian.double(), h_ref, rtol=1e-6, atol=0)
    assert (cross.double() - cross_ref).abs().max() <= 1e-6 * h_ref.abs().max()


@CPU
def test_zero_row_contributes_exactly_zero(monkeypatch):
    x = torch.zeros((1, 8), dtype=torch.float16)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x)
    assert (codes == 0).all()
    torch.testing.assert_close(scale, torch.ones_like(scale))
    hessian, cross, _ = qqq_calibration_moments(x)
    torch.testing.assert_close(hessian, torch.zeros_like(hessian), rtol=0, atol=0)
    torch.testing.assert_close(cross, torch.zeros_like(cross), rtol=0, atol=0)


@CPU
def test_zero_and_nonzero_rows_coexist_without_poisoning_the_moments(monkeypatch):
    x = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.25, 1.0]], dtype=torch.float16)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x)
    assert (codes[0] == 0).all()
    assert scale[0].item() == 1.0
    assert scale[1].item() > 0


@CPU
@pytest.mark.parametrize(
    "value,expected_scale,expected_code",
    [
        # A genuinely zero row keeps the unit scale and decodes to zero.
        (0.0, 1.0, 0),
        # 7u/127 = 0.055u and 63u/127 = 0.496u both round to zero in FP32, so
        # the pre-fix unit scale erased them. The floored scale keeps the codes
        # that reconstruct the input exactly.
        (7 * _FP32_MIN_SUBNORMAL, _FP32_MIN_SUBNORMAL, 7),
        (63 * _FP32_MIN_SUBNORMAL, _FP32_MIN_SUBNORMAL, 63),
        # 64u/127 = 0.504u rounds up to one subnormal step, so this row never
        # needed the floor; it is the control that keeps the quotient path.
        (64 * _FP32_MIN_SUBNORMAL, _FP32_MIN_SUBNORMAL, 64),
        (127 * _FP32_MIN_SUBNORMAL, _FP32_MIN_SUBNORMAL, 127),
        # 200u/127 = 1.575u rounds to 2u, so the row saturates at code 100 and
        # still reconstructs 200u exactly.
        (200 * _FP32_MIN_SUBNORMAL, 2 * _FP32_MIN_SUBNORMAL, 100),
    ],
)
def test_fp32_subnormal_row_keeps_its_codes(monkeypatch, value, expected_scale, expected_code):
    """Issue: a finite subnormal row was misread as an all-zero row."""

    x = torch.tensor([[value]], dtype=torch.float32)
    codes, scale = _assert_helper_matches_deployment(monkeypatch, x)

    assert codes.item() == expected_code
    assert scale.item() == expected_scale
    if value:
        # Every one of these rows is an exact multiple of its scale, so the
        # recovered operand equals the stored input bit for bit.
        assert (codes.float() * scale).item() == x.item()


@CPU
def test_zero_and_subnormal_rows_keep_distinct_scales(monkeypatch):
    """The zero-row convention must not leak into a finite subnormal row."""

    x = torch.tensor(
        [[0.0, 0.0], [7 * _FP32_MIN_SUBNORMAL, 3 * _FP32_MIN_SUBNORMAL]],
        dtype=torch.float32,
    )
    codes, scale = _assert_helper_matches_deployment(monkeypatch, x)

    assert scale[0].item() == 1.0 and scale[1].item() == _FP32_MIN_SUBNORMAL
    assert codes[0].tolist() == [0, 0]
    assert codes[1].tolist() == [7, 3]


@CPU
def test_subnormal_row_projects_to_the_reference_value(monkeypatch):
    """The erased row was invisible to every consumer of the INT8 operand.

    A subnormal activation times a large weight is still representable, so the
    pre-fix zero codes made the whole row contribute exactly nothing.
    """

    x = torch.tensor([[7 * _FP32_MIN_SUBNORMAL, 3 * _FP32_MIN_SUBNORMAL]], dtype=torch.float32)
    codes, scale = _deployed_quant(monkeypatch, x)
    weight = torch.tensor([[2.0**100], [0.0]], dtype=torch.float64)

    projected = (codes.double() * scale.double()) @ weight
    reference = x.double() @ weight
    assert projected.item() != 0
    torch.testing.assert_close(projected, reference, rtol=0, atol=0)


@CPU
def test_torch_linear_forward_keeps_a_subnormal_activation_row(monkeypatch):
    """The next consumer: an exact identity projection must keep the row.

    The fixture is exact end to end, so the assertion is equality rather than a
    tolerance: the weight is the identity, its INT4 codes are ``0``/``1`` with
    a channel scale of ``1/16``, and the channelwise INT8 plane is therefore
    ``16 * identity``. The forward's ``(accum * s1) * s_channel`` order is exact
    for the power-of-two channel scale, so the visible output must reproduce the
    input row bit for bit. The pre-fix unit scale erased the codes, so the whole
    row projected to exactly zero and this test failed at the equality
    assertion, not on a tolerance.
    """

    # Keep the arithmetic in Torch so the assertion describes the activation
    # grid under test rather than a host-specific packed-CPU kernel.
    monkeypatch.setattr(qqq_mod, "_qqq_cpu_kernel", lambda: None)

    in_features, out_features = 128, 128
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(out_features, in_features))
    module = qqq_mod.QQQTorchLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
    )
    module.pack(linear, torch.ones(out_features, 1, dtype=torch.float16))
    module.post_init()
    module.eval()

    torch.testing.assert_close(
        module._unpermute_scales(module.s_channel, module._scale_perm_single_i).reshape(-1),
        torch.full((out_features,), 1.0 / 16.0, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        module._int8_weight_for_forward(),
        (torch.eye(out_features, in_features, dtype=torch.int8) * 16),
    )

    x = torch.zeros(1, in_features, dtype=torch.float32)
    x[0, :2] = torch.tensor(
        [7 * _FP32_MIN_SUBNORMAL, 3 * _FP32_MIN_SUBNORMAL], dtype=torch.float32
    )
    quant_x, scale = module.dynamic_quant(x)
    assert quant_x[0, :2].tolist() == [7, 3]
    assert scale.item() == _FP32_MIN_SUBNORMAL

    with torch.inference_mode():
        output = module(x)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    # An identity weight with channel scale 1/16 is the identity, so the
    # projection has to reproduce the subnormal row exactly.
    torch.testing.assert_close(output, x, rtol=0, atol=0)


@CPU
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_supported_activation_dtypes_use_the_same_grid(monkeypatch, dtype):
    # Values exactly representable in every dtype, so the stored rows are
    # identical and the grid must agree bit for bit.
    base = torch.tensor([[1.0, 0.25], [2.0**-10, 2.0**-12]], dtype=torch.float32)
    x = base.to(dtype)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x)
    fp16_codes, fp16_scale = _deployed_quant(monkeypatch, x.to(torch.float16))
    assert torch.equal(codes, fp16_codes)
    torch.testing.assert_close(scale, fp16_scale, rtol=0, atol=0)


@CPU
def test_bf16_row_keeps_its_own_range(monkeypatch):
    """BF16 has a wider exponent; the grid must follow the stored values."""

    x = torch.tensor([[2.0**-30, 2.0**-32]], dtype=torch.bfloat16)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x)
    assert tuple(codes.flatten().tolist()) == (127, 32)
    assert scale.item() == pytest.approx(2.0**-30 / 127.0, rel=1e-6)


@CPU
def test_cross_moment_uses_the_teacher_independently(monkeypatch):
    x = torch.tensor([[1.0, 0.25]], dtype=torch.float16)
    teacher = torch.tensor([[0.5, -0.125]], dtype=torch.float16)
    codes, scale = _assert_calibration_matches_deployment(monkeypatch, x, teacher=teacher)
    deployed = codes.double() * scale.double()
    torch.testing.assert_close(
        qqq_calibration_moments(x, teacher_inputs=teacher)[1].double(),
        (teacher.double() - deployed).T @ deployed,
        rtol=1e-6,
        atol=1e-9,
    )


@CPU
def test_qqq_quantizer_add_batch_accumulates_deployed_moments():
    """The real calibration consumer must collect the deployed grid.

    ``QQQ.add_batch`` is what the looper calls; a tiny FP16 row whose FP16
    quotient underflows used to raise there before any statistics were stored.
    """

    from gptqmodel.quantization.config import GSQConfig, QQQConfig
    from gptqmodel.quantization.qqq import QQQ

    torch.manual_seed(11)
    layer = torch.nn.Linear(4, 2, bias=False)
    qcfg = QQQConfig(bits=4, group_size=-1, desc_act=False, gsq=GSQConfig(enabled=True, steps=2, candidates=3))
    qqq = QQQ(layer, qcfg)
    assert qqq._gsq_enabled

    x = torch.tensor([[2.0**-20, 2.0**-22, 0.0, 0.0]], dtype=torch.float16)
    # ``add_batch`` transposes to [in,tokens] for a Linear layer.
    qqq.add_batch(x, None)

    ((hessian, cross, count),) = qqq._gsq_moments.values()
    codes, scale = qqq_activation_quantize(x)
    deployed = codes.double() * scale.double()
    assert count == 1
    torch.testing.assert_close(hessian.double(), deployed.T @ deployed, rtol=1e-6, atol=0)
    torch.testing.assert_close(cross.double(), (x.double() - deployed).T @ deployed, rtol=1e-6, atol=1e-12)


@CPU
def test_calibration_rejects_nonfinite_and_nonfloating_inputs():
    with pytest.raises(ValueError):
        qqq_calibration_moments(torch.tensor([[float("nan"), 1.0]], dtype=torch.float16))
    with pytest.raises(ValueError):
        qqq_calibration_moments(torch.tensor([[float("inf"), 1.0]], dtype=torch.float16))
    with pytest.raises(ValueError):
        qqq_calibration_moments(torch.ones((2, 4), dtype=torch.int32))
    with pytest.raises(ValueError):
        qqq_calibration_moments(torch.zeros((0, 4), dtype=torch.float16))


@CPU
def test_shared_helper_rejects_a_zero_width_row():
    with pytest.raises(ValueError):
        qqq_activation_quantize(torch.zeros((2, 0), dtype=torch.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton quantizer requires a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_calibration_matches_the_native_quantizer(dtype):
    """Real CUDA activations: the calibration must describe the Triton operand.

    The Triton path serves CUDA inference, so the CPU fallback's values are not
    evidence for it. Compare codes, scale and moments against
    ``qqq_triton.dynamic_quant`` plus FP64 accumulation.
    """

    from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant as triton_dynamic_quant

    torch.manual_seed(17)
    # Include a very small row (its FP16 quotient is subnormal) and ordinary rows.
    small = torch.zeros((1, 4096), dtype=dtype, device="cuda")
    small[0, 0] = 2.0**-20
    small[0, 1] = 2.0**-22
    bulk = (torch.randn(64, 4096, dtype=torch.float16, device="cuda") * 0.75).to(dtype)
    x = torch.cat([small, bulk], dim=0)

    codes, scale = triton_dynamic_quant(x)
    hessian, cross, count = qqq_calibration_moments(x)
    deployed = codes.double() * scale.double()
    h_ref = deployed.T @ deployed
    cross_ref = (x.double() - deployed).T @ deployed

    assert count == x.shape[0]
    assert (hessian.double() - h_ref).norm() / h_ref.norm() <= 1e-6
    assert (cross.double() - cross_ref).abs().max() <= 1e-6 * h_ref.abs().max()
    calib_codes, calib_scale = qqq_activation_quantize(x)
    assert torch.equal(calib_codes, codes.contiguous())
    torch.testing.assert_close(calib_scale, scale, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton quantizer requires a CUDA device")
def test_triton_quantizer_matches_the_torch_grid():
    """The native quantizer is a second implementation of the same grid."""

    from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant as triton_dynamic_quant

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.tensor(
            [[1.0, 0.25], [0.001, 0.00025], [2.0**-20, 2.0**-22], [0.0, 0.0]],
            dtype=dtype,
            device="cuda",
        )
        torch_codes, torch_scale = qqq_activation_quantize(x)
        triton_codes, triton_scale = triton_dynamic_quant(x)
        assert torch.equal(triton_codes, torch_codes), dtype
        torch.testing.assert_close(triton_scale, torch_scale, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton quantizer requires a CUDA device")
@pytest.mark.parametrize("value,expected_code", [
    (0.0, 0),
    (7 * _FP32_MIN_SUBNORMAL, 7),
    (63 * _FP32_MIN_SUBNORMAL, 63),
    (64 * _FP32_MIN_SUBNORMAL, 64),
])
def test_triton_quantizer_keeps_fp32_subnormal_rows(value, expected_code):
    """The deployed CUDA quantizer must not read a subnormal row as zero."""

    from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant as triton_dynamic_quant

    x = torch.tensor([[value, 0.0]], dtype=torch.float32, device="cuda")
    codes, scale = triton_dynamic_quant(x)
    torch_codes, torch_scale = qqq_activation_quantize(x)

    assert torch.equal(codes, torch_codes)
    torch.testing.assert_close(scale, torch_scale, rtol=0, atol=0)
    assert codes[0, 0].item() == expected_code
    expected_scale = 1.0 if value == 0.0 else _FP32_MIN_SUBNORMAL
    assert scale.item() == expected_scale
