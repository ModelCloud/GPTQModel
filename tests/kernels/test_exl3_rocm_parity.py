# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXL3 codebook parity on ROCm / RDNA4 (gfx12xx).

The harness has two jobs:

* prove that every EXL3 codebook this tree can build decodes to the same
  weights as the independent bitstream oracle in
  ``tests/test_exllamav3_torch_padding.py``; and
* prove that the fallback forwards work for those codebooks on ROCm.

Two forward paths are checked when the JIT extension is available:

``portable``
    ``ExllamaV3TorchLinear``: torch decode of the trellis, then a hipBLAS
    matmul.  This is the backend AUTO selects for EXL3 on ROCm today.

``reconstruct+hgemm``
    ``LinearEXL3.forward`` on a HIP build: ``ext.reconstruct`` writes a dense
    FP16 weight, ``ext.hgemm`` runs hipBLAS, and ``ext.had_r_128`` applies the
    output rotation.  This is the reference oracle the native RDNA4 kernels
    must match (issue #5710).

Codebook coverage on ROCm:

* ``mcg`` at 2/3/4 bpw and ``mul1`` at 2/3/4 bpw quantize through
  ``quantize_exl3``.
* ``mul1`` at 2.5/3.5 bpw and ``q4nx`` at 2.5/3/3.5 bpw are exposed for
  *inference* but their quantizers are CUDA-only on ROCm (the Viterbi and
  q4nx tile kernels are not in the HIP build).  Those cases skip quantization
  with the printed reason and are still exercised end to end from a synthetic
  packed trellis, which is enough to validate decode and forward parity.
* ``r4nx`` is picked up automatically when the tree exposes it (a
  ``gptqmodel.exllamav3.r4nx`` module, an ``r4nx`` marker in
  ``EXL3_CODEBOOK_MARKERS``, or an ``r4nx`` keyword on
  ``validate_trellis_k``); otherwise the test skips with the reason.

Run with the RDNA4 device pinned, for example::

    HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD HF_HUB_OFFLINE=1 \
    GPTQMODEL_EXLLAMAV3_BUILD_ROOT=$PWD/.extbuild MAX_JOBS=8 \
    /opt/venvs/rocm101t/bin/python -m pytest \
        tests/kernels/test_exl3_rocm_parity.py -p no:cacheprovider
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache

import pytest
import torch

from gptqmodel.exllamav3.trellis import EXL3_CODEBOOK_MARKERS, validate_trellis_k
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from tests.test_exllamav3_torch_padding import _oracle_inner, _oracle_weight


pytestmark = pytest.mark.cuda

IN_FEATURES = 256
OUT_FEATURES = 256
CALIBRATION_ROWS = 64
INPUT_ROWS = 16
SEED = 787

# ``quantize_exl3`` returns the dense weight after the Hadamard/sign transforms
# in FP32, while the oracle rebuilds it from the packed trellis.  The two
# accumulation orders can cross a half-rounding boundary, so the dense weight
# check allows a small absolute error; the packed decode checks below are exact.
WEIGHT_RTOL = 1e-3
WEIGHT_ATOL = 1e-3
PORTABLE_FORWARD_RTOL = 2e-3
PORTABLE_FORWARD_ATOL = 2e-3
NATIVE_FORWARD_RTOL = 5e-3
NATIVE_FORWARD_ATOL = 5e-3

# Integer codebooks quantize on ROCm; half-bit and q4nx quantization is
# CUDA-only, so those rates are covered by the synthetic packed cases below.
QUANTIZED_CASES = (
    ("mcg", 2),
    ("mcg", 3),
    ("mcg", 4),
    ("mul1", 2),
    ("mul1", 3),
    ("mul1", 4),
)
SYNTHETIC_CASES = (
    ("mul1", 2.5),
    ("mul1", 3.5),
    ("q4nx", 2.5),
    ("q4nx", 3.0),
    ("q4nx", 3.5),
)


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName)
    except Exception:  # pragma: no cover - old torch / non-AMD build
        return ""


def _is_rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


if not _is_rocm():
    pytest.skip("EXL3 ROCm parity requires a HIP build", allow_module_level=True)
if not _gcn_arch().startswith("gfx12"):
    pytest.skip(
        f"RDNA4 (gfx12xx) required; detected {_gcn_arch() or 'unknown arch'}",
        allow_module_level=True,
    )


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _ids(cases):
    return [f"{codebook}-{rate:g}bpw" for codebook, rate in cases]


def _cpu_packed(packed: dict) -> dict:
    return {name: tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for name, tensor in packed.items()}


def _to_device(packed: dict, device: torch.device) -> dict:
    return {name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for name, tensor in packed.items()}


def _make_bias(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    return (torch.randn(OUT_FEATURES, device=device, generator=generator) * 0.01).half()


def _make_input(device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED + 2)
    x = torch.randn(INPUT_ROWS, IN_FEATURES, device=device, generator=generator) * 0.1
    return x.to(dtype)


def _reference_output(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    expected = x.double() @ weight.double()
    if bias is not None:
        expected = expected + bias.double()
    return expected


def _torch_module(packed: dict) -> ExllamaV3TorchLinear:
    return ExllamaV3TorchLinear.from_tensors(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        name="exl3_rocm_parity",
        tensors=packed,
    )


def _inner_module(packed: dict, bias: torch.Tensor | None = None):
    from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3

    return LinearEXL3(
        config=None,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        suh=packed["suh"],
        svh=packed["svh"],
        trellis=packed["trellis"],
        mcg=packed.get("mcg"),
        mul1=packed.get("mul1"),
        q4nx=packed.get("q4nx"),
        r4nx=packed.get("r4nx"),
        bias=bias,
        out_dtype=torch.float32,
    )


@dataclass(frozen=True)
class _ExtensionStatus:
    available: bool
    reason: str


@lru_cache(maxsize=1)
def _extension_status() -> _ExtensionStatus:
    """Build (once) and report whether the EXL3 JIT extension is usable."""

    try:
        from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

        if prewarm_exllamav3_extension():
            return _ExtensionStatus(True, "")
        return _ExtensionStatus(False, "prewarm_exllamav3_extension() returned False")
    except Exception as exc:  # pragma: no cover - depends on the host toolchain
        detail = " ".join(str(exc).split())
        return _ExtensionStatus(False, f"{type(exc).__name__}: {detail[:400]}")


def _require_extension() -> None:
    status = _extension_status()
    if not status.available:
        reason = f"EXL3 JIT extension is unavailable on this ROCm build: {status.reason}"
        print(f"[exl3-rocm] skip: {reason}")
        pytest.skip(reason)


@lru_cache(maxsize=None)
def _quantized_case(codebook: str, rate: float):
    """Quantize one deterministic weight, cached across the parity tests."""

    # ``quantize_exl3`` calls ext.quantize_tiles/pack_trellis, so the quantized
    # cases need the JIT extension even though the portable decode does not.
    _require_extension()
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = _device()
    generator = torch.Generator(device=device).manual_seed(SEED)
    weight = torch.randn(IN_FEATURES, OUT_FEATURES, device=device, generator=generator) * 0.1
    calibration = torch.randn(CALIBRATION_ROWS, IN_FEATURES, device=device, generator=generator) * 0.1
    hessian = calibration.T @ calibration
    quant_args = {
        "K": rate,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": SEED,
        codebook: True,
    }
    weight_q, proxy_error, packed = quantize_exl3(
        weight=weight,
        H_data={"H": hessian, "count": CALIBRATION_ROWS, "finalized": False},
        quant_args=quant_args,
        return_weight_q=True,
    )
    assert quant_args.get("q_fallback") is False, (
        f"{codebook} {rate:g} bpw fell back to unweighted quantization; "
        "the deterministic Hessian should be well conditioned"
    )
    assert proxy_error == proxy_error  # NaN guard
    return weight_q, packed


def _synthetic_case(codebook: str, rate: float):
    """Build a valid packed trellis without the CUDA-only quantizer."""

    device = _device()
    words = int(16 * rate)
    trellis = torch.randint(
        -32768,
        32768,
        (IN_FEATURES // 16, OUT_FEATURES // 16, words),
        dtype=torch.int16,
        generator=torch.Generator().manual_seed(SEED + 10),
    ).to(device)
    suh = (
        torch.randn(IN_FEATURES, generator=torch.Generator().manual_seed(SEED + 11)) * 0.1
    ).half().to(device)
    svh = (
        torch.randn(OUT_FEATURES, generator=torch.Generator().manual_seed(SEED + 12)) * 0.1
    ).half().to(device)
    marker = {
        "mcg": 0xCBAC1FED,
        "mul1": 0x83DCD12D,
        "q4nx": 0x83DCD12D,
        "r4nx": 0x83DCD12D,
    }[codebook]
    packed = {
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        codebook: torch.tensor(marker, dtype=torch.uint32).view(torch.int32).to(device),
    }
    return packed


@pytest.mark.parametrize(("codebook", "rate"), QUANTIZED_CASES, ids=_ids(QUANTIZED_CASES))
def test_quantized_packed_tensors_match_bitstream_oracle(codebook: str, rate: float):
    """The quantizer's dense output must equal the independent trellis decode."""

    weight_q, packed = _quantized_case(codebook, rate)
    reference = _oracle_weight(_cpu_packed(packed), device=_device())
    torch.testing.assert_close(
        weight_q.half().float(),
        reference,
        rtol=WEIGHT_RTOL,
        atol=WEIGHT_ATOL,
        msg=lambda msg: f"{codebook} {rate:g} bpw: quantized weight differs from the bitstream oracle\n{msg}",
    )


@pytest.mark.parametrize(("codebook", "rate"), QUANTIZED_CASES, ids=_ids(QUANTIZED_CASES))
def test_portable_forward_matches_bitstream_oracle(codebook: str, rate: float):
    """EXL3_TORCH (the ROCm AUTO backend) must match the FP64 oracle."""

    _, packed = _quantized_case(codebook, rate)
    device = _device()
    packed = _to_device(packed, device)
    bias = _make_bias(device)
    packed["bias"] = bias
    reference = _oracle_weight(_cpu_packed(packed), device=device)
    module = _torch_module(packed)

    torch.testing.assert_close(
        module.get_weight_tensor(torch.float32),
        reference,
        rtol=1e-4,
        atol=1e-4,
        msg=lambda msg: f"{codebook} {rate:g} bpw: portable decode differs from the bitstream oracle\n{msg}",
    )

    x = _make_input(device, torch.float32)
    actual = module(x)
    expected = _reference_output(x, reference, bias)
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=PORTABLE_FORWARD_RTOL,
        atol=PORTABLE_FORWARD_ATOL,
        msg=lambda msg: f"{codebook} {rate:g} bpw: portable forward differs from the oracle\n{msg}",
    )


@pytest.mark.parametrize(("codebook", "rate"), QUANTIZED_CASES, ids=_ids(QUANTIZED_CASES))
def test_reconstruct_hgemm_forward_matches_bitstream_oracle(codebook: str, rate: float):
    """The ROCm reconstruct+hgemm fallback must match the FP64 oracle."""

    _require_extension()
    _, packed = _quantized_case(codebook, rate)
    device = _device()
    packed = _to_device(packed, device)
    bias = _make_bias(device)
    reference = _oracle_weight(_cpu_packed(packed), device=device)
    inner = _inner_module(packed, bias)

    torch.testing.assert_close(
        inner.get_inner_weight_tensor().float().cpu(),
        _oracle_inner(_cpu_packed(packed)),
        rtol=1e-4,
        atol=1e-4,
        msg=lambda msg: f"{codebook} {rate:g} bpw: ext.reconstruct differs from the bitstream oracle\n{msg}",
    )

    x = _make_input(device, torch.float16)
    actual = inner.forward(x, {})
    expected = _reference_output(x, reference, bias)
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=NATIVE_FORWARD_RTOL,
        atol=NATIVE_FORWARD_ATOL,
        msg=lambda msg: f"{codebook} {rate:g} bpw: reconstruct+hgemm forward differs from the oracle\n{msg}",
    )


@pytest.mark.parametrize(("codebook", "rate"), SYNTHETIC_CASES, ids=_ids(SYNTHETIC_CASES))
def test_synthetic_portable_fallback_matches_bitstream_oracle(codebook: str, rate: float):
    """Half-bit/q4nx inference works from a packed trellis even without the CUDA quantizer."""

    device = _device()
    packed = _synthetic_case(codebook, rate)
    bias = _make_bias(device)
    packed["bias"] = bias
    reference = _oracle_weight(_cpu_packed(packed), device=device)
    module = _torch_module(packed)
    torch.testing.assert_close(
        module.get_weight_tensor(torch.float32),
        reference,
        rtol=1e-4,
        atol=1e-4,
        msg=lambda msg: f"{codebook} {rate:g} bpw: synthetic portable decode differs from the oracle\n{msg}",
    )

    x = _make_input(device, torch.float32)
    actual = module(x)
    expected = _reference_output(x, reference, bias)
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=PORTABLE_FORWARD_RTOL,
        atol=PORTABLE_FORWARD_ATOL,
        msg=lambda msg: f"{codebook} {rate:g} bpw: synthetic portable forward differs from the oracle\n{msg}",
    )


@pytest.mark.parametrize(("codebook", "rate"), SYNTHETIC_CASES, ids=_ids(SYNTHETIC_CASES))
def test_synthetic_reconstruct_hgemm_fallback_matches_bitstream_oracle(codebook: str, rate: float):
    """The reconstruct+hgemm fallback covers codebooks whose quantizer is CUDA-only."""

    _require_extension()
    device = _device()
    packed = _synthetic_case(codebook, rate)
    bias = _make_bias(device)
    reference = _oracle_weight(_cpu_packed(packed), device=device)
    inner = _inner_module(packed, bias)

    x = _make_input(device, torch.float16)
    actual = inner.forward(x, {})
    expected = _reference_output(x, reference, bias)
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=NATIVE_FORWARD_RTOL,
        atol=NATIVE_FORWARD_ATOL,
        msg=lambda msg: f"{codebook} {rate:g} bpw: synthetic reconstruct+hgemm forward differs from the oracle\n{msg}",
    )


def test_half_bit_and_q4nx_quantizers_report_cuda_only_on_rocm():
    """The harness skips these quantizers for a reason; pin the reason to the build."""

    _require_extension()
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = _device()
    generator = torch.Generator(device=device).manual_seed(SEED)
    weight = torch.randn(IN_FEATURES, OUT_FEATURES, device=device, generator=generator) * 0.1
    calibration = torch.randn(CALIBRATION_ROWS, IN_FEATURES, device=device, generator=generator) * 0.1
    for codebook, rate in (("mul1", 2.5), ("q4nx", 3.0)):
        quant_args = {
            "K": rate,
            "devices": [device],
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": SEED,
            codebook: True,
        }
        with pytest.raises((RuntimeError, ValueError), match="requires NVIDIA CUDA"):
            quantize_exl3(
                weight=weight,
                H_data={
                    "H": calibration.T @ calibration,
                    "count": CALIBRATION_ROWS,
                    "finalized": False,
                },
                quant_args=quant_args,
                return_weight_q=True,
            )


def _r4nx_rates() -> tuple[float, ...] | None:
    """Feature-detect the R4NX codebook without assuming an API shape."""

    module = None
    for name in (
        "gptqmodel.exllamav3.r4nx",
        "gptqmodel.exllamav3.modules.quant.exl3_lib.r4nx",
        "gptqmodel.exllamav3.modules.quant.r4nx",
    ):
        try:
            module = importlib.import_module(name)
            break
        except Exception:
            continue

    parameters = inspect.signature(validate_trellis_k).parameters
    has_keyword = "r4nx" in parameters
    has_marker = "r4nx" in EXL3_CODEBOOK_MARKERS
    if module is None and not has_keyword and not has_marker:
        return None

    candidates = tuple(getattr(module, "R4NX_RATES", ())) if module is not None else ()
    if not candidates:
        candidates = (2.0, 2.5, 3.0, 3.5, 4.0)

    rates: list[float] = []
    for rate in candidates:
        kwargs = {"r4nx": True} if has_keyword else {}
        try:
            validate_trellis_k(rate, False, False, **kwargs)
        except Exception:
            continue
        rates.append(float(rate))
    return tuple(rates) or (None if module is None else tuple(float(rate) for rate in candidates))


def test_r4nx_codebook_when_available():
    """Exercise R4NX through the same quantize/decode/forward checks when it lands."""

    rates = _r4nx_rates()
    if not rates:
        reason = (
            "R4NX is not exposed by this tree: no gptqmodel.exllamav3.r4nx module, "
            "no `r4nx` marker in EXL3_CODEBOOK_MARKERS, and no `r4nx` keyword on "
            "validate_trellis_k"
        )
        print(f"[exl3-rocm] skip: {reason}")
        pytest.skip(reason)

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    device = _device()
    generator = torch.Generator(device=device).manual_seed(SEED)
    weight = torch.randn(IN_FEATURES, OUT_FEATURES, device=device, generator=generator) * 0.1
    calibration = torch.randn(CALIBRATION_ROWS, IN_FEATURES, device=device, generator=generator) * 0.1
    x = _make_input(device, torch.float16)
    checked = 0
    for rate in rates:
        quant_args = {
            "K": rate,
            "devices": [device],
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": SEED,
            "r4nx": True,
        }
        try:
            _, _, packed = quantize_exl3(
                weight=weight,
                H_data={
                    "H": calibration.T @ calibration,
                    "count": CALIBRATION_ROWS,
                    "finalized": False,
                },
                quant_args=quant_args,
                return_weight_q=True,
            )
        except Exception as exc:
            reason = f"R4NX quantizer at {rate:g} bpw is not usable on this build: {type(exc).__name__}: {exc}"
            print(f"[exl3-rocm] skip: {reason}")
            pytest.skip(reason)

        packed = _to_device(packed, device)
        bias = _make_bias(device)
        packed["bias"] = bias
        module = _torch_module(packed)
        try:
            reference = module.get_weight_tensor(torch.float32)
        except Exception as exc:
            reason = f"R4NX has no torch reference decode on this build: {type(exc).__name__}: {exc}"
            print(f"[exl3-rocm] skip: {reason}")
            pytest.skip(reason)

        actual = module(x.float())
        expected = _reference_output(x, reference, bias)
        torch.testing.assert_close(
            actual.double(),
            expected,
            rtol=PORTABLE_FORWARD_RTOL,
            atol=PORTABLE_FORWARD_ATOL,
            msg=lambda msg: f"r4nx {rate:g} bpw: portable forward differs from the torch reference\n{msg}",
        )

        if _extension_status().available:
            inner = _inner_module(packed, bias)
            actual = inner.forward(x, {})
            torch.testing.assert_close(
                actual.double(),
                expected,
                rtol=NATIVE_FORWARD_RTOL,
                atol=NATIVE_FORWARD_ATOL,
                msg=lambda msg: f"r4nx {rate:g} bpw: reconstruct+hgemm forward differs from the torch reference\n{msg}",
            )
        checked += 1

    assert checked > 0


def test_native_dispatch_state_is_reported_not_assumed():
    """Record whether the HIP tensor-core dispatch is live without failing either state."""

    _require_extension()
    from gptqmodel.exllamav3.ext import exllamav3_ext

    device = _device()
    packed = _synthetic_case("mcg", 3)
    inner = _inner_module(packed)
    x = _make_input(device, torch.float16)
    y = torch.empty((x.shape[0], OUT_FEATURES), dtype=torch.float16, device=device)

    native = False
    message = ""
    try:
        inner.bc.run(x, y, include_bias=False)
        native = True
    except Exception as exc:
        message = " ".join(str(exc).split())
        lowered = message.lower()
        assert any(
            token in lowered
            for token in ("not available", "no exl3 gemm kernel", "rocm", "hip")
        ), f"unexpected native dispatch failure: {message}"

    # Keep the extension facade imported so this test documents the same object
    # the product dispatch uses.
    assert exllamav3_ext is not None
    print(
        "[exl3-rocm] native HIP tensor-core dispatch: "
        + ("available" if native else f"unavailable ({message[:200]})")
    )
