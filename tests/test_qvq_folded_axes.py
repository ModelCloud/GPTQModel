# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, qvq_dense_oracle_forward
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    quantize_qvq_linear,
    rht_preprocess_hessian,
    rht_preprocess_weight,
    rht_reconstruct_weight,
)


@pytest.mark.parametrize(("input_hadamard", "output_hadamard"), ((True, True), (False, True), (True, False), (False, False)))
def test_qvq_transform_axis_preprocess_is_dense_exact(input_hadamard, output_hadamard):
    generator = torch.Generator().manual_seed(41000 + input_hadamard * 10 + output_hadamard)
    weight = torch.randn((32, 32), generator=generator)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    sv = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    inner = rht_preprocess_weight(
        weight,
        su,
        sv,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    reconstructed = rht_reconstruct_weight(
        inner,
        su,
        sv,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    torch.testing.assert_close(reconstructed, weight, rtol=2e-6, atol=2e-6)

    samples = torch.randn((96, 32), generator=generator)
    hessian = samples.T @ samples / samples.shape[0]
    transformed_hessian = rht_preprocess_hessian(
        hessian,
        su,
        hadamard=input_hadamard,
    )
    transformed_samples = samples * su
    if input_hadamard:
        from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

        transformed_samples = matmul_hadU(transformed_samples)
    torch.testing.assert_close(
        transformed_hessian,
        transformed_samples.T @ transformed_samples / samples.shape[0],
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(("input_hadamard", "output_hadamard"), ((True, True), (False, True), (True, False), (False, False)))
def test_qvq_p32_quantization_and_runtime_support_folded_axes(input_hadamard, output_hadamard):
    generator = torch.Generator().manual_seed(42000 + input_hadamard * 10 + output_hadamard)
    weight = torch.randn((32, 32), generator=generator) * 0.1
    samples = torch.randn((128, 32), generator=generator)
    hessian = samples.T @ samples / samples.shape[0]
    result = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        seed=7,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )

    assert result.input_hadamard is input_hadamard
    assert result.output_hadamard is output_hadamard
    assert result.weight.shape == weight.shape
    if input_hadamard and output_hadamard:
        assert result.serialization_allowed is True
    else:
        assert result.serialization_allowed is False
        with pytest.raises(RuntimeError, match="cannot be serialized"):
            result.serialized_tensors()

    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=32,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
        tensors={
            "trellis": result.trellis,
            "SU": result.SU,
            "SV": result.SV,
            "bank_ids": result.bank_ids,
            "bank_alt_id": result.bank_alt_id,
        },
        dtype=torch.float32,
    ).eval()
    x = torch.randn((3, 32), generator=generator)
    expected = x @ result.weight.T
    actual = qvq_dense_oracle_forward(layer, x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_qvq_shared_input_seed_changes_only_su_random_stream():
    generator = torch.Generator().manual_seed(42501)
    weight = torch.randn((32, 32), generator=generator) * 0.1
    samples = torch.randn((96, 32), generator=generator)
    hessian = samples.T @ samples / samples.shape[0]

    baseline = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        seed=71,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
    )
    shared_a = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        seed=71,
        input_sign_seed=991,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
    )
    shared_b = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        seed=72,
        input_sign_seed=991,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
    )

    assert torch.equal(shared_a.SU, shared_b.SU)
    assert not torch.equal(baseline.SU, shared_a.SU)
    # Consuming the ordinary module SU draw before generating SV preserves
    # all module-local randomness when a shared SU is selected.
    assert torch.equal(baseline.SV.sign(), shared_a.SV.sign())


def test_qvq_declared_folded_axes_can_be_serialized():
    generator = torch.Generator().manual_seed(42502)
    weight = torch.randn((32, 32), generator=generator) * 0.1
    samples = torch.randn((96, 32), generator=generator)
    result = quantize_qvq_linear(
        weight,
        samples.T @ samples / samples.shape[0],
        bits=2,
        seed=73,
        input_hadamard=True,
        output_hadamard=False,
        allow_folded_axis_serialization=True,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
    )
    assert result.serialization_allowed is True
    assert set(result.serialized_tensors()) >= {"trellis", "SU", "SV"}


@pytest.mark.parametrize(("input_hadamard", "output_hadamard"), ((False, True), (True, False), (False, False)))
def test_qvq_mlx_p32_folded_axes_match_quantized_dense_weight(input_hadamard, output_hadamard):
    mx = pytest.importorskip("mlx.core")
    from gptqmodel.utils.qvq_mlx import QVQMLXLinear

    generator = torch.Generator().manual_seed(43000 + input_hadamard * 10 + output_hadamard)
    weight = torch.randn((32, 32), generator=generator) * 0.1
    samples = torch.randn((96, 32), generator=generator)
    result = quantize_qvq_linear(
        weight,
        samples.T @ samples / samples.shape[0],
        bits=2,
        seed=11,
        rounding="block_ldlq",
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    layer = QVQMLXLinear(
        bits=2,
        in_features=32,
        out_features=32,
        trellis=mx.array(result.trellis.numpy()),
        SU=mx.array(result.SU.numpy()),
        SV=mx.array(result.SV.numpy()),
        bank_ids=mx.array(pack_qvq_binary_bank_ids(result.bank_ids).numpy()),
        v2b2_p32=True,
        bank_alt_id=mx.array(result.bank_alt_id.numpy()),
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    x = torch.randn((4, 32), generator=generator, dtype=torch.float16)
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    expected = (x.float() @ result.weight.float().T).half()
    torch.testing.assert_close(
        torch.from_numpy(np.asarray(actual)),
        expected,
        rtol=1e-3,
        atol=1e-3,
    )
