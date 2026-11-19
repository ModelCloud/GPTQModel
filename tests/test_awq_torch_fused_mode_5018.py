# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`TorchFusedAwqLinear` must clear its training latch on `eval()` (#5018).

The CPU int4pack kernels pick their layout from `self.training` on the first
forward and latch `linear_mode`. #4949 taught `TorchAtenLinear`,
`TorchAtenAwqLinear` and `TorchFusedLinear` to clear the latch when leaving
training mode; `TorchFusedAwqLinear` kept the old one-way transition, so a
single forward taken in training mode (a LoRA step, a `model.train()`
evaluation loop) pinned it to full dense dequantization for the rest of the
process: `eval()` could never recover, and `inference_mode()` forwards stayed
32x slower on AWQ-Llama-3.2-1B fp16.

The cases here are CPU-only and build the packed buffers directly, so no
checkpoint is needed. The AVX512 siblings reset the latch on any mode change
by construction; they are included as controls when this host can run them.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.looper.linear_mode import LinearMode
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear


IN_FEATURES = 256
OUT_FEATURES = 256
GROUP_SIZE = 128


def _kernel(cls):
    """Build one packed int4 module, skipping a kernel this host cannot run."""

    try:
        return cls(
            bits=4,
            group_size=GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            bias=False,
            register_buffers=True,
        )
    except RuntimeError as exc:  # e.g. the AVX512-BF16 kernel on a non-AVX512 host
        pytest.skip(f"{cls.__name__} is not usable on this host: {exc}")


def _packed_awq_module() -> TorchFusedAwqLinear:
    """A `TorchFusedAwqLinear` with non-trivial packed buffers."""

    module = _kernel(TorchFusedAwqLinear)
    generator = torch.Generator().manual_seed(1)
    max_int32 = 2**31 - 1
    module.qweight.copy_(
        torch.randint(0, max_int32, module.qweight.shape, generator=generator).to(torch.int32)
    )
    module.qzeros.copy_(
        torch.randint(0, max_int32, module.qzeros.shape, generator=generator).to(torch.int32)
    )
    module.scales.copy_(
        (torch.rand(module.scales.shape, generator=generator) * 0.01 + 0.001).to(torch.float16)
    )
    return module


def _eval_forward(module, x):
    module.eval()
    with torch.inference_mode():
        return module(x)


INT4PACK_KERNELS = (
    TorchAtenLinear,
    TorchAtenAwqLinear,
    TorchFusedLinear,
    TorchFusedAwqLinear,
    TorchCpuAvx512Linear,
    TorchCpuAvx512AwqLinear,
)


@pytest.mark.parametrize("kernel_cls", INT4PACK_KERNELS, ids=lambda cls: cls.__name__)
def test_int4pack_family_clears_the_train_latch_on_eval(kernel_cls):
    """`eval()` after a training-mode forward must drop the `TRAIN` latch."""

    module = _kernel(kernel_cls)
    module.linear_mode = LinearMode.TRAIN

    module.eval()

    assert module.linear_mode is None, (
        f"{kernel_cls.__name__} kept the `TRAIN` latch across eval(); the next "
        "inference forward would dequantize the full weight"
    )


def test_torch_fused_awq_recovers_the_fused_path_after_a_training_forward():
    """The end-to-end symptom: train forward, `eval()`, then inference."""

    x = torch.randn(4, IN_FEATURES, dtype=torch.float16)
    reference = _eval_forward(_packed_awq_module(), x)

    module = _packed_awq_module()
    module.train()
    with torch.no_grad():
        train_output = module(x)
    assert module.linear_mode == LinearMode.TRAIN
    assert torch.isfinite(train_output).all()

    recovered = _eval_forward(module, x)

    assert module.linear_mode == LinearMode.INFERENCE, (
        "a forward taken in training mode pinned the module to the dense path; "
        "`eval()` must let the next forward choose the fused layout again (#5018)"
    )
    # The recovered fused path is the same computation as a fresh eval module.
    assert torch.equal(recovered, reference)
    # ...and it is not the degenerate all-zero output of an uninitialised module.
    assert float(reference.abs().max()) > 0


def test_torch_fused_awq_training_forward_still_uses_the_dense_path():
    """Clearing the latch on `eval()` must not change training behaviour."""

    module = _packed_awq_module()
    module.train()
    x = torch.randn(2, IN_FEATURES, dtype=torch.float16)

    with torch.no_grad():
        output = module(x)

    assert module.linear_mode == LinearMode.TRAIN
    # The dense path dequantizes through `awq_weight_dequantize`; it must not
    # have built the fused operator or the packed scales/zeros.
    assert getattr(module, "torch_fused_op", None) is None
    assert getattr(module, "scales_and_zeros", None) is None
    assert torch.isfinite(output).all()


def test_torch_fused_awq_alternating_modes_stay_correct():
    """Repeated train/eval cycles keep recovering the fused layout."""

    x = torch.randn(3, IN_FEATURES, dtype=torch.float16)
    reference = _eval_forward(_packed_awq_module(), x)
    module = _packed_awq_module()

    for _ in range(3):
        module.train()
        with torch.no_grad():
            module(x)
        assert module.linear_mode == LinearMode.TRAIN

        recovered = _eval_forward(module, x)
        assert module.linear_mode == LinearMode.INFERENCE
        assert torch.equal(recovered, reference)
