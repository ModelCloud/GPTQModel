# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4604: GPTAQ/FOEM must retry the Hessian inverse on the host.

GPTQ, QQQ and EXL3 wrap ``hessian_inverse`` in an accelerator-allocation
guard, so a cuSOLVER/hipSOLVER workspace failure (``*SOLVER_STATUS_ALLOC_FAILED``)
finishes the module on CPU instead of aborting the run. GPTAQ and FOEM
re-implement the block pass and called ``hessian_inverse`` (plus GPTAQ's
native-delta product) unguarded, so the same failure killed the whole
quantization.

These tests force a one-shot failure in the real Cholesky inverse and compare
the retried result with a run that never leaves the host. The comparison is
bitwise: the retry moves the Hessian, the weight, the native delta, the
act-order permutation and the static-group quantizers to the host, so it must
reproduce the host reference exactly.
"""

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fallback only engages for accelerator allocations",
)

ROWS, COLUMNS, GROUP_SIZE = 8, 64, 16
SAMPLES = 64
BLOCKSIZE = 16
ALPHA = 0.25
ALLOC_FAILURE = (
    "hipsolver error: HIPSOLVER_STATUS_ALLOC_FAILED, "
    "when calling `hipsolverDnSpotrf(...)`"
)
REAL_CHOLESKY = torch.linalg.cholesky
REAL_TRIU = torch.Tensor.triu


def _weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4604)
    return torch.randn(ROWS, COLUMNS, generator=generator) * 0.05


def _hessian() -> torch.Tensor:
    """Positive-definite Hessian with a scrambled diagonal.

    The scrambled diagonal makes ``argsort(diag(H), descending=True)`` a
    non-identity permutation, so a dropped reorder or a stale-device index
    tensor cannot pass unnoticed.
    """

    generator = torch.Generator().manual_seed(4604)
    skew = torch.randn(COLUMNS, COLUMNS, generator=generator)
    skew = (skew - skew.T) / 2
    diagonal = torch.linspace(1.0, 2.0, COLUMNS)[
        torch.randperm(COLUMNS, generator=generator)
    ]
    return torch.diag(diagonal) + 1e-3 * skew


def _dXXT() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4605)
    return torch.randn(COLUMNS, COLUMNS, generator=generator) * 0.01


def _task(
    kind: str,
    device,
    *,
    ordering: str = "desc_act",
    static_groups: bool = False,
    group_size: int = GROUP_SIZE,
):
    layer = torch.nn.Linear(
        COLUMNS, ROWS, bias=False, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        layer.weight.copy_(_weight().to(device))

    named = NamedModule(
        layer, name="proj", full_name="model.layers.0.proj", layer_index=0
    )
    named.state[NATIVE_INPUTS_STATE_KEY] = []

    common = {
        "bits": 4,
        "group_size": group_size,
        "static_groups": static_groups,
        "desc_act": ordering == "desc_act",
        "act_group_aware": ordering == "act_group_aware",
    }
    if kind == "gptaq":
        qcfg = QuantizeConfig(gptaq=GPTAQConfig(alpha=ALPHA), **common)
        cls = GPTAQ
    else:
        qcfg = QuantizeConfig(foem=FOEMConfig(alpha=ALPHA), **common)
        cls = FOEM

    task = cls(named, qcfg)
    task.quantizer.configure(perchannel=True)
    task.H = _hessian().to(device)
    task.dXXT = _dXXT().to(device)
    task.nsamples = SAMPLES
    task._gptaq_batch_request_count = SAMPLES
    return task


def _fail_first_cuda_cholesky(monkeypatch) -> dict:
    state = {"failed": False}

    def flaky(hessian, *args, **kwargs):
        if hessian.device.type == "cuda" and not state["failed"]:
            state["failed"] = True
            raise RuntimeError(ALLOC_FAILURE)
        return REAL_CHOLESKY(hessian, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", flaky)
    return state


def _fail_first_cuda_triu(monkeypatch) -> dict:
    """Fail inside GPTAQ's native-delta product, after the inverse succeeded."""

    state = {"failed": False}

    def flaky(tensor, *args, **kwargs):
        if tensor.is_cuda and not state["failed"]:
            state["failed"] = True
            raise RuntimeError(ALLOC_FAILURE)
        return REAL_TRIU(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "triu", flaky)
    return state


def _capture_fallback_warnings(monkeypatch) -> list:
    messages = []
    monkeypatch.setattr(
        gptq_mod.log,
        "warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )
    return messages


def _assert_matches_host_reference(result, reference) -> None:
    # The module's device is restored after the host retry, and both the
    # quantized weight and the act-order map match a run that never left the
    # host. Before the fix this aborted with the solver RuntimeError.
    assert result[0].device.type == "cuda"
    assert torch.equal(result[0].cpu(), reference[0].cpu())
    assert torch.equal(result[3].cpu(), reference[3].cpu())


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize(
    "ordering", ["desc_act", "act_group_aware", "natural"]
)
@pytest.mark.parametrize("static_groups", [False, True], ids=["dynamic", "static"])
def test_hessian_inverse_oom_retries_on_cpu(
    monkeypatch, kind, ordering, static_groups
):
    reference = _task(
        kind, "cpu", ordering=ordering, static_groups=static_groups
    ).quantize(blocksize=BLOCKSIZE)

    task = _task(kind, "cuda", ordering=ordering, static_groups=static_groups)
    state = _fail_first_cuda_cholesky(monkeypatch)
    messages = _capture_fallback_warnings(monkeypatch)

    result = task.quantize(blocksize=BLOCKSIZE)

    assert state["failed"] is True
    assert "falling back to CPU" in "\n".join(messages)
    _assert_matches_host_reference(result, reference)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
@pytest.mark.parametrize("static_groups", [False, True], ids=["dynamic", "static"])
def test_ungrouped_desc_act_retry_keeps_host_numerics(
    monkeypatch, kind, static_groups
):
    # Ungrouped ``desc_act`` keeps the parameters measured before the reorder,
    # so the retry must rebuild that measurement from the natural order.
    reference = _task(
        kind, "cpu", ordering="desc_act", static_groups=static_groups,
        group_size=-1,
    ).quantize(blocksize=BLOCKSIZE)

    task = _task(
        kind, "cuda", ordering="desc_act", static_groups=static_groups,
        group_size=-1,
    )
    state = _fail_first_cuda_cholesky(monkeypatch)

    result = task.quantize(blocksize=BLOCKSIZE)

    assert state["failed"] is True
    _assert_matches_host_reference(result, reference)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_native_delta_product_oom_retries_on_cpu(monkeypatch, kind):
    # ``P = alpha * ((dXXT @ Hinv.T).triu(1)) @ Hinv`` is a second
    # ``columns ** 2`` allocation that can fail after a successful inverse.
    reference = _task(kind, "cpu").quantize(blocksize=BLOCKSIZE)

    task = _task(kind, "cuda")
    state = _fail_first_cuda_triu(monkeypatch)
    messages = _capture_fallback_warnings(monkeypatch)

    result = task.quantize(blocksize=BLOCKSIZE)

    assert state["failed"] is True
    assert "falling back to CPU" in "\n".join(messages)
    _assert_matches_host_reference(result, reference)


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_numerical_inverse_failure_does_not_retry(monkeypatch, kind):
    task = _task(kind, "cuda")
    calls = {"cuda": 0, "cpu": 0}

    def failing_hessian_inverse(self, hessian):
        calls[hessian.device.type] += 1
        raise torch._C._LinAlgError(
            "The leading minor of order 1 is not positive definite"
        )

    monkeypatch.setattr(GPTQ, "hessian_inverse", failing_hessian_inverse)

    with pytest.raises(torch._C._LinAlgError):
        task.quantize(blocksize=BLOCKSIZE)

    assert calls == {"cuda": 1, "cpu": 0}
