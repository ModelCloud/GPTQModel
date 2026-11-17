# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
"""Coverage for the ROCm quantize-phase BLAS backend override."""

import threading
import time
from contextlib import contextmanager

import pytest
import torch

from gptqmodel.utils import rocm
from gptqmodel.utils.kernel import KernelConfig, configure_kernel


_BACKEND_NAMES = {
    "rocblas": "Cublas",
    "hipblas": "Cublas",
    "cublas": "Cublas",
    "hipblaslt": "Cublaslt",
    "cublaslt": "Cublaslt",
    "ck": "Ck",
}


class _FakeBackend:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_BlasBackend.{self.name}"


class _FakeTorchBlas:
    """Minimal stand-in for ``torch.backends.cuda.preferred_blas_library``."""

    def __init__(self, current="Cublaslt", refuse=None):
        self.current = _FakeBackend(current)
        self.refuse = refuse
        self.calls = []

    def preferred_blas_library(self, value=None):
        if value is None:
            return self.current
        # The real setter accepts both backend names and the backend enum.
        name = getattr(value, "name", value).lower()
        self.calls.append(name)
        if self.refuse is not None and name == self.refuse:
            raise RuntimeError("refused")
        self.current = _FakeBackend(_BACKEND_NAMES[name])
        return None


def _install(monkeypatch, fake):
    monkeypatch.setattr(rocm, "_preferred_blas_library_getter", lambda: fake.preferred_blas_library)


@pytest.fixture(autouse=True)
def _reset_guard(monkeypatch):
    """The guard is process-global; never leak state between tests."""

    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_DEPTH", 0)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_PREVIOUS", None)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_TARGET", None)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_RESTORE", False)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_WAIT_WARNED", False)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_WAIT_SECONDS", 1800.0)
    monkeypatch.setattr(rocm._QUANT_BLAS_GUARD_THREAD_LOCAL, "depth", 0, raising=False)


def test_switch_is_scoped_to_the_context(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublas"

    assert fake.calls == ["hipblas", "cublaslt"]
    assert fake.current.name == "Cublaslt"


def _raise_from_quantize():
    raise RuntimeError("quantize blew up")


def test_restore_happens_on_error(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with pytest.raises(RuntimeError):
        with rocm.quant_blas_backend():
            assert fake.current.name == "Cublas"
            _raise_from_quantize()

    assert fake.calls == ["hipblas", "cublaslt"]
    assert fake.current.name == "Cublaslt"


def test_env_can_keep_hipblaslt(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig(rocm_quant_blas="hipblaslt"))

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == []


def test_user_selected_backend_is_respected(monkeypatch):
    fake = _FakeTorchBlas(current="Ck")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        assert fake.current.name == "Ck"

    assert fake.calls == []


def test_non_rocm_hosts_are_untouched(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", False)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == []


def test_refused_switch_keeps_the_original_backend(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt", refuse="hipblas")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == ["hipblas"]
    assert fake.current.name == "Cublaslt"


def test_missing_torch_api_is_tolerated(monkeypatch):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setattr(rocm, "_preferred_blas_library_getter", lambda: None)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        pass


def _gptq_hessian_recorder(monkeypatch):
    """Patch GPTQ's Hessian path to record every BLAS-override entry."""

    from gptqmodel.quantization import gptq

    entered: list[object] = []

    @contextmanager
    def recorder(device=None):
        entered.append(device)
        yield

    monkeypatch.setattr(gptq, "quant_blas_backend", recorder)
    return gptq, entered


def test_gptq_hessian_accumulation_enters_the_override(monkeypatch):
    """Determinism comes from the fp32 X^T X GEMM, so pin only that phase."""

    gptq, entered = _gptq_hessian_recorder(monkeypatch)
    accumulator = gptq.GPTQ.__new__(gptq.GPTQ)
    accumulator.columns = 4
    accumulator.preferred_staging_dtype = lambda dtype, device: dtype
    accumulator.resolve_hessian_chunk_size = lambda rows, stage: None

    out = accumulator.compute_hessian_xtx(torch.randn(3, 4, dtype=torch.float32))

    assert out.shape == (4, 4)
    assert entered, "the Hessian X^T X GEMM must run on the deterministic rocBLAS path"


def test_chunked_hessian_accumulation_enters_the_override_per_chunk(monkeypatch):
    gptq, entered = _gptq_hessian_recorder(monkeypatch)
    accumulator = gptq.GPTQ.__new__(gptq.GPTQ)
    accumulator.columns = 4
    accumulator.preferred_staging_dtype = lambda dtype, device: dtype
    accumulator.resolve_hessian_chunk_size = lambda rows, stage: 2

    @contextmanager
    def borrow(source, rows):
        yield source

    accumulator.borrow_materialized_chunk_fp32 = borrow

    out = accumulator.compute_hessian_xtx(torch.randn(5, 4, dtype=torch.float32))

    assert out.shape == (4, 4)
    assert len(entered) == 3


@pytest.mark.parametrize(
    "step",
    ["GPTQ", "QQQ", "VPTQ", "MACAQ", "EXL3", "FP8_GSQ", "PARO"],
)
def test_hessian_methods_need_the_deterministic_pin(step):
    """Every Hessian builder and Paro must accumulate on rocBLAS (#4123, #4460)."""

    from types import SimpleNamespace

    from gptqmodel.models.base import _needs_deterministic_hessian
    from gptqmodel.models.quantization_run_plan import ProcessorStep

    assert _needs_deterministic_hessian(
        SimpleNamespace(processors=(getattr(ProcessorStep, step),))
    )


@pytest.mark.parametrize("step", ["AWQ", "WEIGHT_ONLY"])
def test_methods_without_a_hessian_keep_hipblaslt(step):
    """AWQ and RTN keep the faster hipBLASLt kernels."""

    from types import SimpleNamespace

    from gptqmodel.models.base import _needs_deterministic_hessian
    from gptqmodel.models.quantization_run_plan import ProcessorStep

    assert not _needs_deterministic_hessian(
        SimpleNamespace(processors=(getattr(ProcessorStep, step),))
    )


def test_nested_scopes_switch_once_and_restore_once(monkeypatch):
    """Concurrent Hessian workers share one process-wide switch."""

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend():
        with rocm.quant_blas_backend():
            assert fake.current.name == "Cublas"
        assert fake.current.name == "Cublas", "the outer scope still holds rocBLAS"

    assert fake.current.name == "Cublaslt"
    assert fake.calls == ["hipblas", "cublaslt"]


def test_non_hessian_run_pins_hipblaslt(monkeypatch):
    """AWQ/RTN record ownership of hipBLASLt so a Hessian run cannot flip it."""

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend(deterministic=False):
        assert fake.current.name == "Cublaslt"
        assert rocm._QUANT_BLAS_GUARD_TARGET == rocm._QUANT_BLAS_TARGET_HIPBLASLT

    assert fake.current.name == "Cublaslt"
    assert fake.calls == [], "the fast backend needs no setter call"
    assert rocm._QUANT_BLAS_GUARD_DEPTH == 0
    assert rocm._QUANT_BLAS_GUARD_TARGET is None


def test_conflicting_backends_serialize(monkeypatch):
    """A Hessian run and a non-Hessian run must not flip the backend mid-run."""

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    hessian_entered = threading.Event()
    release_hessian = threading.Event()
    seen = []

    def hessian_run():
        with rocm.quant_blas_backend(deterministic=True):
            seen.append(("hessian_start", fake.current.name))
            hessian_entered.set()
            release_hessian.wait(timeout=10)
            seen.append(("hessian_end", fake.current.name))

    def awq_run():
        with rocm.quant_blas_backend(deterministic=False):
            seen.append(("awq", fake.current.name))

    hessian_thread = threading.Thread(target=hessian_run)
    hessian_thread.start()
    assert hessian_entered.wait(timeout=10)
    awq_thread = threading.Thread(target=awq_run)
    awq_thread.start()
    # The AWQ run must wait for the rocBLAS run instead of sharing (and later
    # losing) its backend.
    time.sleep(0.3)
    assert ("awq", "Cublas") not in seen
    assert not any(item[0] == "awq" for item in seen)

    release_hessian.set()
    hessian_thread.join(timeout=10)
    awq_thread.join(timeout=10)

    assert seen == [
        ("hessian_start", "Cublas"),
        ("hessian_end", "Cublas"),
        ("awq", "Cublaslt"),
    ]
    assert fake.calls == ["hipblas", "cublaslt"]
    assert rocm._QUANT_BLAS_GUARD_DEPTH == 0


def test_nested_scope_with_a_different_target_does_not_wait(monkeypatch):
    """A run must not deadlock on its own outer pin."""

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    with rocm.quant_blas_backend(deterministic=True):
        with rocm.quant_blas_backend(deterministic=False):
            assert fake.current.name == "Cublas", "the nested scope adopts the outer pin"
        assert fake.current.name == "Cublas"

    assert fake.current.name == "Cublaslt"
    assert rocm._QUANT_BLAS_GUARD_DEPTH == 0


def test_wait_timeout_adopts_the_active_backend(monkeypatch):
    """A wedged run must not hang a caller forever; the fallback is loud."""

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setattr(rocm, "_QUANT_BLAS_GUARD_WAIT_SECONDS", 0.0)
    configure_kernel(KernelConfig())
    warnings = []
    monkeypatch.setattr(rocm.log, "warning", lambda *args, **kwargs: warnings.append(args))

    hessian_entered = threading.Event()
    release_hessian = threading.Event()
    observed = []

    def hessian_run():
        with rocm.quant_blas_backend(deterministic=True):
            hessian_entered.set()
            release_hessian.wait(timeout=10)

    def awq_run():
        with rocm.quant_blas_backend(deterministic=False):
            observed.append(fake.current.name)

    hessian_thread = threading.Thread(target=hessian_run)
    hessian_thread.start()
    assert hessian_entered.wait(timeout=10)
    awq_thread = threading.Thread(target=awq_run)
    awq_thread.start()
    awq_thread.join(timeout=10)
    release_hessian.set()
    hessian_thread.join(timeout=10)

    assert observed == ["Cublas"], "the late run adopts the active backend"
    assert warnings and "waited" in warnings[0][0]
    assert rocm._QUANT_BLAS_GUARD_DEPTH == 0


def test_paro_selects_the_deterministic_pin_on_mocked_rocm(monkeypatch):
    """Paro's rotation-optimization GEMMs must run on rocBLAS (#4460).

    hipBLASLt chooses GEMM kernels per process, so two identically seeded
    Paro runs produced different checkpoints. ROCm hardware is not needed
    to validate the selection: mock the platform and assert the run plan maps
    to a rocBLAS pin.
    """

    from types import SimpleNamespace

    from gptqmodel.models.base import _needs_deterministic_hessian
    from gptqmodel.models.quantization_run_plan import ProcessorStep

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    run_plan = SimpleNamespace(processors=(ProcessorStep.PARO,))
    assert _needs_deterministic_hessian(run_plan)

    with rocm.quant_blas_backend(deterministic=_needs_deterministic_hessian(run_plan)):
        assert fake.current.name == "Cublas", "Paro must run on rocBLAS"

    assert fake.calls == ["hipblas", "cublaslt"]
    assert fake.current.name == "Cublaslt"
    assert rocm._QUANT_BLAS_GUARD_DEPTH == 0


def test_awq_plus_eora_plan_pins_rocblas_for_the_adapter_gram(monkeypatch):
    """#4614: EoRA's fp32 input Gram must not run on per-process hipBLASLt kernels.

    A GPTQ primary is in the deterministic set, so GPTQ+EoRA was already pinned
    by accident. An AWQ primary is not, so the AWQ + EoRA plan stayed on
    hipBLASLt and produced a different adapter in every process.
    """

    from gptqmodel.adapter.adapter import Lora
    from gptqmodel.models.base import _needs_deterministic_hessian
    from gptqmodel.models.quantization_run_plan import ProcessorStep, QuantizationRunPlan
    from gptqmodel.quantization.config import FORMAT, AWQConfig

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    run_plan = QuantizationRunPlan.build(
        AWQConfig(bits=4, group_size=128, adapter=Lora(rank=8, path="/tmp/lora-4614")),
        format_code=FORMAT.GEMM,
        pack_backend=None,
        loop_backend=None,
        batch_size=1,
        needs_lora=True,
    )
    assert run_plan.processors == (ProcessorStep.AWQ, ProcessorStep.EORA)
    assert _needs_deterministic_hessian(run_plan)

    with rocm.quant_blas_backend(deterministic=_needs_deterministic_hessian(run_plan)):
        assert fake.current.name == "Cublas", "AWQ + EoRA must run on rocBLAS"

    # A plain AWQ run still keeps the faster hipBLASLt kernels.
    assert not _needs_deterministic_hessian(
        QuantizationRunPlan.build(
            AWQConfig(bits=4, group_size=128),
            format_code=FORMAT.GEMM,
            pack_backend=None,
            loop_backend=None,
            batch_size=1,
            needs_lora=False,
        )
    )


def test_eora_gram_runs_on_the_pinned_backend(monkeypatch):
    """The EoRA Gram matmul must execute while the run-level pin is active.

    Recording the backend from inside the real ``eora_process_input`` call
    proves the pin reaches the actual ``X^T X`` GEMM, not just the run-plan
    predicate. Two calls with the same input must also be bitwise identical
    within one process.
    """

    from types import SimpleNamespace

    from gptqmodel.eora import eora as eora_mod
    from gptqmodel.models.base import _needs_deterministic_hessian
    from gptqmodel.models.quantization_run_plan import ProcessorStep

    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    configure_kernel(KernelConfig())

    observed = []
    real_matmul = torch.matmul

    def recording_matmul(left, right, *args, **kwargs):
        observed.append(fake.current.name)
        return real_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(eora_mod.torch, "matmul", recording_matmul)

    run_plan = SimpleNamespace(processors=(ProcessorStep.AWQ, ProcessorStep.EORA))
    inputs = (torch.randn(1, 8, 16, dtype=torch.float32),)

    with rocm.quant_blas_backend(deterministic=_needs_deterministic_hessian(run_plan)):
        _, first, _ = eora_mod.eora_process_input(inputs, "layer.0.q_proj", 1, torch.device("cpu"))
        _, second, _ = eora_mod.eora_process_input(inputs, "layer.0.q_proj", 1, torch.device("cpu"))

    assert observed == ["Cublas", "Cublas"], "the Gram matmul must run on rocBLAS"
    assert torch.equal(first, second)
