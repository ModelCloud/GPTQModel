# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pin the GGUF_CPP CPU activation-quantization envelope and its exact selector (#4316).

`ggml_mul_mat` quantizes activations to Q8_0/Q8_1/Q8_K before the integer dot
product. For Gaussian activations the CPU error stays within the documented
``GGML_NATIVE_MATMUL_ERROR_BOUND`` (2e-2), but one large channel in a 256-element
Q8_K block widens the step for every value in that block, so outlier channels
exceed it. Before this change the CPU path was always native and the only escape
was an environment variable.

The tests below use the issue repro (a synthetic Q4_K_M tensor) and assert:

* Gaussian activations stay within the documented envelope;
* outlier channels exceed it on the native path (the bug is real);
* ``GGUFCppKernel(exact_matmul=True)`` selects the exact path and matches
  ``GGUFTorchLinear`` bitwise;
* the default is unchanged for callers that pass nothing.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    GGML_NATIVE_MATMUL_ERROR_BOUND,
    GGML_NATIVE_MATMUL_OUTLIER_ERROR_ENVELOPE,
    GGUFCppKernel,
)


IN_FEATURES = 4096
OUT_FEATURES = 128
ROWS = 64
OUTLIER_CHANNELS = 16


def _require_cpu_bridge() -> None:
    try:
        ok, error = GGUFCppKernel.cached_validate_once()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"llama-cpp-python GGML bridge unavailable: {exc}")
    if not ok:
        pytest.skip(f"GGML CPU bridge unavailable: {error}")


def _build_pair(*, exact_matmul=None):
    """Build a native kernel and an exact GGUFTorchLinear on the same Q4_K_M bytes."""

    generator = torch.Generator().manual_seed(0)
    cpp = GGUFCppKernel(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        exact_matmul=exact_matmul,
    )
    exact = GGUFTorchLinear(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
    )
    raw = torch.randint(0, 256, cpp.qweight.shape, generator=generator, dtype=torch.uint8)
    # Q4_K block: FP16 d and dmin followed by 12 bytes of scales/codes. Keep the
    # two scales small so the synthetic weights are finite and well-conditioned.
    raw.view(OUT_FEATURES, -1, 144)[..., 0:4] = torch.tensor([1e-3, 1e-3], dtype=torch.float16).view(torch.uint8)
    for module in (cpp, exact):
        module.qweight.copy_(raw)
        module.eval()
    return cpp, exact, generator


def _activations(generator):
    x = torch.randn(ROWS, IN_FEATURES, generator=generator)
    xo = x.clone()
    xo[:, torch.randperm(IN_FEATURES, generator=generator)[:OUTLIER_CHANNELS]] *= 100.0
    return x, xo


def _relative_errors(output, reference):
    diff = (output.double() - reference.double()).abs()
    relative = (diff.max() / reference.double().abs().max()).item()
    worst_row = (diff.amax(1) / reference.double().abs().amax(1)).max().item()
    return relative, worst_row


def _run(module, x):
    with torch.inference_mode():
        return module(x.clone())


def test_gaussian_activations_stay_within_the_documented_envelope(monkeypatch):
    _require_cpu_bridge()
    monkeypatch.delenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", raising=False)
    cpp, exact, generator = _build_pair()
    x, _ = _activations(generator)

    relative, worst_row = _relative_errors(_run(cpp, x), _run(exact, x))

    assert relative <= GGML_NATIVE_MATMUL_ERROR_BOUND, (
        f"Gaussian relative error {relative:.4f} exceeds the documented {GGML_NATIVE_MATMUL_ERROR_BOUND:.0e}"
    )
    assert worst_row <= GGML_NATIVE_MATMUL_ERROR_BOUND, (
        f"Gaussian worst-row error {worst_row:.4f} exceeds the documented {GGML_NATIVE_MATMUL_ERROR_BOUND:.0e}"
    )


def test_outlier_channels_exceed_the_documented_envelope_on_the_native_path(monkeypatch):
    """The issue repro: 16/4096 channels x100 breaks the 2e-2 claim."""

    _require_cpu_bridge()
    monkeypatch.delenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", raising=False)
    cpp, exact, generator = _build_pair()
    _, xo = _activations(generator)

    relative, worst_row = _relative_errors(_run(cpp, xo), _run(exact, xo))

    assert relative > GGML_NATIVE_MATMUL_ERROR_BOUND, (
        f"outlier relative error {relative:.4f} did not reproduce the >2e-2 bug"
    )
    assert worst_row > GGML_NATIVE_MATMUL_ERROR_BOUND, (
        f"outlier worst-row error {worst_row:.4f} did not reproduce the >2e-2 bug"
    )
    # Pin the observed magnitude without presenting it as a universal bound.
    assert worst_row <= GGML_NATIVE_MATMUL_OUTLIER_ERROR_ENVELOPE, (
        f"outlier worst-row error {worst_row:.4f} exceeds the recorded envelope "
        f"{GGML_NATIVE_MATMUL_OUTLIER_ERROR_ENVELOPE:.0e}"
    )


def test_exact_matmul_selector_takes_the_exact_path(monkeypatch):
    """`exact_matmul=True` must be the GGUF_TORCH path, not the native kernel."""

    _require_cpu_bridge()
    monkeypatch.delenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", raising=False)
    cpp, exact, generator = _build_pair(exact_matmul=True)
    x, xo = _activations(generator)

    assert cpp.exact_matmul is True
    assert cpp._use_ggml_cpu_matmul(x) is False
    assert torch.equal(_run(cpp, x), _run(exact, x))
    assert torch.equal(_run(cpp, xo), _run(exact, xo))

    # An explicit constructor argument wins over the process-wide override.
    monkeypatch.setenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", "1")
    assert cpp._use_ggml_cpu_matmul(x) is False


def test_default_path_is_unchanged(monkeypatch):
    """Callers that pass nothing keep the capability-based native choice."""

    _require_cpu_bridge()
    monkeypatch.delenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", raising=False)
    cpp, exact, generator = _build_pair()
    x, xo = _activations(generator)

    assert cpp.exact_matmul is None
    assert cpp._use_ggml_cpu_matmul(x) is True
    # The default really is the lossy path: it must not equal the exact result
    # on the outlier input.
    assert not torch.equal(_run(cpp, xo), _run(exact, xo))

    # `exact_matmul=False` is an explicit request for the native path, and the
    # environment variable still overrides the default in both directions.
    native = GGUFCppKernel(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        exact_matmul=False,
    )
    assert native._use_ggml_cpu_matmul(x) is True
    monkeypatch.setenv("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL", "0")
    assert cpp._use_ggml_cpu_matmul(x) is False
