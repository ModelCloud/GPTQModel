# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""SmoothMSE must weight its shrink search by weight saliency (#5286).

``mse_optimal_quant`` used to minimize the uniform mean squared weight error.
Clipping a row's largest weight is cheap under that objective: one element's
error buys a 20% smaller step for the other 127, so the search accepted the
clipped endpoint. The layer output is dominated by those large weights, so
end-to-end PPL regressed even though the per-row weight MSE fell.

Calibration-free consumers now weight each squared error by
``|w| + mean(|w|)``; GPTQ/QQQ fallbacks use the captured Hessian diagonal,
which is the exact input-channel activation energy. These tests pin the
consumer-level outcome, not only the helper's return value.
"""

import pytest
import torch

import gptqmodel.quantization.gptq as gptq_module
import gptqmodel.quantization.qqq as qqq_module
import gptqmodel.quantization.rtn as rtn_module
from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    GPTQConfig,
    QQQConfig,
    RTNConfig,
    SmoothMSE,
)
from gptqmodel.quantization.fallback_smooth import mse_optimal_quant, smooth_applies
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.rtn import RTN


MAXQ = 15
GROUP_SIZE = 128
DEVICES = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _rtn_dequant(block: torch.Tensor, smooth, *, sym: bool = True) -> torch.Tensor:
    source = torch.nn.Linear(block.shape[1], block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)
    config = RTNConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=sym,
        smooth=smooth,
        offload_to_disk=False,
    )
    dequant, *_ = RTN(source, config).quantize()
    return dequant


def _unshrunk_dequant(block: torch.Tensor, *, sym: bool = True) -> torch.Tensor:
    """Independent float64 ``p = 1`` grid, the plain RTN baseline."""
    ref = block.double()
    if sym:
        hi = ref.abs().amax(dim=1, keepdim=True)
        lo = -hi
        zero = torch.full_like(hi, (MAXQ + 1) / 2.0)
    else:
        zero = torch.zeros_like(ref[:, :1])
        lo = torch.minimum(ref.amin(dim=1, keepdim=True), zero)
        hi = torch.maximum(ref.amax(dim=1, keepdim=True), zero)
        zero = None
    scale = ((hi - lo) / MAXQ).clamp_min(1e-8)
    z = zero if sym else torch.round(-lo / scale)
    codes = (ref / scale + z).round().clamp(0, MAXQ)
    return (codes - z) * scale


def _saliency(block: torch.Tensor) -> torch.Tensor:
    magnitude = block.double().abs()
    return magnitude + magnitude.mean(dim=1, keepdim=True)


def _weighted_error(dequant: torch.Tensor, block: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error = (dequant.double() - block.double()).pow(2)
    return (error * weights).sum(dim=1) / weights.sum(dim=1)


def _output_error(dequant: torch.Tensor, block: torch.Tensor, activations: torch.Tensor) -> float:
    delta = dequant.double() - block.double()
    return (activations.double() @ delta.T).pow(2).sum().item()


def _heavy_tail_block(rows: int = 8) -> torch.Tensor:
    """Gaussian rows with one dominating weight, the shape that broke PPL."""
    torch.manual_seed(5286)
    block = torch.randn(rows, GROUP_SIZE) * 0.5
    block[:, 7] = 12.0
    return block


@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
def test_rtn_smooth_mse_does_not_clip_a_heavy_tail_maximum(sym):
    block = _heavy_tail_block()
    activations = torch.ones(1, GROUP_SIZE)
    activations[0, 7] = 20.0

    plain = _rtn_dequant(block, None, sym=sym)
    smoothed = _rtn_dequant(block, SmoothMSE(), sym=sym)

    # The unshrunk grid is the plain RTN control. Clipping the 12.0 outlier
    # raises the output error by more than 2x even though the weight MSE falls.
    assert _output_error(smoothed, block, activations) <= _output_error(
        plain, block, activations
    ) + 1e-6, "SmoothMSE clipped the heavy-tail maximum and raised the layer output error"

    # The largest weight must stay on the unshrunk grid (within rounding).
    plain_max = (plain[:, 7] - block[:, 7]).abs().max()
    smoothed_max = (smoothed[:, 7] - block[:, 7]).abs().max()
    assert smoothed_max <= plain_max + 1e-6

    # Contract: no worse than the unshrunk baseline under the consumer's
    # saliency-weighted objective, computed independently in float64.
    weights = _saliency(block)
    baseline = _unshrunk_dequant(block, sym=sym)
    searched_error = _weighted_error(smoothed, block, weights)
    baseline_error = _weighted_error(baseline, block, weights)
    assert bool((searched_error <= baseline_error * (1.0 + 1e-6) + 1e-9).all())


@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
def test_rtn_smooth_mse_still_clips_a_gaussian_row(sym):
    """Control: the saliency objective must not disable the smoother."""
    torch.manual_seed(0)
    block = torch.randn(16, GROUP_SIZE)

    plain = _rtn_dequant(block, None, sym=sym)
    smoothed = _rtn_dequant(block, SmoothMSE(), sym=sym)

    assert (smoothed - block).pow(2).mean() < (plain - block).pow(2).mean() - 1e-9


@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
def test_weighted_search_is_no_worse_than_the_unshrunk_baseline(sym):
    from gptqmodel.quantization.fallback_smooth import element_saliency_weights

    block = _heavy_tail_block()
    weights = element_saliency_weights(block)
    qcfg = RTNConfig(bits=4, group_size=GROUP_SIZE, sym=sym, smooth=SmoothMSE())

    searched, _scale, _zero = mse_optimal_quant(
        block, qcfg, MAXQ, steps=32, maxshrink=0.8, weights=weights
    )
    baseline = _unshrunk_dequant(block, sym=sym)
    weights = _saliency(block)

    searched_error = _weighted_error(searched, block, weights)
    baseline_error = _weighted_error(baseline, block, weights)
    assert bool((searched_error <= baseline_error * (1.0 + 1e-6) + 1e-9).all())


def test_gptq_fallback_uses_the_hessian_diagonal():
    """The fallback must protect the channel the calibration Hessian marks."""
    block = _heavy_tail_block()
    source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)

    config = GPTQConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothMSE()),
        offload_to_disk=False,
    )
    task = GPTQ(source, config)
    task.quantizer.configure(perchannel=True)

    # The outlier column carries all of the captured input energy.
    hessian = torch.zeros(GROUP_SIZE, GROUP_SIZE)
    hessian[7, 7] = 1.0
    task.H = hessian
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN, blocksize=GROUP_SIZE)

    baseline = _unshrunk_dequant(block)
    weights = hessian.diagonal().double().reshape(1, -1).expand_as(block)
    searched_error = _weighted_error(dequant, block, weights)
    baseline_error = _weighted_error(baseline, block, weights)
    assert bool((searched_error <= baseline_error * (1.0 + 1e-6) + 1e-9).all())
    assert (dequant[:, 7] - block[:, 7]).abs().max() <= (baseline[:, 7] - block[:, 7]).abs().max() + 1e-6


def test_gptq_fallback_follows_the_hessian_not_weight_magnitude():
    """A channel the Hessian marks can justify a step the magnitude proxy refuses.

    Column 0 is a 20.0 outlier and column 1 is a 1.3 weight. The magnitude
    proxy keeps ``p = 1`` to protect the outlier; a Hessian that only observes
    column 1 prefers the smaller step that quantizes 1.3 better. On the old
    uniform objective both cases kept ``p = 1`` and this assertion fails.
    """
    torch.manual_seed(0)
    block = torch.randn(8, GROUP_SIZE) * 0.2
    block[:, 0] = 20.0
    block[:, 1] = 1.3

    source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)
    config = GPTQConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothMSE()),
        offload_to_disk=False,
    )
    task = GPTQ(source, config)
    task.quantizer.configure(perchannel=True)

    hessian = torch.zeros(GROUP_SIZE, GROUP_SIZE)
    hessian[1, 1] = 1.0
    task.H = hessian
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN, blocksize=GROUP_SIZE)

    weights = hessian.diagonal().double().reshape(1, -1).expand_as(block)
    baseline = _unshrunk_dequant(block)
    assert _weighted_error(dequant, block, weights).mean() < _weighted_error(
        baseline, block, weights
    ).mean() - 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_gptq_fallback_uses_batches_added_after_finalize_hessian(monkeypatch, device):
    """A finalized Hessian must not hide batches captured after it (#5365).

    ``finalize_hessian()`` materializes ``self.H`` from the samples seen so
    far. Later ``add_batch()`` calls only update the partials. The fallback
    must weight its search by the diagonal reduced from the live capture, not
    by the older ``self.H`` snapshot.
    """

    torch.manual_seed(0)
    block = torch.randn(8, GROUP_SIZE) * 0.2
    block[:, 0] = 20.0
    block[:, 1] = 1.3

    first = torch.zeros(1, GROUP_SIZE, device=device)
    first[0, 0] = 1.0
    second = torch.zeros(1, GROUP_SIZE, device=device)
    second[0, 1] = 100.0

    def make_task():
        source = torch.nn.Linear(GROUP_SIZE, block.shape[0], bias=False).to(device)
        with torch.no_grad():
            source.weight.copy_(block)
        config = GPTQConfig(
            bits=4,
            group_size=GROUP_SIZE,
            sym=True,
            fallback=Fallback(
                threshold=100, strategy=FallbackStrategy.RTN, smooth=SmoothMSE()
            ),
            offload_to_disk=False,
        )
        task = GPTQ(source, config)
        task.quantizer.configure(perchannel=True)
        task.expected_nsamples = 2
        return task

    control = make_task()
    control.add_batch(first, None)
    control.add_batch(second, None)

    early = make_task()
    early.add_batch(first, None)
    early.finalize_hessian()
    early.add_batch(second, None)

    # The trigger: H holds only the first batch and the second is pending.
    assert control.H is None
    assert early.H is not None
    assert early._hessian_dirty
    assert control.nsamples == early.nsamples == 2

    # Independently sum the squared activations of the two explicit rows.
    expected = torch.zeros(GROUP_SIZE, device=device)
    expected[0] = 1.0
    expected[1] = 100.0**2

    captured = _spy_mse_weights(monkeypatch, gptq_module)

    control_quantized, *_ = control.quantize(blocksize=GROUP_SIZE)
    assert len(captured) == 1
    control_weights = captured.pop()

    early_quantized, *_ = early.quantize(blocksize=GROUP_SIZE)
    assert len(captured) == 1
    early_weights = captured.pop()

    _assert_weights_follow_hessian(control_weights, expected)
    _assert_weights_follow_hessian(early_weights, expected)
    torch.testing.assert_close(early_quantized, control_quantized, rtol=0, atol=0)


def _captured_inputs(rows: int = 32) -> torch.Tensor:
    """Calibration rows with one channel carrying most of the energy."""

    torch.manual_seed(5300)
    inputs = torch.randn(rows, GROUP_SIZE)
    inputs[:, 3] *= 25.0
    return inputs


def _spy_mse_weights(monkeypatch, module) -> list:
    captured = []
    original = module.mse_optimal_quant

    def spy(block, qcfg, maxq, **kwargs):
        captured.append(kwargs.get("weights"))
        return original(block, qcfg, maxq, **kwargs)

    monkeypatch.setattr(module, "mse_optimal_quant", spy)
    return captured


def _expected_diagonal(task) -> torch.Tensor:
    """The reduction ``_captured_hessian_diagonal`` performs, from the partials."""

    total = task.nsamples
    diagonal = sum(
        partial.diagonal() for partial in task._device_hessian_partials.values()
    )
    return (diagonal * (2.0 / float(total))).float()


def _assert_weights_follow_hessian(weights, expected):
    assert weights is not None
    assert weights.numel() == expected.numel()
    torch.testing.assert_close(
        weights / weights.sum(),
        expected / expected.sum(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gptq_quantize_fallback_uses_the_captured_hessian_diagonal(monkeypatch):
    """The real ``quantize()`` path must not drop H before the fallback (#5300).

    Singleton modules (``o_proj``/``down_proj``) have no shared-input H, so
    the fallback used to clear the partials and silently switch to the
    weight-magnitude proxy. Capture what ``mse_optimal_quant`` receives.
    """

    inputs = _captured_inputs()
    source = torch.nn.Linear(GROUP_SIZE, 8, bias=False)
    config = GPTQConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        fallback=Fallback(
            threshold=10**9, strategy=FallbackStrategy.RTN, smooth=SmoothMSE()
        ),
        offload_to_disk=False,
    )
    task = GPTQ(source, config)
    task.quantizer.configure(perchannel=True)
    task.expected_nsamples = inputs.shape[0]
    for batch in inputs.split(8):
        task.add_batch(batch, None)
    assert task.H is None

    expected = _expected_diagonal(task)
    captured = _spy_mse_weights(monkeypatch, gptq_module)
    task.quantize(blocksize=GROUP_SIZE)

    assert captured, "the fallback never reached mse_optimal_quant"
    _assert_weights_follow_hessian(captured[0], expected)


def test_qqq_quantize_fallback_uses_the_captured_hessian_diagonal(monkeypatch):
    """QQQ released H before the threshold fallback, so SmoothMSE saw no H."""

    inputs = _captured_inputs()
    source = torch.nn.Linear(GROUP_SIZE, 8, bias=False)
    config = QQQConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        fallback=Fallback(threshold=10**9, smooth=SmoothMSE()),
        offload_to_disk=False,
    )
    task = QQQ(source, config)
    task.quantizer.configure(
        bits=config.bits, perchannel=True, sym=config.sym, groupsize=config.group_size
    )
    task.expected_nsamples = inputs.shape[0]
    for batch in inputs.split(8):
        task.add_batch(batch, None)
    assert task.H is None

    expected = _expected_diagonal(task)
    captured = _spy_mse_weights(monkeypatch, qqq_module)
    task.quantize(blocksize=GROUP_SIZE)

    assert captured, "the fallback never reached mse_optimal_quant"
    _assert_weights_follow_hessian(captured[0], expected)


def _rtn_grid(block, smooth, *, group_size=GROUP_SIZE):
    source = torch.nn.Linear(block.shape[1], block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)
    config = RTNConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        smooth=smooth,
        offload_to_disk=False,
    )
    dequant, *_ = RTN(source, config).quantize()
    return dequant


def _gptq_fallback_grid(block, smooth, *, group_size=GROUP_SIZE):
    source = torch.nn.Linear(block.shape[1], block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)
    config = GPTQConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=smooth),
        offload_to_disk=False,
    )
    task = GPTQ(source, config)
    task.quantizer.configure(perchannel=True)
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN, blocksize=GROUP_SIZE)
    return dequant


def _qqq_fallback_grid(block, smooth, *, group_size=GROUP_SIZE):
    source = torch.nn.Linear(block.shape[1], block.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(block)
    config = QQQConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        fallback=Fallback(strategy=FallbackStrategy.RTN, smooth=smooth),
        offload_to_disk=False,
    )
    task = QQQ(source, config)
    task.quantizer.configure(
        bits=config.bits, perchannel=True, sym=config.sym, groupsize=config.group_size
    )
    dequant, *_ = task._fallback_quantize(FallbackStrategy.RTN)
    return dequant


GRID_BUILDERS = [
    pytest.param(_rtn_grid, id="rtn"),
    pytest.param(_gptq_fallback_grid, id="gptq"),
    pytest.param(_qqq_fallback_grid, id="qqq"),
]


@pytest.mark.parametrize("build_grid", GRID_BUILDERS)
def test_smooth_mse_group_size_threshold_gates_the_search(build_grid):
    """A disabled SmoothMSE must not change the grid (#5366).

    The RTN loop and the GPTQ/QQQ fallback loops entered ``mse_optimal_quant``
    whenever the smoother was a ``SmoothMSE``, so ``group_size_threshold`` only
    gated the non-MSE branch. An above-threshold smoother must produce exactly
    the no-smoother grid, while the equal-threshold control must still search.
    """

    torch.manual_seed(5366)
    block = torch.randn(16, GROUP_SIZE)

    plain = build_grid(block, None)
    disabled = build_grid(block, SmoothMSE(group_size_threshold=GROUP_SIZE * 2))
    enabled = build_grid(block, SmoothMSE(group_size_threshold=GROUP_SIZE))

    assert torch.equal(disabled, plain), (
        "SmoothMSE ran below group_size_threshold and changed the quantized weights"
    )
    assert not torch.equal(enabled, plain), (
        "the equal-threshold control never reached the MSE search"
    )


@pytest.mark.parametrize("build_grid", GRID_BUILDERS)
def test_smooth_mse_threshold_uses_the_resolved_whole_row_size(build_grid):
    """``group_size=-1`` resolves to the row width before the threshold gate."""

    torch.manual_seed(5367)
    block = torch.randn(16, GROUP_SIZE)

    plain = build_grid(block, None, group_size=-1)
    disabled = build_grid(
        block, SmoothMSE(group_size_threshold=GROUP_SIZE + 1), group_size=-1
    )
    enabled = build_grid(
        block, SmoothMSE(group_size_threshold=GROUP_SIZE), group_size=-1
    )

    assert torch.equal(disabled, plain), (
        "a whole-row group must still honour group_size_threshold"
    )
    assert not torch.equal(enabled, plain), (
        "a whole-row group at the threshold must still reach the MSE search"
    )


def test_default_smooth_mse_is_not_gated_by_the_clipping_threshold():
    """``SmoothMSE()`` must not inherit the clipping smoothers' g128 default (#5530).

    The search runs per group, so it has no reason to skip small groups. The
    shared ``128`` default belongs to the clipping smoothers; inheriting it made
    ``SmoothMSE()`` a silent no-op at the common g32/g64 sizes while
    ``quantize_config.json`` still recorded ``smooth: mse``.
    """

    assert SmoothMSE().group_size_threshold == 0
    assert smooth_applies(SmoothMSE(), 32)
    assert smooth_applies(SmoothMSE(), 64)
    # An explicit threshold keeps its documented meaning.
    assert not smooth_applies(SmoothMSE(group_size_threshold=128), 64)
    assert smooth_applies(SmoothMSE(group_size_threshold=128), 128)


def test_default_smooth_mse_reaches_the_rtn_search_at_group_size_64(monkeypatch):
    """The RTN loop must reach ``mse_optimal_quant`` at g64 without a threshold (#5530)."""

    captured = _spy_mse_weights(monkeypatch, rtn_module)
    torch.manual_seed(5530)
    block = torch.randn(16, 64)

    _rtn_grid(block, SmoothMSE(), group_size=64)
    assert captured, "the default SmoothMSE search was skipped at group_size=64"

    captured.clear()
    _rtn_grid(block, SmoothMSE(group_size_threshold=128), group_size=64)
    assert not captured, "an explicit above-group threshold must still skip the search"


def test_default_smooth_mse_reaches_the_gptq_fallback_at_group_size_64(monkeypatch):
    """The GPTQ/QQQ fallback loops must search at g64 without a threshold (#5530)."""

    captured = _spy_mse_weights(monkeypatch, gptq_module)
    torch.manual_seed(5531)
    block = torch.randn(16, 64)

    _gptq_fallback_grid(block, SmoothMSE(), group_size=64)
    assert captured, "the default SmoothMSE search was skipped at group_size=64"


SMOOTH_CONFIG_FORMS = [SmoothMSE(), "mse", {"type": "mse"}]


@pytest.mark.parametrize("smooth", SMOOTH_CONFIG_FORMS, ids=["object", "string", "dict"])
def test_default_smooth_mse_threshold_agrees_across_config_forms(smooth):
    """Every way of asking for the MSE smoother must resolve threshold 0 (#5572).

    #5541 only changed the ``SmoothMSE.__init__`` default. The dict/string
    builder reads the dataclass field default, which stayed at the clipping
    smoothers' 128, so saved JSON/YAML and CLI configs silently skipped the
    search at g32/g64.
    """

    assert smooth_applies(smooth, 32)
    assert smooth_applies(smooth, 64)

    config = RTNConfig(bits=4, group_size=64, smooth=smooth)
    assert config.smooth.group_size_threshold == 0
    assert smooth_applies(config.smooth, 32)
    assert smooth_applies(config.smooth, 64)

    fallback = Fallback(strategy=FallbackStrategy.RTN, smooth=smooth)
    assert fallback.smooth.group_size_threshold == 0


@pytest.mark.parametrize("smooth", SMOOTH_CONFIG_FORMS, ids=["object", "string", "dict"])
def test_default_smooth_mse_from_every_config_form_reaches_the_rtn_search_at_g64(monkeypatch, smooth):
    """The parsed forms must actually run the RTN search at g64 (#5572)."""

    captured = _spy_mse_weights(monkeypatch, rtn_module)
    torch.manual_seed(5572)
    block = torch.randn(16, 64)

    _rtn_grid(block, smooth, group_size=64)
    assert captured, "the default SmoothMSE search was skipped at group_size=64"
