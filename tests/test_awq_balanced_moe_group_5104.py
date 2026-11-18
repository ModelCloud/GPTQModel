# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scale groups must survive ``VramStrategy.BALANCED`` MoE placement (#5104).

``build_subset_plan`` distributes the routed experts of one norm-anchored group
round-robin over the MoE device pool, so a single scale group's members live on
every device of the pool. AWQ's scale search and fold assumed one device:

* ``_accumulate_awq_weight_mean`` allocated ``w_sum`` on ``layers[0]`` and added
  every other member's channel sum into it without moving it;
* ``_compute_best_scale`` applied the shared channel scale with
  ``torch.mul(..., out=fc.weight)`` and ``fc.weight.div_(scales_view)``;
* ``apply_scale`` moved every member of a group onto ``layers[0]``'s device
  before folding, which copies a whole expert family onto one device.

The first is a hard ``RuntimeError`` before any weight is touched. The other two
crash on the cross-device scale, and co-locating defeats the memory the
balanced pool exists to save. These tests pin all three on two real devices.
"""

import types

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _compute_awq_weight_mean,
)
from gptqmodel.quantization.awq.quantize.scale import apply_scale, scale_ln_fcs
from gptqmodel.quantization.config import FORMAT, AWQConfig


pytestmark = [pytest.mark.gpu]


_NUM_CHANNELS = 8


@pytest.fixture(scope="module")
def two_devices():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices to place one scale group across devices")
    return torch.device("cuda:0"), torch.device("cuda:1")


def _linear(in_features, out_features, device):
    layer = nn.Linear(in_features, out_features, bias=False).to(device)
    layer.weight.requires_grad_(False)
    return layer


def _processor() -> AWQProcessor:
    return AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(bits=4, group_size=4, format=FORMAT.GEMM),
        calibration=None,
        gptq_model=types.SimpleNamespace(rotary_embedding=None),
        model=None,
        calculate_w_wq_diff=False,
    )


def _per_member_weight_mean(layers, group_size):
    """Independent host oracle: normalised magnitude mean over every row."""

    total = None
    rows = 0
    for layer in layers:
        weight = layer.weight.detach().to("cpu", torch.float32)
        flat = weight.reshape(-1, group_size)
        normalised = flat.abs() / (flat.abs().amax(dim=1, keepdim=True) + 1e-6)
        channel_sum = normalised.reshape(weight.shape).sum(dim=0)
        total = channel_sum if total is None else total + channel_sum
        rows += weight.shape[0]
    return total / rows


def test_weight_mean_accepts_members_on_two_devices(two_devices):
    """The accumulator gathers each member's sum onto its own device."""

    first, second = two_devices
    torch.manual_seed(0)
    layers = [
        _linear(16, 4, first),
        _linear(16, 4, second),
        _linear(16, 4, first),
    ]

    # The defect: `w_sum += <member sum>.to(...)` was missing, so adding the
    # second device's reduction raised before any scale could be searched.
    with pytest.raises(RuntimeError):
        broken = torch.zeros(16, dtype=torch.float32, device=layers[0].weight.device)
        broken += layers[1].weight.detach().abs().sum(dim=0, dtype=torch.float32)

    mean = _compute_awq_weight_mean(layers, 8)
    expected = _per_member_weight_mean(layers, 8)

    assert mean.device == first
    torch.testing.assert_close(mean.to("cpu").float(), expected, rtol=1e-6, atol=1e-7)


def test_scale_ln_fcs_applies_the_scale_on_each_consumer_device(two_devices):
    """Both consumers receive the same channel scale, each on its own device."""

    first, second = two_devices
    torch.manual_seed(0)
    norm = nn.LayerNorm(_NUM_CHANNELS).to(first)
    near = _linear(_NUM_CHANNELS, 4, first)
    far = _linear(_NUM_CHANNELS, 4, second)
    scales = torch.linspace(0.5, 2.0, _NUM_CHANNELS, device=first)

    near_before = near.weight.detach().clone()
    far_before = far.weight.detach().clone()

    with pytest.raises(RuntimeError):
        # The defect: the scale stayed on `layers[0]`'s device.
        far.weight.mul_(scales.view(1, -1))

    effective = scale_ln_fcs(norm, [near, far], scales)

    # Closed form on both devices; the reduction is the one `scale_ln_fcs`
    # returns, not a value read back from the fold.
    for layer, original in ((near, near_before), (far, far_before)):
        expected = original * effective.to(original.device).view(1, -1)
        torch.testing.assert_close(layer.weight, expected, rtol=1e-6, atol=1e-7)
    assert near.weight.device == first
    assert far.weight.device == second


class _SplitMoeLayer(nn.Module):
    """A ``post_attention_layernorm``-anchored group split over two devices."""

    def __init__(self, first, second):
        super().__init__()
        torch.manual_seed(0)
        self.post_attention_layernorm = nn.LayerNorm(_NUM_CHANNELS).to(first)
        self.experts = nn.ModuleList(
            [_linear(_NUM_CHANNELS, 4, first), _linear(_NUM_CHANNELS, 4, second)]
        )


def _apply_split_group(layer):
    scales = torch.linspace(0.5, 2.0, _NUM_CHANNELS, device=layer.experts[0].weight.device)
    before = [expert.weight.detach().clone() for expert in layer.experts]
    apply_scale(
        layer,
        [("post_attention_layernorm", ["experts.0", "experts.1"], scales, None)],
    )
    return before, scales


def test_apply_scale_does_not_co_locate_a_norm_anchored_group(two_devices, monkeypatch):
    """A balanced MoE fold keeps every expert on its planned device.

    Co-locating a norm-anchored group copies the whole expert family onto one
    device, which is exactly the memory the balanced pool exists to save and can
    exhaust that device. The fold must instead move only the norm and apply the
    shared channel scale on each consumer's own device.
    """

    import gptqmodel.quantization.awq.quantize.scale as scale_module

    first, second = two_devices
    layer = _SplitMoeLayer(first, second)
    moved = []
    real_move_to = scale_module.move_to

    def recording_move_to(obj, device, dtype=None):
        moved.append(obj)
        return real_move_to(obj, device, dtype=dtype)

    monkeypatch.setattr(scale_module, "move_to", recording_move_to)
    before, scales = _apply_split_group(layer)

    # The defect moved every consumer onto `layers[0]`'s device before folding.
    experts = list(layer.experts)
    assert not any(obj is expert for obj in moved for expert in experts), (
        "AWQ co-located a norm-anchored MoE group; a balanced pool must fold "
        "each consumer on its planned device"
    )

    assert layer.experts[0].weight.device == first
    assert layer.experts[1].weight.device == second

    # The norm owns a plain gain here, so the folded scale is exactly the
    # searched one and each expert's rows are multiplied by it.
    for expert, original in zip(layer.experts, before):
        expected = original * scales.to(original.device).view(1, -1)
        torch.testing.assert_close(expert.weight, expected, rtol=1e-6, atol=1e-7)


def test_balanced_moe_scale_search_scores_a_group_split_over_devices(two_devices):
    """The 20-candidate grid runs with members on both devices and restores them."""

    first, second = two_devices
    torch.manual_seed(0)
    near = _linear(16, 8, first)
    far = _linear(16, 8, second)
    far.weight.copy_(near.weight.detach())

    class Inspect(nn.Module):
        """Runs each member on its own device, like the routed expert loop."""

        def __init__(self):
            super().__init__()
            self.near = near
            self.far = far

        def forward(self, x):
            out = self.near(x)
            moved = self.far(x.to(second))
            return out + moved.to(out.device)

    module = Inspect()
    x = torch.randn(3, 16, device=first)
    with torch.no_grad():
        fp16_output = module(x).detach().clone()

    originals = [near.weight.detach().clone(), far.weight.detach().clone()]
    processor = _processor()
    processor._quant_batch_size = 0
    processor.max_chunk_memory = 1 << 20
    processor.duo_scaling = False
    processor._scale_context = types.SimpleNamespace(prev_scale=None)
    # Identity quantization keeps the objective comparable without a packer;
    # the cross-device multiply/divide under test is unchanged.
    processor._pseudo_quantize_tensor_into = lambda src, dst, **_kwargs: dst.copy_(src)

    # The defect: `torch.mul(master, scales_view, out=far.weight)` and
    # `far.weight.div_(scales_view)` raised with the scale on `cuda:0`.
    with pytest.raises(RuntimeError):
        broken = torch.zeros(1, 16, device=first)
        torch.mul(originals[1], broken, out=far.weight)

    scales, loss = processor._compute_best_scale(
        x,
        torch.linspace(0.2, 1.4, 16, device=first),
        torch.linspace(1.6, 0.3, 16, device=first),
        module,
        [near, far],
        fp16_output,
        {},
        replay_batch_size=0,
        prepared_forward=(x, {}),
        bits=[4, 4],
        group_sizes=[8, 8],
        syms=[True, True],
    )

    assert scales.shape == (16,)
    assert torch.isfinite(scales).all()
    # Identity quantization: every candidate reproduces the reference, so the
    # reconstruction error is only the s / s round trip.
    assert loss < 1e-8
    # Both devices still hold exactly the pre-search values.
    for layer, original in zip((near, far), originals):
        assert torch.equal(layer.weight, original)
    assert near.weight.device == first
    assert far.weight.device == second
