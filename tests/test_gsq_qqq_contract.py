"""QQQ candidate-grid prerequisites; these fixtures make no quality claim."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization.gsq_qqq import qqq_candidate_values


@pytest.mark.parametrize("ratio", [0.5, 1.5, 17.0])
def test_grouped_candidate_grid_matches_packed_reload(ratio):
    # Every stored nibble, including ties and INT8 saturation, in two groups.
    codes = torch.arange(16).repeat(16).reshape(1, 256).expand(64, -1)
    channel = torch.full((64,), 0.25, dtype=torch.float32)
    scales = torch.full((64, 2), ratio * 0.25, dtype=torch.float16)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    linear.weight.data.copy_((codes - 8) * scales.repeat_interleave(128, dim=1))

    def make_module():
        return QQQTorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                              in_features=256, out_features=64, bias=False)

    packed = make_module()
    packed.pack(linear, scales, channel)
    loaded = make_module()
    loaded.load_state_dict(packed.state_dict(), strict=True)
    assert torch.equal(loaded._unpack_weight_codes(), codes.T)
    integer_weight, stored_channel = loaded._dequantize_weight_for_torch()
    expected_integer = ((codes.T - 8).float() * ratio).round().clamp(-128, 127)
    assert torch.equal(integer_weight, expected_integer)
    assert torch.equal(stored_channel, channel.reshape(1, -1))
    deployed = integer_weight * stored_channel
    affine = (codes.T - 8).float() * (ratio * 0.25)
    assert not torch.equal(deployed, affine)

    decoded = qqq_candidate_values(codes, scales, group_size=128, in_features=256,
                                   channel_scales=channel)
    assert torch.equal(decoded.T, deployed)
    candidates = codes.unsqueeze(-1).expand(-1, -1, 3)
    candidate_values = qqq_candidate_values(candidates, scales, group_size=128, in_features=256,
                                            channel_scales=channel)
    assert torch.equal(candidate_values, decoded.unsqueeze(-1).expand_as(candidate_values))


def test_channelwise_candidate_grid_matches_packed_reload():
    signed = torch.arange(-7, 8).repeat(18)[:256].reshape(1, 256).expand(64, -1)
    scales = torch.full((64, 1), 0.03125, dtype=torch.float16)
    linear = torch.nn.Linear(256, 64, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(signed * scales)
    packed = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    packed.pack(linear, scales)
    loaded = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                            in_features=256, out_features=64, bias=False)
    loaded.load_state_dict(packed.state_dict(), strict=True)
    packed = loaded
    integer_weight, channel = packed._dequantize_weight_for_torch()
    codes = packed._unpack_weight_codes().T
    decoded = qqq_candidate_values(codes, scales, group_size=-1, in_features=256)
    assert torch.equal(decoded.T, integer_weight * channel)
    assert torch.equal(decoded, linear.weight.float())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 16.0, 1.5])
def test_candidate_decoder_rejects_invalid_codes(bad):
    with pytest.raises(ValueError, match="nibbles"):
        qqq_candidate_values(torch.full((1, 128), bad), torch.ones(1, 1),
                             group_size=-1, in_features=128)


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_candidate_decoder_rejects_invalid_scales(group_size, bad):
    groups = 1 if group_size == -1 else 2
    with pytest.raises(ValueError, match="scales"):
        qqq_candidate_values(torch.zeros(1, 256), torch.full((1, groups), bad),
                             group_size=group_size, in_features=256, channel_scales=torch.ones(1))


def test_candidate_mixture_has_gradient_over_decoded_values():
    codes = torch.tensor([8, 9, 10]).expand(1, 256, 3)
    values = qqq_candidate_values(codes, torch.full((1, 2), 0.125),
                                  group_size=128, in_features=256, channel_scales=torch.tensor([0.25]))
    assert torch.equal(values[0, 0], torch.tensor([0.0, 0.0, 0.25]))
    logits = torch.zeros_like(values, requires_grad=True)
    mixture = (logits.softmax(-1) * values).sum(-1)
    mixture.sum().backward()
    expected = (values - values.mean(-1, keepdim=True)) / 3
    torch.testing.assert_close(logits.grad, expected)
