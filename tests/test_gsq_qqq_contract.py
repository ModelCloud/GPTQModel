"""QQQ candidate-grid prerequisites; these fixtures make no quality claim."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear


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
