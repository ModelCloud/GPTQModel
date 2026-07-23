# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.gptq_processor import snapshot_eora_reconstructed_weight


def test_eora_reconstructed_weight_snapshot_survives_dense_rematerialization():
    dense = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=torch.bfloat16,
    )
    reconstructed = torch.tensor(
        [[0.75, -1.5, 2.5], [3.5, 4.5, -5.5]],
        dtype=torch.bfloat16,
    )
    module = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    module.weight.data = reconstructed

    saved = snapshot_eora_reconstructed_weight(module.weight.data)
    expected = reconstructed.clone()

    assert saved.device.type == "cpu"
    assert saved.data_ptr() != module.weight.data.data_ptr()

    # Lazy checkpoint materialization copies the dense tensor into the existing
    # parameter storage. The module and the original reconstructed tensor are
    # overwritten, but EoRA's saved reconstruction must remain unchanged.
    module.weight.data.copy_(dense)

    torch.testing.assert_close(module.weight.data, dense, rtol=0, atol=0)
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)
