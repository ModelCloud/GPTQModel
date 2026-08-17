# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization import QVQConfig


def test_qvq_mps_config_forces_serial_calibration_forwarding():
    mps_config = QVQConfig(bits=2, device="mps", auto_forward_data_parallel=True, offload_to_disk=False)
    cpu_config = QVQConfig(bits=2, device="cpu", auto_forward_data_parallel=True, offload_to_disk=False)

    assert mps_config.auto_forward_data_parallel is False
    assert cpu_config.auto_forward_data_parallel is True
    assert torch.device(mps_config.device).type == "mps"
