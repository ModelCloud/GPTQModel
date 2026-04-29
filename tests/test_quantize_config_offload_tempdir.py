# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import gptqmodel.quantization.config as quant_config_module
from gptqmodel.quantization import EXL3Config, QuantizeConfig


def test_quantize_config_offload_path_defaults_to_tempdir(monkeypatch):
    class FakeTemporaryDirectory:
        def __init__(self, *, prefix):
            assert prefix == "gptqmodel_"
            self.name = "/tmp/gptqmodel_quant_cfg"

        def cleanup(self):
            return None

    monkeypatch.setattr(quant_config_module, "_SharedTemporaryDirectory", FakeTemporaryDirectory)

    cfg = QuantizeConfig(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_quant_cfg"
    assert isinstance(cfg._offload_temp_dir, FakeTemporaryDirectory)


def test_exl3_config_offload_path_defaults_to_tempdir(monkeypatch):
    class FakeTemporaryDirectory:
        def __init__(self, *, prefix):
            assert prefix == "gptqmodel_"
            self.name = "/tmp/gptqmodel_exl3_cfg"

        def cleanup(self):
            return None

    monkeypatch.setattr(quant_config_module, "_SharedTemporaryDirectory", FakeTemporaryDirectory)

    cfg = EXL3Config(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_exl3_cfg"
    assert isinstance(cfg._offload_temp_dir, FakeTemporaryDirectory)
