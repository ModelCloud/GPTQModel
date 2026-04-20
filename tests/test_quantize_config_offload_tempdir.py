# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import shutil

import gptqmodel.quantization.config as quant_config_module
from gptqmodel.quantization import EXL3Config, QuantizeConfig


def test_quantize_config_offload_path_defaults_to_tempdir(monkeypatch):
    registered = []

    def fake_mkdtemp(prefix):
        assert prefix == "gptqmodel_"
        return "/tmp/gptqmodel_quant_cfg"

    def fake_register(func, *args, **kwargs):
        registered.append((func, args, kwargs))

    monkeypatch.setattr(quant_config_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(quant_config_module.atexit, "register", fake_register)

    cfg = QuantizeConfig(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_quant_cfg"
    assert registered == [
        (shutil.rmtree, ("/tmp/gptqmodel_quant_cfg",), {"ignore_errors": True}),
    ]


def test_exl3_config_offload_path_defaults_to_tempdir(monkeypatch):
    registered = []

    def fake_mkdtemp(prefix):
        assert prefix == "gptqmodel_"
        return "/tmp/gptqmodel_exl3_cfg"

    def fake_register(func, *args, **kwargs):
        registered.append((func, args, kwargs))

    monkeypatch.setattr(quant_config_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(quant_config_module.atexit, "register", fake_register)

    cfg = EXL3Config(offload_to_disk=True)

    assert cfg.offload_to_disk_path == "/tmp/gptqmodel_exl3_cfg"
    assert registered == [
        (shutil.rmtree, ("/tmp/gptqmodel_exl3_cfg",), {"ignore_errors": True}),
    ]
