# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gptqmodel import QuantizeConfig, TelemetryConfig
from gptqmodel.utils.device_telemetry import (
    clear_device_telemetry_records,
    device_telemetry_enabled,
    device_telemetry_scope,
    emit_device_telemetry,
    get_device_telemetry_records,
    with_quantization_device_telemetry,
)
from gptqmodel.utils.threads import AsyncManager, SerialWorker
from gptqmodel.utils.threadx import DeviceThreadPool


def test_nested_config_roundtrip():
    assert QuantizeConfig().telemetry == TelemetryConfig(device=False)
    for value in [TelemetryConfig(device=True), {"device": True}]:
        config = QuantizeConfig(telemetry=value)
        saved = config.to_dict()
        assert saved["meta"]["telemetry"] == {"device": True}
        assert QuantizeConfig.from_quant_config(saved).telemetry == config.telemetry


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_invalid_nested_device_flag(invalid):
    with pytest.raises(ValueError, match="must be a boolean"):
        QuantizeConfig(telemetry={"device": invalid})


def test_old_env_has_no_effect(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_DEVICE_TELEMETRY", "1")
    with device_telemetry_scope(False):
        assert not device_telemetry_enabled()


def test_quantization_scope_resets_after_exception():
    model = SimpleNamespace(
        quantize_config=QuantizeConfig(telemetry=TelemetryConfig(device=True))
    )

    @with_quantization_device_telemetry
    def fail(model):
        assert device_telemetry_enabled()
        raise RuntimeError("injected")

    assert not device_telemetry_enabled()
    with pytest.raises(RuntimeError, match="injected"):
        fail(model)
    assert not device_telemetry_enabled()


def test_concurrent_runs_and_reused_device_workers_are_isolated():
    pool = DeviceThreadPool(devices=["cpu"], empty_cache_every_n=0)
    barrier = threading.Barrier(2)
    clear_device_telemetry_records()

    @with_quantization_device_telemetry
    def quantize(model):
        enabled = model.quantize_config.telemetry.device
        barrier.wait(timeout=10)

        def probe():
            emit_device_telemetry("scope_probe", enabled=enabled)
            return device_telemetry_enabled()

        assert pool.submit("cpu", probe).result(timeout=10) == enabled
        assert pool.submit_serial("cpu", probe).result(timeout=10) == enabled

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [
                executor.submit(
                    quantize,
                    SimpleNamespace(
                        quantize_config=QuantizeConfig(
                            telemetry=TelemetryConfig(device=enabled)
                        )
                    ),
                )
                for enabled in [True, False]
            ]
            for job in jobs:
                job.result(timeout=20)
        assert pool.submit("cpu", device_telemetry_enabled).result(timeout=10) is False
        records = get_device_telemetry_records()
        assert len(records) == 2 and all(record["enabled"] for record in records)
    finally:
        pool.shutdown(wait=True)
        clear_device_telemetry_records()


def test_async_and_serial_workers_capture_and_reset_scope():
    manager = AsyncManager(threads=1)
    serial = SerialWorker()
    observed = []
    try:
        for enabled in [True, False, True, False]:
            with device_telemetry_scope(enabled):
                assert (
                    manager.submit(device_telemetry_enabled).result(timeout=10)
                    == enabled
                )
                serial.submit(lambda: observed.append(device_telemetry_enabled()))
        serial.join()
        assert observed == [True, False, True, False]
    finally:
        manager.shutdown()
        serial.shutdown()
