# SPDX-License-Identifier: Apache-2.0
"""Shared scheduler continuation for calibration and weight-only execution."""

import torch


class DeviceAssignmentState:
    def execution_state_dict(self) -> dict:
        with self._quant_device_lock:
            return {
                "version": 1,
                "next_device": self._quant_device_rr,
                "module_devices": {name: str(device) for name, device in self._module_device_map.items()},
            }

    def load_execution_state_dict(self, state: dict) -> None:
        if (
            not isinstance(state, dict) or state.get("version") != 1
            or type(state.get("next_device")) is not int or state["next_device"] < 0
        ):
            raise ValueError("invalid execution scheduler continuation")
        devices = state.get("module_devices")
        allowed = {str(device) for device in self._quant_devices} | {"cpu"}
        if not isinstance(devices, dict) or any(
            not isinstance(name, str) or not isinstance(device, str) or device not in allowed
            for name, device in devices.items()
        ):
            raise ValueError("execution continuation contains an unavailable device")
        restored = {name: torch.device(device) for name, device in devices.items()}
        with self._quant_device_lock:
            self._quant_device_rr = state["next_device"]
            self._module_device_map = restored
