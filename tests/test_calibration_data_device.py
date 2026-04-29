# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Unit tests for the calibration_data_device feature.

Tests the actual implementation code paths with mocked dependencies,
not extracted copies of the logic.

This feature allows specifying where calibration data is stored during quantization:
- None (default): original behavior - CPU initially, then DEVICE_0
- "cpu": store calibration data on CPU
- "cuda:1" or any torch.device: store on specific device
- "balanced": distribute across available compute devices via round-robin
"""

import os
import types
import unittest

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization import QuantizeConfig


# Model path for integration tests - can be overridden via environment variable
_INTEGRATION_MODEL_ID = os.environ.get(
    "GPTQMODEL_TEST_MODEL_ID",
    "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
)


# ============================================================================
# CONFIG PARSING TESTS (CPU-only)
# ============================================================================

class TestCalibrationDataDeviceConfig(unittest.TestCase):
    """Test calibration_data_device configuration parsing in QuantizeConfig.__post_init__."""

    def test_default_is_none(self):
        """Default calibration_data_device should be None."""
        config = QuantizeConfig(bits=4, group_size=128)
        self.assertIsNone(config.calibration_data_device)

    def test_explicit_none(self):
        """Explicitly setting None should work."""
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device=None)
        self.assertIsNone(config.calibration_data_device)

    def test_balanced_string_normalized(self):
        """String 'balanced' should be preserved as lowercase 'balanced'."""
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device="balanced")
        self.assertEqual(config.calibration_data_device, "balanced")

        # Test case normalization
        config_upper = QuantizeConfig(bits=4, group_size=128, calibration_data_device="BALANCED")
        self.assertEqual(config_upper.calibration_data_device, "balanced")

        config_mixed = QuantizeConfig(bits=4, group_size=128, calibration_data_device="Balanced")
        self.assertEqual(config_mixed.calibration_data_device, "balanced")

    def test_cpu_device_string_converted_to_torch_device(self):
        """String 'cpu' should be converted to torch.device('cpu')."""
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device="cpu")
        self.assertIsInstance(config.calibration_data_device, torch.device)
        self.assertEqual(config.calibration_data_device.type, "cpu")

    def test_cuda_device_string_converted_to_torch_device(self):
        """String 'cuda:0' should be converted to torch.device('cuda:0')."""
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device="cuda:0")
        self.assertIsInstance(config.calibration_data_device, torch.device)
        self.assertEqual(config.calibration_data_device.type, "cuda")
        self.assertEqual(config.calibration_data_device.index, 0)

    def test_cuda_indexed_device_string(self):
        """String 'cuda:1' should be converted to torch.device('cuda:1')."""
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device="cuda:1")
        self.assertIsInstance(config.calibration_data_device, torch.device)
        self.assertEqual(config.calibration_data_device.type, "cuda")
        self.assertEqual(config.calibration_data_device.index, 1)

    def test_torch_device_input_preserved(self):
        """torch.device input should be preserved/normalized."""
        device = torch.device("cuda:2")
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device=device)
        self.assertIsInstance(config.calibration_data_device, torch.device)
        self.assertEqual(config.calibration_data_device.type, "cuda")
        self.assertEqual(config.calibration_data_device.index, 2)

    def test_cuda_without_index_normalized_to_index_0(self):
        """torch.device('cuda') without index should be normalized to cuda:0 via _canonical_device."""
        # String input
        config = QuantizeConfig(bits=4, group_size=128, calibration_data_device="cuda")
        self.assertIsInstance(config.calibration_data_device, torch.device)
        self.assertEqual(config.calibration_data_device.type, "cuda")
        self.assertEqual(config.calibration_data_device.index, 0)

        # torch.device input without index
        config2 = QuantizeConfig(bits=4, group_size=128, calibration_data_device=torch.device("cuda"))
        self.assertIsInstance(config2.calibration_data_device, torch.device)
        self.assertEqual(config2.calibration_data_device.type, "cuda")
        self.assertEqual(config2.calibration_data_device.index, 0)


# ============================================================================
# STAGE INPUTS CAPTURE TESTS - Test real cache_inputs method
# ============================================================================

class _DummyLooperForCapture:
    """Minimal fake ModuleLooper for testing StageInputsCapture."""

    def __init__(self, gptq_model):
        self.gptq_model = gptq_model

    def _batch_row_count(self, batch_inputs):
        if not batch_inputs:
            return 0
        primary = batch_inputs[0]
        if isinstance(primary, torch.Tensor) and primary.ndim > 0:
            return max(int(primary.shape[0]), 0)
        return 1


def _create_mock_logger():
    """Create a mock logger for StageInputsCapture tests."""
    class MockLogger:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def pb(self, *args, **kwargs):
            class MockPB:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return MockPB()

    return MockLogger()


def _create_mock_thread_pool():
    """Create a mock DEVICE_THREAD_POOL for tests."""
    class MockThreadPool:
        @staticmethod
        def read_lock(device):
            class MockLock:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return MockLock()

    return MockThreadPool()


def test_stage_capture_cpu_device_stores_inputs_on_cpu(monkeypatch):
    """
    Test StageInputsCapture: with calibration_data_device='cpu',
    inputs are captured on CPU.

    Tests: stage_inputs_capture.py lines 85-108, 132-165
    """
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture

    # Create a minimal model where calling model(input_ids=...) triggers the hooked layer
    class MinimalModel(nn.Module):
        def __init__(self, hooked_layer):
            super().__init__()
            self.hooked_layer = hooked_layer
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # Trigger the hooked layer - this will raise StopForward
            self.hooked_layer(input_ids.float())
            return None

    # Create the layer that will be hooked
    hooked_layer = nn.Linear(4, 4)
    hooked_layer.full_name = "test_layer"

    model = MinimalModel(hooked_layer)

    # Create fake GPT-QModel with calibration_data_device="cpu"
    class FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del kwargs
            return target_submodule

        def get_base_modules(self, model):
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            pass

    gptq_model = FakeGPTQModel()
    looper = _DummyLooperForCapture(gptq_model)

    # Mock external dependencies
    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.DEVICE_THREAD_POOL",
        _create_mock_thread_pool(),
    )

    # Mock ctx to be a proper context manager that combines multiple context managers
    def mock_ctx(*context_managers):
        class CombinedContextManager:
            def __enter__(self):
                for cm in context_managers:
                    cm.__enter__()
                return self

            def __exit__(self, *args):
                for cm in reversed(context_managers):
                    cm.__exit__(*args)
                return False
        return CombinedContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.ctx",
        mock_ctx,
    )

    # Mock device_ctx to be a proper context manager
    def mock_device_ctx(device):
        class DeviceContextManager:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        return DeviceContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.device_ctx",
        mock_device_ctx,
    )

    capture = StageInputsCapture(looper, logger=_create_mock_logger())

    # Create calibration data
    calibration_data = [
        {"input_ids": torch.randint(0, 100, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    ]

    # Run cache_inputs - this will trigger store_input_hook which captures inputs
    result = capture.cache_inputs(
        layers=[hooked_layer],
        calibration_data=calibration_data,
        use_cache=False,
    )

    # Verify inputs were captured on CPU
    assert len(result.layer_inputs) == 1, "Should have captured one batch"
    assert len(result.layer_inputs[0]) == 1, "Should have one input tensor"
    assert result.layer_inputs[0][0].device.type == "cpu", \
        f"Input should be on CPU, got {result.layer_inputs[0][0].device}"


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices"
)
def test_stage_capture_balanced_mode_applies_compute_device_filter(monkeypatch):
    """
    Test StageInputsCapture: in balanced mode, compute_device_filter is applied
    to determine which devices are used for round-robin distribution.

    Tests: stage_inputs_capture.py lines 91-104
    """
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture

    # Simulate 4 devices, cuda:0 will be filtered out
    mock_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
        torch.device("cuda:3"),
    ]

    def fake_select_forward_devices(_base_device):
        return mock_devices

    def compute_device_filter(devices):
        # Filter out cuda:0 (e.g., reserved for model layers)
        return [d for d in devices if d != torch.device("cuda:0")]

    # Patch at source module since import happens inside the function
    monkeypatch.setattr(
        "gptqmodel.utils.looper_helpers.select_forward_devices",
        fake_select_forward_devices,
    )

    # Create a minimal model
    class MinimalModel(nn.Module):
        def __init__(self, hooked_layer):
            super().__init__()
            self.hooked_layer = hooked_layer
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            self.hooked_layer(input_ids.float())
            return None

    hooked_layer = nn.Linear(4, 4)
    hooked_layer.full_name = "test_layer"
    model = MinimalModel(hooked_layer)

    class FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cuda:0"),
                calibration_data_device="balanced",
                compute_device_filter=compute_device_filter,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del kwargs
            return target_submodule

        def get_base_modules(self, model):
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            pass

    gptq_model = FakeGPTQModel()
    looper = _DummyLooperForCapture(gptq_model)

    # Track which devices are actually used
    used_devices = []
    __import__('gptqmodel.utils.model', fromlist=['move_to']).move_to

    def tracking_move_to(obj, device, **kwargs):
        if isinstance(obj, torch.Tensor):
            used_devices.append(device)
        # For testing, just clone the tensor (don't actually move to GPU)
        return obj.detach().clone() if device.type == "cpu" else obj

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.DEVICE_THREAD_POOL",
        _create_mock_thread_pool(),
    )

    # Mock ctx to be a proper context manager that combines multiple context managers
    def mock_ctx(*context_managers):
        class CombinedContextManager:
            def __enter__(self):
                for cm in context_managers:
                    cm.__enter__()
                return self

            def __exit__(self, *args):
                for cm in reversed(context_managers):
                    cm.__exit__(*args)
                return False
        return CombinedContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.ctx",
        mock_ctx,
    )

    # Mock device_ctx to be a proper context manager
    def mock_device_ctx(device):
        class DeviceContextManager:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        return DeviceContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.device_ctx",
        mock_device_ctx,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.move_to",
        tracking_move_to,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.nested_move_to",
        lambda obj, device, **kw: obj,
    )

    capture = StageInputsCapture(looper, logger=_create_mock_logger())

    # Create multiple calibration batches to test round-robin
    calibration_data = [
        {"input_ids": torch.randint(0, 100, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        for _ in range(3)
    ]

    result = capture.cache_inputs(
        layers=[hooked_layer],
        calibration_data=calibration_data,
        use_cache=False,
    )

    # Verify cuda:0 is NOT used (it was filtered out by compute_device_filter)
    assert torch.device("cuda:0") not in used_devices, \
        f"cuda:0 should be filtered out, got devices: {used_devices}"

    # Verify we captured the expected number of batches
    assert len(result.layer_inputs) == 3, f"Should have captured 3 batches, got {len(result.layer_inputs)}"


def test_stage_capture_balanced_mode_empty_filter_fallback(monkeypatch):
    """
    Test StageInputsCapture: if compute_device_filter returns empty list,
    it falls back to using all devices from select_forward_devices.

    Tests: stage_inputs_capture.py lines 99-101
    """
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture

    mock_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]

    def fake_select_forward_devices(_base_device):
        return mock_devices

    def compute_device_filter_returns_empty(devices):
        return []  # Invalid filter returning empty

    # Patch at source module since import happens inside the function
    monkeypatch.setattr(
        "gptqmodel.utils.looper_helpers.select_forward_devices",
        fake_select_forward_devices,
    )

    class MinimalModel(nn.Module):
        def __init__(self, hooked_layer):
            super().__init__()
            self.hooked_layer = hooked_layer
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            self.hooked_layer(input_ids.float())
            return None

    hooked_layer = nn.Linear(4, 4)
    hooked_layer.full_name = "test_layer"
    model = MinimalModel(hooked_layer)

    class FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cuda:0"),
                calibration_data_device="balanced",
                compute_device_filter=compute_device_filter_returns_empty,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del kwargs
            return target_submodule

        def get_base_modules(self, model):
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            pass

    gptq_model = FakeGPTQModel()
    looper = _DummyLooperForCapture(gptq_model)

    used_devices = []

    def tracking_move_to(obj, device, **kwargs):
        if isinstance(obj, torch.Tensor):
            used_devices.append(device)
        return obj.detach().clone() if device.type == "cpu" else obj

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.DEVICE_THREAD_POOL",
        _create_mock_thread_pool(),
    )

    # Mock ctx to be a proper context manager that combines multiple context managers
    def mock_ctx(*context_managers):
        class CombinedContextManager:
            def __enter__(self):
                for cm in context_managers:
                    cm.__enter__()
                return self

            def __exit__(self, *args):
                for cm in reversed(context_managers):
                    cm.__exit__(*args)
                return False
        return CombinedContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.ctx",
        mock_ctx,
    )

    # Mock device_ctx to be a proper context manager
    def mock_device_ctx(device):
        class DeviceContextManager:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        return DeviceContextManager()

    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.device_ctx",
        mock_device_ctx,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.move_to",
        tracking_move_to,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.stage_inputs_capture.nested_move_to",
        lambda obj, device, **kw: obj,
    )

    capture = StageInputsCapture(looper, logger=_create_mock_logger())

    calibration_data = [
        {"input_ids": torch.randint(0, 100, (1, 4)), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    ]

    result = capture.cache_inputs(
        layers=[hooked_layer],
        calibration_data=calibration_data,
        use_cache=False,
    )

    # Should still work despite empty filter (fallback to all_devices)
    assert len(result.layer_inputs) == 1, "Should have captured one batch"


# ============================================================================
# MODULE LOOPER INIT TESTS - Device assignment via _quant_devices
# ============================================================================

class _DummyGPTQModelForLooper:
    """Minimal fake GPT-QModel for testing ModuleLooper."""

    capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
    capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
    prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs

    def __init__(
        self,
        calibration_data_device=None,
        compute_device_filter=None,
        quant_device="cuda:0",
        auto_forward_data_parallel=True,
        dense_vram_strategy="exclusive",
        dense_vram_strategy_devices=None,
        moe_vram_strategy="exclusive",
        moe_vram_strategy_devices=None,
    ):
        self.quantize_config = types.SimpleNamespace(
            device=torch.device(quant_device),
            calibration_data_device=calibration_data_device,
            compute_device_filter=compute_device_filter,
            auto_forward_data_parallel=auto_forward_data_parallel,
            dense_vram_strategy=dense_vram_strategy,
            dense_vram_strategy_devices=dense_vram_strategy_devices,
            moe_vram_strategy=moe_vram_strategy,
            moe_vram_strategy_devices=moe_vram_strategy_devices,
            moe_routing_bypass=lambda: False,
            moe=None,
            gc_mode=None,
        )
        self.support_batch_quantize = False
        self.layer_callback = None
        self.lm_head = None
        self.quant_region_timer = None
        self.moe_lifecycle_hooks = None
        # Don't set supported dense/MoE strategy lists so getattr uses default values


def test_compute_device_filter_applied_to_quant_devices(monkeypatch):
    """
    Test that compute_device_filter is applied to _quant_devices in ModuleLooper.__init__.

    This test verifies the interaction between compute_device_filter and calibration_data_device
    in the ModuleLooper initialization.
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    all_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]

    # Filter excludes cuda:0 (e.g., reserved for model layers)
    def compute_device_filter(devices):
        return [d for d in devices if d != torch.device("cuda:0")]

    def fake_select_forward_devices(_base_device):
        return all_devices

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device="balanced",
        compute_device_filter=compute_device_filter,
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    # Verify cuda:0 is filtered out from quant_devices
    assert torch.device("cuda:0") not in looper._quant_devices
    assert torch.device("cuda:1") in looper._quant_devices
    assert torch.device("cuda:2") in looper._quant_devices


def test_compute_device_filter_with_empty_result_uses_all_devices(monkeypatch):
    """
    Test that if compute_device_filter returns empty list, all devices are used.

    Tests: The fallback behavior when filter returns nothing.
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    all_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]

    def compute_device_filter_returns_empty(devices):
        return []  # Invalid filter returning empty

    def fake_select_forward_devices(_base_device):
        return all_devices

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device="balanced",
        compute_device_filter=compute_device_filter_returns_empty,
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    # Should still have devices (fallback behavior in __init__)
    assert len(looper._quant_devices) > 0


# ============================================================================
# FORWARD BATCH TESTS - Balanced mode batch-to-device assignment
# ============================================================================


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices"
)
def test_balanced_mode_assigns_batches_to_input_devices(monkeypatch):
    """
    Test that in balanced mode, batches are assigned to the device
    where their input already resides.

    Tests: module_looper.py _run_forward_batches_parallel balanced mode logic
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    all_devices = [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]

    def fake_select_forward_devices(_base_device):
        return all_devices

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device="balanced",
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    # Track which device each batch was executed on
    batch_to_device = {}

    def fake_clone_module_for_devices(_module, target_devices, progress_callback=None):
        return {device: object() for device in target_devices}

    class DummyProcessor:
        num_batches = 4

        def _set_current_batch_index(self, _index):
            pass

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    def fake_forward_batch_worker(*args, **kwargs):
        batch_idx = args[2]
        return batch_idx, None, None

    def fake_submit(device, fn, *args, **kwargs):
        batch_idx = args[2]
        batch_to_device[batch_idx] = device
        return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.clone_module_for_devices",
        fake_clone_module_for_devices,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.forward_batch_worker",
        fake_forward_batch_worker,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit",
        fake_submit,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit_serial",
        fake_submit,
    )

    module = torch.nn.Linear(1, 1)
    processor = DummyProcessor()

    # Batches 0,1 on cuda:0, batches 2,3 on cuda:1
    layer_inputs = [
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
        [torch.zeros(1, 1, device=torch.device("cuda:1"))],
        [torch.zeros(1, 1, device=torch.device("cuda:1"))],
    ]

    looper._run_forward_batches_parallel(
        module=module,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}, {}, {}, {}],
        position_ids=[],
        attention_masks=[torch.zeros(1, 1)] * 4,
        cur_layer_device=torch.device("cuda:0"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=False,
        reuse_kv=False,
        devices=all_devices,
    )

    # Verify batches were assigned to their input device
    assert batch_to_device.get(0) == torch.device("cuda:0"), "Batch 0 should run on cuda:0"
    assert batch_to_device.get(1) == torch.device("cuda:0"), "Batch 1 should run on cuda:0"
    assert batch_to_device.get(2) == torch.device("cuda:1"), "Batch 2 should run on cuda:1"
    assert batch_to_device.get(3) == torch.device("cuda:1"), "Batch 3 should run on cuda:1"


def test_balanced_mode_fallback_when_input_device_not_in_forward_devices(monkeypatch):
    """
    Test that in balanced mode, if a batch's input device is not in
    forward_devices, it falls back to round-robin assignment.

    Tests: module_looper.py _run_forward_batches_parallel fallback logic
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    # Only cuda:1 and cuda:2 available for forward
    forward_devices = [
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]

    def fake_select_forward_devices(_base_device):
        return forward_devices

    def compute_device_filter(devices):
        return forward_devices

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device="balanced",
        compute_device_filter=compute_device_filter,
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    batch_to_device = {}

    def fake_clone_module_for_devices(_module, target_devices, progress_callback=None):
        return {device: object() for device in target_devices}

    class DummyProcessor:
        num_batches = 4

        def _set_current_batch_index(self, _index):
            pass

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    def fake_forward_batch_worker(*args, **kwargs):
        batch_idx = args[2]
        return batch_idx, None, None

    def fake_submit(device, fn, *args, **kwargs):
        batch_idx = args[2]
        batch_to_device[batch_idx] = device
        return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.clone_module_for_devices",
        fake_clone_module_for_devices,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.forward_batch_worker",
        fake_forward_batch_worker,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit",
        fake_submit,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit_serial",
        fake_submit,
    )

    module = torch.nn.Linear(1, 1)
    processor = DummyProcessor()

    # All inputs on cuda:0 which is NOT in forward_devices
    layer_inputs = [
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
        [torch.zeros(1, 1, device=torch.device("cuda:0"))],
    ]

    looper._run_forward_batches_parallel(
        module=module,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}, {}, {}, {}],
        position_ids=[],
        attention_masks=[torch.zeros(1, 1)] * 4,
        cur_layer_device=torch.device("cuda:1"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=False,
        reuse_kv=False,
        devices=forward_devices,
    )

    # All batches should be assigned to either cuda:1 or cuda:2 (fallback)
    # cuda:0 should never be used as it's not in forward_devices
    for batch_idx in range(4):
        assert batch_to_device[batch_idx] in forward_devices, \
            f"Batch {batch_idx} should be on forward device, got {batch_to_device[batch_idx]}"
        assert batch_to_device[batch_idx] != torch.device("cuda:0"), \
            f"Batch {batch_idx} should not be on cuda:0"


# ============================================================================
# OUTPUT DEVICE PRESERVATION TESTS
# ============================================================================

def test_output_moved_to_input_device_in_single_mode(monkeypatch):
    """
    Test that in _run_forward_batches_single, outputs are moved
    back to the same device as the input.

    Tests: module_looper.py _run_forward_batches_single output device preservation
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    def fake_select_forward_devices(_base_device):
        return [torch.device("cuda:0")]

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device=torch.device("cpu"),
        auto_forward_data_parallel=False,  # Force single mode
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    class DummyProcessor:
        num_batches = 1

        def _set_current_batch_index(self, _index):
            pass

    processor = DummyProcessor()

    # Input is on CPU (calibration data device)
    layer_inputs = [[torch.zeros(1, 1, device=torch.device("cpu"))]]

    # Use is_lm_head_module=True to avoid passing attention_mask/use_cache kwargs
    # that plain nn.Linear doesn't accept
    module = torch.nn.Linear(1, 1)

    outputs = looper._run_forward_batches_single(
        module=module,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=True,  # Avoid kwargs that nn.Linear doesn't accept
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
    )

    # Output should be on CPU (same as input)
    assert len(outputs) == 1, "Should have one output"
    assert outputs[0][0].device == torch.device("cpu"), \
        f"Output should be on CPU (input device), got {outputs[0][0].device}"


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices"
)
def test_output_moved_to_input_device_in_parallel_mode(monkeypatch):
    """
    Test that in _run_forward_batches_parallel, outputs are moved
    back to the same device as each batch's input.

    Tests: module_looper.py _run_forward_batches_parallel output device preservation
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    all_devices = [torch.device("cuda:0"), torch.device("cuda:1")]

    def fake_select_forward_devices(_base_device):
        return all_devices

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device="balanced",
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    def fake_clone_module_for_devices(_module, target_devices, progress_callback=None):
        return {device: object() for device in target_devices}

    class DummyProcessor:
        num_batches = 2

        def _set_current_batch_index(self, _index):
            pass

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    # Create outputs that return tensors on cuda:0 (compute device)
    # These will be moved back to input device by the code
    output_tensors = [
        torch.ones(1, 4, device=torch.device("cuda:0")),
        torch.ones(1, 4, device=torch.device("cuda:0")),
    ]

    def fake_forward_batch_worker(*args, **kwargs):
        batch_idx = args[2]
        need_output = kwargs.get("need_output", False)
        if need_output:
            return batch_idx, (output_tensors[batch_idx],), None
        return batch_idx, None, None

    def fake_submit(device, fn, *args, **kwargs):
        return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.clone_module_for_devices",
        fake_clone_module_for_devices,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.forward_batch_worker",
        fake_forward_batch_worker,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit",
        fake_submit,
    )
    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.DEVICE_THREAD_POOL.submit_serial",
        fake_submit,
    )

    module = torch.nn.Linear(4, 4)
    processor = DummyProcessor()

    # Inputs on different devices
    layer_inputs = [
        [torch.zeros(1, 4, device=torch.device("cuda:0"))],
        [torch.zeros(1, 4, device=torch.device("cuda:1"))],
    ]

    outputs = looper._run_forward_batches_parallel(
        module=module,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}, {}],
        position_ids=[],
        attention_masks=[torch.zeros(1, 1), torch.zeros(1, 1)],
        cur_layer_device=torch.device("cuda:0"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        devices=all_devices,
    )

    # Verify outputs are on same device as their inputs
    assert len(outputs) == 2, "Should have two outputs"
    assert outputs[0][0].device == torch.device("cuda:0"), \
        f"Batch 0 output should be on cuda:0, got {outputs[0][0].device}"
    assert outputs[1][0].device == torch.device("cuda:1"), \
        f"Batch 1 output should be on cuda:1, got {outputs[1][0].device}"


def test_auto_forward_data_parallel_false_uses_single_mode(monkeypatch):
    """
    Test that when auto_forward_data_parallel=False,
    _run_forward_batches uses the single (serial) path.

    Tests: module_looper.py _run_forward_batches mode selection
    """
    from gptqmodel.looper.module_looper import ModuleLooper

    def fake_select_forward_devices(_base_device):
        return [torch.device("cuda:0"), torch.device("cuda:1")]

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.select_forward_devices",
        fake_select_forward_devices,
    )

    gptq_model = _DummyGPTQModelForLooper(
        calibration_data_device=torch.device("cpu"),
        auto_forward_data_parallel=False,
    )

    looper = ModuleLooper(model=gptq_model, processors=[])

    single_called = [False]

    def patched_single(**kwargs):
        single_called[0] = True
        return []

    # Patch the forward_executor's run_single method (not looper's non-existent _run_forward_batches_single)
    looper._forward_executor.run_single = patched_single

    class DummyProcessor:
        num_batches = 1

        def _set_current_batch_index(self, _index):
            pass

    processor = DummyProcessor()
    module = torch.nn.Linear(1, 1)
    layer_inputs = [[torch.zeros(1, 1)]]

    looper._run_forward_batches(
        module=module,
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=False,
        reuse_kv=False,
    )

    assert single_called[0], "Should have called _run_forward_batches_single"


# ============================================================================
# INTEGRATION TESTS (Marked for GPU execution)
# ============================================================================


def _skip_if_model_missing(model_id: str):
    """Skip test if model path doesn't exist, matching test_failsafe.py pattern."""
    if not os.path.isdir(model_id):
        pytest.skip(f"Model path missing: {model_id}")


@pytest.mark.cuda
@pytest.mark.integration
class TestCalibrationDataDeviceIntegration:
    """
    Integration tests for calibration_data_device with actual quantization.
    These tests require GPU and load real models.
    Run with: pytest -m "cuda and integration" tests/test_calibration_data_device.py
    """

    NATIVE_MODEL_ID = _INTEGRATION_MODEL_ID
    DATASET_SIZE = 4

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        import gc

        # Skip if model is missing (matching test_failsafe.py pattern)
        _skip_if_model_missing(self.NATIVE_MODEL_ID)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Use longer text samples to pass calibration_data_min_length check (default=10 tokens)
        # These samples are long enough to not be filtered out
        self.calibration_data = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes in typography and keyboard layouts.",
            "Machine learning is transforming technology by enabling computers to learn from data and make predictions or decisions without being explicitly programmed for each specific task.",
            "Natural language processing enables computers to understand text and human language, allowing for applications like chatbots, translation services, and sentiment analysis tools.",
            "Quantization reduces model size while maintaining accuracy by converting floating-point numbers to lower precision representations, making models faster and more memory efficient.",
        ]

        yield

        # Teardown: aggressive cleanup to prevent VRAM leak between tests
        self.calibration_data = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _cleanup_model(self, model):
        """Immediately delete model and free GPU memory."""
        if model is not None:
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _create_quantize_config(self, calibration_data_device, **kwargs):
        """Create QuantizeConfig with memory-efficient defaults for integration tests."""
        defaults = {
            "bits": 4,
            "group_size": 128,
            "calibration_data_device": calibration_data_device,
            "offload_to_disk": False,
            # Memory-efficient settings for multi-test runs
            "auto_forward_data_parallel": False,
            "wait_for_submodule_finalizers": True,
            "gc_mode": "on_stage_end",
        }
        defaults.update(kwargs)
        return QuantizeConfig(**defaults)

    def test_calibration_data_device_cpu_integration(self, tmp_path):
        """Integration test: quantization with calibration_data_device='cpu'."""
        from gptqmodel import GPTQModel

        quantize_config = self._create_quantize_config(calibration_data_device="cpu")

        model = GPTQModel.load(self.NATIVE_MODEL_ID, quantize_config=quantize_config)
        model.quantize(self.calibration_data, batch_size=1)

        save_path = str(tmp_path / "quantized")
        model.save(save_path)
        self._cleanup_model(model)

        loaded = GPTQModel.load(save_path)
        assert loaded is not None

        inp = self.tokenizer("Hello", return_tensors="pt").to(loaded.device)
        output = loaded.generate(**inp, max_new_tokens=5)
        assert output is not None
        self._cleanup_model(loaded)

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires at least 2 CUDA devices"
    )
    def test_calibration_data_device_balanced_integration(self, tmp_path):
        """Integration test: quantization with calibration_data_device='balanced'."""
        from gptqmodel import GPTQModel

        quantize_config = self._create_quantize_config(calibration_data_device="balanced")

        model = GPTQModel.load(self.NATIVE_MODEL_ID, quantize_config=quantize_config)
        model.quantize(self.calibration_data, batch_size=1)

        save_path = str(tmp_path / "quantized_balanced")
        model.save(save_path)
        self._cleanup_model(model)

        loaded = GPTQModel.load(save_path)
        assert loaded is not None
        self._cleanup_model(loaded)

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires at least 2 CUDA devices"
    )
    def test_calibration_data_device_cuda1_with_compute_filter_integration(self, tmp_path):
        """Integration test: quantization with calibration_data_device='cuda:1' and compute_device_filter."""
        from gptqmodel import GPTQModel

        # Filter that excludes cuda:1 from compute devices
        def compute_device_filter(devices):
            return [d for d in devices if d != torch.device("cuda:1")]

        quantize_config = self._create_quantize_config(
            calibration_data_device="cuda:1",
            compute_device_filter=compute_device_filter,
        )

        model = GPTQModel.load(self.NATIVE_MODEL_ID, quantize_config=quantize_config)
        model.quantize(self.calibration_data, batch_size=1)

        save_path = str(tmp_path / "quantized_cuda1")
        model.save(save_path)
        self._cleanup_model(model)

        loaded = GPTQModel.load(save_path)
        assert loaded is not None
        self._cleanup_model(loaded)
