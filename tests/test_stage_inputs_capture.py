# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture


class FakeLayer(nn.Module):
    """Minimal decoder layer stand-in for module-path tests."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class FakeModel(nn.Module):
    """Tiny model tree for testing fallback path resolution."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])


class TestStageInputsCapture(unittest.TestCase):
    """Micro-tests for the layer label / module_path resolution in `StageInputsCapture`."""

    def _make_capture(self, layer, layer_names=None, gptq_model_model=None):
        layer_list = [layer]
        layer_names_list = layer_names

        quantize_config = MagicMock()
        quantize_config.device = torch.device("cpu")
        quantize_config.calibration_data_device = None
        quantize_config.offload_to_disk = False
        quantize_config.compute_device_filter = None

        gptq_model = MagicMock()
        gptq_model.model = gptq_model_model or FakeModel()
        gptq_model.quantize_config = quantize_config
        gptq_model.get_modules_with_direct_meta_tensors.return_value = []
        gptq_model.get_base_modules.return_value = []
        gptq_model.get_input_embeddings.return_value = None
        gptq_model.get_input_embeddings_name.return_value = None
        gptq_model.shell_module_materialize.return_value = layer
        gptq_model.shell_direct_meta_materialize.return_value = None
        gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
        gptq_model.quant_region_timer = None

        looper = MagicMock()
        looper.gptq_model = gptq_model

        capture = StageInputsCapture(looper)
        return capture, layer_list, layer_names_list, gptq_model

    def test_cache_inputs_uses_caller_supplied_layer_name(self):
        """`layer_names[0]` should win and become the materialization module_path."""

        layer = FakeLayer()
        capture, layers, layer_names, gptq_model = self._make_capture(
            layer, layer_names=["model.layers.42"]
        )

        capture.cache_inputs(
            layers=layers,
            calibration_data=[],
            use_cache=False,
            layer_names=layer_names,
        )

        gptq_model.shell_module_materialize.assert_called_once()
        call_kwargs = gptq_model.shell_module_materialize.call_args[1]
        self.assertEqual(call_kwargs["module_path"], "model.layers.42")

    def test_cache_inputs_falls_back_to_named_modules_resolution(self):
        """Without `layer_names` and `full_name`, scan `named_modules()` for dotted path."""

        model = FakeModel()
        layer = model.layers[0]
        capture, layers, _, gptq_model = self._make_capture(layer, gptq_model_model=model)

        capture.cache_inputs(
            layers=layers,
            calibration_data=[],
            use_cache=False,
        )

        call_kwargs = gptq_model.shell_module_materialize.call_args[1]
        self.assertEqual(call_kwargs["module_path"], "layers.0")

    def test_cache_inputs_falls_back_to_class_name_for_orphan_layer(self):
        """If the layer cannot be found in the model tree, the display label falls back to its class name,
        but the materialization module_path is left unset so the underlying resolver can either find it or fail loudly."""

        layer = FakeLayer()
        capture, layers, _, gptq_model = self._make_capture(layer, gptq_model_model=FakeModel())
        # Detach the layer so it is not the same object as the one in the model tree.
        self.assertIsNot(layer, gptq_model.model.layers[0])

        capture.cache_inputs(
            layers=layers,
            calibration_data=[],
            use_cache=False,
        )

        call_kwargs = gptq_model.shell_module_materialize.call_args[1]
        self.assertIsNone(call_kwargs["module_path"])

    def test_cache_inputs_prefers_full_name_attribute(self):
        """A layer-level `full_name` attribute is respected when `layer_names` is absent."""

        layer = FakeLayer()
        layer.full_name = "custom.path.layer_0"
        capture, layers, _, gptq_model = self._make_capture(layer, gptq_model_model=FakeModel())

        capture.cache_inputs(
            layers=layers,
            calibration_data=[],
            use_cache=False,
        )

        call_kwargs = gptq_model.shell_module_materialize.call_args[1]
        self.assertEqual(call_kwargs["module_path"], "custom.path.layer_0")

    def test_cache_inputs_warns_when_caller_name_differs_from_full_name(self):
        """A caller-supplied `layer_names[0]` wins, but a mismatch against `full_name` is logged."""

        layer = FakeLayer()
        layer.full_name = "legacy.path.layer_0"
        capture, layers, layer_names, gptq_model = self._make_capture(
            layer,
            layer_names=["model.layers.42"],
            gptq_model_model=FakeModel(),
        )
        mock_logger = MagicMock()
        capture.logger = mock_logger

        capture.cache_inputs(
            layers=layers,
            calibration_data=[],
            use_cache=False,
            layer_names=layer_names,
        )

        call_kwargs = gptq_model.shell_module_materialize.call_args[1]
        self.assertEqual(call_kwargs["module_path"], "model.layers.42")
        mock_logger.warn.assert_called_once()
        message = mock_logger.warn.call_args[0][0]
        self.assertIn("model.layers.42", message)
        self.assertIn("legacy.path.layer_0", message)

    def test_forward_device_prefers_materialized_embedding_device(self):
        layer = FakeLayer()
        capture, _, _, gptq_model = self._make_capture(layer)
        embedding = nn.Embedding(8, 4)
        gptq_model.get_input_embeddings.return_value = embedding

        resolved = capture._resolve_forward_device(
            {"input_ids": torch.tensor([[1, 2, 3]])},
            fallback=torch.device("meta"),
        )

        self.assertEqual(resolved, embedding.weight.device)

    def test_forward_device_uses_fallback_for_unmaterialized_embedding(self):
        layer = FakeLayer()
        capture, _, _, gptq_model = self._make_capture(layer)
        gptq_model.get_input_embeddings.return_value = nn.Embedding(8, 4, device="meta")

        resolved = capture._resolve_forward_device(
            {"input_ids": torch.tensor([[1, 2, 3]])},
            fallback=torch.device("cpu"),
        )

        self.assertEqual(resolved, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
