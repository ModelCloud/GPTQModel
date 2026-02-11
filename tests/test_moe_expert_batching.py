import unittest
from unittest.mock import MagicMock, patch

import torch

from gptqmodel.looper.stage_subset import run_subset_stage
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    ExpertsRoutingOverride,
    GcMode,
    MoEConfig,
)


class TestMoEExpertBatching(unittest.TestCase):
    def setUp(self):
        self.looper = MagicMock()
        self.processor = MagicMock()
        self.module = MagicMock()
        self.layer_inputs = [MagicMock()]
        self.layer_input_kwargs = [MagicMock()]
        self.position_ids = [MagicMock()]
        self.attention_masks = [MagicMock()]
        self.cur_layer_device = torch.device("cpu")
        self.full = {}
        self.shared_kv_cache_dict = {}
        self.pb = MagicMock()

        # Setup config with ExpertsRoutingBypass routing strategy
        self.looper.gptq_model.quantize_config.moe = MoEConfig(routing=ExpertsRoutingBypass())
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = None
        self.looper.gptq_model.quantize_config.gc_mode = GcMode.ON_STAGE_END
        self.looper.gptq_model.quantize_config.auto_forward_data_parallel = False

        # Setup mocks
        self.looper._is_attention_module_name.return_value = False
        self.looper._extract_moe_group_key.return_value = "moe.experts"
        self.looper._moe_subset_threshold = 2
        # Mock device preparation to return proper torch.device
        self.looper._prepare_named_module_for_quantization.return_value = torch.device("cpu")
        self.looper._vram_strategy = None

        self.processor.name.return_value = "GPTQProcessor"
        self.processor.require_fwd = True
        # Mock processor tasks
        self.processor.tasks = {}
        # Explicitly set fwd_after_process to match GPTQProcessor default
        self.processor.fwd_after_process = True

        # Create fake subset
        self.subset = {f"expert.{i}": MagicMock() for i in range(10)}

        # Setup default return values
        self.looper._run_forward_batches.return_value = [torch.tensor([1.0])]
        self.looper._resolve_batch_total.return_value = 1
        self.looper._collect_row_counts.return_value = [1]

    def _run_subset_stage(self, subset):
        """Helper to run subset stage with given subset."""
        run_subset_stage(
            looper=self.looper,
            processor=self.processor,
            module=self.module,
            layer_inputs=self.layer_inputs,
            layer_input_kwargs=self.layer_input_kwargs,
            position_ids=self.position_ids,
            attention_masks=self.attention_masks,
            cur_layer_device=self.cur_layer_device,
            is_lm_head_module=False,
            layer_descriptor="layer.0",
            layer_title="title",
            layer_index=0,
            layers_prefix="model.layers",
            subset=subset,
            subset_index=0,
            subset_total=1,
            full=self.full,
            failsafe=False,
            shared_kv_cache_dict=self.shared_kv_cache_dict,
            pb=self.pb,
        )

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_no_batching_when_batch_size_is_none(self, mock_empty_cache):
        """When batch_size is None, all experts should be processed in one batch."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = None

        self._run_subset_stage(self.subset)

        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_no_batching_when_batch_size_is_zero(self, mock_empty_cache):
        """When batch_size is 0, batching should be disabled."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 0

        self._run_subset_stage(self.subset)

        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_batching_with_expert_groups(self, mock_empty_cache):
        """Test batching when modules are processed by module count."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 2

        # Create 10 experts with 2 modules each (gate_proj, up_proj)
        subset = {}
        for i in range(10):
            gate_name = f"model.layers.0.experts.{i}.gate_proj"
            up_name = f"model.layers.0.experts.{i}.up_proj"
            subset[gate_name] = MagicMock()
            subset[up_name] = MagicMock()

        # Mock group key extraction to return expert group key
        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}"
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key

        self._run_subset_stage(subset)

        # 20 total modules (10 experts × 2 modules) with batch_size 2 modules = 10 batches
        # Each batch calls _run_forward_batches once, and torch_empty_cache is called 3 times per batch
        # (once after forward pass, once after quant pass, once after chunk processing)
        self.assertEqual(self.looper._run_forward_batches.call_count, 10)
        self.assertEqual(mock_empty_cache.call_count, 30)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_batching_with_odd_number_of_experts(self, mock_empty_cache):
        """Test batching with odd number of experts that don't divide evenly."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 3

        # Create 7 experts
        subset = {}
        for i in range(7):
            name = f"model.layers.0.experts.{i}.gate_proj"
            subset[name] = MagicMock()

        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}"
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key

        self._run_subset_stage(subset)

        # 7 experts (modules) with batch_size 3 = 3 batches (3 + 3 + 1)
        # Each batch calls _run_forward_batches once, and torch_empty_cache is called 3 times per batch
        # (once after forward pass, once after quant pass, once after chunk processing)
        self.assertEqual(self.looper._run_forward_batches.call_count, 3)
        self.assertEqual(mock_empty_cache.call_count, 9)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_batching_when_batch_size_exceeds_expert_count(self, mock_empty_cache):
        """When batch_size > number of experts, all should be in one batch."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 100

        # Create 5 experts
        subset = {}
        for i in range(5):
            name = f"model.layers.0.experts.{i}.gate_proj"
            subset[name] = MagicMock()

        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}"
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key

        self._run_subset_stage(subset)

        # Should process all in one batch
        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_batching_one_expert_per_batch(self, mock_empty_cache):
        """Test with batch_size=1, meaning one module per batch."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 1

        # Create 4 experts
        subset = {}
        for i in range(4):
            name = f"model.layers.0.experts.{i}.gate_proj"
            subset[name] = MagicMock()

        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}"
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key

        self._run_subset_stage(subset)

        # 4 experts with batch_size 1 = 4 batches
        # Each batch calls _run_forward_batches once, and torch_empty_cache is called 3 times per batch
        # (once after forward pass, once after quant pass, once after chunk processing)
        self.assertEqual(self.looper._run_forward_batches.call_count, 4)
        self.assertEqual(mock_empty_cache.call_count, 12)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_no_batching_when_not_using_bypass_routing(self, mock_empty_cache):
        """When using a different routing strategy, batching should be disabled."""
        # Use ExpertsRoutingOverride instead of ExpertsRoutingBypass
        self.looper.gptq_model.quantize_config.moe = MoEConfig(
            routing=ExpertsRoutingOverride(num_experts_per_tok=2)
        )

        self._run_subset_stage(self.subset)

        # Should process all in one batch since no batch_size is available
        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_no_batching_when_moe_is_none(self, mock_empty_cache):
        """When moe config is None, batching should be disabled."""
        self.looper.gptq_model.quantize_config.moe = None

        self._run_subset_stage(self.subset)

        self.assertEqual(self.looper._run_forward_batches.call_count, 1)

    @patch('gptqmodel.looper.stage_subset.torch_empty_cache')
    def test_batching_with_non_expert_modules(self, mock_empty_cache):
        """Test batching when subset contains both expert and non-expert modules."""
        self.looper.gptq_model.quantize_config.moe.routing.batch_size = 2

        # Create 4 experts + 2 non-expert modules
        subset = {}
        for i in range(4):
            name = f"model.layers.0.experts.{i}.gate_proj"
            subset[name] = MagicMock()

        # Add non-expert modules
        subset["model.layers.0.norm"] = MagicMock()
        subset["model.layers.0.input_layernorm"] = MagicMock()

        def get_group_key(name):
            parts = name.split('.')
            if "experts" in parts:
                idx = parts.index("experts")
                return f"{'.'.join(parts[:idx+2])}"
            return None
        self.looper._extract_moe_group_key.side_effect = get_group_key

        self._run_subset_stage(subset)

        # 6 total modules (4 expert + 2 non-expert) with batch_size 2 modules = 3 batches
        self.assertEqual(self.looper._run_forward_batches.call_count, 3)

if __name__ == '__main__':
    unittest.main()
