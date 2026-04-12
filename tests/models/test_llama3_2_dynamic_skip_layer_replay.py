# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pcre
import pytest
import torch
import torch.nn as nn
from model_test import ModelTest
from safetensors import safe_open

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear


LAYER0_AND_LAYER2_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?!(?:0|2)\.)\d+\."
# Saved GPTQ checkpoints represent quantized linears with these tensor names.
GPTQ_TENSOR_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")
# Dynamically skipped layers must remain in native half precision on disk.
HALF_PRECISION_DTYPES = {"F16", "BF16"}
_LAYER_INDEX_RE = pcre.compile(r"\.layers\.(\d+)\.")
_QUANTIZED_TENSOR_RE = pcre.compile(
    r"^(model\.layers\.(\d+)\..*)\.(qweight|qzeros|scales|g_idx)$"
)


class TestLlama3_2DynamicSkipLayerReplay(ModelTest):
    """Exercise dynamic full-layer skips across a quantized -> skipped -> quantized chain."""

    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is required for Llama-3.2 GPTQ Marlin integration tests",
    )

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    LOAD_BACKEND = BACKEND.MARLIN
    KERNEL_INFERENCE = {MarlinLinear}
    DYNAMIC = {
        f"-:{LAYER0_AND_LAYER2_ONLY_NEGATIVE_MATCH}": {},
    }
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3987,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3234,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3643,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda",
            },
            "evalution_suite_kwargs": {
                "batch_size": 24,
                "max_new_tokens": 96,
                "stream": True,
                "max_rows": 128,
            },
            "acc,num": {
                "value": 0.390625,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3166,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3430,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    def _assert_dynamic_config_targets_only_layers_0_and_2(self, cfg) -> None:
        assert cfg.dynamic == self.DYNAMIC
        assert cfg.dynamic_get("model.layers.0.self_attn.q_proj") is not False
        assert cfg.dynamic_get("model.layers.1.self_attn.q_proj") is False
        assert cfg.dynamic_get("model.layers.2.self_attn.q_proj") is not False
        assert cfg.dynamic_get("model.layers.3.self_attn.q_proj") is False

    def _assert_only_layers_0_and_2_quantized(self, model) -> None:
        quantized_layer_names = {0: [], 1: [], 2: []}
        unexpected_quantized = []

        for name, module in model.named_modules():
            if not isinstance(module, BaseQuantLinear):
                continue

            layer_match = _LAYER_INDEX_RE.search(name)
            if layer_match is None:
                continue

            layer_idx = int(layer_match.group(1))
            if layer_idx in quantized_layer_names:
                quantized_layer_names[layer_idx].append(name)
            else:
                unexpected_quantized.append(name)

        assert quantized_layer_names[0], "Expected quantized modules in layer 0."
        assert not quantized_layer_names[1], (
            "Layer 1 should be fully skipped by QuantizeConfig.dynamic, "
            f"but found quantized modules: {quantized_layer_names[1][:8]}"
        )
        assert quantized_layer_names[2], "Expected quantized modules in layer 2."
        assert not unexpected_quantized, (
            "Only layers 0 and 2 should be quantized, "
            f"but found additional quantized modules: {unexpected_quantized[:8]}"
        )

    @staticmethod
    def _layer_index_from_module_name(module_name: str) -> int | None:
        """Return the transformer layer index encoded in a module/tensor name."""
        layer_match = _LAYER_INDEX_RE.search(module_name)
        if layer_match is None:
            return None
        return int(layer_match.group(1))

    @staticmethod
    def _saved_module_name_candidates(module_name: str) -> list[str]:
        """Generate candidate tensor prefixes for wrapped module names."""
        candidates = [module_name]
        trimmed_name = module_name
        while trimmed_name.startswith("model."):
            trimmed_name = trimmed_name[len("model."):]
            if trimmed_name:
                candidates.append(trimmed_name)
        return candidates

    def _resolve_saved_module_name(
        self,
        module_name: str,
        tensor_dtypes: dict[str, str],
        required_suffix: str,
    ) -> str:
        """Map an in-memory module path to the saved checkpoint tensor prefix."""
        for candidate in self._saved_module_name_candidates(module_name):
            if f"{candidate}.{required_suffix}" in tensor_dtypes:
                return candidate
        raise AssertionError(
            f"Could not resolve saved tensor prefix for `{module_name}` with suffix `{required_suffix}`."
        )

    def _collect_saved_safetensor_dtypes(self, model_path: str) -> dict[str, str]:
        """Read tensor dtype metadata from saved safetensor shards without loading weights."""
        shard_paths = sorted(Path(model_path).rglob("*.safetensors"))
        assert shard_paths, f"No safetensors shards found under `{model_path}`."

        tensor_dtypes = {}
        for shard_path in shard_paths:
            with safe_open(str(shard_path), framework="pt") as shard:
                for key in shard.keys():
                    tensor_dtypes[key] = str(shard.get_slice(key).get_dtype())
        return tensor_dtypes

    def _assert_saved_checkpoint_preserves_dynamic_layer_selection(self, model) -> None:
        """Verify the saved checkpoint only GPTQ-serializes layers 0 and 2."""
        model_path = self._resolve_quantized_model_path(model)
        assert model_path, "Expected the quantized model to expose a saved checkpoint path."

        tensor_dtypes = self._collect_saved_safetensor_dtypes(model_path)
        expected_quantized_layers = {0, 2}
        quantized_module_names = []
        native_linear_module_names = []

        for name, module in model.named_modules():
            layer_idx = self._layer_index_from_module_name(name)
            if layer_idx is None:
                continue

            if isinstance(module, BaseQuantLinear):
                quantized_module_names.append((name, layer_idx))
            elif isinstance(module, nn.Linear):
                native_linear_module_names.append((name, layer_idx))

        assert quantized_module_names, "Expected at least one quantized linear module in the saved model."
        assert native_linear_module_names, "Expected skipped layers to retain native linear weights in the saved model."

        unexpected_quantized_keys = []
        for tensor_name in tensor_dtypes:
            match = _QUANTIZED_TENSOR_RE.match(tensor_name)
            if match is None:
                continue
            layer_idx = int(match.group(2))
            if layer_idx not in expected_quantized_layers:
                unexpected_quantized_keys.append(tensor_name)
        assert not unexpected_quantized_keys, (
            "Only layers 0 and 2 should have GPTQ-style tensors on disk, "
            f"but found additional quantized tensors: {unexpected_quantized_keys[:8]}"
        )

        for module_name, layer_idx in quantized_module_names:
            assert layer_idx in expected_quantized_layers, (
                f"Only layers 0 and 2 should be quantized, but found `{module_name}` in layer {layer_idx}."
            )
            saved_module_name = self._resolve_saved_module_name(
                module_name,
                tensor_dtypes,
                required_suffix="qweight",
            )
            for suffix in GPTQ_TENSOR_SUFFIXES:
                tensor_key = f"{saved_module_name}.{suffix}"
                assert tensor_key in tensor_dtypes, (
                    f"Missing saved GPTQ tensor `{tensor_key}` for quantized module `{module_name}`."
                )

            assert tensor_dtypes[f"{saved_module_name}.scales"] in HALF_PRECISION_DTYPES, (
                f"Expected `{saved_module_name}.scales` to be saved in half precision, "
                f"but found `{tensor_dtypes[f'{saved_module_name}.scales']}`."
            )
            for suffix in ("qweight", "qzeros", "g_idx"):
                dtype_name = tensor_dtypes[f"{saved_module_name}.{suffix}"]
                assert dtype_name.startswith(("I", "U")), (
                    f"Expected `{saved_module_name}.{suffix}` to use an integer dtype, but found `{dtype_name}`."
                )
            assert f"{saved_module_name}.weight" not in tensor_dtypes, (
                f"Quantized module `{module_name}` should not be saved with a native `.weight` tensor."
            )

        for module_name, layer_idx in native_linear_module_names:
            assert layer_idx not in expected_quantized_layers, (
                f"Module `{module_name}` in quantized layer {layer_idx} unexpectedly remained native."
            )
            saved_module_name = self._resolve_saved_module_name(
                module_name,
                tensor_dtypes,
                required_suffix="weight",
            )
            tensor_key = f"{saved_module_name}.weight"
            assert tensor_key in tensor_dtypes, (
                f"Missing native weight tensor `{tensor_key}` for skipped module `{module_name}`."
            )
            assert tensor_dtypes[tensor_key] in HALF_PRECISION_DTYPES, (
                f"Expected `{tensor_key}` to remain in bf16/f16, but found `{tensor_dtypes[tensor_key]}`."
            )
            for suffix in GPTQ_TENSOR_SUFFIXES:
                unexpected_key = f"{saved_module_name}.{suffix}"
                assert unexpected_key not in tensor_dtypes, (
                    f"Skipped module `{module_name}` should not have saved GPTQ tensor `{unexpected_key}`."
                )

    def _run_dynamic_skip_replay_eval(self) -> None:
        cfg = self._build_quantize_config()
        self._assert_dynamic_config_targets_only_layers_0_and_2(cfg)

        self.model, _, _ = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
        )
        self.check_kernel(self.model, self.KERNEL_INFERENCE)
        self._assert_only_layers_0_and_2_quantized(self.model)
        self._assert_saved_checkpoint_preserves_dynamic_layer_selection(self.model)

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and len(eval_records) == 1 and target_backend in eval_records:
            task_results = eval_records[target_backend]
        else:
            task_results = self.evaluate_model(
                model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=self.DELETE_QUANTIZED_MODEL,
            )

        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)

    def test_llama3_2_dynamic_skip_layer_replay(self):
        self._run_dynamic_skip_replay_eval()
