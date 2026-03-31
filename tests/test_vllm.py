# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import importlib.util  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from tests.eval import evaluate, get_eval_task_metrics, import_evalution  # noqa: E402

from .models.model_test import ModelTest  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


class TestLoadVLLM(ModelTest):
    TASK_NAME = "arc_challenge"

    @classmethod
    def setUpClass(self):
        if ((importlib.util.find_spec("flashinfer") is None and importlib.util.find_spec("flashinfer-python") is None) or
                importlib.util.find_spec("vllm") is None):
            raise unittest.SkipTest(
                "flashinfer and vllm are required by this test. install via `pip install gptqmodel['vllm']`"
            )

        try:
            import vllm._C  # noqa: F401,E402
        except Exception as exc:
            raise unittest.SkipTest(f"vllm runtime unavailable: {exc}")
        try:
            import_evalution()
        except ValueError as exc:
            raise unittest.SkipTest(str(exc))
        self.MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.SHARDED_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-sharded"
        self.NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        for model_path in (self.MODEL_ID, self.SHARDED_MODEL_ID, self.NATIVE_MODEL_ID):
            if not Path(model_path).exists():
                raise unittest.SkipTest(f"missing local model path: {model_path}")

    def release_vllm_model(self):
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402
        except Exception:
            torch_empty_cache()
            return

        destroy_model_parallel()
        torch_empty_cache()

    def assert_evalution_vllm(self, model_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = evaluate(
                model_or_id_or_path=model_path,
                tasks=[self.TASK_NAME],
                batch_size=1,
                output_path=f"{tmp_dir}/result.json",
                llm_backend="vllm",
                model_args={
                    "enforce_eager": False,
                    "gpu_memory_utilization": 0.8,
                    "tensor_parallel_size": 1,
                },
                suite_kwargs={
                    "max_rows": 2,
                    "num_fewshot": 1,
                },
            )

        metrics = get_eval_task_metrics(results, self.TASK_NAME)
        self.assertTrue(metrics, f"Expected Evalution metrics for task {self.TASK_NAME}")
        self.assertEqual(results["engine"]["execution"]["generation_backend"], "vllm_generate")

    def test_evalution_vllm(self):
        try:
            self.assert_evalution_vllm(self.MODEL_ID)
        finally:
            self.release_vllm_model()

    def test_evalution_sharded_vllm(self):
        try:
            self.assert_evalution_vllm(self.SHARDED_MODEL_ID)
        finally:
            self.release_vllm_model()

    def test_dynamic(self):
        tokenizer = AutoTokenizer.from_pretrained(self.NATIVE_MODEL_ID, use_fast=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        calibration_dataset = self.load_dataset(tokenizer, self.DATASET_SIZE)

        # support dynamic override of bits, group_size, desc_act, sym for each layer/module match
        #
        dynamic = {
            # `.*\.` matches the layers_node prefix
            # layer index start at 0
            r"-:model\.layers\.0\..*": {},  # skip 0 layers
            r".*\.18\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 18 gate and up module
            r".*\.19\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 19 gate and up module
            r".*\.20\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 20 gate and up module
            r".*\.21\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 21 gate and up module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic=dynamic,
            group_size=128,
        )
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        model.quantize(calibration_dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model

            inspect_model = GPTQModel.load(tmp_dir)
            for name, submodule in inspect_model.named_modules():
                if name == 'model.model.layers.0.self_attn.q_proj' and isinstance(submodule, BaseQuantLinear):  # module 0 was skipped
                    raise ValueError("first layer should be native module")
            del inspect_model

            try:
                self.assert_evalution_vllm(tmp_dir)
            finally:
                self.release_vllm_model()
