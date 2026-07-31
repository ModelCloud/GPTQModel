# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import json
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, LlamaConfig  # noqa: E402

from gptqmodel.nn_modules.qlinear.torch import TorchLinear  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


class TestIntegration(unittest.TestCase):
    INFERENCE_PROMPT = "Which city is the capital of France? The city name is "
    INFERENCE_RESULT_KEYWORDS = ["paris", "eiffel", "country"]

    @classmethod
    def setUpClass(cls):
        pass

    def _test_load_quantized_model_gptq_v1(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        self.assertInference(model=model, tokenizer=tokenizer)

        del model
        torch_empty_cache()

    def _test_load_quantized_model_gptq_v2(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        self.assertInference(model=model, tokenizer=tokenizer)

        del model
        torch_empty_cache()

    def _test_quantize(self, device_map):
        model_id = "/monster/data/model/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = [
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, use_safetensors=False,
                                                               quantization_config=gptq_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            del quantized_model

            model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map)

            generate_str = ModelTest.generate_stable_with_limit(
                model,
                tokenizer,
                "gptqmodel is",
                max_new_tokens=30,
                skip_special_tokens=False,
            )

            self.assertIn("is a good", generate_str.lower())

            del model
            torch_empty_cache()

    def test_load_quantized_model_gptq_v1_torch_fused(self):
        self._test_load_quantized_model_gptq_v1(device_map="cpu")

    def test_load_quantized_model_gptq_v1_cuda(self):
        self._test_load_quantized_model_gptq_v1(device_map="cuda")

    def test_load_quantized_model_gptq_v2_torch_fused(self):
        self._test_load_quantized_model_gptq_v2(device_map="cpu")

    def test_load_quantized_model_gptq_v2_cuda(self):
        self._test_load_quantized_model_gptq_v2(device_map="cuda")

    def test_quantize_torch_fused(self):
        self._test_quantize(device_map="cpu")

    def test_quantize_cuda(self):
        self._test_quantize(device_map="cuda")

    def test_transformers_native_gptqmodel_load_bridge(self):
        from optimum.gptq import quantizer as optimum_quantizer

        if getattr(optimum_quantizer, "_gptqmodel_load_prepare_model", None) is None:
            self.skipTest("requires the Optimum native GPTQModel load bridge")

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        source_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        source_model.model.layers[0].self_attn.q_proj = TorchLinear(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            in_features=32,
            out_features=32,
            bias=False,
            pack_dtype=torch.int32,
            register_buffers=True,
            dtype=torch.float16,
            name="model.layers.0.self_attn.q_proj",
        )
        source_model.model.layers[0].self_attn.q_proj.scales.fill_(1)

        # The manifest keeps gate_proj dense even though dynamic rules mention it.
        quantization_config = {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
            "format": "gptq",
            "quant_method": "gptq",
            "dynamic": {
                r"+:^model\.layers\.0\.self_attn\.q_proj$": {"bits": 4, "group_size": 32},
                r"+:^model\.layers\.0\.mlp\.gate_proj$": {"bits": 4, "group_size": 32},
            },
            "meta": {"quantizer": ["gptqmodel:test"]},
        }
        source_model.config.quantization_config = GPTQConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            format="gptq",
            backend="gptq_torch",
            meta=quantization_config["meta"],
        ).to_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_model.save_pretrained(tmp_dir, safe_serialization=True)
            with open(os.path.join(tmp_dir, "quantize_config.json"), "w", encoding="utf-8") as config_file:
                json.dump(quantization_config, config_file)

            model = AutoModelForCausalLM.from_pretrained(
                tmp_dir,
                device_map={"": "cpu"},
                dtype=torch.float16,
                local_files_only=True,
            )

        q_proj = model.model.layers[0].self_attn.q_proj
        gate_proj = model.model.layers[0].mlp.gate_proj
        self.assertIsInstance(q_proj, TorchLinear)
        self.assertEqual(q_proj.group_size, 32)
        self.assertIsInstance(gate_proj, torch.nn.Linear)
        self.assertEqual(gate_proj.out_features, 48)
        self.assertEqual(model.config.quantization_config.dynamic, quantization_config["dynamic"])

        meta_tensors = [
            name
            for name, tensor in (*model.named_parameters(), *model.named_buffers())
            if tensor.is_meta
        ]
        self.assertEqual(meta_tensors, [])
        self.assertTrue(torch.isfinite(model.model.rotary_emb.inv_freq).all())

        del model
        del source_model
        torch_empty_cache()

    def assertInference(self, model, tokenizer=None, keywords=None, prompt=INFERENCE_PROMPT):
        # gptqmodel can auto init tokenizer internally
        if keywords is None:
            keywords = self.INFERENCE_RESULT_KEYWORDS
        if tokenizer is None:
            tokenizer = model.tokenizer

        generated = self.generate(model, tokenizer, prompt).lower()
        for k in keywords:
            if k.lower() in generated:
                self.assertTrue(True)
                return
        self.assertTrue(False, f"none of keywords were found in generated: {generated}")

    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = self.INFERENCE_PROMPT
        output = ModelTest.generate_stable_with_limit(
            model,
            tokenizer,
            prompt,
            min_new_tokens=10,
            max_new_tokens=30,
            skip_special_tokens=False,
        )
        print(f"Result is: >>\n{output}\n<<")
        return output

    def test_llm_awq(self):
        model_name = "ModelCloud/opt-125m-llm-awq"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        with torch.no_grad():
            result = ModelTest.generate_stable_with_limit(
                model,
                tokenizer,
                "The capital city of France is named",
                max_new_tokens=128,
            )
            print("result:", result)

            if "paris" not in result.lower() and "city" not in result.lower() and "food" not in result.lower() and "market" not in result.lower():
                raise AssertionError(" `paris` not found in `result`")
