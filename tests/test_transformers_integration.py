import tempfile
import unittest

from gptqmodel.integration import integration
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


class TestTransformersIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        integration.patch_hf()

    def _test_load_quantized_model_gptq_v1(self, device_map):
        model_id_or_path = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id_or_path,
                                                               device_map=device_map,)
        generate_str = tokenizer.decode(quantized_model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(quantized_model.device))[0])
        expect_str = "<s> The capital of France is is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is"
        self.assertEqual(generate_str[:50], expect_str[:50])

    def _test_load_quantized_model_gptq_v2(self, device_map):
        model_id_or_path = "/monster/data/model/opt-125m/quant/2024-12-02_13-28-10_subcircularly_autogptq_version_pr640_bit4_group128_seq2048_batch16/damp0.1_descTrue_gptq_v2_symTrue_pack_dataFalse_mseTrue_mse_norm2.4_mse_grid100_mse_maxshrink0.8/c40_gr0_dic0_sen0_det0_rate0_native0_lm_compression1024_text_reduction0/opt_125m_gptqv2"
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        quantized_model = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
                                                               device_map=device_map,)
        generate_str = tokenizer.decode(quantized_model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(quantized_model.device))[0])
        expect_str = "</s>The capital of France is is found velvetJustice ten for bowel Tuesday"

        self.assertEqual(generate_str[:len(expect_str)], expect_str)

    def _test_quantize(self, device_map):
        model_id = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = [
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map,
                                                               quantization_config=gptq_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            del quantized_model

            model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map)

            generate_str = tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0])

            expect_str = "</s>gptqmodel is a good way to get a good way for a good way for a good way."

            print('generate_str',generate_str)
            print('expect_str',expect_str)

            diff_count = len(set(generate_str.split()).symmetric_difference(expect_str.split()))

            self.assertLessEqual(diff_count, 2)

    def test_load_quantized_model_gptq_v1_ipex(self):
        self._test_load_quantized_model_gptq_v1(device_map="cpu")

    def test_load_quantized_model_gptq_v1_cuda(self):
        self._test_load_quantized_model_gptq_v1(device_map="cuda")

    def test_load_quantized_model_gptq_v2_ipex(self):
        self._test_load_quantized_model_gptq_v2(device_map="cpu")

    def test_load_quantized_model_gptq_v2_cuda(self):
        self._test_load_quantized_model_gptq_v2(device_map="cuda")

    def test_quantize_ipex(self):
        self._test_quantize(device_map="cpu")

    def test_quantize_cuda(self):
        self._test_quantize(device_map="cuda")
